"""Store only incomplete QSA index-key groups after CP prompt extend."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "KS0", "KS2", "RS0", "RS1"])
def _store_pending_tail(
    keys, rope, lengths, extend_lengths, starts, requests, key_ring, rope_ring,
    N, D: tl.constexpr, R: tl.constexpr, RING_SIZE: tl.constexpr,
    KS0, KS2,
    RS0, RS1,
    ROPE_1D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_R: tl.constexpr,
):
    request = tl.program_id(0)
    length = tl.load(lengths + request)
    extend_length = tl.load(extend_lengths + request)
    start = tl.load(starts + request)
    request_slot = tl.load(requests + request)
    tail = length % R
    member = tl.arange(0, BLOCK_R)
    row = start + extend_length - tail + member
    position = length - tail + member
    slot = request_slot.to(tl.int64) * RING_SIZE + position % RING_SIZE
    valid = (member < tail) & (extend_length >= tail) & (row >= 0) & (row < N) & (request_slot > 0)
    dim = tl.arange(0, BLOCK_D)
    value = tl.load(keys + row[:, None] * KS0 + dim[None, :] * KS2,
                    valid[:, None] & (dim[None, :] < D), other=0)
    tl.store(key_ring + slot[:, None] * D + dim[None, :], value,
             valid[:, None] & (dim[None, :] < D))
    for axis in tl.static_range(3):
        if ROPE_1D:
            position = tl.load(rope + row * RS0, valid, other=0)
        else:
            position = tl.load(rope + axis * RS0 + row * RS1, valid, other=0)
        tl.store(rope_ring + slot * 3 + axis, position, valid)


def store_qsa_pending_tail(token_k, rope_positions, metadata, layer_id):
    """Keep each request's final 0..ratio-1 members, without touching dummy slots.

    Ordinary extend starts at a compression-group boundary. Completed groups
    read directly from token_k; only the unfinished tail must survive in the
    request ring. Existing batch tensors supply the compact layout on device.
    """
    pool = metadata.token_to_kv_pool
    if token_k.shape[0] == 0 or metadata.sequence_lengths.numel() == 0:
        return
    assert token_k.shape[1] == 1
    assert rope_positions.ndim in (1, 2)
    if rope_positions.ndim == 2:
        assert rope_positions.shape[0] == 3
    _store_pending_tail[(metadata.sequence_lengths.numel(),)](
        token_k, rope_positions, metadata.sequence_lengths,
        metadata.extend_seq_lens, metadata.extend_start_loc,
        metadata.req_pool_indices, pool.get_qsa_key_state_buffer(layer_id),
        pool.qsa_rope_position_buffer,
        N=token_k.shape[0], D=token_k.shape[2], R=metadata.compress_ratio,
        RING_SIZE=pool.qsa_ring_size,
        KS0=token_k.stride(0), KS2=token_k.stride(2),
        RS0=rope_positions.stride(0),
        RS1=rope_positions.stride(1) if rope_positions.ndim == 2 else 0,
        ROPE_1D=rope_positions.ndim == 1,
        BLOCK_D=triton.next_power_of_2(token_k.shape[2]),
        BLOCK_R=triton.next_power_of_2(metadata.compress_ratio), num_warps=4,
    )
