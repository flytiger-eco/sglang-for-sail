"""DeepseekV4 CSA prefill via FlashMLA sparse forward.

Ported from vLLM's `vllm/v1/attention/ops/deepseek_v4_ops/cache_utils.py` and
`vllm/model_executor/layers/deepseek_v4_attention.py:_forward_prefill`.

The SGLang DSv4 paged FP8 K cache uses the same per-token byte layout as vLLM
(448 fp8 nope + 128 bf16 rope + 8 ue8m0 scales with a 1-byte pad), and the same
intra-page layout (page_size*576 token bytes followed by page_size*8 scale
bytes, padded to 576-byte multiples) — see
`python/sglang/srt/layers/attention/dsv4/index_buf_accessor.py` writer and the
assertion at `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py:104-117`.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.nsa.triton_kernel import _supports_fp8

# FlashMLA sparse prefill kernel asserts `params.topk % B_TOPK == 0`. B_TOPK is
# 64 for the h_q=64 kernel and 128 for h_q=128; pad to 128 to satisfy both.
# Extra slots stay as -1 sentinels and combined_lens caps the valid range via
# `topk_length`, so padding is a no-op at kernel level.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128

# Bound the bf16 KV gather workspace by chunking prefill requests.
PREFILL_CHUNK_SIZE = 4


@triton.jit
def _fused_fp8_ue8m0_dequant(x_uint8, ue8m0_uint8):
    """Fused FP8 e4m3fn + UE8M0 scale → float32 via IEEE754 bit construction.

    Merges the FP8 exponent and UE8M0 scale exponent into a single integer
    add, producing float32 directly — no tl.exp2 or float multiply.

    Args:
        x_uint8: FP8 e4m3fn bytes (SEEEEMMM, bias=7).
        ue8m0_uint8: UE8M0 scale byte (value = 2^(ue8m0 - 127)), scalar
                     broadcast across the block.
    Returns:
        float32 tensor = fp8_value * ue8m0_scale.

   Preconditions (caller's responsibility):
        1 <= fp8_exp + ue8m0 - 7 <= 254 for nonzero values. Amax-based
        quantizers satisfy this by construction. Out-of-range ue8m0
        (e.g. corrupted cache bytes) is NOT detected and yields garbage
        bit patterns — rule this out first when debugging.

    Numerical notes:
        - Exactly bit-identical to ``fp8.to(f32) * exp2(ue8m0 - 127)`` for
          all FP8 *normal* values: multiplying by a power of two only
          shifts the exponent and introduces no rounding.
        - FP8 subnormals/zero (exp_bits == 0) are flushed to zero (FTZ).
          Max e4m3 subnormal is 0.875 * 2^-6 ≈ 0.0137, i.e. ~3e-5 of
          fp8_max=448 —- negligible but deliberate divergence from native FP8 hardware decode.
        - e4m3fn NaN codepoints (0x7F / 0xFF) decode to a *finite* value
          of ~480 * scale instead of NaN (this trick treats exp=15,
          mant=7 as a normal number). 

    """
    x_i32 = x_uint8.to(tl.int32)
    sign = x_i32 >> 7
    exp = (x_i32 >> 3) & 0xF
    mant = x_i32 & 0x7

    ue8m0 = ue8m0_uint8.to(tl.int32)
    # f32 exponent field = fp8_exp + ue8m0 - 7  (bias: 134 - 127)
    f32_bits = (sign << 31) | ((exp + ue8m0 - 7) << 23) | (mant << 20)
    result = f32_bits.to(tl.float32, bitcast=True)
    # FP8 subnormal/zero (exp==0): negligibly small in quantized KV cache.
    return tl.where(exp != 0, result, 0.0)


@triton.jit
def _dequantize_and_gather_k_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    max_blocks_per_seq: tl.constexpr,
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8 (incl. 1 pad)
    quant_block: tl.constexpr,  # 64
    cache_block_size: tl.constexpr,  # swa=128 / c4=64
    token_data_size: tl.constexpr,  # 576
    block_stride: tl.constexpr,  # bytes per cache page
    output_dim: tl.constexpr,  # 512
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,  # 7
    USE_FP8_NATIVE: tl.constexpr = True,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        pos = start_pos + i

        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)

        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size

        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )

        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim

        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block

            if qblock_start < fp8_dim:
                offsets = qblock_start + tl.arange(0, quant_block)
                mask = offsets < fp8_dim

                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)

                if USE_FP8_NATIVE:
                    x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
                    x_float = x_fp8.to(tl.float32)

                    encoded_scale = tl.load(token_scale_ptr + qblock_idx)
                    exponent = encoded_scale.to(tl.float32) - 127.0
                    scale = tl.exp2(exponent)

                    x_dequant = x_float * scale
                else:
                    ue8m0 = tl.load(token_scale_ptr + qblock_idx)
                    x_dequant = _fused_fp8_ue8m0_dequant(x_uint8, ue8m0)

                tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

        bf16_output_offset = fp8_dim
        bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))

        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


def dequantize_and_gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: Optional[torch.Tensor],
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    """Gather + dequantize per-request slices of a DSv4 paged FP8 K cache.

    Args:
        out: [num_reqs, max_num_tokens, 576] bf16.
        k_cache: [num_blocks, block_bytes] uint8 (raw pool buffer).
        seq_lens: [num_reqs] int32, total seq len in cache pool units.
        gather_lens: [num_reqs] int32 or None. If None, gather full seq_lens.
        block_table: [num_reqs, max_blocks_per_seq] int32, physical page ids.
        block_size: tokens per cache page (swa=128, c4=64).
        offset: starting column in `out` for the gathered tokens.
    """
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    FP8_MAX = 448.0
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=k_cache.stride(0),
        output_dim=512,
        fp8_max=FP8_MAX,
        n_quant_blocks=7,
        USE_FP8_NATIVE=_supports_fp8(),
    )


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        offset = tl.arange(0, PADDED_TOP_K)
        mask = offset < topk_len
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
        )
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + offset,
            topk_indices + M * batch_idx,
            mask=mask,
        )
        offset = tl.arange(0, WINDOW_SIZE)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + offset,
            M * batch_idx + N + offset + pos - swa_len + 1 - gather_start,
            mask=offset < swa_len,
        )

        combined_len = topk_len + swa_len
        tl.store(combined_lens_ptr + token_idx, combined_len)


@triton.jit
def _combine_full_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_MAX_TOPK: tl.constexpr,
):
    """C128 (HCA) variant of combine_topk_swa: no topk selection; every
    compressed token in [0, (pos+1)//COMPRESS_RATIO) is attended to. We
    synthesize the sequential indices instead of loading from a topk_indices
    tensor."""
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        # Full enumeration: every causally-visible compressed token.
        topk_len = (pos + 1) // COMPRESS_RATIO
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        # Sequential compressed indices: M*batch + [0, topk_len)
        offset = tl.arange(0, PADDED_MAX_TOPK)
        mask = offset < topk_len
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + offset,
            M * batch_idx + offset,
            mask=mask,
        )
        # SWA window indices (identical layout to combine_topk_swa).
        offset = tl.arange(0, WINDOW_SIZE)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + offset,
            M * batch_idx + N + offset + pos - swa_len + 1 - gather_start,
            mask=offset < swa_len,
        )

        combined_len = topk_len + swa_len
        tl.store(combined_lens_ptr + token_idx, combined_len)


def combine_full_swa_indices(
    num_q_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    max_topk_len: int,
    M: int,
    N: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate full-enumeration compressed indices and SWA window indices
    for C128 (HCA) prefill.

    `max_topk_len` is the per-request upper bound on `(pos+1) // compress_ratio`,
    typically `ceil(max_model_len / compress_ratio)` (= N).
    """
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (max_topk_len + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_q_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )
    combined_lens = torch.empty(
        num_q_tokens,
        dtype=torch.int32,
        device=device,
    )

    NUM_WORKERS = 128
    _combine_full_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        PADDED_MAX_TOPK=triton.next_power_of_2(max(max_topk_len, 1)),
    )
    return combined_indices, combined_lens


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-token topk compressed indices and SWA window indices.

    Returns (combined_indices, combined_lens) where combined_indices points
    into the chunk's bf16 KV workspace (shape `[chunk, M, 576]` viewed flat as
    `[chunk*M, 1, 576]`).
    """
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


def forward_prefill_sparse(
    *,
    q: torch.Tensor,
    attn_sink: torch.Tensor,
    sm_scale: float,
    output: torch.Tensor,
    swa_k_cache: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    local_topk_indices: Optional[torch.Tensor],
    prefill_seq_lens: torch.Tensor,
    prefill_gather_lens: torch.Tensor,
    prefill_query_start_loc: torch.Tensor,
    prefill_query_start_loc_cpu: list[int],
    compressed_block_table: torch.Tensor,
    swa_block_table: torch.Tensor,
    compressed_block_size: int,
    swa_block_size: int,
    compress_ratio: int,
    window_size: int,
    sparse_topk: Optional[int],
    max_model_len: int,
    max_seq_len_in_batch: Optional[int] = None,
    max_qo_len_in_batch: Optional[int] = None,
    attn_tp_rank: int = 0,
    attn_tp_size: int = 1,
    chunk_size: int = PREFILL_CHUNK_SIZE,
) -> None:
    """DSv4 CSA / HCA prefill attention via FlashMLA sparse forward.

    Mirrors `vllm/.../deepseek_v4_attention.py:_forward_prefill`. Writes the
    attention output in-place into `output` (shape (T, H, head_dim_v=512) bf16).

    Two compression modes supported:
      - C4 (CSA, compress_ratio=4): pass `local_topk_indices` (LOGICAL c4
        positions emitted by the indexer via `out_raw_indices`) and
        `sparse_topk` (e.g. 512 or 1024). Each query token attends to the
        topk-selected compressed tokens + SWA window.
      - C128 (HCA, compress_ratio=128): pass `local_topk_indices=None` and
        `sparse_topk=None`. Each query token attends to the full prefix of
        compressed tokens [0, (pos+1)//128) + SWA window — no topk selection.

    Args:
        q: (T, H, D=512) bf16.
        attn_sink: (n_heads_global,) float32 — full global sink, will be
            sliced to the local TP rank's heads.
        output: (T, H, 512) bf16, pre-allocated.
        swa_k_cache: raw SWA pool FP8 buffer (num_swa_blocks, swa_page_bytes) uint8.
        compressed_k_cache: raw C4 / C128 pool FP8 buffer.
        local_topk_indices: (T, sparse_topk) int32 of LOGICAL compressed
            positions for C4; None for C128 full enumeration.
        prefill_seq_lens: (num_prefill_reqs,) int32 full seq lens.
        prefill_gather_lens: (num_prefill_reqs,) int32 = min(seq_len, qo_len+window-1).
        prefill_query_start_loc: (num_prefill_reqs+1,) int32 cumulative qo lens.
        compressed_block_table: (num_prefill_reqs, max_blocks) int32 — pool
            page ids; for both C4 and C128 this is reusable from
            core_attn_metadata.page_table directly because every main pool
            256-token page corresponds to one C4 page (64 token) or one C128
            page (2 token), sharing the same page id.
        swa_block_table: (num_prefill_reqs, max_swa_blocks) int32.
        compressed_block_size: tokens per page in the compressed pool
            (page_size // compress_ratio = 64 for C4, 2 for C128).
        swa_block_size: tokens per SWA page (= 128).
        compress_ratio: 4 (C4 / CSA) or 128 (C128 / HCA).
        window_size: SWA window (128).
        sparse_topk: cap on topk_len for C4; ignored for C128.
        max_model_len: upper bound on max_seq_len, used to size the workspace.
    """
    import flash_mla

    assert compress_ratio in (
        4,
        128,
    ), f"only C4 (4) or C128 (128) supported, got {compress_ratio=}"
    if compress_ratio == 4:
        assert (
            local_topk_indices is not None and sparse_topk is not None
        ), "C4 prefill requires local_topk_indices + sparse_topk"
    else:
        assert (
            local_topk_indices is None
        ), "C128 (HCA) prefill must not pass local_topk_indices"

    num_prefill_reqs = prefill_seq_lens.shape[0]
    if num_prefill_reqs == 0:
        return

    # head_dim is the BF16 element count (= qk_nope_head_dim 448 + qk_rope_head_dim 64
    # = 512 for DSv4). Don't confuse this with the FP8 cache's 576-byte per-token
    # layout (448B FP8 nope + 128B BF16 rope + 8B UE8M0 scales) — after dequant
    # we end up with 512 BF16 elements per token.
    head_dim = q.shape[-1]
    assert (
        head_dim == 512
    ), f"expected DSv4 MQA head_dim=512 (nope 448 + rope 64), got {head_dim=}"

    device = q.device

    # Size workspace by the actual batch maxima when available, falling back
    # to max_model_len. With long-context models this cuts the workspace by
    # >10x for short-input workloads (e.g. 8K input vs 128K model context),
    # which is the difference between fitting and OOMing under tight
    # mem-fraction-static budgets.
    seq_len_for_ws = (
        min(max_seq_len_in_batch, max_model_len)
        if max_seq_len_in_batch is not None
        else max_model_len
    )
    qo_len_for_ws = (
        min(max_qo_len_in_batch, max_model_len)
        if max_qo_len_in_batch is not None
        else max_model_len
    )

    # Compressed-region pool size per request — must hold every compressed
    # token of the longest request (seq_len_for_ws // compress_ratio).
    N = (seq_len_for_ws + compress_ratio - 1) // compress_ratio

    # M bounds the concatenated workspace per request: compressed region (N) +
    # window (SWA) + the longest qo extent in this batch.
    M = N + window_size + qo_len_for_ws

    # Workspace: chunk of bf16 KV. Allocated once per call; the allocator will
    # reuse this hot path's memory across layers.
    kv = torch.empty(
        (chunk_size, M, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )

    num_chunks = (num_prefill_reqs + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * chunk_size
        c_end = min(c_start + chunk_size, num_prefill_reqs)
        cs = c_end - c_start

        # Gather compressed (C4 / C128) KV → kv[:cs, 0:N) of workspace
        dequantize_and_gather_k_cache(
            out=kv[:cs],
            k_cache=compressed_k_cache,
            seq_lens=prefill_seq_lens[c_start:c_end] // compress_ratio,
            gather_lens=None,
            block_table=compressed_block_table[c_start:c_end],
            block_size=compressed_block_size,
            offset=0,
        )

        # Gather SWA KV → kv[:cs, N:N+gather_len)
        dequantize_and_gather_k_cache(
            out=kv[:cs],
            k_cache=swa_k_cache,
            seq_lens=prefill_seq_lens[c_start:c_end],
            gather_lens=prefill_gather_lens[c_start:c_end],
            block_table=swa_block_table[c_start:c_end],
            block_size=swa_block_size,
            offset=N,
        )

        # Combine compressed + SWA indices per query token, rebased to workspace.
        q_start = prefill_query_start_loc_cpu[c_start]
        q_end = prefill_query_start_loc_cpu[c_end]

        if local_topk_indices is not None:
            # C4 (CSA) — read topk indices from the indexer.
            combined_indices, combined_lens = combine_topk_swa_indices(
                local_topk_indices[q_start:q_end],
                prefill_query_start_loc[c_start : c_end + 1],
                prefill_seq_lens[c_start:c_end],
                prefill_gather_lens[c_start:c_end],
                window_size,
                compress_ratio,
                sparse_topk,
                M,
                N,
            )
        else:
            # C128 (HCA) — synthesize sequential [0, (pos+1)//128) indices.
            combined_indices, combined_lens = combine_full_swa_indices(
                num_q_tokens=q_end - q_start,
                query_start_loc=prefill_query_start_loc[c_start : c_end + 1],
                seq_lens=prefill_seq_lens[c_start:c_end],
                gather_lens=prefill_gather_lens[c_start:c_end],
                window_size=window_size,
                compress_ratio=compress_ratio,
                max_topk_len=N,
                M=M,
                N=N,
                device=device,
            )

        q_chunk = q[q_start:q_end]
        # q_chunk: (q_end - q_start, h_q_kernel = n_local, head_dim) bf16
        # attn_sink: (n_local,) float32 — already sliced to local-rank heads

        flash_mla.flash_mla_sparse_fwd(
            q=q_chunk,
            kv=kv.view(-1, 1, head_dim),
            indices=combined_indices.unsqueeze(1),
            sm_scale=sm_scale,
            attn_sink=attn_sink,
            topk_length=combined_lens,
            out=output[q_start:q_end],
        )
