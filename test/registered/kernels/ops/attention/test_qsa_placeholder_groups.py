"""Regression coverage for padded QSA compression groups and short extends."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.qsa_indexer import qsa_index_k_compress_store
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

RATIO = 4
DIM = 128
SENTINEL = 17.0
CASES = [
    ((1,), (0,)),
    ((2,), (0,)),
    ((3,), (0,)),
    ((4,), (0,)),
    ((64,), (0,)),
    ((1,), (8192,)),
    ((2,), (8192,)),
    ((3,), (8192,)),
    ((1, 2, 3, 4, 7), (0, 0, 0, 0, 0)),
    ((1, 4, 3), (8192, 8192, 8192)),
]


def _plan(lengths, prefixes):
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = SimpleNamespace(qsa_compress_ratio=RATIO)
    total = sum(lengths)
    width = max(n + p for n, p in zip(lengths, prefixes))
    stride = (width + 63) // 64 * 64
    table = torch.arange(
        64, 64 + len(lengths) * stride, device="cuda", dtype=torch.int32
    ).reshape(len(lengths), stride)
    writes, positions, rows, members = backend._qsa_build_write_plan(
        forward_batch=SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: False),
            extend_seq_lens=torch.tensor(lengths, device="cuda"),
            input_ids=torch.zeros(total, device="cuda", dtype=torch.long),
        ),
        speculative_paged=False,
        token_slot_table=table,
        sequence_lengths=torch.tensor(
            [n + p for n, p in zip(lengths, prefixes)], device="cuda"
        ),
    )
    groups = (members[:, None] + torch.arange(RATIO, device="cuda")).int()
    torch.manual_seed(total)
    keys = torch.randn(total, DIM, device="cuda", dtype=torch.bfloat16)
    rope = torch.zeros(total, 3, device="cuda", dtype=torch.int64)
    output = torch.full(
        (int(writes.max().item()) + 2, DIM),
        SENTINEL,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return keys, rope, groups, writes, output, positions, rows, members


def _kernel_args(keys, rope, groups, writes, output):
    cos_sin = torch.cat(
        [torch.ones(1, DIM // 2), torch.zeros(1, DIM // 2)], dim=1
    ).cuda()
    return (
        keys,
        groups,
        rope,
        cos_sin,
        torch.zeros(DIM // 2, device="cuda", dtype=torch.int32),
        torch.zeros(DIM, device="cuda", dtype=keys.dtype),
        writes,
        output,
        RATIO,
        DIM,
        1e-6,
        True,
    )


def _assert_fused_result(keys, groups, writes, output):
    valid = writes != 0
    expected = torch.full_like(output, SENTINEL)
    pooled = keys[groups[valid].long()].float().mean(dim=1).to(keys.dtype)
    normalized = pooled.float() * torch.rsqrt(
        pooled.float().square().mean(dim=-1, keepdim=True) + 1e-6
    )
    expected[writes[valid].long()] = normalized.to(keys.dtype)
    torch.testing.assert_close(output, expected, rtol=0.01, atol=0.015625)
    assert torch.all(output[0] == SENTINEL).item()


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 64])
def test_fused_reserved_group(rows):
    keys, rope, groups, writes, output, *_ = _plan((rows,), (0,))
    qsa_index_k_compress_store(*_kernel_args(keys, rope, groups, writes, output))
    torch.cuda.synchronize()
    _assert_fused_result(keys, groups, writes, output)


@pytest.mark.parametrize("lengths,prefixes", CASES)
def test_fused_planner_and_graph_replay(lengths, prefixes):
    keys, rope, groups, writes, output, *_ = _plan(lengths, prefixes)
    args = _kernel_args(keys, rope, groups, writes, output)
    # Warm the exact specialization before capture, then exercise changing
    # device-side validity with fixed tensor shapes on the same graph.
    qsa_index_k_compress_store(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qsa_index_k_compress_store(*args)
    output.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    _assert_fused_result(keys, groups, writes, output)
    writes.zero_()
    groups.fill_(2**30)
    output.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(output == SENTINEL).item()


def test_fused_placeholder_needs_no_source_storage():
    # Empty source allocations make any accidental source read invalid,
    # independently of allocator padding or sanitizer availability.
    keys = torch.empty(0, DIM, device="cuda", dtype=torch.bfloat16)
    rope = torch.empty(0, 3, device="cuda", dtype=torch.int64)
    groups = torch.full((5, RATIO), 2**30, device="cuda", dtype=torch.int32)
    writes = torch.zeros(5, device="cuda", dtype=torch.int32)
    output = torch.full((2, DIM), SENTINEL, device="cuda", dtype=keys.dtype)
    qsa_index_k_compress_store(*_kernel_args(keys, rope, groups, writes, output))
    torch.cuda.synchronize()
    assert torch.all(output == SENTINEL).item()


@pytest.mark.parametrize("lengths,prefixes", CASES)
@pytest.mark.parametrize("graph_mode", [False, True])
def test_fallback_placeholder_gathers(lengths, prefixes, graph_mode):
    keys, rope, groups, writes, output, positions, rows, members = _plan(
        lengths, prefixes
    )
    keys = keys[:, None, :]
    output = output[:, None, :]
    pool = SimpleNamespace(
        qsa_rope_position_buffer=rope,
        get_qsa_key_state_buffer=lambda layer: keys,
        set_qsa_compressed_k_buffer=lambda layer, loc, value: output.index_copy_(
            0, loc.long(), value
        ),
    )
    indexer = SimpleNamespace(
        layer_id=0,
        compress_ratio=RATIO,
        _use_fused_compress=lambda pool: False,
        _rope_from_matrix=lambda value: value.T,
        normalize_compressed_keys=lambda pooled, positions: pooled,
    )
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        is_cuda_graph=False,
        write_locs=writes,
        compress_member_rows=members,
        compress_group_positions=positions,
        compress_sequence_ids=rows,
        extend_rope_matrix=rope,
        graph_write_locs=writes,
        graph_ring_group_locs=groups,
    )
    if graph_mode:
        groups[writes == 0] = 2**30
        run = lambda: QSAIndexer._compress_decode_cuda_graph(indexer, metadata)
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graph.replay()
    else:
        QSAIndexer.update_key_state_and_compress(
            indexer, keys, None, None, metadata, state_stored=True
        )
    torch.cuda.synchronize()
    valid = writes != 0
    expected = keys[groups[valid].long()].float().mean(dim=1).to(keys.dtype)
    torch.testing.assert_close(output[writes[valid].long()], expected, rtol=0, atol=0)
