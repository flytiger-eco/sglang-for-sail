"""Physical-pool FA3 QSA adapter correctness, including graph replay metadata."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.fa3 import (
    fa3_sparse_attention,
    quantize_q_fp8,
    resolve_fa3_qsa,
)


@pytest.fixture(autouse=True)
def strict_fp32_reference():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def require_fa3_qsa():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        resolve_fa3_qsa()
    except (ImportError, RuntimeError):
        pytest.skip("QSA-enabled FA3 wheel required")


def inputs(heads=6, kv_heads=1, prefix=0, dtype=torch.bfloat16):
    torch.manual_seed(41)
    device = "cuda"
    rows, width, topk = 8, 64, 19
    q = torch.randn(rows, heads, 256, device=device, dtype=dtype)
    k = torch.randn(160, kv_heads, 256, device=device, dtype=dtype)
    v = torch.randn_like(k)
    # Two independent requests, unequal lengths, non-contiguous physical pages.
    slots = torch.randperm(160, device=device)[: 2 * width].reshape(2, width).int()
    owners = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1], device=device, dtype=torch.int32)
    pos = torch.tensor([prefix+i for i in [0, 1, 2, 0, 1, 2, 3, 4]], device=device)
    idx = torch.randint(width, (rows, topk), device=device, dtype=torch.int32)
    idx[:, ::4] = -1  # Holes are compacted, not just counted.
    idx[:, 1] = pos.int()  # Every query has at least one valid key.
    idx[:, 2] = 65  # Out-of-range logical indices must not access another request.
    return q, k, v, idx, owners, slots, pos


def reference(q, k, v, idx, owners, slots, pos, scale):
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    result = torch.empty_like(q)
    for row in range(q.shape[0]):
        logical = idx[row].long()
        valid = (logical >= 0) & (logical <= pos[row]) & (logical < slots.shape[1])
        selected = slots[owners[row], logical[valid]].long()
        kk = k[selected].repeat_interleave(q.shape[1] // k.shape[1], dim=1).to(dtype)
        vv = v[selected].repeat_interleave(q.shape[1] // k.shape[1], dim=1).to(dtype)
        prob = (torch.einsum("hd,thd->ht", q[row].to(dtype), kk) * scale).softmax(-1)
        result[row] = torch.einsum("ht,thd->hd", prob, vv)
    return result


@pytest.mark.parametrize("heads,kv_heads", [(6, 1), (12, 2), (32, 1)])
@pytest.mark.parametrize("prefix", [0, 32])
def test_fa3_physical_qsa(heads, kv_heads, prefix):
    require_fa3_qsa()
    args = inputs(heads, kv_heads, prefix)
    actual = fa3_sparse_attention(*args, scale=0.07)
    expected = reference(*args, scale=0.07)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


def test_fa3_qsa_graph_replay_updates_selections():
    require_fa3_qsa()
    args = inputs(prefix=32)
    for _ in range(3):
        fa3_sparse_attention(*args, scale=0.0625)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fa3_sparse_attention(*args, scale=0.0625)
    q, k, v, idx, owners, slots, pos = args
    idx[:, 3::4] = -1
    pos.sub_(7)
    idx[:, 1] = pos.int()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual, reference(*args, scale=0.0625), atol=0.02, rtol=0.02
    )


@pytest.mark.parametrize("dtype", [torch.float16])
def test_fa3_qsa_rejects_unsupported_pool_dtype(dtype):
    require_fa3_qsa()
    q, k, v, *rest = inputs()
    with pytest.raises(ValueError, match="matching BF16 or E4M3"):
        fa3_sparse_attention(q, k.to(dtype), v.to(dtype), *rest, scale=0.0625)


def test_fa3_qsa_empty_selection_is_zero():
    require_fa3_qsa()
    args = inputs(prefix=32)
    args[3][0].fill_(-1)
    actual = fa3_sparse_attention(*args, scale=0.0625)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]))


def test_fa3_backend_graph_uses_real_request_table():
    require_fa3_qsa()
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )

    args = inputs(prefix=32)
    q, k, v, idx, owners, slots, pos = args
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _: k, get_value_buffer=lambda _: v
    )
    backend.req_to_token_pool = SimpleNamespace(req_to_token=slots)
    backend.forward_metadata = SimpleNamespace(
        token_to_batch_idx=torch.arange(q.shape[0], device=q.device),
        row_req_pool_indices=owners.clone(),
        token_slot_table=torch.zeros((q.shape[0], 1), device=q.device),
        is_cuda_graph=True,
    )
    batch = SimpleNamespace(positions=pos)
    layer = SimpleNamespace(layer_id=0, scaling=0.0625)
    for _ in range(3):
        backend._forward_fa3_attention(q, layer, batch, idx)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = backend._forward_fa3_attention(q, layer, batch, idx)
    # Reassign rows to other requests during replay, keeping graph pointers fixed.
    owners.copy_(1 - owners)
    backend.forward_metadata.row_req_pool_indices.copy_(owners)
    pos.sub_(4)
    idx[:, 1] = pos.int()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual, reference(*args, scale=0.0625), atol=0.02, rtol=0.02
    )


@pytest.mark.parametrize("heads,kv_heads", [(g * kv, kv) for g in range(1, 33) for kv in (1, 2, 3)])
@pytest.mark.parametrize("pool_size", [160, 192])
def test_fa3_fp8_kv_matches_quantized_reference(heads, kv_heads, pool_size):
    require_fa3_qsa()
    q, k, v, idx, owners, slots, pos = inputs(heads, kv_heads, prefix=32)
    if pool_size != k.shape[0]:
        k = torch.cat([k, k[: pool_size - k.shape[0]]]).contiguous()
        v = torch.cat([v, v[: pool_size - v.shape[0]]]).contiguous()
    k, v = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    # A wide range of Q scales makes omitted/incorrect descales visible.
    q[0].zero_()
    q[1].mul_(0.01)
    q[2].mul_(100)
    q8, descale = quantize_q_fp8(q, kv_heads)
    restored = (
        q8.cpu().double().reshape(q.shape[0], kv_heads, heads // kv_heads, 256)
        * descale.cpu().double()[:, :, None, None]
    ).reshape(q.shape)
    expected = reference(
        restored, k.cpu().double(), v.cpu().double(),
        idx.cpu(), owners.cpu(), slots.cpu(), pos.cpu(), scale=0.0625,
    )
    actual = fa3_sparse_attention(q, k, v, idx, owners, slots, pos, scale=0.0625)
    assert torch.isfinite(actual).all()
    # Same FP8 criterion as hopper/test_qsa_cases.py::_run_fp8_general_qsa.
    relative = (actual.cpu().double() - expected).norm() / expected.norm().clamp_min(1e-12)
    assert relative <= 0.1



@pytest.mark.parametrize("heads", [3, 6, 12])
def test_fa3_fp8_graph_replay_updates_query_scale(heads):
    require_fa3_qsa()
    q, k, v, *rest = inputs(heads=heads, prefix=32)
    k, v = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
    for _ in range(3):
        fa3_sparse_attention(q, k, v, *rest, scale=0.0625)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fa3_sparse_attention(q, k, v, *rest, scale=0.0625)
    q.mul_(3)
    graph.replay()
    torch.cuda.synchronize()
    expected = fa3_sparse_attention(q, k, v, *rest, scale=0.0625)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)


@pytest.mark.parametrize("heads,kv_heads", [(0, 1), (1, 0), (7, 2), (33, 1)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_fa3_rejects_invalid_head_layout(heads, kv_heads, dtype):
    require_fa3_qsa()
    q, k, v, *rest = inputs(heads, kv_heads, prefix=32)
    with pytest.raises(ValueError, match="positive Q/KV head counts"):
        fa3_sparse_attention(q, k.to(dtype), v.to(dtype), *rest, scale=0.0625)


@pytest.mark.parametrize("k_scale,v_scale", [(0.25, 2.0), (2.0, 0.125)])
@pytest.mark.parametrize("heads", [3, 6, 12])
def test_fa3_backend_preserves_fp8_kv_scales(k_scale, v_scale, heads):
    require_fa3_qsa()
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        QwenSparseAttnBackend,
    )

    q, k, v, idx, owners, slots, pos = inputs(heads=heads, prefix=32)
    k = (k / k_scale).to(torch.float8_e4m3fn)
    v = (v / v_scale).to(torch.float8_e4m3fn)
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _: k, get_value_buffer=lambda _: v
    )
    backend.forward_metadata = SimpleNamespace(
        token_to_batch_idx=owners, token_slot_table=slots, is_cuda_graph=False
    )
    layer = SimpleNamespace(
        layer_id=0, scaling=0.0625,
        k_scale_float=k_scale, v_scale_float=v_scale,
    )
    actual = backend._forward_fa3_attention(
        q, layer, SimpleNamespace(positions=pos), idx
    )
    # Powers-of-two scales can be folded into Q/output without extra rounding.
    # This controls for the native FP8 probability quantization while detecting
    # a missing or swapped per-layer KV descale at the backend boundary.
    expected = fa3_sparse_attention(
        q * k_scale, k, v, idx, owners, slots, pos, scale=0.0625
    ) * v_scale
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)


@pytest.mark.parametrize("heads", [3, 6, 12])
def test_fa3_fp8_preserves_constant_values(heads):
    require_fa3_qsa()
    q, k, v, *rest = inputs(heads=heads, prefix=32)
    k = k.to(torch.float8_e4m3fn)
    v = torch.ones_like(v).to(torch.float8_e4m3fn)
    out = fa3_sparse_attention(q, k, v, *rest, scale=0.0625)
    torch.testing.assert_close(out, torch.ones_like(out), atol=0.002, rtol=0.002)
