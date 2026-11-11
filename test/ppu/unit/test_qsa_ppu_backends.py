"""QSA ports of 645eb6b / 5684ecf: dispatch, quantization and sparse layouts."""

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as backend_module
from sglang.srt.layers.attention.qsa import mqa
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    QwenSparseAttnMetadata,
)
from sglang.srt.utils import get_device_sm, is_ppu


@pytest.fixture(autouse=True)
def clear_resolvers():
    avg = mqa._resolve_fp8_avg_mqa_logits
    fa3 = backend_module._resolve_ppu_fa3_qsa_with_kvcache
    avg.cache_clear()
    fa3.cache_clear()
    yield
    avg.cache_clear()
    fa3.cache_clear()


@pytest.mark.parametrize("sm,ppu", [(80, True), (90, True), (89, False)])
def test_avg_logits_rejects_unvalidated_devices(monkeypatch, sm, ppu):
    import sglang.srt.utils as utils

    monkeypatch.setattr(utils, "is_ppu", lambda: ppu)
    monkeypatch.setattr(utils, "get_device_sm", lambda: sm)
    assert mqa._resolve_fp8_avg_mqa_logits() is None


def test_missing_avg_logits_falls_back(monkeypatch):
    import sglang.srt.utils as utils

    monkeypatch.setattr(utils, "is_ppu", lambda: True)
    monkeypatch.setattr(utils, "get_device_sm", lambda: 89)
    monkeypatch.setitem(sys.modules, "deep_gemm", ModuleType("deep_gemm"))
    assert mqa._resolve_fp8_avg_mqa_logits() is None


@pytest.mark.parametrize(
    "heads,dim,scale",
    [(8, 128, None), (4, 32, None), (4, 256, None), (4, 128, 2.0)],
)
def test_unsupported_avg_logits_uses_original_scoring(monkeypatch, heads, dim, scale):
    monkeypatch.setenv("SGLANG_SAIL_QSA_INDEXER_FP8_AVG_LOGITS", "1")
    monkeypatch.setattr(mqa, "HAS_TILELANG", False)
    q = SimpleNamespace(is_cuda=True, ndim=3, shape=(2, heads, dim))
    expected = object()
    monkeypatch.setattr(mqa, "torch_qsa_mqa_prefill", lambda *args: expected)
    assert mqa.qsa_mqa_prefill(q, None, None, None, scale) is expected


def _require_890p():
    if not is_ppu() or get_device_sm() != 89:
        pytest.skip("requires a PPU-890P")


@pytest.mark.parametrize("width", [128, 512])
def test_fused_quant_zero_and_strided_rows(width):
    _require_890p()
    torch.manual_seed(41)
    x = torch.randn(17, width * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    x[0].zero_()
    quantized, scales = mqa._fp8_e4m3_per_row_quant(x)
    expected_scales = x.float().abs().amax(-1) / 448.0
    torch.testing.assert_close(scales, expected_scales)
    restored = quantized.float() * scales[:, None]
    assert torch.isfinite(restored).all()
    assert torch.count_nonzero(restored[0]) == 0
    # E4M3 has three mantissa bits; allow one rounding unit after scaling.
    torch.testing.assert_close(restored, x.float(), rtol=0.063, atol=1e-5)


@pytest.mark.parametrize(
    "rows,keys,dim",
    [(17, 91, 128), (33, 65, 64), (3, 7, 128), (129, 1024, 128)],
)
def test_avg_logits_matches_library_accuracy_contract(rows, keys, dim):
    _require_890p()
    if mqa._resolve_fp8_avg_mqa_logits() is None:
        pytest.fail("installed DeepGEMM does not provide fp8_mqa_avg_logits")
    torch.manual_seed(42)
    q = torch.randn(rows, 4, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(keys, 1, dim, device="cuda", dtype=torch.bfloat16)
    q[0].zero_()
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    starts[rows // 2 :] = keys // 2
    ends = starts + torch.arange(rows, device="cuda", dtype=torch.int32) % (keys // 2)

    # DeepGEMM's MQA tests compare the FP8 result to the original BF16 formula
    # with calc_diff < 1e-3. FP32 output storage does not promise pointwise
    # FP32 equivalence. Use an independent CPU FP64 oracle and retain the
    # measured pointwise errors as diagnostics instead of hiding them.
    def reference(q_cpu, k_cpu):
        scores = torch.einsum("mhd,nd->mnh", q_cpu, k_cpu)
        scores = torch.relu(scores).sum(-1) / math.sqrt(dim)
        columns = torch.arange(keys)[None, :]
        mask = (columns >= starts.cpu()[:, None]) & (columns < ends.cpu()[:, None])
        return scores.masked_fill(~mask, -float("inf")), mask

    expected, valid = reference(q.cpu().double(), k[:, 0].cpu().double())
    q8, qs = mqa._fp8_e4m3_per_row_quant(q.reshape(rows, -1))
    k8, ks = mqa._fp8_e4m3_per_row_quant(k[:, 0])
    quantized_reference, _ = reference(
        (q8.float().cpu().double() * qs.cpu().double()[:, None]).reshape_as(q),
        k8.float().cpu().double() * ks.cpu().double()[:, None],
    )
    actual = mqa.deep_gemm_fp8_avg_mqa_prefill(q, k, starts, ends)
    assert actual.dtype == torch.float32
    actual = actual.cpu().double()
    assert torch.equal(torch.isneginf(actual), ~valid)
    assert torch.isfinite(actual[valid]).all()
    x, y = actual.masked_fill(~valid, 0), expected.masked_fill(~valid, 0)
    denominator = (x.square() + y.square()).sum()
    diff = float((x - y).square().sum() / denominator) if denominator else 0.0
    print(
        {
            "shape": (rows, keys, dim),
            "library_calc_diff_vs_bf16": diff,
            "max_abs_vs_bf16": float((actual[valid] - expected[valid]).abs().max()),
            "max_abs_vs_quantized_fp64": float(
                (actual[valid] - quantized_reference[valid]).abs().max()
            ),
            "pointwise_rtol1e-4_atol1e-4_vs_quantized_fp64": torch.allclose(
                actual, quantized_reference, rtol=1e-4, atol=1e-4
            ),
        }
    )
    assert diff < 1e-3, f"DeepGEMM MQA accuracy contract violated: {diff}"


@pytest.mark.parametrize("rows,keys", [(0, 5), (3, 0)])
def test_avg_logits_empty_inputs(monkeypatch, rows, keys):
    monkeypatch.setattr(mqa, "_resolve_fp8_avg_mqa_logits", lambda: object())
    q = torch.zeros(rows, 4, 128)
    k = torch.zeros(keys, 1, 128)
    lengths = torch.zeros(rows, dtype=torch.int32)
    out = mqa.deep_gemm_fp8_avg_mqa_prefill(q, k, lengths, lengths)
    assert out.shape == (rows, keys)
    assert out.dtype == torch.float32


@pytest.mark.parametrize("per_query", [False, True])
def test_fa3_slot_mapping_handles_metadata_and_padding(monkeypatch, per_query):
    captured = {}

    def fake_fa3(**kwargs):
        captured.update(kwargs)
        return torch.zeros_like(kwargs["q"])

    monkeypatch.setattr(
        backend_module, "_resolve_ppu_fa3_qsa_with_kvcache", lambda: fake_fa3
    )
    backend = QwenSparseAttnBackend()
    # No row_req_pool_indices: current metadata's slot table is authoritative.
    mapping = torch.tensor([0, 0, 1], dtype=torch.int32)
    lengths = torch.tensor([3, 2], dtype=torch.int32)
    table = torch.tensor([[7, 2, 5], [1, 8, 0]], dtype=torch.int32)
    if per_query:
        table = table[mapping.long()]
        lengths = lengths[mapping.long()]
        mapping = torch.arange(3, dtype=torch.int32)
    metadata = QwenSparseAttnMetadata(
        sequence_lengths=lengths,
        token_to_batch_idx=mapping,
        token_slot_table=table,
        indexer_metadata=None,
    )
    indices = torch.tensor([[0, -1, 99], [2, 1, -1], [1, 0, 2]], dtype=torch.int32)
    q = torch.zeros(3, 4, 256, dtype=torch.bfloat16)
    kv = torch.zeros(9, 1, 256, dtype=q.dtype)
    cu_q = torch.tensor([0, 1, 2, 3] if per_query else [0, 2, 3], dtype=torch.int32)
    out = backend._forward_fa3_qsa_sparse(
        q,
        kv,
        kv,
        SimpleNamespace(scaling=1 / 16),
        metadata,
        indices,
        cu_q,
        1 if per_query else 2,
    )
    expected = torch.tensor([[7, -1, -1], [5, 2, -1], [8, 1, -1]], dtype=torch.int32)
    torch.testing.assert_close(captured["page_table"], expected.unsqueeze(1))
    assert captured["qsa_allow_aiu"] is False
    assert captured["cache_seqlens"] is lengths
    assert captured["cu_seqlens_q"] is cu_q
    assert out.shape == (3, 1024)


def test_fa3_quantized_cache_uses_original_backend(monkeypatch):
    monkeypatch.setenv("SGLANG_SAIL_QSA_FA3_SPARSE_ATTN", "1")
    monkeypatch.setattr(backend_module, "is_ppu", lambda: True)
    q = torch.zeros(1, 4, 256, dtype=torch.bfloat16)
    kv = torch.empty(2, 1, 256, dtype=torch.float8_e4m3fn)
    assert not QwenSparseAttnBackend()._fa3_qsa_supported(
        q, SimpleNamespace(head_dim=256, tp_q_head_num=4), kv, kv
    )


@pytest.mark.parametrize("mode", ["decode", "extend"])
def test_fa3_sparse_matches_reference(mode):
    _require_890p()
    if backend_module._resolve_ppu_fa3_qsa_with_kvcache() is None:
        pytest.skip("installed FA3 is not built with QSA support")
    torch.manual_seed(43)
    device = "cuda"
    rows = 3
    q = torch.randn(rows, 4, 256, device=device, dtype=torch.bfloat16)
    k = torch.randn(11, 1, 256, device=device, dtype=q.dtype)
    v = torch.randn_like(k)
    indices = torch.tensor(
        [[0, -1, -1], [1, 0, -1], [2, 0, 1]], device=device, dtype=torch.int32
    )
    slots = torch.tensor([7, 2, 9], device=device, dtype=torch.int32)
    extend = mode == "extend"
    metadata = QwenSparseAttnMetadata(
        sequence_lengths=torch.tensor(
            [3] if extend else [1, 2, 3], device=device, dtype=torch.int32
        ),
        token_to_batch_idx=(
            torch.zeros(rows, device=device, dtype=torch.int32)
            if extend
            else torch.arange(rows, device=device, dtype=torch.int32)
        ),
        token_slot_table=slots[None, :].expand(1 if extend else rows, -1),
        indexer_metadata=None,
    )
    cu_q = torch.tensor(
        [0, 3] if extend else [0, 1, 2, 3], device=device, dtype=torch.int32
    )
    scale = 1 / math.sqrt(256)
    actual = QwenSparseAttnBackend()._forward_fa3_qsa_sparse(
        q,
        k,
        v,
        SimpleNamespace(scaling=scale),
        metadata,
        indices,
        cu_q,
        rows if extend else 1,
    )
    expected = []
    for row in range(rows):
        selected = slots[indices[row][indices[row] >= 0].long()].long()
        scores = (
            torch.einsum("hd,nd->hn", q[row].float(), k[selected, 0].float()) * scale
        )
        expected.append(torch.softmax(scores, -1) @ v[selected, 0].float())
    expected = torch.stack(expected).reshape(rows, -1).to(q.dtype)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
