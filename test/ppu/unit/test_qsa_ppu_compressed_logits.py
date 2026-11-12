"""Latest PPU AVG ABI, compressed coordinates and bounded strided top-k."""

import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa import mqa
from sglang.srt.layers.attention.qsa import qsa_indexer as indexer_module
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
from sglang.srt.utils import get_device_sm, is_sail


@pytest.mark.parametrize("legacy", [False, True])
def test_resolver_checks_native_schema(monkeypatch, legacy):
    import sglang.srt.utils as utils

    monkeypatch.setattr(utils, "is_sail", lambda: True)
    monkeypatch.setattr(utils, "get_device_sm", lambda: 89)
    native = lambda: None
    native.__doc__ = (
        "fp8_mqa_avg_logits(q, kv_s, ks, ke, clean_logits=True, max_seqlen_k=0)"
        if legacy
        else "fp8_mqa_avg_logits(q, kv_s, ks, ke, max_seqlen_k, q_scale=None)"
    )
    module = ModuleType("deep_gemm")
    module.fp8_mqa_avg_logits = native
    monkeypatch.setitem(sys.modules, "deep_gemm", module)
    mqa._resolve_fp8_avg_mqa_logits.cache_clear()
    try:
        assert mqa._resolve_fp8_avg_mqa_logits() is (None if legacy else native)
    finally:
        mqa._resolve_fp8_avg_mqa_logits.cache_clear()


def test_native_helper_passes_capacity_and_omits_removed_keyword(monkeypatch):
    captured = {}
    expected = object()

    def native(*args, **kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(mqa, "_resolve_fp8_avg_mqa_logits", lambda: native)
    q = torch.ones(3, 4, 128, dtype=torch.bfloat16)
    k = torch.ones(7, 1, 128, dtype=torch.bfloat16)
    bounds = torch.zeros(3, dtype=torch.int32)
    assert (
        mqa.deep_gemm_fp8_avg_mqa_prefill(q, k, bounds, bounds, max_seqlen_k=5)
        is expected
    )
    assert "clean_logits" not in captured
    assert captured["max_seqlen_k"] == 5
    assert captured["logits_dtype"] == torch.float32
    assert captured["q_scale"].shape == (3,)


@pytest.mark.parametrize("capacity", [None, 0, -1, 1.5])
def test_native_helper_rejects_invalid_capacity(monkeypatch, capacity):
    monkeypatch.setattr(mqa, "_resolve_fp8_avg_mqa_logits", lambda: object())
    with pytest.raises(ValueError, match="positive integer"):
        mqa.deep_gemm_fp8_avg_mqa_prefill(
            torch.ones(1, 4, 128),
            torch.ones(1, 1, 128),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            max_seqlen_k=capacity,
        )


def test_selector_preserves_request_local_indices_and_accounts_for_stride(monkeypatch):
    # Two packed requests, including nonzero K starts and an empty window.
    torch.manual_seed(19)
    q = torch.randn(6, 4, 128, dtype=torch.bfloat16)
    k = torch.randn(14, 1, 128, dtype=torch.bfloat16)
    starts = torch.tensor([0, 0, 0, 6, 6, 6], dtype=torch.int32)
    ends = torch.tensor([0, 1, 6, 6, 9, 14], dtype=torch.int32)
    positions = torch.tensor([0, 3, 23, 0, 11, 31])
    lengths = torch.tensor([24, 24, 24, 32, 32, 32], dtype=torch.int32)
    indexer = SimpleNamespace(block_topk=512, token_topk=2048, compress_ratio=4)
    monkeypatch.setattr(
        indexer_module, "can_use_deep_gemm_fp8_avg_mqa", lambda q: False
    )
    expected = indexer_module.QSAIndexer.select_prefill_tokens(
        indexer, q, k, starts, ends, positions, lengths
    )
    monkeypatch.setattr(indexer_module, "can_use_deep_gemm_fp8_avg_mqa", lambda q: True)
    widths, capacities = [], []

    def chunk_size(rows, width, heads):
        widths.append(width)
        return 2

    def scorer(q_chunk, keys, ks, ke, *, max_seqlen_k):
        capacities.append(max_seqlen_k)
        global_scores = mqa.torch_qsa_mqa_prefill(q_chunk, keys, ks, ke)
        storage = torch.full((q_chunk.shape[0], max_seqlen_k + 37), float("nan"))
        compressed = storage[:, :max_seqlen_k]
        for row in range(q_chunk.shape[0]):
            compressed[row, : ke[row] - ks[row]] = global_scores[row, ks[row] : ke[row]]
        return compressed

    monkeypatch.setattr(indexer_module, "_qsa_prefill_row_chunk_size", chunk_size)
    monkeypatch.setattr(indexer_module, "deep_gemm_fp8_avg_mqa_prefill", scorer)
    actual = indexer_module.QSAIndexer.select_prefill_tokens(
        indexer, q, k, starts, ends, positions, lengths, max_seqlen_k=8
    )
    assert widths == [256]  # FP32 native stride, not total packed K=14.
    assert capacities == [8, 8, 8]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "native_requested,dtype,expected_backend",
    [
        (False, torch.bfloat16, "relative"),
        (True, torch.bfloat16, "native"),
        (True, torch.float8_e4m3fn, "relative"),
    ],
)
def test_selector_reuses_host_bound_and_keeps_relative_fallback(
    monkeypatch, native_requested, dtype, expected_backend
):
    # Model only the CUDA tensor protocol at the selector boundary. All scoring
    # is mocked, so this dispatch regression needs no accelerator or native op.
    data = torch.empty(2, 4, 128, dtype=dtype)

    class Query:
        is_cuda = True
        shape = data.shape
        device = data.device

        def __init__(self):
            self.dtype = data.dtype

        def __getitem__(self, selection):
            return data[selection]

    q = Query()
    k = torch.empty(14, 1, 128, dtype=dtype)
    starts = torch.tensor([0, 7], dtype=torch.int32)
    ends = torch.tensor([3, 14], dtype=torch.int32)
    indexer = SimpleNamespace(block_topk=512, token_topk=2048, compress_ratio=4)
    calls = []

    def native(query, keys, ks, ke, *, max_seqlen_k):
        calls.append(("native", max_seqlen_k))
        return torch.zeros(query.shape[0], max_seqlen_k)

    def relative(query, keys, ks, ke, max_row_keys):
        calls.append(("relative", max_row_keys))
        return torch.zeros(query.shape[0], max_row_keys)

    def topk(logits, ks, ke, *, topk):
        assert ks.tolist() == [0, 0]
        assert ke.tolist() == [3, 7]
        return torch.full((logits.shape[0], topk), -1, dtype=torch.int32)

    def reject_global(*args, **kwargs):
        pytest.fail("the original relative fallback must keep its host bound")

    monkeypatch.setattr(indexer_module, "HAS_TILELANG", True)
    monkeypatch.setattr(
        indexer_module, "can_use_deep_gemm_fp8_avg_mqa", lambda _: native_requested
    )
    monkeypatch.setattr(indexer_module, "deep_gemm_fp8_avg_mqa_prefill", native)
    monkeypatch.setattr(indexer_module, "qsa_mqa_prefill_relative", relative)
    monkeypatch.setattr(indexer_module, "qsa_mqa_prefill", reject_global)
    monkeypatch.setattr(indexer_module, "qsa_fast_topk", topk)
    monkeypatch.setattr(
        indexer_module,
        "expand_qsa_block_indices",
        lambda blocks, *args, **kwargs: torch.zeros(
            blocks.shape[0], 2051, dtype=torch.int32
        ),
    )
    output = indexer_module.QSAIndexer.select_prefill_tokens(
        indexer,
        q,
        k,
        starts,
        ends,
        torch.tensor([11, 27]),
        torch.tensor([28, 28]),
        max_row_keys=7,
    )
    assert calls == [(expected_backend, 7)]
    assert output.shape == (2, 2051)


@pytest.mark.parametrize("rows,keys,dim", [(3, 7, 128), (17, 1100, 128), (7, 777, 64)])
def test_native_compressed_logits_and_poisoned_padding(rows, keys, dim, tmp_path):
    if not is_sail() or get_device_sm() != 89:
        pytest.skip("requires PPU-890P")
    assert mqa._resolve_fp8_avg_mqa_logits() is not None
    torch.manual_seed(93)
    q = torch.randn(rows, 4, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(keys, 1, dim, device="cuda", dtype=torch.bfloat16)
    q[1].zero_()  # Ties must not require identical atomic output order.
    capacity = min(keys, 777)
    starts = torch.zeros(rows, dtype=torch.int32)
    starts[rows // 2 :] = keys // 2
    lengths = torch.tensor([0, 1, 511, 512, 513, 777], dtype=torch.int32)
    lengths = lengths.repeat((rows + 5) // 6)[:rows]
    lengths = torch.minimum(lengths, keys - starts).clamp_max(capacity)
    ends = starts + lengths
    raw = mqa.deep_gemm_fp8_avg_mqa_prefill(
        q, k, starts.cuda(), ends.cuda(), max_seqlen_k=capacity
    )
    torch.cuda.synchronize()
    assert raw.shape == (rows, capacity) and raw.dtype == torch.float32
    assert raw.stride(1) == 1 and raw.stride(0) % 256 == 0

    # Independent FP64 oracle for the actual quantized Q/K, excluding FP8
    # rounding relative to BF16 (covered separately without relaxing its UT).
    q8, qs = mqa._fp8_e4m3_per_row_quant(q.reshape(rows, -1))
    k8, ks = mqa._fp8_e4m3_per_row_quant(k[:, 0])
    q_ref = (q8.cpu().double() * qs.cpu().double()[:, None]).reshape(rows, 4, dim)
    k_ref = k8.cpu().double() * ks.cpu().double()[:, None]
    global_ref = torch.relu(torch.einsum("mhd,nd->mnh", q_ref, k_ref)).sum(
        -1
    ) / math.sqrt(dim)
    raw_cpu = raw.cpu().double()
    numerical_errors = []
    measured, expected = [], []
    for row, length in enumerate(lengths.tolist()):
        x = raw_cpu[row, :length]
        y = global_ref[row, starts[row] : starts[row] + length]
        measured.append(x)
        expected.append(y)
        try:
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-4)
        except AssertionError as error:
            numerical_errors.append(f"row={row}: {error}")
    measured, expected = torch.cat(measured), torch.cat(expected)
    assert torch.isfinite(measured).all()

    # Separate framework integration from native arithmetic: identical
    # quantized tensors/scales sent directly to the installed API must give
    # exactly the wrapper's valid values. Do not weaken the FP64 assertion.
    direct = (
        mqa._resolve_fp8_avg_mqa_logits()(
            q8.reshape(rows, 4, dim),
            (k8, ks),
            starts.cuda(),
            ends.cuda(),
            max_seqlen_k=capacity,
            q_scale=qs,
            logits_dtype=torch.float32,
        )
        .cpu()
        .double()
    )
    valid = torch.arange(capacity)[None, :] < lengths[:, None]
    torch.testing.assert_close(raw_cpu[valid], direct[valid], rtol=0, atol=0)

    # Both real native padding and deliberately poisoned, unaligned padding
    # must be ignored. Never require unspecified entries to contain -inf.
    storage = torch.full((rows, capacity + 37), float("nan"), device="cuda")
    poisoned = storage[:, :capacity]
    for row, length in enumerate(lengths.tolist()):
        poisoned[row, :length] = raw[row, :length]
    for scores in (raw, poisoned):
        indices = qsa_fast_topk(
            scores, torch.zeros_like(lengths).cuda(), lengths.cuda(), 512
        ).cpu()
        for row, length in enumerate(lengths.tolist()):
            count = min(length, 512)
            chosen = indices[row, :count].long()
            assert (indices[row, count:] == -1).all()
            assert chosen.unique().numel() == count
            assert ((chosen >= 0) & (chosen < length)).all()
            wanted = torch.topk(raw_cpu[row, :length], count).values.sort().values
            torch.testing.assert_close(
                raw_cpu[row, chosen].sort().values, wanted, rtol=0, atol=0
            )

    metrics = {
        "shape": [rows, keys, dim],
        "direct_api_bitwise_match": True,
        "bounded_topk_and_poisoned_padding_passed": True,
        "fp64_pointwise_rtol": 1e-4,
        "fp64_pointwise_atol": 1e-4,
        "fp64_pointwise_passed": not numerical_errors,
        "valid_values": measured.numel(),
        "max_abs_vs_quantized_fp64": float((measured - expected).abs().max())
        if measured.numel()
        else 0.0,
        "normalized_squared_diff_vs_quantized_fp64": float(
            (measured - expected).square().sum()
            / (measured.square() + expected.square()).sum()
        )
        if measured.numel()
        else 0.0,
        "pointwise_failures": numerical_errors,
    }
    output_dir = Path(os.environ.get("PPU_DEV_RUN_DIR", str(tmp_path)))
    (output_dir / f"native-mqa-diagnostic-{rows}-{keys}-{dim}.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(metrics)
    assert not numerical_errors, "QUANTIZED_FP64_POINTWISE_MISMATCH: " + "\n".join(
        numerical_errors
    )
