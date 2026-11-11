from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
    DeepseekSparseAttnBackendKPoolMixin,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    _validate_flashmla_sparse_q8_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

RAGGED = TopkTransformMethod.RAGGED
PAGED = TopkTransformMethod.PAGED


def _make_backend(prefill_impl: str, *, fp8_kv_cache: bool):
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.dsa_kv_cache_store_fp8 = fp8_kv_cache
    backend.dsa_prefill_impl = prefill_impl
    backend.attn_dcp_size = 1
    return backend


@pytest.mark.parametrize(
    "dcp_size,dcp_rank,expected",
    [(1, 0, [65536, 65536]), (8, 0, [1, 1025]), (8, 7, [0, 1024])],
)
def test_capture_keeps_dcp_visible_lengths(monkeypatch, dcp_size, dcp_rank, expected):
    """Schedule capacity must not become the local TopK's visible key count."""
    backend = _make_backend("fa3", fp8_kv_cache=False)
    backend.device = torch.device("cpu")
    backend.attn_dcp_size = dcp_size
    backend.attn_dcp_rank = dcp_rank
    backend.physical_page_size = 64
    backend.real_page_span = 64 * dcp_size
    backend.dsa_index_topk = 2048
    backend.dsa_index_kpool = 1
    backend.speculative_num_draft_tokens = 2
    backend.dsa_drop_wide_page_table = False
    backend.dsa_decode_impl = "fa3"
    backend.enable_auto_select_prefill_impl = False
    backend._is_in_breakable_cuda_graph = lambda: False
    backend._is_in_tc_piecewise_cuda_graph = lambda: False
    backend._get_device_sm = lambda: 100
    backend._is_blackwell = lambda: True
    backend.dsa_topk_backend = dsa_backend.DSATopKBackend.TORCH
    backend.req_to_token = torch.zeros((2, 65536), dtype=torch.int32)
    backend._arange_buf = torch.arange(3, dtype=torch.int32)
    backend.decode_cuda_graph_metadata = {
        "page_table": backend.req_to_token.clone(),
        "cu_seqlens_q": backend._arange_buf,
    }
    # Isolate the external DeepGEMM scheduler; metadata construction stays real.
    backend._build_paged_mqa_schedule_metadata = lambda lengths: lengths.clone()
    monkeypatch.setattr(dsa_backend, "is_cuda", lambda: True)
    seq = torch.tensor([1, 8193], dtype=torch.int32)
    backend._build_forward_metadata_cuda_graph(
        2, None, torch.arange(2), seq, seq, ForwardMode.DECODE, None
    )

    metadata = backend.forward_metadata
    torch.testing.assert_close(
        metadata.paged_mqa_ctx_lens_2d,
        torch.tensor(expected, dtype=torch.int32)[:, None],
    )
    torch.testing.assert_close(
        metadata.paged_mqa_schedule_metadata,
        torch.full((2, 1), 65536, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    "prefill_impl,fp8_kv_cache,forward_mode,expected",
    [
        # The q8 helper only exists on the RAGGED route, whatever the pool dtype.
        ("flashmla_sparse_q8", False, ForwardMode.EXTEND, RAGGED),
        ("flashmla_sparse_q8", True, ForwardMode.EXTEND, RAGGED),
        # bf16 flashmla_sparse reads the pool directly through PAGED slot ids.
        ("flashmla_sparse", False, ForwardMode.EXTEND, PAGED),
        ("flashmla_sparse", True, ForwardMode.EXTEND, RAGGED),
        ("flashmla_sparse_q8", False, ForwardMode.DECODE, PAGED),
        ("flashmla_sparse_q8", False, ForwardMode.TARGET_VERIFY, PAGED),
        ("flashmla_sparse_q8", False, ForwardMode.DRAFT_EXTEND_V2, PAGED),
    ],
)
def test_topk_transform_method(prefill_impl, fp8_kv_cache, forward_mode, expected):
    backend = _make_backend(prefill_impl, fp8_kv_cache=fp8_kv_cache)
    assert backend.get_topk_transform_method(forward_mode) == expected


@pytest.mark.parametrize("kv_cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_q8_validator_accepts_bf16_and_fp8_kv_on_sm90(kv_cache_dtype):
    _validate_flashmla_sparse_q8_backend(
        prefill_impl="flashmla_sparse_q8",
        decode_impl="tilelang",
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_store_fp8=kv_cache_dtype == torch.float8_e4m3fn,
        qk_rope_head_dim=64,
        device_sm_major=9,
    )


@pytest.mark.parametrize(
    "prefill_impl,decode_impl,kv_cache_dtype,store_fp8,rope,device_sm_major",
    [
        ("flashmla_sparse_q8", "tilelang", torch.float16, False, 64, 9),
        ("flashmla_sparse_q8", "trtllm", torch.bfloat16, False, 64, 10),
        ("tilelang", "flashmla_sparse_q8", torch.bfloat16, False, 64, 9),
        # An fp8 pool on a NoPE model is not supported yet.
        ("flashmla_sparse_q8", "flashmla_kv", torch.float8_e4m3fn, True, 0, 9),
    ],
)
def test_q8_validator_rejects_unsupported_configs(
    prefill_impl, decode_impl, kv_cache_dtype, store_fp8, rope, device_sm_major
):
    with pytest.raises(ValueError):
        _validate_flashmla_sparse_q8_backend(
            prefill_impl=prefill_impl,
            decode_impl=decode_impl,
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_store_fp8=store_fp8,
            qk_rope_head_dim=rope,
            device_sm_major=device_sm_major,
        )


@pytest.mark.parametrize(
    "extra",
    [
        pytest.param({"hisparse_enabled": True}, id="hisparse"),
        pytest.param({"mixed_chunk_enabled": True}, id="mixed_chunk"),
    ],
)
def test_q8_validator_rejects_combinations_that_used_to_fail_mid_forward(extra):
    """Both used to surface only once a batch of the wrong shape was formed --
    HiSparse as an UnboundLocalError, mixed chunks as a NotImplementedError
    hours into serving. The launch flags are visible at construction."""
    kwargs = dict(
        prefill_impl="flashmla_sparse_q8",
        decode_impl="tilelang",
        kv_cache_dtype=torch.bfloat16,
        kv_cache_store_fp8=False,
        qk_rope_head_dim=0,
        device_sm_major=9,
    )
    kwargs.update(extra)
    with pytest.raises(ValueError):
        _validate_flashmla_sparse_q8_backend(**kwargs)


def test_q8_validator_rejects_an_unpacked_fp8_pool():
    """An fp8 dtype whose pool is not stored packed stays in the raw 576-dim
    layout that the bf16 prefix gather cannot read; it previously surfaced as a
    message-less assert on the first prefix extend."""
    with pytest.raises(ValueError):
        _validate_flashmla_sparse_q8_backend(
            prefill_impl="flashmla_sparse_q8",
            decode_impl="tilelang",
            kv_cache_dtype=torch.float8_e4m3fn,
            kv_cache_store_fp8=False,
            qk_rope_head_dim=64,
            device_sm_major=9,
        )


def _make_padder():
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend._q8kv8_topk_pad_buf = None
    backend._q8kv8_topk_pad_width = None
    return backend


def test_pad_topk_width_pads_kpool_width_with_minus_one():
    indices = torch.arange(2 * 2051, dtype=torch.int32).view(2, 2051)
    padded = _make_padder()._pad_topk_width_buffered(indices, multiple=128)
    assert padded.shape == (2, 2176) and padded.is_contiguous()
    assert torch.equal(padded[:, :2051], indices)
    assert (padded[:, 2051:] == -1).all()


@pytest.mark.parametrize("width", [2048, 2176])
def test_pad_topk_width_is_zero_copy_on_aligned_widths(width):
    indices = torch.zeros((3, width), dtype=torch.int32)
    assert _make_padder()._pad_topk_width_buffered(indices, multiple=128) is indices


def test_pad_topk_width_rounds_64_aligned_width_to_128():
    """The SM90 kernel walks top-k in pairs of 64-wide blocks; a width that is
    only a multiple of 64 (tilelang's padding) is still rejected."""
    indices = torch.zeros((1, 2112), dtype=torch.int32)
    padded = _make_padder()._pad_topk_width_buffered(indices, multiple=128)
    assert padded.shape == (1, 2176)


def test_pad_topk_width_reuses_the_buffer_without_leaking_stale_ids():
    """The pad columns are -1-filled once at allocation and never rewritten, so a
    later call with fewer rows must still read -1 there rather than ids left by
    an earlier, larger call."""
    padder = _make_padder()
    first = padder._pad_topk_width_buffered(
        torch.full((4, 2051), 7, dtype=torch.int32), multiple=128
    )
    assert (first[:, 2051:] == -1).all()
    buf = padder._q8kv8_topk_pad_buf
    second = padder._pad_topk_width_buffered(
        torch.full((2, 2051), 9, dtype=torch.int32), multiple=128
    )
    assert padder._q8kv8_topk_pad_buf is buf, "fewer rows must not reallocate"
    assert (second[:, :2051] == 9).all()
    assert (second[:, 2051:] == -1).all()


def test_pad_topk_width_does_not_share_a_buffer_across_different_real_widths():
    """Two different real widths can pad to the same total: 2051 and 2100 both
    round to 2176 at multiple=128. Reusing the buffer across them would leave the
    wider call's real ids in columns the narrower call must read as padding, and
    the SM90 kernel's -1 clamp would consume them as genuine top-k indices."""
    padder = _make_padder()
    wide = padder._pad_topk_width_buffered(
        torch.arange(1, 2100 + 1, dtype=torch.int32).unsqueeze(0), multiple=128
    )
    assert wide.shape == (1, 2176)
    narrow = padder._pad_topk_width_buffered(
        torch.full((1, 2051), 7, dtype=torch.int32), multiple=128
    )
    assert narrow.shape == (1, 2176)
    assert (narrow[:, :2051] == 7).all()
    assert (narrow[:, 2051:] == -1).all(), (
        "stale ids from the wider call leaked into the pad band"
    )


def test_kpool_tail_guard_admits_q8_for_prefill_only():
    backend = SimpleNamespace(
        dsa_index_kpool=4, device_sm_major=9, dsa_kv_cache_store_fp8=False
    )
    indices = torch.full((2, 2051), -1, dtype=torch.int32)
    mixin = DeepseekSparseAttnBackendKPoolMixin
    assert (
        mixin._resolve_kpool_tail_backend(backend, indices, "flashmla_sparse_q8")
        == "flashmla_sparse_q8"
    )
    assert (
        mixin._resolve_kpool_tail_backend(backend, indices, "flashmla_sparse") == "fa3"
    )
    mixin._check_kpool_tail_backend(backend, indices, "flashmla_sparse_q8", "prefill")
    with pytest.raises(NotImplementedError):
        mixin._check_kpool_tail_backend(
            backend, indices, "flashmla_sparse_q8", "decode"
        )
    with pytest.raises(NotImplementedError):
        mixin._check_kpool_tail_backend(backend, indices, "flashmla_kv", "prefill")


def test_kpool_tail_guard_still_rejects_q8_over_a_packed_fp8_pool():
    """Gathering packed fp8 rows through kpool tail columns is untested, so the
    guard must stay in force there even though q8 prefill is admitted on bf16."""
    backend = SimpleNamespace(
        dsa_index_kpool=4, device_sm_major=9, dsa_kv_cache_store_fp8=True
    )
    indices = torch.full((2, 2051), -1, dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend(
            backend, indices, "flashmla_sparse_q8", "prefill"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
