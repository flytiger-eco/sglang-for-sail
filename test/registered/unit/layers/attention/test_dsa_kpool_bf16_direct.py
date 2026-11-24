"""CPU gate tests for kpool bfloat16 direct output (the ``bf16 直出`` change).

The pooled-score producer can now ask DeepGEMM for bfloat16 logits instead of
float32, which removes the fp32 buffer and the dtype conversion inside the
DeepSelect staging pack. That is only safe while the request predicts the serve
gate exactly, so this file locks three things:

  * the width is requested on no call the route will not serve -- switch off,
    non-PPU device, missing wheel, shape outside the measured band, unsupported
    bucket -- and the historical call shape stays keyword-free otherwise;
  * a library build without the ``logits_dtype`` parameter, or a branch that has
    no such parameter at all (AITER, tilelang), never receives it, and the gate
    width handed to the library always matches the width asked for (newer builds
    infer the output width from the gate dtype instead of exposing a parameter);
  * the gate that does fire is honoured: when the fused kernel runs after all,
    bfloat16 scores are upcast to float32 and one log line records it.

The numeric consequence of the narrower width is measured against the wheel on
PPU in test/registered/kernels/ops/attention/test_deep_select_kpool_ppu.py.
"""

import inspect
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from sglang.kernels.ops.attention.dsa import deep_select as ds  # noqa: E402
from sglang.srt.layers.attention.dsa import (  # noqa: E402
    dsa_indexer_kpool as ikp,
)
from sglang.srt.layers.attention.dsa import (  # noqa: E402
    dsa_topk_backend as tb,
)
from sglang.srt.environ import envs  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

GROUP_TOPK = 512
# Marks "the caller did not name the width", so a test can tell an omitted keyword
# from an explicit float32 one.
_DEFAULT = object()
# The measured served band, read from the shipped constants so a threshold change
# cannot pass these tests by accident.
BAND_ROWS = ds.KPOOL_SERVE_MIN_ROWS
BAND_COLS = ds.KPOOL_SERVE_MIN_COLS


@contextmanager
def _route(*, switch=True, ppu=True, importable=True):
    """Patch the environment facts the bfloat16 predicate reads."""
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(envs.SGLANG_DSA_DEEPSELECT, "get", return_value=switch)
        )
        stack.enter_context(mock.patch.object(tb, "is_ppu", return_value=ppu))
        stack.enter_context(
            mock.patch.object(tb, "_deep_select_importable", return_value=importable)
        )
        yield


def _gate_serves(rows, cols, group_topk, *, dtype=torch.bfloat16):
    """The shipped serve gate on a real tensor of that width, without calling the wheel."""
    logits = torch.zeros(rows, cols, dtype=dtype)
    return tb._deepselect_kpool_skip_reason(logits, group_topk) is None


# Outside the band on exactly one axis, on a bucket the wheel does not serve, and
# inside the band: the shapes the predicate and the gate must agree about.
_SHAPES = [
    (BAND_ROWS, BAND_COLS, GROUP_TOPK, True),
    (BAND_ROWS - 1, BAND_COLS, GROUP_TOPK, False),
    (BAND_ROWS, BAND_COLS - 1, GROUP_TOPK, False),
    (BAND_ROWS, BAND_COLS, 96, False),
    (BAND_ROWS * 2, BAND_COLS * 2, GROUP_TOPK, True),
]


@pytest.mark.parametrize("switch", [True, False])
@pytest.mark.parametrize("ppu", [True, False])
@pytest.mark.parametrize("rows,cols,group_topk,in_band", _SHAPES)
def test_bf16_is_requested_only_on_calls_the_route_will_serve(
    rows, cols, group_topk, in_band, switch, ppu
):
    """Prediction == takeover, and anything else keeps the historical width."""
    expected = in_band and switch and ppu
    with _route(switch=switch, ppu=ppu):
        predicted = tb.kpool_deepselect_will_serve(
            rows, cols, group_topk, logits_dtype=torch.bfloat16
        )
        gate = _gate_serves(rows, cols, group_topk)
        width = ikp._kpool_scoring_dtype(rows, cols, group_topk)
    assert predicted is expected, f"predicate {predicted} != expected {expected}"
    # The gate itself checks the device but not the switch, so it agrees with the
    # prediction whenever the device also allows a takeover.
    assert gate is (in_band and ppu)
    assert width is (torch.bfloat16 if expected else torch.float32)


def test_missing_wheel_stops_the_request_but_never_the_run():
    """Without the wheel the producer keeps float32, so the fused route is unchanged."""
    with _route(importable=False):
        assert not tb.kpool_deepselect_will_serve(
            BAND_ROWS, BAND_COLS, GROUP_TOPK, logits_dtype=torch.bfloat16
        )
        assert (
            ikp._kpool_scoring_dtype(BAND_ROWS, BAND_COLS, GROUP_TOPK)
            is torch.float32
        )


def test_float32_is_still_a_servable_width():
    """The emitted width is a producer choice, not a takeover condition."""
    with _route():
        assert tb.kpool_deepselect_will_serve(
            BAND_ROWS, BAND_COLS, GROUP_TOPK, logits_dtype=torch.float32
        )
        assert _gate_serves(BAND_ROWS, BAND_COLS, GROUP_TOPK, dtype=torch.float32)
        assert not tb.kpool_deepselect_will_serve(
            BAND_ROWS, BAND_COLS, GROUP_TOPK, logits_dtype=torch.float16
        )


def _kernel_pair(*, modern):
    """Two stand-in scoring kernels: one exposes ``logits_dtype``, one does not.

    Records what the producer actually handed over, so the assertions below are
    about the shipped call shape and not about the helper's return value. A build
    without the parameter models the newer library, which infers the output width
    from the gate dtype.
    """
    calls = []

    if modern:

        def scoring(
            q,
            kv,
            weights,
            starts,
            ends,
            clean_logits=True,
            logits_dtype=_DEFAULT,
        ):
            sent = logits_dtype is not _DEFAULT
            width = torch.float32 if not sent else logits_dtype
            calls.append(
                {
                    "weights": weights,
                    "clean_logits": clean_logits,
                    "keyword_sent": sent,
                    "logits_dtype": width,
                }
            )
            return torch.zeros(2, 2, dtype=width)

    else:

        def scoring(q, kv, weights, starts, ends, clean_logits=True):
            calls.append(
                {
                    "weights": weights,
                    "clean_logits": clean_logits,
                    # Reaching this branch with the keyword would have raised TypeError
                    # before the call, so the flag can only ever be false here.
                    "keyword_sent": False,
                    "logits_dtype": weights.dtype,
                }
            )
            return torch.zeros(2, 2, dtype=weights.dtype)

    return scoring, calls


def test_bf16_request_is_dropped_for_builds_without_the_parameter():
    modern, _ = _kernel_pair(modern=True)
    legacy, _ = _kernel_pair(modern=False)
    assert ikp._kpool_bf16_request(legacy, torch.bfloat16) == (torch.float32, {})
    assert ikp._kpool_bf16_request(modern, torch.bfloat16) == (
        torch.bfloat16,
        {"logits_dtype": torch.bfloat16},
    )
    # A float32 request keeps today's call shape, keyword absent.
    assert ikp._kpool_bf16_request(modern, torch.float32) == (torch.float32, {})


def _call_fp8_mqa_logits(*, modern, logits_dtype, weights_dtype):
    scoring, calls = _kernel_pair(modern=modern)
    fake = SimpleNamespace(fp8_mqa_logits=scoring, int8_mqa_logits=scoring)
    rows, cols = BAND_ROWS, BAND_COLS
    with mock.patch.object(ikp, "deep_gemm", fake), mock.patch.object(
        ikp, "is_hip", return_value=False
    ):
        out = ikp.IndexerKPool._fp8_mqa_logits(
            SimpleNamespace(use_int8=False),
            torch.zeros(rows, 4, dtype=torch.float32),
            (torch.zeros(cols, dtype=torch.float32), None),
            None,
            torch.zeros(rows, 4, dtype=weights_dtype),
            torch.zeros(rows, dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
            clean_logits=True,
            logits_dtype=logits_dtype,
        )
    return out, calls[-1]


def test_in_band_call_hands_the_library_matching_widths():
    out, call = _call_fp8_mqa_logits(
        modern=True, logits_dtype=torch.bfloat16, weights_dtype=torch.float32
    )
    assert call["keyword_sent"], call
    assert call["logits_dtype"] is torch.bfloat16
    assert call["weights"].dtype is torch.bfloat16, "library gate-width contract"
    assert call["clean_logits"] is True, "the historical masking request is untouched"
    assert out.dtype is torch.bfloat16


def test_out_of_band_call_is_the_historical_call_shape():
    """A float32 request sends no keyword at all, whatever the caller's gate width was."""
    for weights_dtype in (torch.float32, torch.bfloat16):
        out, call = _call_fp8_mqa_logits(
            modern=True, logits_dtype=torch.float32, weights_dtype=weights_dtype
        )
        assert not call["keyword_sent"], call
        assert call["weights"].dtype is torch.float32
        assert out.dtype is torch.float32


def test_legacy_build_never_sees_the_keyword_even_when_bf16_was_predicted():
    """A stale wheel on a served-shape call degrades to float32 instead of raising."""
    out, call = _call_fp8_mqa_logits(
        modern=False, logits_dtype=torch.bfloat16, weights_dtype=torch.float32
    )
    assert not call["keyword_sent"], "the keyword must not reach a legacy build"
    assert call["weights"].dtype is torch.float32
    assert out.dtype is torch.float32


def test_bf16_capability_probe_reads_the_signature_or_the_docstring():
    modern, _ = _kernel_pair(modern=True)
    legacy, _ = _kernel_pair(modern=False)
    assert tb.kpool_bf16_scoring_supported(modern)
    assert not tb.kpool_bf16_scoring_supported(legacy)

    def doc_only():
        """fp8_mqa_logits(q, kv, weights, ..., logits_dtype)"""
        return None

    def _no_signature(_fn):
        raise ValueError("no signature found for builtin")

    with mock.patch.object(inspect, "signature", _no_signature):
        assert tb.kpool_bf16_scoring_supported(doc_only), (
            "an extension whose signature cannot be inspected may still advertise "
            "the parameter in its docstring"
        )
        assert not tb.kpool_bf16_scoring_supported(
            lambda *a, **kw: None
        ), "and a docstring without the parameter must not enable bf16"


def test_kpool_weights_follow_the_emitted_logits_width():
    """No copy when the caller already handed over the required width."""
    fp32 = torch.zeros(4, 8, dtype=torch.float32)
    bf16 = torch.zeros(4, 8, dtype=torch.bfloat16)
    assert ikp._kpool_weights_for_logits(fp32, torch.float32) is fp32
    assert ikp._kpool_weights_for_logits(bf16, torch.bfloat16) is bf16
    assert ikp._kpool_weights_for_logits(fp32, torch.bfloat16).dtype is torch.bfloat16
    assert ikp._kpool_weights_for_logits(bf16, torch.float32).dtype is torch.float32


def test_fused_fallback_upcasts_bf16_scores_and_logs_once():
    """Band-out and contract-fallback routes hand the incumbent kernel float32."""
    from sglang.srt.layers.attention.dsa import kpool_fp8_index as kpi

    bf16 = torch.zeros(4, 8, dtype=torch.bfloat16)
    with mock.patch.object(kpi, "_log_kpool_route_once") as record:
        got = kpi._kpool_scores_for_fallback(bf16)
    assert got.dtype is torch.float32
    assert record.call_count == 1, "a silent width mismatch must be visible"

    fp32 = torch.zeros(4, 8, dtype=torch.float32)
    with mock.patch.object(kpi, "_log_kpool_route_once") as record:
        assert kpi._kpool_scores_for_fallback(fp32) is fp32
    assert record.call_count == 0, "the historical route must stay log-free"


def test_entry_point_still_refuses_unsupported_inputs():
    """The relaxed dtype gate must not open a silent PyTorch fallback."""
    from sglang.srt.layers.attention.dsa import kpool_fp8_index as kpi

    lengths = torch.full((4,), 64, dtype=torch.int32)
    with pytest.raises(NotImplementedError) as info:
        kpi.topk_from_pooled_history_logits(
            torch.zeros(4, 64, dtype=torch.float16), lengths, 4, 2048
        )
    assert "bfloat16" in str(info.value)

    with pytest.raises(NotImplementedError) as info:
        kpi.topk_from_pooled_history_logits(
            torch.zeros(4, 64, dtype=torch.bfloat16), lengths, 4, 400
        )
    assert "group_topk" in str(info.value), "unknown buckets are still refused outright"
