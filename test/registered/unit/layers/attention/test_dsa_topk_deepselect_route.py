"""CPU route checks for the SGLANG_DSA_DEEPSELECT switch on DSA top-k.

Three routes are locked here, because only one of them is reachable per configuration:
  * PAGED / bf16   -- what the prefill best config really takes (bf16 KV store keeps
    get_topk_transform_method on PAGED, and the platform hook turns top-k v2 off);
  * RAGGED legacy  -- what prefill-CP reaches when the v2 gate is bypassed;
  * RAGGED v2      -- the fp8-KV ragged entry point.

The PAGED route additionally carries a work-shape dispatch (ruling seq=101 option D):
calls outside the measured band stay on the fused kernel by design, which is a separate
outcome from the contract fallbacks and gets its own marker. Both sides of that band are
locked below, against the shipped predicate and not only a stubbed answer.
"""

import logging
import sys
from contextlib import ExitStack
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.dsa.deep_select import (
    PAGED_SERVE_MIN_COLS,
    PAGED_SERVE_MIN_ROWS,
)
from sglang.kernels.ops.attention.dsa.deep_select import (
    paged_topk_shape_reason as _real_shape_reason,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa import dsa_topk_backend
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

_ROWS, _COLS, _TOPK = 3, 64, 8
_TABLE_COLS = 16
_DSV4 = "sglang.kernels.ops.attention.dsv4.topk"
_DSV4_PKG = "sglang.kernels.ops.attention.dsv4"
_NATIVE = "sglang.kernels.ops.attention.dsa.deep_select"


class _Meta:
    def __init__(self, seq_lens_cpu=None):
        self.indexer_seq_lens_cpu = seq_lens_cpu
        self.token_to_batch_idx = None
        self.real_page_table = torch.zeros(_ROWS, 8, dtype=torch.int32)
        # Per-request page table (page_size = 1); the route expands it per query row.
        self.page_table_1 = torch.arange(
            _ROWS * _TABLE_COLS, dtype=torch.int32
        ).reshape(_ROWS, _TABLE_COLS)
        self.dsa_extend_seq_lens_cpu = None


def _inputs(dtype=torch.bfloat16):
    torch.manual_seed(0)
    logits = torch.randn(_ROWS, _COLS, dtype=dtype)
    lengths = torch.full((_ROWS,), _COLS // 2, dtype=torch.int32)
    offsets = torch.tensor([0, 100, 200], dtype=torch.int32)
    starts = torch.tensor([0, 1, 2], dtype=torch.int32)
    return logits, lengths, offsets, starts


def _fake_v2(calls):
    module = ModuleType(_DSV4)

    def topk_transform_ragged_v2(
        scores, seq_lens, *, out_offsets, out_indices, row_starts
    ):
        calls["v2"] += 1
        out_indices.fill_(7)

    module.topk_transform_ragged_v2 = topk_transform_ragged_v2
    return module


def _fake_dsv4_package(calls):
    """Stand in for the fused bf16 PAGED prefill kernel the PAGED route replaces."""
    module = ModuleType(_DSV4_PKG)

    def top_k_per_row_prefill_bf16(
        scores, row_starts, row_ends, page_tables, out_page_indices, page_size, out_raw
    ):
        calls["dsv4_bf16"] += 1
        # Pins the OFF arm: the fused kernel is fed the caller's BF16 scores and a
        # per-query page table of the same width the gather below must reproduce.
        assert scores.dtype == torch.bfloat16
        assert page_tables.shape[0] == scores.shape[0]
        assert page_size == 1
        out_page_indices.fill_(5)

    module.top_k_per_row_prefill_bf16 = top_k_per_row_prefill_bf16
    return module


def _fake_native(calls, expected, logits):
    """Stand in for the deep_select wheel so routing is observable on CPU.

    The wheel only accepts BF16/FP32 row windows as given, so what it receives
    must be the caller's own logits tensor: recording dtype and identity here is
    what proves the route intercepts scores before any fp32 upcast.
    """
    module = ModuleType(_NATIVE)

    def topk_transform_ragged(
        scores, seq_lens, topk, out_offsets, row_starts, *, max_seq_len=None
    ):
        calls["native"] += 1
        calls["scores_dtype"] = str(scores.dtype)
        calls["scores_is_caller_logits"] = scores is logits
        calls["scores_shape"] = tuple(scores.shape)
        calls["row_starts"] = None if row_starts is None else row_starts.tolist()
        calls["max_seq_len"] = max_seq_len
        return expected

    def paged_topk_skip_reason(scores, topk):
        return calls["paged_layout_reason"]

    def paged_topk_shape_reason(scores):
        calls["paged_shape_calls"] += 1
        reason = calls["paged_shape_reason"]
        # "real" routes the composed path through the shipped predicate instead of a
        # fixed answer, so a test can lock the band and the wiring at the same time.
        return _real_shape_reason(scores) if reason == "real" else reason

    def topk_transform_paged(scores, seq_lens, topk, row_starts, page_tables):
        calls["native_paged"] += 1
        calls["paged_scores_dtype"] = str(scores.dtype)
        calls["paged_scores_is_caller_logits"] = scores is logits
        calls["paged_scores_shape"] = tuple(scores.shape)
        calls["paged_row_starts"] = None if row_starts is None else row_starts.tolist()
        calls["paged_table_shape"] = tuple(page_tables.shape)
        calls["paged_table"] = page_tables.clone()
        # None models the wheel refusing the layout (rows starting at different columns).
        return expected if calls["paged_serves"] else None

    module.topk_transform_ragged = topk_transform_ragged
    module.paged_topk_skip_reason = paged_topk_skip_reason
    module.topk_transform_paged = topk_transform_paged
    return module


def _sgl_kernel_stub(calls):
    module = ModuleType("sgl_kernel")

    def fast_topk_transform_ragged_fused(**kwargs):
        calls["legacy"] += 1
        # The pre-existing legacy route upcasts BF16 scores to FP32 first.
        assert kwargs["score"].dtype == torch.float32
        return torch.full((_ROWS, _TOPK), 3, dtype=torch.int32)

    def fast_topk_transform_fused(**kwargs):
        calls["legacy_paged"] += 1
        return torch.full((_ROWS, _TOPK), 4, dtype=torch.int32)

    module.fast_topk_transform_ragged_fused = fast_topk_transform_ragged_fused
    module.fast_topk_transform_fused = fast_topk_transform_fused
    return module


def _patches(dtype, switch, ppu, seq_lens_cpu=None):
    calls = {
        "v2": 0,
        "native": 0,
        "legacy": 0,
        "native_paged": 0,
        "dsv4_bf16": 0,
        "legacy_paged": 0,
        "paged_serves": True,
        "paged_layout_reason": None,
        "paged_shape_reason": None,
        "paged_shape_calls": 0,
        "paged_table": None,
        "paged_table_shape": None,
        "paged_scores_dtype": None,
        "paged_scores_is_caller_logits": None,
        "paged_scores_shape": None,
        "paged_row_starts": None,
        "row_starts": None,
        "max_seq_len": None,
        "scores_dtype": None,
        "scores_is_caller_logits": None,
        "scores_shape": None,
    }
    logits, lengths, offsets, starts = _inputs(dtype)
    expected = torch.full((_ROWS, _TOPK), 11, dtype=torch.int32)
    modules = {
        _DSV4: _fake_v2(calls),
        _DSV4_PKG: _fake_dsv4_package(calls),
        _NATIVE: _fake_native(calls, expected, logits),
        "sgl_kernel": _sgl_kernel_stub(calls),
    }
    return (
        calls,
        logits,
        lengths,
        offsets,
        starts,
        expected,
        patch.dict(sys.modules, modules),
        patch.object(envs.SGLANG_DSA_DEEPSELECT, "get", return_value=switch),
        patch.object(dsa_topk_backend, "is_ppu", return_value=ppu),
        patch.object(dsa_topk_backend, "_DEEPSELECT_LOGGED", set()),
        _Meta(seq_lens_cpu),
    )


def _run(dtype, switch, ppu, seq_lens_cpu=None, *, batch_idx_list=None):
    (
        calls,
        logits,
        lengths,
        offsets,
        starts,
        expected,
        m,
        sw,
        dev,
        seen,
        meta,
    ) = _patches(dtype, switch, ppu, seq_lens_cpu)
    with m, sw, dev, seen:
        if batch_idx_list is None:
            out = DSATopKBackend.SGL_KERNEL.topk_transform(
                logits,
                lengths,
                _TOPK,
                TopkTransformMethod.RAGGED,
                meta,
                topk_indices_offset=offsets,
                row_starts=starts,
            )
        else:
            out = DSATopKBackend.SGL_KERNEL.topk_transform(
                logits,
                lengths,
                _TOPK,
                TopkTransformMethod.RAGGED,
                meta,
                topk_indices_offset=offsets,
                row_starts=starts,
                batch_idx_list=batch_idx_list,
            )
    return calls, out, expected


def _run_paged(
    dtype,
    switch,
    ppu,
    *,
    row_starts=...,
    serves=True,
    layout_reason=None,
    shape_reason=None,
    batch_idx_list=None,
    cu_seqlens_q_topk="auto",
    topk_v2=...,
):
    """Drive the PAGED branch, which is what the case's prefill config really takes."""
    (
        calls,
        logits,
        lengths,
        offsets,
        starts,
        expected,
        m,
        sw,
        dev,
        seen,
        meta,
    ) = _patches(dtype, switch, ppu)
    calls["paged_serves"] = serves
    calls["paged_layout_reason"] = layout_reason
    calls["paged_shape_reason"] = shape_reason
    if cu_seqlens_q_topk == "auto":
        # One request covering every query row: the expansion must map all rows to
        # request 0, which is what the fused kernel would have done.
        cu_seqlens_q_topk = torch.tensor([0, _ROWS], dtype=torch.int32)
    gates = [m, sw, dev, seen]
    if topk_v2 is not ...:
        # The PPU platform hook turns the top-k v2 routes off, which is what the real
        # case run does; tests can pin either side of that gate.
        gates.append(
            patch.object(envs.SGLANG_OPT_USE_TOPK_V2, "get", return_value=topk_v2)
        )
    with ExitStack() as stack:
        for gate in gates:
            stack.enter_context(gate)
        out = DSATopKBackend.SGL_KERNEL.topk_transform(
            logits,
            lengths,
            _TOPK,
            TopkTransformMethod.PAGED,
            meta,
            cu_seqlens_q_topk=cu_seqlens_q_topk,
            row_starts=starts if row_starts is ... else row_starts,
            batch_idx_list=batch_idx_list,
        )
    return calls, out, expected, logits, meta


# ---------------------------------------------------------------- ragged routes


def test_switch_off_keeps_the_existing_v2_ragged_kernel():
    calls, out, _ = _run(torch.bfloat16, switch=False, ppu=True)
    assert calls["native"] == 0
    assert calls["v2"] == 1
    assert int(out.max()) == 7


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_switch_on_but_non_bf16_scores_stay_on_legacy(dtype):
    calls, out, _ = _run(dtype, switch=True, ppu=True)
    assert calls["native"] == 0
    assert calls["v2"] == 1
    assert int(out.max()) == 7


def test_switch_on_non_ppu_device_is_not_routed():
    calls, out, _ = _run(torch.bfloat16, switch=True, ppu=False)
    assert calls["native"] == 0
    assert calls["v2"] == 1


def test_switch_on_ppu_bf16_replaces_the_topk_result():
    calls, out, expected = _run(torch.bfloat16, switch=True, ppu=True)
    assert calls["native"] == 1
    assert calls["v2"] == 0
    assert torch.equal(out, expected)


def test_routed_scores_are_the_callers_bf16_logits_not_an_upcast():
    calls, out, _ = _run(torch.bfloat16, switch=True, ppu=True)
    assert calls["scores_is_caller_logits"] is True
    assert calls["scores_dtype"] == "torch.bfloat16"
    assert calls["scores_shape"] == (_ROWS, _COLS)


def test_prefill_cp_route_intercepts_before_the_fp32_upcast():
    # batch_idx_list != None is the prefill-CP shape (the case's prefill best
    # config). It never reaches the v2 ragged helper, so this is where DeepSelect
    # must catch BF16 scores ahead of fast_topk_transform_ragged_fused's upcast.
    calls, out, expected = _run(
        torch.bfloat16, switch=True, ppu=True, batch_idx_list=[0, 1, 2]
    )
    assert calls["native"] == 1
    assert calls["legacy"] == 0, "DeepSelect must take the call before the fp32 cast"
    assert calls["scores_dtype"] == "torch.bfloat16"
    assert calls["scores_is_caller_logits"] is True
    assert torch.equal(out, expected)


def test_prefill_cp_route_upcasts_when_the_switch_is_off():
    calls, out, _ = _run(
        torch.bfloat16, switch=False, ppu=True, batch_idx_list=[0, 1, 2]
    )
    assert calls["native"] == 0
    # The stub asserts the legacy entry really receives fp32 scores, so this also
    # pins the OFF arm on the pre-upcast behaviour.
    assert calls["legacy"] == 1
    assert int(out.max()) == 3


def test_switch_on_bounds_packing_with_cpu_seq_lens():
    lengths_cpu = torch.tensor([17, 23], dtype=torch.int64)
    calls, out, _ = _run(
        torch.bfloat16, switch=True, ppu=True, seq_lens_cpu=lengths_cpu
    )
    assert calls["native"] == 1
    assert calls["max_seq_len"] == 23
    # Several requests share the ragged KV, so per-row window starts must survive.
    assert calls["row_starts"] == [0, 1, 2]


def test_single_request_ragged_window_starts_at_column_zero():
    calls, out, _ = _run(
        torch.bfloat16, switch=True, ppu=True, seq_lens_cpu=torch.tensor([61])
    )
    assert calls["native"] == 1
    assert calls["max_seq_len"] == 61
    assert calls["row_starts"] is None


# ----------------------------------------------------------------- paged route


def test_paged_switch_off_keeps_the_fused_dsv4_bf16_kernel():
    calls, out, _, _, _ = _run_paged(torch.bfloat16, switch=False, ppu=True)
    assert calls["dsv4_bf16"] == 1
    assert calls["native_paged"] == 0
    assert int(out.max()) == 5


def test_paged_switch_on_replaces_the_fused_kernel():
    calls, out, expected, _, _ = _run_paged(torch.bfloat16, switch=True, ppu=True)
    assert calls["native_paged"] == 1
    assert calls["dsv4_bf16"] == 0, "DeepSelect must take the call ahead of the kernel"
    assert torch.equal(out, expected)


def test_paged_route_passes_the_callers_bf16_scores_untouched():
    calls, out, _, logits, _ = _run_paged(torch.bfloat16, switch=True, ppu=True)
    assert calls["paged_scores_is_caller_logits"] is True
    assert calls["paged_scores_dtype"] == "torch.bfloat16"
    assert calls["paged_scores_shape"] == (_ROWS, _COLS)


def test_paged_route_expands_the_page_table_like_the_fused_kernel():
    # Row r belongs to request r searched in cu_seqlens_q_topk; with one request the
    # expanded table must be request 0's row repeated, exactly as _dsv4_bf16 builds it.
    calls, out, _, _, meta = _run_paged(torch.bfloat16, switch=True, ppu=True)
    assert calls["paged_table_shape"] == (_ROWS, _TABLE_COLS)
    assert torch.equal(
        calls["paged_table"], meta.page_table_1[0:1].expand(_ROWS, _TABLE_COLS)
    )


def test_paged_route_keeps_row_starts_for_the_window():
    starts = torch.tensor([4, 4, 4], dtype=torch.int32)
    calls, out, _, _, _ = _run_paged(
        torch.bfloat16, switch=True, ppu=True, row_starts=starts
    )
    assert calls["native_paged"] == 1
    assert calls["paged_row_starts"] == [4, 4, 4]


def test_paged_wheel_refusal_falls_back_to_the_fused_kernel(caplog):
    # Rows starting at different columns cannot be served (no begin support); the
    # route must hand the call back to the existing kernel instead of guessing.
    with caplog.at_level(logging.WARNING, logger=dsa_topk_backend.__name__):
        calls, out, _, _, _ = _run_paged(
            torch.bfloat16, switch=True, ppu=True, serves=False
        )
    assert calls["native_paged"] == 1
    assert calls["dsv4_bf16"] == 1, "a refused layout must stay on the old kernel"
    assert int(out.max()) == 5
    skipped = [r for r in caplog.records if "is on but skipped" in r.getMessage()]
    assert skipped and "different columns" in skipped[0].getMessage()


_SHAPE_KEEP = "kept on the fused kernel by the deep_select shape dispatch"


def test_paged_shape_dispatch_stays_on_the_fused_kernel(caplog):
    # A call outside the measured band is *designed* to stay on the fused kernel, so it
    # must not reach the wheel, and it must not be reported as a missed precondition --
    # the gates count those two outcomes differently on purpose.
    with caplog.at_level(logging.INFO, logger=dsa_topk_backend.__name__):
        calls, out, _, _, _ = _run_paged(
            torch.bfloat16,
            switch=True,
            ppu=True,
            shape_reason=(
                f"rows={_ROWS} cols={_COLS} below the served band "
                f"(rows>={PAGED_SERVE_MIN_ROWS} and cols>={PAGED_SERVE_MIN_COLS})"
            ),
        )
    assert calls["paged_shape_calls"] == 1
    assert calls["native_paged"] == 0, (
        "a shape-dispatched call must never reach the wheel"
    )
    assert calls["dsv4_bf16"] == 1, "the fused kernel must keep serving this call"
    assert int(out.max()) == 5
    messages = [r.getMessage() for r in caplog.records]
    assert [m for m in messages if _SHAPE_KEEP in m], (
        "the dispatch itself needs a marker"
    )
    assert [m for m in messages if _SHAPE_KEEP in m and f"rows={_ROWS}" in m]
    assert not [m for m in messages if "is on but skipped" in m]


def test_paged_shape_gate_uses_the_shipped_predicate_below_the_band():
    # Composed path with the real predicate: the tiny fixture is below the band, so the
    # wheel must stay untouched even though every other precondition is satisfied.
    calls, out, _, _, _ = _run_paged(
        torch.bfloat16, switch=True, ppu=True, shape_reason="real"
    )
    assert calls["native_paged"] == 0
    assert calls["dsv4_bf16"] == 1
    assert int(out.max()) == 5


def test_paged_shape_gate_uses_the_shipped_predicate_inside_the_band():
    # Same fixture with the band lowered to its size: the dispatch is then the only
    # thing that was standing between the call and DeepSelect, which locks the upper
    # side of the threshold through the same wiring, not a different stub.
    from sglang.kernels.ops.attention.dsa import deep_select as real_native

    with (
        patch.object(real_native, "PAGED_SERVE_MIN_ROWS", _ROWS),
        patch.object(real_native, "PAGED_SERVE_MIN_COLS", _COLS),
    ):
        calls, out, expected, _, _ = _run_paged(
            torch.bfloat16, switch=True, ppu=True, shape_reason="real"
        )
    assert calls["native_paged"] == 1, "inside the band the call must be served"
    assert calls["dsv4_bf16"] == 0
    assert torch.equal(out, expected)


@pytest.mark.parametrize(
    "rows,cols,served",
    [
        (PAGED_SERVE_MIN_ROWS, PAGED_SERVE_MIN_COLS, True),
        (PAGED_SERVE_MIN_ROWS - 1, PAGED_SERVE_MIN_COLS, False),
        (PAGED_SERVE_MIN_ROWS, PAGED_SERVE_MIN_COLS - 1, False),
        (1, 65536, False),
        (16, PAGED_SERVE_MIN_COLS, False),
        (256, 32768, False),
        (1024, 8192, False),
        (7808, 64000, True),
        (PAGED_SERVE_MIN_ROWS, 131072, True),
    ],
)
def test_served_band_edges_are_the_measured_crossover(rows, cols, served):
    """Lock the dispatch band to the measured crossovers, not to a round guess.

    Measured on ZW-M890P with the shipping path (one 42us window-start host read
    included): 256 x 32768 +8.2% and 1024 x 8192 +10.6% lose to the fused kernel, while
    512 x 32768 loses 6.2% (repeated at reps=9 and reps=21) and 7808 x 64000 wins 78.0%.
    Meta tensors keep this assertion host-only, which is the point of the dispatch.
    """
    scores = torch.empty(rows, cols, dtype=torch.bfloat16, device="meta")
    reason = _real_shape_reason(scores)
    if served:
        assert reason is None
    else:
        assert reason is not None
        assert f"rows={rows} cols={cols}" in reason


def test_served_band_constants_are_the_documented_pair():
    assert (PAGED_SERVE_MIN_ROWS, PAGED_SERVE_MIN_COLS) == (512, 32768)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_paged_non_bf16_scores_stay_on_the_existing_kernels(dtype):
    calls, out, _, _, _ = _run_paged(dtype, switch=True, ppu=True)
    assert calls["native_paged"] == 0
    # Only BF16 scores are served by the dsv4 bf16 kernel; FP32/FP16 keep going to the
    # native fused transform. Both are the pre-existing behaviours being preserved.
    assert calls["dsv4_bf16"] == 0
    assert calls["legacy_paged"] == 1
    assert int(out.max()) == 4


def test_paged_layout_rejected_by_wheel_contract(caplog):
    with caplog.at_level(logging.WARNING, logger=dsa_topk_backend.__name__):
        calls, out, _, _, _ = _run_paged(
            torch.bfloat16,
            switch=True,
            ppu=True,
            layout_reason="scores row stride 130B not aligned to 1024B",
        )
    assert calls["native_paged"] == 0
    assert calls["dsv4_bf16"] == 1
    skipped = [r for r in caplog.records if "is on but skipped" in r.getMessage()]
    assert skipped and "aligned" in skipped[0].getMessage()


def test_paged_non_ppu_device_is_not_routed():
    calls, out, _, _, _ = _run_paged(torch.bfloat16, switch=True, ppu=False)
    assert calls["native_paged"] == 0
    assert calls["dsv4_bf16"] == 1


def test_paged_hit_marker_is_logged_once(caplog):
    with caplog.at_level(logging.INFO, logger=dsa_topk_backend.__name__):
        calls, out, _, _, _ = _run_paged(torch.bfloat16, switch=True, ppu=True)
    hits = [
        r
        for r in caplog.records
        if "PAGED top-k served by PPU deep_select" in r.getMessage()
    ]
    assert len(hits) == 1
    assert "rows=3" in hits[0].getMessage()
    assert not [r for r in caplog.records if "is on but skipped" in r.getMessage()]


def test_paged_switch_off_logs_no_marker(caplog):
    with caplog.at_level(logging.INFO, logger=dsa_topk_backend.__name__):
        calls, out, _, _, _ = _run_paged(torch.bfloat16, switch=False, ppu=True)
    assert calls["native_paged"] == 0
    assert not [
        r
        for r in caplog.records
        if "deep_select" in r.getMessage() or "DEEPSELECT" in r.getMessage()
    ]


def test_paged_route_uses_the_row_start_free_path_when_starts_are_absent():
    # row_starts=None means every row starts at column 0: the window is the length.
    calls, out, _, _, _ = _run_paged(
        torch.bfloat16, switch=True, ppu=True, row_starts=None, topk_v2=False
    )
    assert calls["native_paged"] == 1
    assert calls["paged_row_starts"] is None


def test_skip_guard_rejects_inputs_outside_the_wheel_contract():
    offsets = torch.empty(2, dtype=torch.int32)
    # K > 4096 and non-2D/non-BF16 scores are outside the wheel's contract.
    assert (
        dsa_topk_backend._deepselect_ragged_skip_reason(
            torch.empty(2, 8, dtype=torch.bfloat16), 5000, offsets
        )
        is not None
    )
    assert (
        dsa_topk_backend._deepselect_ragged_skip_reason(
            torch.empty(2, 8, dtype=torch.float32), 8, offsets
        )
        is not None
    )
    assert (
        dsa_topk_backend._deepselect_ragged_skip_reason(
            torch.empty(2, 8, 1, dtype=torch.bfloat16), 8, offsets
        )
        is not None
    )
    assert (
        dsa_topk_backend._deepselect_ragged_skip_reason(
            torch.empty(2, 8, dtype=torch.bfloat16), 8, None
        )
        is not None
    )
    assert (
        dsa_topk_backend._deepselect_ragged_skip_reason(
            torch.empty(2, 8, dtype=torch.bfloat16), 8, offsets
        )
        is None
    )


def test_paged_skip_guard_rejects_inputs_outside_the_wheel_contract():
    table = torch.empty(2, 8, dtype=torch.int32)
    cu = torch.tensor([0, 2], dtype=torch.int32)
    assert (
        dsa_topk_backend._deepselect_paged_skip_reason(
            torch.empty(2, 8, dtype=torch.bfloat16), 5000, table, cu
        )
        is not None
    )
    assert (
        dsa_topk_backend._deepselect_paged_skip_reason(
            torch.empty(2, 8, 1, dtype=torch.bfloat16), 8, table, cu
        )
        is not None
    )
    assert (
        dsa_topk_backend._deepselect_paged_skip_reason(
            torch.empty(2, 8, dtype=torch.float16), 8, table, cu
        )
        is not None
    )
    assert (
        dsa_topk_backend._deepselect_paged_skip_reason(
            torch.empty(2, 8, dtype=torch.bfloat16), 8, None, cu
        )
        is not None
    )
    assert (
        dsa_topk_backend._deepselect_paged_skip_reason(
            torch.empty(2, 8, dtype=torch.bfloat16), 8, table, None
        )
        is not None
    )
    assert (
        dsa_topk_backend._deepselect_paged_skip_reason(
            torch.empty(2, 8, dtype=torch.bfloat16), 8, table, cu
        )
        is None
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
