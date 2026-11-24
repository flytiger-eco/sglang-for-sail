"""PPU wheel integration: kpool (index_kpool>1) group-level top-k through DeepSelect.

The Flash weights select pool-level top-k inside ``fast_kpool_topk_transform_fused``.
The DeepSelect route replaces only that selection, so every test here compares the two
routes on the same pooled logits -- float32, and bfloat16 for the direct-output width the
producer asks for once the route is predicted to serve -- and asserts the downstream token
set, after the shared group expansion, tail append and page/offset transform, still
matches.
"""

import logging
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.attention.dsa import deep_select as ds
from sglang.kernels.ops.attention.dsa.deep_select import (
    kpool_group_topk_shape_reason,
    topk_kpool_groups,
)
from sglang.kernels.ops.attention.dsa.kpool_topk_transform import (
    fast_kpool_topk_transform_fused,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_ppu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not is_ppu(), reason="Requires the PPU deep_select wheel"
)

POOL = 4
TOPK = 2048
GROUP_TOPK = TOPK // POOL
# Page-table entries are row * TABLE_BASE + column, so a page id maps back to the
# column it came from and any indexing mistake shows up as a wrong column.
TABLE_BASE = 100003


def _fused_group_ids(scores, lengths, row_starts):
    """Pool ids the incumbent fused kernel selects, recovered from its token output."""
    out = fast_kpool_topk_transform_fused(
        scores,
        lengths.to(torch.int32),
        POOL,
        TOPK,
        None,
        None,
        row_starts.to(torch.int32) if row_starts is not None else None,
        None,
        None,
    )
    return out[:, :TOPK].reshape(out.shape[0], GROUP_TOPK, POOL)[:, :, 0] // POOL


def _set_agreement(a, b):
    """Mean per-row |A and B| / |A| over non-empty rows, plus the worst row."""
    inter = total = 0
    worst = 1.0
    for i in range(a.shape[0]):
        sa = {int(v) for v in a[i].tolist() if v >= 0}
        sb = {int(v) for v in b[i].tolist() if v >= 0}
        if not sa:
            continue
        worst = min(worst, len(sa & sb) / len(sa))
        inter += len(sa & sb)
        total += len(sa)
    return inter / max(total, 1), worst


def _selected_score_values(scores, out):
    """Per-row multiset of the score values a top-k output picked, padding as -inf.

    Page ids are ``row * TABLE_BASE + token_column`` (see TABLE_BASE), so the pooled
    column decodes back with a remainder and a division. Comparing values instead of
    columns is the invariant that survives a tied selection boundary.
    """
    cols = torch.remainder(out.clamp(min=0), TABLE_BASE) // POOL
    vals = torch.gather(scores.cpu(), 1, cols.clamp(min=0).to(torch.int64))
    return vals.masked_fill(out.cpu() < 0, float("-inf")).sort(dim=1).values


@pytest.mark.parametrize(
    "rows,width",
    [(2048, 16384), (2048, 8192), (8, 1025), (16, 4097)],
)
@pytest.mark.parametrize("distribution", ["random", "ties", "tight"])
@torch.inference_mode()
def test_kpool_group_ids_match_fused_kernel(rows, width, distribution):
    torch.manual_seed(width)
    device = "cuda"
    if distribution == "random":
        scores = torch.randn(rows, width, dtype=torch.float32, device=device)
    elif distribution == "tight":
        scores = torch.randn(rows, width, dtype=torch.float32, device=device) * 0.02
    else:
        scores = (
            torch.randint(0, 64, (rows, width), device=device).to(torch.float32) / 8.0
        )
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)

    got = topk_kpool_groups(scores, lengths, GROUP_TOPK).cpu()
    ref = _fused_group_ids(scores, lengths, None).cpu()

    assert int((got[0] >= 0).sum()) == GROUP_TOPK, "padding must sit past the picks"
    if distribution == "ties":
        # With many equal scores both routes are free to pick any member of the tied
        # boundary group, so index identity is not the contract -- the selected score
        # values are. Index agreement is printed for the phase-3 record only.
        mean, _ = _set_agreement(ref, got)
        ref_scores = torch.gather(scores.cpu(), 1, ref.clamp(min=0))
        got_scores = torch.gather(scores.cpu(), 1, got.clamp(min=0))
        assert torch.equal(
            ref_scores.sort(dim=1).values, got_scores.sort(dim=1).values
        ), (
            f"tie groups differ in score values (rows={rows} width={width}); "
            f"index agreement was {mean:.4f}"
        )
    else:
        # BF16 selection churns a small number of boundary pools per row; the design
        # target is 99.5% and below 99% the phase-3 gate stops this route.
        mean, worst = _set_agreement(ref, got)
        assert mean >= 0.99, f"pool-set agreement {mean:.4f} < 0.99 ({rows}x{width})"


@pytest.mark.parametrize("rows,width,shift", [(64, 8192, 512), (2048, 9216, 1024)])
@torch.inference_mode()
def test_kpool_group_ids_honor_row_window_starts(rows, width, shift):
    """A shifted window must be searched, and ids stay window-relative like the kernel's."""
    torch.manual_seed(shift)
    device = "cuda"
    length = width - shift
    scores = torch.randn(rows, width, dtype=torch.float32, device=device)
    lengths = torch.full((rows,), length, dtype=torch.int32, device=device)
    starts = torch.full((rows,), shift, dtype=torch.int32, device=device)

    got = topk_kpool_groups(scores, lengths, GROUP_TOPK, row_starts=starts)
    ref = _fused_group_ids(scores, lengths, starts)
    assert int(got.max()) < length, "ids stay inside the row window"
    # Relative ids mean both routes describe the same absolute columns once the start
    # is applied -- which is exactly what the fused kernel's caller does downstream.
    mean, _ = _set_agreement(ref.cpu(), got.cpu())
    assert mean >= 0.99, f"windowed pool-set agreement {mean:.4f} < 0.99"
    # Independent reference: what was searched is the window [shift, shift + length),
    # not the row head. A missing window shift would show up here as a large gap.
    window_ref = torch.topk(
        scores[:, shift : shift + length], GROUP_TOPK, dim=1
    ).indices.cpu()
    mean_w, _ = _set_agreement(window_ref, got.cpu())
    assert mean_w >= 0.99, f"window reference agreement {mean_w:.4f} < 0.99"


@torch.inference_mode()
def test_kpool_group_ids_short_row_matches_reference_exactly():
    """A row with no more pools than the budget is emitted in order, then padded."""
    device = "cuda"
    rows, width, length = 8, 300, 120
    torch.manual_seed(7)
    scores = torch.randn(rows, width, dtype=torch.float32, device=device)
    lengths = torch.full((rows,), length, dtype=torch.int32, device=device)

    got = topk_kpool_groups(scores, lengths, GROUP_TOPK)
    expected = torch.arange(length, dtype=torch.int32, device=device)
    assert torch.equal(got[:, :length], expected.unsqueeze(0).expand(rows, length))
    assert bool((got[:, length:] == -1).all())
    ref = _fused_group_ids(scores, lengths, None)
    assert torch.equal(ref[:, :length], got[:, :length])


@torch.inference_mode()
def test_pooled_topk_end_to_end_matches_fused_route():
    """The public entry point must return the same token columns with the switch on."""
    from sglang.srt.layers.attention.dsa import kpool_fp8_index as kpi

    device = "cuda"
    rows, pool_cols = 2048, 8192
    seq_len = pool_cols * POOL
    torch.manual_seed(11)
    logits = torch.randn(rows, pool_cols, dtype=torch.float32, device=device)
    group_lengths = torch.full((rows,), pool_cols, dtype=torch.int32, device=device)
    seq_lens = torch.full((rows,), seq_len, dtype=torch.int32, device=device)
    # Every pooled token column gets its own identifiable page-table entry.
    page_table = (
        torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1) * TABLE_BASE
        + torch.arange(pool_cols * POOL, device=device, dtype=torch.int64).unsqueeze(0)
    ).to(torch.int32)

    served_rows, served_cols = ds.KPOOL_SERVE_MIN_ROWS, ds.KPOOL_SERVE_MIN_COLS
    ds.KPOOL_SERVE_MIN_ROWS, ds.KPOOL_SERVE_MIN_COLS = 4, 64
    try:
        with mock.patch.object(envs.SGLANG_DSA_DEEPSELECT, "get", return_value=True):
            on = kpi.topk_from_pooled_history_logits(
                logits,
                group_lengths,
                POOL,
                TOPK,
                page_table=page_table,
                seq_lens=seq_lens,
            ).cpu()
        off = kpi.topk_from_pooled_history_logits(
            logits,
            group_lengths,
            POOL,
            TOPK,
            page_table=page_table,
            seq_lens=seq_lens,
        ).cpu()
    finally:
        ds.KPOOL_SERVE_MIN_ROWS, ds.KPOOL_SERVE_MIN_COLS = served_rows, served_cols

    assert on.shape == off.shape == (rows, TOPK + POOL - 1)
    # Page ids decode back to columns through the TABLE_BASE convention.
    on_cols = torch.remainder(on.clamp(min=0), TABLE_BASE)
    off_cols = torch.remainder(off.clamp(min=0), TABLE_BASE)
    mean, worst = _set_agreement(off_cols, on_cols)
    assert mean >= 0.99, f"end-to-end token-set agreement {mean:.4f} < 0.99"
    # The forced tail columns must exist on both routes (pooled rows are closed here,
    # so the tail width is the (pool_size - 1) padding columns of -1).
    assert bool((on[:, TOPK:] == -1).all()) and bool((off[:, TOPK:] == -1).all())


@torch.inference_mode()
def test_pooled_topk_matches_fused_route_with_page_table_row_index():
    """Prefill calls address the page table through the request pool, not by query row.

    The entry used to refuse those calls outright, which is exactly the shape the
    served band exists for, so the row-index gather has to reproduce the fused
    kernel's per-query page row -- including which request row a column came from.
    """
    from sglang.srt.layers.attention.dsa import kpool_fp8_index as kpi

    device = "cuda"
    rows, pool_cols, req_rows = 2048, 8192, 8
    token_cols = pool_cols * POOL
    seq_len = token_cols
    torch.manual_seed(17)
    logits = torch.randn(rows, pool_cols, dtype=torch.float32, device=device)
    group_lengths = torch.full((rows,), pool_cols, dtype=torch.int32, device=device)
    seq_lens = torch.full((rows,), seq_len, dtype=torch.int32, device=device)
    # Request-pool table: the high part names the request row, the low part the column.
    page_table = (
        torch.arange(req_rows, device=device, dtype=torch.int64).unsqueeze(1)
        * TABLE_BASE
        + torch.arange(token_cols, device=device, dtype=torch.int64).unsqueeze(0)
    ).to(torch.int32)
    row_index = torch.arange(rows, device=device, dtype=torch.int32) % req_rows

    served_rows, served_cols = ds.KPOOL_SERVE_MIN_ROWS, ds.KPOOL_SERVE_MIN_COLS
    ds.KPOOL_SERVE_MIN_ROWS, ds.KPOOL_SERVE_MIN_COLS = 4, 64
    torch.cuda.reset_peak_memory_stats()
    base_bytes = torch.cuda.memory_allocated()
    try:
        with mock.patch.object(envs.SGLANG_DSA_DEEPSELECT, "get", return_value=True):
            on = kpi.topk_from_pooled_history_logits(
                logits,
                group_lengths,
                POOL,
                TOPK,
                page_table=page_table,
                seq_lens=seq_lens,
                page_table_row_index=row_index,
            ).cpu()
        peak_bytes = torch.cuda.max_memory_allocated() - base_bytes
        torch.cuda.reset_peak_memory_stats()
        base_off = torch.cuda.memory_allocated()
        off = kpi.topk_from_pooled_history_logits(
            logits,
            group_lengths,
            POOL,
            TOPK,
            page_table=page_table,
            seq_lens=seq_lens,
            page_table_row_index=row_index,
        ).cpu()
        peak_off = torch.cuda.max_memory_allocated() - base_off
    finally:
        ds.KPOOL_SERVE_MIN_ROWS, ds.KPOOL_SERVE_MIN_COLS = served_rows, served_cols

    # Materialising one page-table row per query would cost rows x token_cols x 4 bytes
    # (256 MiB at this shape) on top of everything else; the indexed route must not.
    # Its peak is also required to stay in the same order as the fused route it mirrors,
    # so the shared expansion chain is not quietly copying the table either.
    copy_cost = rows * token_cols * page_table.element_size()
    result_cost = rows * (TOPK + POOL - 1) * 4
    assert peak_bytes < copy_cost, (
        f"row-index route allocated {peak_bytes / 2**20:.1f} MiB peak, a full per-query "
        f"table copy alone would be {copy_cost / 2**20:.1f} MiB"
    )
    assert peak_bytes <= peak_off * 2 + result_cost, (
        f"indexed peak {peak_bytes / 2**20:.1f} MiB vs fused route "
        f"{peak_off / 2**20:.1f} MiB"
    )
    assert peak_off >= result_cost, (
        f"fused-route peak {peak_off / 2**20:.1f} MiB below the "
        f"{result_cost / 2**20:.1f} MiB result -- measurement not meaningful"
    )

    assert on.shape == off.shape == (rows, TOPK + POOL - 1)
    on_cols = torch.remainder(on.clamp(min=0), TABLE_BASE)
    off_cols = torch.remainder(off.clamp(min=0), TABLE_BASE)
    mean, _worst = _set_agreement(off_cols, on_cols)
    assert mean >= 0.99, f"row-index end-to-end token-set agreement {mean:.4f} < 0.99"
    # The request row must survive the gather: every served column came from the pool
    # row this query is mapped to. A missing or misaligned gather fails here.
    on_req = torch.div(on.clamp(min=0), TABLE_BASE, rounding_mode="floor")
    expected = row_index.cpu().unsqueeze(1).expand_as(on_req)
    valid = on >= 0
    assert bool((on_req[valid] == expected[valid]).all()), (
        "page row index lost in gather"
    )
    assert bool((on[:, TOPK:] == -1).all()) and bool((off[:, TOPK:] == -1).all())


@torch.inference_mode()
def test_bf16_logits_select_the_same_pools_as_fp32():
    """The bf16 direct producer changes the score width, so the picks must not move.

    This is the band-internal consistency gate for the bf16 route: the same values
    handed over at float32 (today's route) and at bfloat16 (what the scoring kernel
    will emit after the change) have to agree on the pooled top-k, with the
    boundary churn measured and reported row by row.
    """
    device = "cuda"
    rows, width = 2048, 8192
    torch.manual_seed(23)
    scores = torch.randn(rows, width, dtype=torch.float32, device=device)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)

    fp32 = topk_kpool_groups(scores, lengths, GROUP_TOPK).cpu()
    bf16 = topk_kpool_groups(scores.to(torch.bfloat16), lengths, GROUP_TOPK).cpu()

    mean, worst = _set_agreement(fp32, bf16)
    print(f"[kpool bf16] rows={rows} cols={width} mean={mean:.4f} worst={worst:.4f}")
    assert mean >= 0.99, f"bf16-vs-fp32 pool agreement {mean:.4f} < 0.99"
    assert int((bf16[0] >= 0).sum()) == GROUP_TOPK, "padding must sit past the picks"
    # A tight distribution is where bf16 rounding can actually reorder candidates.
    tight = scores * 0.02
    tight_mean, tight_worst = _set_agreement(
        topk_kpool_groups(tight, lengths, GROUP_TOPK).cpu(),
        topk_kpool_groups(tight.to(torch.bfloat16), lengths, GROUP_TOPK).cpu(),
    )
    print(f"[kpool bf16] tight mean={tight_mean:.4f} worst={tight_worst:.4f}")
    assert tight_mean >= 0.99, f"tight bf16 agreement {tight_mean:.4f} < 0.99"


@torch.inference_mode()
def test_bf16_logits_end_to_end_match_the_fp32_served_route():
    """Token columns out of the public entry point must move no more than the picks."""
    from sglang.srt.layers.attention.dsa import kpool_fp8_index as kpi

    device = "cuda"
    rows, pool_cols = 2048, 8192
    torch.manual_seed(29)
    logits = torch.randn(rows, pool_cols, dtype=torch.float32, device=device)
    group_lengths = torch.full((rows,), pool_cols, dtype=torch.int32, device=device)
    seq_lens = torch.full((rows,), pool_cols * POOL, dtype=torch.int32, device=device)
    page_table = (
        torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1) * TABLE_BASE
        + torch.arange(pool_cols * POOL, device=device, dtype=torch.int64).unsqueeze(0)
    ).to(torch.int32)

    with mock.patch.object(envs.SGLANG_DSA_DEEPSELECT, "get", return_value=True):
        fp32 = kpi.topk_from_pooled_history_logits(
            logits, group_lengths, POOL, TOPK, page_table=page_table, seq_lens=seq_lens
        ).cpu()
        bf16 = kpi.topk_from_pooled_history_logits(
            logits.to(torch.bfloat16),
            group_lengths,
            POOL,
            TOPK,
            page_table=page_table,
            seq_lens=seq_lens,
        ).cpu()

    assert bf16.shape == fp32.shape == (rows, TOPK + POOL - 1)
    mean, worst = _set_agreement(
        torch.remainder(fp32.clamp(min=0), TABLE_BASE),
        torch.remainder(bf16.clamp(min=0), TABLE_BASE),
    )
    print(f"[kpool bf16] end-to-end mean={mean:.4f} worst={worst:.4f}")
    assert mean >= 0.99, f"bf16 end-to-end token agreement {mean:.4f} < 0.99"
    assert bool((bf16[:, TOPK:] == -1).all())


@torch.inference_mode()
def test_bf16_hit_marker_reports_the_emitted_width(caplog):
    """The served log line is the A/B evidence gate, so it must name the width."""
    from sglang.srt.layers.attention.dsa import dsa_topk_backend as tb

    device = "cuda"
    rows, width = 2048, 8192
    scores = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    tb._DEEPSELECT_LOGGED.discard(tb._DEEPSELECT_KPOOL_HIT)
    with (
        caplog.at_level(logging.INFO),
        mock.patch.object(envs.SGLANG_DSA_DEEPSELECT, "get", return_value=True),
    ):
        assert tb.deepselect_kpool_group_ids(scores, lengths, GROUP_TOPK) is not None
    hit = [
        r.getMessage()
        for r in caplog.records
        if "kpool top-k served by PPU deep_select" in r.getMessage()
    ]
    assert hit, "a served call must leave a greppable marker"
    assert "logits=torch.bfloat16" in hit[0], hit[0]


@torch.inference_mode()
def test_bf16_scores_outside_the_band_run_the_fused_kernel_in_fp32(caplog):
    """Band-out calls cannot reach the wheel, so the bf16 width must be upcast first."""
    from sglang.srt.layers.attention.dsa import kpool_fp8_index as kpi

    device = "cuda"
    rows, pool_cols = 8, 4096
    torch.manual_seed(31)
    logits = torch.randn(rows, pool_cols, dtype=torch.float32, device=device)
    group_lengths = torch.full((rows,), pool_cols, dtype=torch.int32, device=device)
    seq_lens = torch.full((rows,), pool_cols * POOL, dtype=torch.int32, device=device)
    page_table = (
        torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1) * TABLE_BASE
        + torch.arange(pool_cols * POOL, device=device, dtype=torch.int64).unsqueeze(0)
    ).to(torch.int32)
    kwargs = dict(page_table=page_table, seq_lens=seq_lens)

    with (
        caplog.at_level(logging.INFO),
        mock.patch.object(envs.SGLANG_DSA_DEEPSELECT, "get", return_value=True),
    ):
        kpi._KPOOL_ROUTE_LOGGED.clear()
        via_fp32 = kpi.topk_from_pooled_history_logits(
            logits, group_lengths, POOL, TOPK, **kwargs
        ).cpu()
        caplog.clear()
        kpi._KPOOL_ROUTE_LOGGED.clear()
        via_bf16 = kpi.topk_from_pooled_history_logits(
            logits.to(torch.bfloat16), group_lengths, POOL, TOPK, **kwargs
        ).cpu()
    fallback = [
        r.getMessage()
        for r in caplog.records
        if "fell back to the fp32 selection path" in r.getMessage()
    ]
    assert fallback, "an unserved bfloat16 call must say that it upcast"
    # The fallback is a plain width change, not a second numeric path: the selected
    # score *values* must be identical to a call that was handed the same rounded
    # values at float32. Column identity is not required, because a bf16-rounded row
    # has ties at the selection boundary and the incumbent kernel is free to answer
    # any member of the tied boundary group (the same churn the ties case of
    # test_kpool_group_ids_match_fused_kernel already allows for).
    rounded = logits.to(torch.bfloat16).to(torch.float32)
    same_as_fp32 = kpi.topk_from_pooled_history_logits(
        rounded.clone(), group_lengths, POOL, TOPK, **kwargs
    ).cpu()
    again = kpi.topk_from_pooled_history_logits(
        rounded.clone(), group_lengths, POOL, TOPK, **kwargs
    ).cpu()
    print(
        "[kpool bf16] fused route bit-reproducible on identical input: "
        f"{torch.equal(same_as_fp32, again)}"
    )
    assert torch.equal(
        _selected_score_values(rounded, via_bf16),
        _selected_score_values(rounded, same_as_fp32),
    ), "the fp32 upcast changed which scores were selected"
    # bf16 rounding churns boundary pools; that is the whole difference from fp32.
    mean, _ = _set_agreement(
        torch.remainder(via_fp32.clamp(min=0), TABLE_BASE),
        torch.remainder(via_bf16.cpu().clamp(min=0), TABLE_BASE),
    )
    assert mean >= 0.99, f"band-out bf16 fallback agreement {mean:.4f} < 0.99"


@torch.inference_mode()
def test_bf16_staging_buffer_is_reused_not_reallocated(monkeypatch):
    """The staging block is allocated once per device and only grown, never per call."""
    device = torch.device("cuda")
    ds._kpool_pack_buffers.clear()

    real_empty = torch.empty
    allocations = {"n": 0}

    def counting_empty(*args, **kwargs):
        allocations["n"] += 1
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", counting_empty)
    first = ds._kpool_pack_buffer(64, 8192, device)
    assert allocations["n"] == 1, "the first request allocates the staging block once"
    again = ds._kpool_pack_buffer(64, 8192, device)
    assert again.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    assert allocations["n"] == 1, "a repeated request must not allocate"
    smaller = ds._kpool_pack_buffer(8, 1024, device)
    assert smaller.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    assert allocations["n"] == 1, "a smaller request must reuse the cached block"
    assert first.stride(0) % (ds._alignment_bytes() // first.element_size()) == 0
    monkeypatch.undo()

    # Through the real selection path the cached address must stay stable, so no
    # staging allocation happens per DSA layer or per CUDA graph replay.
    scores = torch.randn(64, 8192, dtype=torch.float32, device=device)
    lengths = torch.full((64,), 8192, dtype=torch.int32, device=device)
    topk_kpool_groups(scores, lengths, GROUP_TOPK)
    torch.cuda.synchronize()
    key = device.index if device.index is not None else torch.cuda.current_device()
    cached_ptr = ds._kpool_pack_buffers[key].data_ptr()
    base = torch.cuda.memory_allocated(device)
    for _ in range(3):
        topk_kpool_groups(scores, lengths, GROUP_TOPK)
        torch.cuda.synchronize()
    assert ds._kpool_pack_buffers[key].data_ptr() == cached_ptr, "staging address moved"
    grew = torch.cuda.memory_allocated(device) - base
    assert grew < 8 * 1024 * 1024, (
        f"staging churn grew {grew / 1e6:.1f} MB across 3 calls"
    )
    ds._kpool_pack_buffers.clear()


def test_kpool_shape_reason_boundaries():
    assert kpool_group_topk_shape_reason(2048, 8192, GROUP_TOPK) is None
    assert "below the kpool served band" in kpool_group_topk_shape_reason(
        128, 8192, GROUP_TOPK
    )
    assert "below the kpool served band" in kpool_group_topk_shape_reason(
        2048, 4096, GROUP_TOPK
    )
    assert "not a served kpool bucket" in kpool_group_topk_shape_reason(4096, 40960, 96)


def test_disabled_switch_never_touches_the_wheel():
    from sglang.srt.layers.attention.dsa import dsa_topk_backend as tb

    def _boom(*args, **kwargs):
        raise AssertionError("must not be reached while the switch is off")

    with (
        mock.patch.object(envs.SGLANG_DSA_DEEPSELECT, "get", return_value=False),
        mock.patch.object(ds, "topk_kpool_groups", _boom),
    ):
        logits = torch.randn(4096, 16384, dtype=torch.float32)
        lengths = torch.full((4096,), 16384, dtype=torch.int32)
        assert tb.deepselect_kpool_group_ids(logits, lengths, GROUP_TOPK) is None, (
            "the switch off must short-circuit before any wheel call"
        )
