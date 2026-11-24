"""PPU wheel integration: kpool (index_kpool>1) group-level top-k through DeepSelect.

The Flash weights select pool-level top-k inside ``fast_kpool_topk_transform_fused``.
The DeepSelect route replaces only that selection, so every test here compares the two
routes on the same float32 pooled logits and asserts the downstream token set -- after
the shared group expansion, tail append and page/offset transform -- still matches.
"""

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
        assert torch.equal(ref_scores.sort(dim=1).values, got_scores.sort(dim=1).values), (
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
    window_ref = (
        torch.topk(scores[:, shift : shift + length], GROUP_TOPK, dim=1).indices.cpu()
    )
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
        with mock.patch.object(
            envs.SGLANG_DSA_DEEPSELECT, "get", return_value=True
        ):
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
        torch.arange(req_rows, device=device, dtype=torch.int64).unsqueeze(1) * TABLE_BASE
        + torch.arange(token_cols, device=device, dtype=torch.int64).unsqueeze(0)
    ).to(torch.int32)
    row_index = torch.arange(rows, device=device, dtype=torch.int32) % req_rows

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
                page_table_row_index=row_index,
            ).cpu()
        off = kpi.topk_from_pooled_history_logits(
            logits,
            group_lengths,
            POOL,
            TOPK,
            page_table=page_table,
            seq_lens=seq_lens,
            page_table_row_index=row_index,
        ).cpu()
    finally:
        ds.KPOOL_SERVE_MIN_ROWS, ds.KPOOL_SERVE_MIN_COLS = served_rows, served_cols

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
    assert bool((on_req[valid] == expected[valid]).all()), "page row index lost in gather"
    assert bool((on[:, TOPK:] == -1).all()) and bool((off[:, TOPK:] == -1).all())


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
    assert grew < 8 * 1024 * 1024, f"staging churn grew {grew / 1e6:.1f} MB across 3 calls"
    ds._kpool_pack_buffers.clear()


def test_kpool_shape_reason_boundaries():
    assert kpool_group_topk_shape_reason(2048, 8192, GROUP_TOPK) is None
    assert "below the kpool served band" in kpool_group_topk_shape_reason(
        128, 8192, GROUP_TOPK
    )
    assert "below the kpool served band" in kpool_group_topk_shape_reason(
        2048, 4096, GROUP_TOPK
    )
    assert "not a served kpool bucket" in kpool_group_topk_shape_reason(
        4096, 40960, 96
    )


def test_disabled_switch_never_touches_the_wheel():
    from sglang.srt.layers.attention.dsa import dsa_topk_backend as tb

    def _boom(*args, **kwargs):
        raise AssertionError("must not be reached while the switch is off")

    with mock.patch.object(
        envs.SGLANG_DSA_DEEPSELECT, "get", return_value=False
    ), mock.patch.object(ds, "topk_kpool_groups", _boom):
        logits = torch.randn(4096, 16384, dtype=torch.float32)
        lengths = torch.full((4096,), 16384, dtype=torch.int32)
        assert (
            tb.deepselect_kpool_group_ids(logits, lengths, GROUP_TOPK) is None
        ), "the switch off must short-circuit before any wheel call"
