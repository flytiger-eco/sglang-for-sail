"""PPU wheel integration: BF16 row windows, offsets, alignment and padding."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.deep_select import (
    MAX_TOPK,
    paged_topk_skip_reason,
    topk_transform_paged,
    topk_transform_ragged,
)
from sglang.srt.utils import is_ppu
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not is_ppu(), reason="Requires the PPU deep_select wheel"
)

# Page table entries are row * TABLE_BASE + column, so a page id can be mapped back to
# the column it came from: that makes any indexing mistake visible as a wrong column.
TABLE_BASE = 100003


@pytest.mark.parametrize("topk", [1, 8, 512, 2048])
@pytest.mark.parametrize(
    "width,layout",
    [
        (356, "zero"),
        (1025, "none"),
        (8193, "unaligned"),
        (16385, "packed"),
        (65537, "packed"),
        (65537, "packed_compact"),
        (1025, "unaligned_storage"),
        (1025, "strided_columns"),
    ],
)
@pytest.mark.parametrize("distribution", ["random", "ties", "negative_inf"])
@torch.inference_mode()
def test_deep_select_ragged_windows(topk, width, layout, distribution):
    torch.manual_seed(width)
    rows, device = 4, "cuda"
    padded_width = (width + 511) // 512 * 512
    if layout == "unaligned_storage":
        scores = torch.empty(rows, padded_width, dtype=torch.bfloat16, device=device)[
            :, 1 : width + 1
        ]
    elif layout == "strided_columns":
        scores = torch.empty(
            rows, padded_width * 2, dtype=torch.bfloat16, device=device
        )[:, : width * 2 : 2]
    else:
        scores = torch.empty(rows, padded_width, dtype=torch.bfloat16, device=device)[
            :, :width
        ]
    scores.fill_(1000)
    if layout in ("none", "zero"):
        starts_cpu = [0, 0, 0, 0]
    elif layout in ("packed", "packed_compact"):
        starts_cpu = [0, 3, width // 4 + 1, width // 2 + 3]
    else:
        starts_cpu = [0, 1, 2, 3]
    lengths_cpu = [
        0,
        min(topk - 1, width - starts_cpu[1]),
        min(topk + 1, width - starts_cpu[2]),
        width - starts_cpu[3],
    ]
    for row, (start, length) in enumerate(zip(starts_cpu, lengths_cpu)):
        values = torch.randn(length, device=device)
        if distribution == "ties":
            values.fill_(0.5)
        elif distribution == "negative_inf":
            values.fill_(-torch.inf)
            values[: min(length, 3)] = torch.arange(min(length, 3), device=device)
        scores[row, start : start + length] = values
    before = scores.clone()
    starts = torch.tensor(starts_cpu, dtype=torch.int32, device=device)
    lengths = torch.tensor(lengths_cpu, dtype=torch.int32, device=device)
    offsets = torch.tensor([100, 20000, 40000, 60000], dtype=torch.int32, device=device)

    out = topk_transform_ragged(
        scores,
        lengths,
        topk,
        offsets,
        None if layout == "none" else starts,
        **({"max_seq_len": max(lengths_cpu)} if layout == "packed_compact" else {}),
    )
    torch.cuda.synchronize()
    assert out.shape == (rows, topk)
    assert out.dtype == torch.int32
    assert out.is_contiguous()
    assert torch.equal(scores, before)
    for row, (start, length) in enumerate(zip(starts_cpu, lengths_cpu)):
        valid = out[row] != -1
        assert int(valid.sum()) == min(topk, length)
        indices = out[row, valid].long() - offsets[row]
        assert bool(((indices >= 0) & (indices < length)).all())
        assert indices.unique().numel() == indices.numel()
        window = before[row, start : start + length].float()
        expected = window.topk(min(topk, length)).values.sort().values
        assert torch.equal(window[indices].sort().values, expected)


@torch.inference_mode()
def test_deep_select_empty_rows():
    scores = torch.empty((0, 1024), dtype=torch.bfloat16, device="cuda")
    lengths = torch.empty(0, dtype=torch.int32, device="cuda")
    out = topk_transform_ragged(scores, lengths, 2048, lengths)
    assert out.shape == (0, 2048)
    assert out.dtype == torch.int32


def _paged_case(rows, width, start, lengths_cpu, seed):
    """BF16 score matrix, a shared window start and an invertible per-query page table."""
    torch.manual_seed(seed)
    scores = torch.randn((rows, width), dtype=torch.float32, device="cuda").to(
        torch.bfloat16
    )
    # Columns outside every row's window are set to the largest BF16 value, so a route
    # that ignores the window bound shows up as a wrong pick immediately.
    scores[:, start + max(lengths_cpu) :] = torch.finfo(torch.bfloat16).max
    row_starts = torch.full((rows,), start, dtype=torch.int32, device="cuda")
    lengths = torch.tensor(lengths_cpu, dtype=torch.int32, device="cuda")
    tables = (
        torch.arange(rows, dtype=torch.int32, device="cuda").unsqueeze(1) * TABLE_BASE
        + torch.arange(width, dtype=torch.int32, device="cuda").unsqueeze(0)
    ).contiguous()
    return scores, row_starts, lengths, tables


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("width", [1024, 8192])
@pytest.mark.parametrize("start", [0, 8, 64])
@torch.inference_mode()
def test_deep_select_paged_matches_the_fused_kernel(topk, width, start):
    """The PAGED route must return what the fused dsv4 bf16 kernel returns.

    BF16 scores contain ties, and both kernels are free to pick any tied index, so the
    contract is checked in three layers: identical selected score multiset per row
    (always), identical selected column set on rows whose k-th value is unique, and
    identical padding counts everywhere.
    """
    from sglang.kernels.ops.attention.dsv4.topk import top_k_per_row_prefill_bf16

    rows = 8
    lengths_cpu = [
        0,
        min(topk - 1, width - start),
        min(topk + 1, width - start),
        width - start,
        width - start,
        max(width - start - topk, 1),
        min(2 * topk, width - start),
        width - start,
    ]
    scores, row_starts, lengths, tables = _paged_case(
        rows, width, start, lengths_cpu, width
    )
    row_ends = row_starts + lengths
    before = scores.clone()

    expected = scores.new_full((rows, topk), -1, dtype=torch.int32)
    top_k_per_row_prefill_bf16(scores, row_starts, row_ends, tables, expected, 1, None)
    got = topk_transform_paged(scores, lengths, topk, row_starts, tables)
    torch.cuda.synchronize()

    assert got is not None, "the layout is inside the wheel contract"
    assert (
        got.shape == (rows, topk) and got.dtype == torch.int32 and got.is_contiguous()
    )
    assert torch.equal(scores, before), "the caller's scores must not be modified"

    exp_cols = expected.long() - (
        torch.arange(rows, device="cuda").unsqueeze(1) * TABLE_BASE
    )
    exp_cols = torch.where(expected < 0, torch.full_like(exp_cols, -1), exp_cols)
    got_cols = got.long() - (
        torch.arange(rows, device="cuda").unsqueeze(1) * TABLE_BASE
    )
    got_cols = torch.where(got < 0, torch.full_like(got_cols, -1), got_cols)

    assert int((expected < 0).sum()) == int((got < 0).sum())
    tie_free_rows = 0
    for row, length in enumerate(lengths_cpu):
        n = min(topk, length)
        assert int((got[row] >= 0).sum()) == n, f"row {row} selected count"
        assert int((expected[row] >= 0).sum()) == n
        if n == 0:
            continue
        window = before[row, start : start + length].float()
        kth = window.topk(n).values[-1]
        # Selected score multisets must be identical: the real semantic contract.
        # Both arms index the page table with the position inside the window, which is
        # exactly how the local `window` slice is addressed.
        got_pos = got_cols[row, :n].long().clamp(min=0, max=max(length - 1, 0))
        exp_pos = exp_cols[row, :n].long().clamp(min=0, max=max(length - 1, 0))
        assert torch.equal(
            torch.gather(window, 0, got_pos).sort().values,
            torch.gather(window, 0, exp_pos).sort().values,
        )
        if int((window == kth).sum()) == 1:
            # No tie at the cut-off: the chosen columns are fully determined.
            tie_free_rows += 1
            assert set(got_cols[row, :n].tolist()) == set(exp_cols[row, :n].tolist()), (
                f"row {row} columns must match exactly on a tie-free row"
            )
            assert set(got_pos.tolist()) == set(exp_pos.tolist())
    assert tie_free_rows > 0, "the fixture must contain at least one tie-free row"


@torch.inference_mode()
def test_deep_select_paged_rejects_non_constant_row_starts():
    rows, width, topk = 4, 1024, 8
    scores, _, lengths, tables = _paged_case(rows, width, 0, [width] * rows, seed=1)
    row_starts = torch.tensor([0, 8, 16, 24], dtype=torch.int32, device="cuda")
    assert topk_transform_paged(scores, lengths, topk, row_starts, tables) is None, (
        "the wheel has no begin offset, so mixed starts must not be served"
    )


@torch.inference_mode()
def test_deep_select_paged_empty_rows():
    scores = torch.empty((0, 1024), dtype=torch.bfloat16, device="cuda")
    lengths = torch.empty(0, dtype=torch.int32, device="cuda")
    tables = torch.empty((0, 1024), dtype=torch.int32, device="cuda")
    out = topk_transform_paged(scores, lengths, 2048, None, tables)
    assert out is not None and out.shape == (0, 2048)
    assert out.dtype == torch.int32


def test_paged_topk_skip_reason_tracks_the_wheel_contract():
    good = torch.empty(2, 1024, dtype=torch.bfloat16)
    assert paged_topk_skip_reason(good, 2048) is None
    assert paged_topk_skip_reason(good, 0) is not None
    assert paged_topk_skip_reason(good, MAX_TOPK + 1) is not None
    assert paged_topk_skip_reason(good.float(), 8) is not None
    assert paged_topk_skip_reason(good[0], 8) is not None
    assert paged_topk_skip_reason(good[:, ::2], 8) is not None
    # A row stride that is not a multiple of the documented alignment is rejected.
    storage = torch.empty(2, 1025, dtype=torch.bfloat16)
    assert paged_topk_skip_reason(storage[:, :1024], 8) is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
