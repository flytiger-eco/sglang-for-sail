"""INT8 paged indexer logits with several query rows per request.

The verify forward of DSpark/MTP hands the low-ratio INT8 indexer one group of
``rows_per_request`` rows per request, and the DeepGEMM plan is built for those
groups. This pins the kernel contract: a grouped call must reproduce the
per-row calls on every row's visible columns. Rows of one request carry
non-decreasing lengths (later draft positions see at least as much context),
which is what lets the schedule size the K walk from the group's last row.

Runs on PPU (where the low-ratio indexer is INT8) and skips elsewhere.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.utils import get_device_sm, is_sail

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sail() or get_device_sm() >= 89,
    reason="the low-ratio indexer is INT8 only on PPU below SM89",
)

HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64
GROUP = 6


def _page_table_pool(batch: int, pages: int, device: str):
    # Per page: page_size * 128 INT8 K bytes, then page_size FP32 scales.
    flat = torch.zeros(
        (batch * pages, PAGE_SIZE * 132), dtype=torch.uint8, device=device
    )
    k = torch.randint(
        -64, 64, (batch * pages, PAGE_SIZE, HEAD_DIM), dtype=torch.int8, device=device
    )
    flat[:, : PAGE_SIZE * HEAD_DIM] = k.reshape(batch * pages, -1).view(torch.uint8)
    flat[:, PAGE_SIZE * HEAD_DIM :] = (
        torch.tensor([0, 0, 128, 63], dtype=torch.uint8, device=device)
        .repeat_interleave(PAGE_SIZE)
        .repeat(batch * pages, 1)
    )
    return flat


def test_grouped_rows_match_per_row_calls():
    import deep_gemm

    from sglang.kernels.ops.attention.dsv4.int8_indexer import int8_paged_index_logits

    device = "cuda"
    torch.manual_seed(7)
    batch, pages, width = 3, 2, PAGE_SIZE * 2
    flat = _page_table_pool(batch, pages, device)
    cache = flat
    page_table = torch.arange(batch * pages, dtype=torch.int32, device=device).view(
        batch, pages
    )
    # Per row view, as the indexer metadata stores it.
    row_table = page_table.repeat_interleave(GROUP, dim=0)

    q = torch.randint(
        -16, 16, (batch * GROUP, HEADS, HEAD_DIM), dtype=torch.int8, device=device
    )
    weights = torch.rand(batch * GROUP, HEADS, device=device) * 0.05
    # Non-decreasing within a request, as verify produces them.
    lens = torch.tensor(
        [[64, 72, 80, 88, 96, 104], [64, 72, 80, 88, 96, 128], [128] * 6],
        dtype=torch.int32,
        device=device,
    ).reshape(-1)
    grouped_lens = lens.view(batch, GROUP)

    def plan_of(lens_arg):
        return deep_gemm.get_paged_mqa_logits_metadata(
            lens_arg,
            PAGE_SIZE,
            deep_gemm.get_num_sms(),
            metadata_extra=(1, HEADS, HEAD_DIM, 1),
        )

    grouped = int8_paged_index_logits(
        q,
        weights,
        cache,
        lens,
        row_table,
        plan_of(grouped_lens),
        width,
        PAGE_SIZE,
        rows_per_request=GROUP,
    )
    assert grouped.shape == (batch * GROUP, width)

    per_row = torch.cat(
        [
            int8_paged_index_logits(
                q[row : row + 1],
                weights[row : row + 1],
                cache,
                lens[row : row + 1],
                row_table[row : row + 1],
                plan_of(lens[row : row + 1].view(1, 1)),
                width,
                PAGE_SIZE,
            )
            for row in range(batch * GROUP)
        ]
    )

    visible = torch.arange(width, device=device)[None, :] < lens[:, None]
    assert torch.isfinite(grouped[visible]).all()
    assert torch.equal(grouped[visible], per_row[visible])
