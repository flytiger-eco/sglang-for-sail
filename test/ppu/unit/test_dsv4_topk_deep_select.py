"""DeepSelect-backed DSV4 page-table top-k against the torch reference.

Runs on any device where the precompiled ``deep_select`` wheel is importable
(the PPU build), and skips elsewhere. Selection happens in bf16, so the
reference is taken on the bf16-rounded scores. Score rows hold distinct bf16
values, so the selected index sets are unambiguous.
"""

from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    is_deep_select_available,
    topk_transform_paged_deep_select,
    topk_transform_paged_torch,
    topk_transform_ragged_deep_select,
    topk_transform_sparse_deep_select,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_deep_select_available()),
    reason="requires the deep_select wheel on a CUDA-compatible device",
)


def _distinct_bf16_rows(
    rows: int, width: int, device: torch.device, seed: int = 0
) -> torch.Tensor:
    # Every finite bf16 bit pattern is a distinct value; sample without
    # replacement so a row has no ties at the top-k boundary.
    bits = torch.arange(1 << 16, dtype=torch.int32, device=device).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    gen = torch.Generator(device=device).manual_seed(seed)
    picks = torch.stack(
        [
            torch.randperm(values.numel(), generator=gen, device=device)[:width]
            for _ in range(rows)
        ]
    )
    return values[picks]


def _reference(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor | None,
    k: int,
    page_size: int,
):
    rows = scores.shape[0]
    page_indices = torch.empty((rows, k), dtype=torch.int32, device=scores.device)
    raw_indices = torch.empty_like(page_indices)
    wants_pages = page_tables is not None
    if not wants_pages:
        # Identity page table at the requested page size over the score width.
        num_pages = (scores.shape[1] + page_size - 1) // page_size
        page_tables = (
            torch.arange(num_pages, dtype=torch.int32, device=scores.device)
            .expand(rows, num_pages)
            .contiguous()
        )
    topk_transform_paged_torch(
        scores,
        seq_lens,
        page_tables,
        page_indices,
        page_size,
        raw_indices,
    )
    if not wants_pages:
        return raw_indices
    return page_indices, raw_indices


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("k", [64, 100, 2048])
def test_matches_torch_reference(dtype: torch.dtype, k: int) -> None:
    device = torch.device("cuda")
    rows, width, page_size = 7, 4096, 64
    scores = _distinct_bf16_rows(rows, width, device).to(dtype)
    seq_lens = torch.tensor(
        [4096, 3000, 2048, 100, 1, 0, 4096], dtype=torch.int32, device=device
    )
    page_tables = torch.randint(
        0, 32, (rows, width // page_size), device=device, dtype=torch.int32
    )

    ref_page, ref_raw = _reference(
        scores.to(torch.bfloat16), seq_lens, page_tables, k, page_size
    )

    page_indices = torch.empty((rows, k), dtype=torch.int32, device=device)
    raw_indices = torch.empty_like(page_indices)
    topk_transform_paged_deep_select(
        scores, seq_lens, page_tables, page_indices, page_size, raw_indices
    )

    # The wheel returns the selected set in no particular order.
    torch.testing.assert_close(
        raw_indices.sort(dim=-1).values, ref_raw.sort(dim=-1).values
    )
    torch.testing.assert_close(
        page_indices.sort(dim=-1).values, ref_page.sort(dim=-1).values
    )


@pytest.mark.parametrize("width", [4000, 4096])
def test_unaligned_score_row_stride(width: int) -> None:
    # 4000 bf16 columns is not a multiple of the wheel's 1024-byte row stride,
    # so the adapter must materialize a padded copy.
    device = torch.device("cuda")
    rows, k = 3, 128
    scores = _distinct_bf16_rows(rows, width, device, seed=2)
    seq_lens = torch.tensor([width, width - 1, 7], dtype=torch.int32, device=device)
    page_tables = torch.randint(
        0, 8, (rows, width // 64 + 1), device=device, dtype=torch.int32
    )
    ref_page, ref_raw = _reference(scores, seq_lens, page_tables, k, 64)
    page_indices = torch.empty((rows, k), dtype=torch.int32, device=device)
    raw_indices = torch.empty_like(page_indices)
    topk_transform_paged_deep_select(
        scores, seq_lens, page_tables, page_indices, 64, raw_indices
    )
    torch.testing.assert_close(
        raw_indices.sort(dim=-1).values, ref_raw.sort(dim=-1).values
    )
    torch.testing.assert_close(
        page_indices.sort(dim=-1).values, ref_page.sort(dim=-1).values
    )


@pytest.mark.parametrize("k", [64, 128])
def test_sparse_compact_transform(k: int) -> None:
    # topk_transform_sparse contract: compact block-packed bf16 columns, a
    # per-row valid prefix, and column j -> blocks[b, j // 8] * 8 + j % 8.
    device = torch.device("cuda")
    rows, topk_blocks, block = 4, 16, 8
    width = topk_blocks * block
    logits = _distinct_bf16_rows(rows, width, device, seed=3)
    gen = torch.Generator(device=device).manual_seed(4)
    blocks = torch.stack(
        [
            torch.randperm(128, generator=gen, device=device)[:topk_blocks]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    valid_lens = torch.tensor(
        [width, width - 5, 29, 0], dtype=torch.int32, device=device
    )

    columns = torch.arange(width, device=device)
    masked = logits.masked_fill(columns[None, :] >= valid_lens[:, None], -torch.inf)
    picked = masked.topk(k, dim=-1, sorted=False).indices
    reach = picked < valid_lens[:, None]
    blk = blocks.gather(1, (picked.clamp_min(0) // block).to(torch.int64))
    expected = (
        torch.where(reach, blk * block + picked % block, -1)
        .to(torch.int32)
        .sort(dim=-1)
        .values
    )

    out = torch.empty((rows, k), dtype=torch.int32, device=device)
    topk_transform_sparse_deep_select(logits, valid_lens, blocks, out)
    torch.testing.assert_close(out.sort(dim=-1).values, expected)


def test_raw_indices_without_page_table() -> None:
    device = torch.device("cuda")
    rows, width, k = 4, 1024, 128
    scores = _distinct_bf16_rows(rows, width, device, seed=1)
    seq_lens = torch.tensor([1024, 512, 8, 0], dtype=torch.int32, device=device)

    ref_raw = _reference(scores, seq_lens, None, k, 64)
    raw_indices = torch.empty((rows, k), dtype=torch.int32, device=device)
    topk_transform_paged_deep_select(scores, seq_lens, None, raw_indices, 64, None)

    torch.testing.assert_close(
        raw_indices.sort(dim=-1).values, ref_raw.sort(dim=-1).values
    )


def test_ragged_offsets_match_reference() -> None:
    # Prefill contract: row i takes the top-k of its first lens[i] columns and
    # writes the position plus its request base, -1 past the valid count.
    device = torch.device("cuda")
    rows, width, k = 5, 2500, 100
    scores = _distinct_bf16_rows(rows, width, device, seed=5)
    lens = torch.tensor([2500, 1800, 0, 1, 700], dtype=torch.int32, device=device)
    offsets = torch.tensor([0, 100000, 7, 3, 42], dtype=torch.int32, device=device)

    columns = torch.arange(width, device=device)
    masked = scores.masked_fill(columns[None, :] >= lens[:, None], -torch.inf)
    picked = masked.topk(k, dim=-1, sorted=False).indices
    reach = picked < lens[:, None]
    expected = (
        torch.where(reach, picked + offsets[:, None], -1)
        .to(torch.int32)
        .sort(dim=-1)
        .values
    )

    out = torch.empty((rows, k), dtype=torch.int32, device=device)
    topk_transform_ragged_deep_select(scores, lens, out, offsets)
    torch.testing.assert_close(out.sort(dim=-1).values, expected)


def test_paged_falls_back_without_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    # Without the wheel the bf16 path must degrade to the fp32 V1 top-k, which
    # converts exactly: same selected set, no dtype error.
    from sglang.kernels.ops.attention.dsv4 import topk as topk_ops

    monkeypatch.setattr(topk_ops, "is_deep_select_available", lambda: False)
    device = torch.device("cuda")
    rows, width, k, page_size = 5, 4096, 64, 64
    scores = _distinct_bf16_rows(rows, width, device, seed=6)
    seq_lens = torch.tensor([4096, 1000, 63, 8, 0], dtype=torch.int32, device=device)
    page_tables = torch.randint(
        0, 16, (rows, width // page_size), device=device, dtype=torch.int32
    )
    ref_page, ref_raw = _reference(scores, seq_lens, page_tables, k, page_size)

    page_indices = torch.empty((rows, k), dtype=torch.int32, device=device)
    raw_indices = torch.empty_like(page_indices)
    topk_ops.topk_transform_paged(
        scores, seq_lens, page_tables, page_indices, page_size, raw_indices
    )
    torch.testing.assert_close(
        raw_indices.sort(dim=-1).values, ref_raw.sort(dim=-1).values
    )
    torch.testing.assert_close(
        page_indices.sort(dim=-1).values, ref_page.sort(dim=-1).values
    )


def test_ragged_falls_back_without_wheel(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang.kernels.ops.attention import dsv4 as dsv4_ops
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import topk_ragged

    monkeypatch.setattr(dsv4_ops, "is_deep_select_available", lambda: False)
    device = torch.device("cuda")
    rows, width, k = 4, 2048, 64
    scores = _distinct_bf16_rows(rows, width, device, seed=7)
    lens = torch.tensor([2048, 900, 3, 0], dtype=torch.int32, device=device)
    offsets = torch.tensor([0, 5000, 11, 2], dtype=torch.int32, device=device)

    columns = torch.arange(width, device=device)
    masked = scores.masked_fill(columns[None, :] >= lens[:, None], -torch.inf)
    picked = masked.topk(k, dim=-1, sorted=False).indices
    reach = picked < lens[:, None]
    expected = (
        torch.where(reach, picked + offsets[:, None], -1)
        .to(torch.int32)
        .sort(dim=-1)
        .values
    )

    out = torch.empty((rows, k), dtype=torch.int32, device=device)
    topk_ragged(scores, lens, out=out, offsets=offsets)
    torch.testing.assert_close(out.sort(dim=-1).values, expected)
