"""DeepSelect entry points against the torch reference, on the PPU wheel.

``deep_select.topk`` and ``deep_select.topk_page_transform`` are the upstream
DeepSelect API; on PPU their backend is the precompiled wheel instead of the
JIT. These tests pin that replacement: the selection must equal the torch
reference, including the page-table transform, ragged offsets, unaligned row
strides and the fp32 fallback used when no backend exists.

Selection happens in bf16, so the reference is taken on the bf16-rounded
scores; rows hold distinct bf16 values, so the selected sets are unambiguous.
"""

from __future__ import annotations

import pytest
import torch

import sglang.kernels.ops.attention.deep_select as ds


def _wheel_present() -> bool:
    return ds._wheel_available()


pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and _wheel_present()),
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


def _reference_page(scores, seq_lens, page_tables, k, page_size):
    rows = scores.shape[0]
    device = scores.device
    raw = torch.full((rows, k), -1, dtype=torch.int32, device=device)
    for row in range(rows):
        n = int(seq_lens[row])
        if n > 0:
            take = min(k, n)
            raw[row, :take] = torch.topk(
                scores[row, :n], take, sorted=False
            ).indices.to(torch.int32)
    cols = raw.to(torch.int64).clamp_min(0)
    slots = page_tables.gather(1, cols // page_size) * page_size + cols % page_size
    return torch.where(raw >= 0, slots, torch.full_like(slots, -1)).to(torch.int32)


@pytest.mark.parametrize("k", [64, 100, 2048])
def test_page_transform_matches_torch(k: int) -> None:
    device, rows, width, page_size = torch.device("cuda"), 7, 4096, 64
    scores = _distinct_bf16_rows(rows, width, device)
    seq_lens = torch.tensor(
        [4096, 3000, 2048, 100, 1, 0, 4096], dtype=torch.int32, device=device
    )
    page_tables = torch.randint(
        0, 32, (rows, width // page_size), device=device, dtype=torch.int32
    )

    expected = _reference_page(scores, seq_lens, page_tables, k, page_size)
    got = ds.topk_page_transform(
        scores, k, page_table=page_tables, page_size=page_size, end=seq_lens
    )
    torch.testing.assert_close(got.sort(dim=-1).values, expected.sort(dim=-1).values)


@pytest.mark.parametrize("width", [4000, 4096])
def test_page_transform_unaligned_row_stride(width: int) -> None:
    # 4000 bf16 columns is not a multiple of the wheel's 1024 B run, so the
    # dispatch materializes a padded copy before selecting.
    device, rows, k, page_size = torch.device("cuda"), 3, 128, 64
    scores = _distinct_bf16_rows(rows, width, device, seed=2)
    seq_lens = torch.tensor([width, width - 1, 7], dtype=torch.int32, device=device)
    page_tables = torch.randint(
        0, 8, (rows, width // page_size + 1), device=device, dtype=torch.int32
    )

    expected = _reference_page(scores, seq_lens, page_tables, k, page_size)
    got = ds.topk_page_transform(
        scores, k, page_table=page_tables, page_size=page_size, end=seq_lens
    )
    torch.testing.assert_close(got.sort(dim=-1).values, expected.sort(dim=-1).values)


def test_wheel_rejects_fp32_scores() -> None:
    device = torch.device("cuda")
    scores = torch.randn(4, 1024, device=device, dtype=torch.float32)
    page_table = torch.zeros((4, 16), dtype=torch.int32, device=device)
    with pytest.raises(TypeError, match="bf16 scores only"):
        ds.topk(scores, 64)
    with pytest.raises(TypeError, match="bf16 scores only"):
        ds.topk_page_transform(scores, 64, page_table=page_table, page_size=64)


def test_ragged_offsets_match_reference() -> None:
    # Prefill contract: row i takes the top-k of its first lens[i] columns and
    # writes the position plus its request base, -1 past the valid count.
    device, rows, width, k = torch.device("cuda"), 5, 2500, 100
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
    ds.topk(
        scores,
        k,
        end=lens,
        output_idx=out,
        output_idx_offset=offsets,
        idx_oob_fill_value=-1,
    )
    torch.testing.assert_close(out.sort(dim=-1).values, expected)


def test_fp32_fallback_without_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    # A device with neither the wheel nor a supported JIT arch ranks the exact
    # fp32 values on the top-k v2 kernel: same selected set, no dtype error.
    monkeypatch.setattr(ds, "_wheel_available", lambda: False)
    monkeypatch.setattr(ds, "is_deep_select_supported", lambda: False)
    device, rows, width, k = torch.device("cuda"), 4, 2048, 64
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
    ds.topk(
        scores,
        k,
        end=lens,
        output_idx=out,
        output_idx_offset=offsets,
        idx_oob_fill_value=-1,
    )
    torch.testing.assert_close(out.sort(dim=-1).values, expected)
