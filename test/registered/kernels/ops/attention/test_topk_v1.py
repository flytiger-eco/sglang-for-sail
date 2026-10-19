"""Correctness tests for the DeepSeek-V4 JIT top-k transform v1.

``topk_transform_paged`` dispatches to ``deepseek_v4/topk_v1.cuh`` on CUDA. Its
radix select first bins scores by the high byte of their ordered FP16 value and
stashes the threshold bin in shared memory (8192 entries). These tests put the
threshold bin on both sides of that capacity, with strictly higher values
emitted before it, ties at the cutoff, and clusters that only separate in one
of the lower three FP32 key bytes. They cover runtime ``topk`` values, the
``seq_len <= topk`` path, score rows padded past ``seq_lens``, a noncontiguous
score view, and both the page-table mapped and raw outputs.

Selections are compared with ``torch.topk`` as value multisets: any subset of
values tied at the cutoff is a valid selection, but indices must be distinct
and in range, and the mapped output must be the page-table transform of the
raw output.

A second group covers the 2048-entry output cap: the runtime ``topk`` ladder
with and without a page table, an all-ties row that must fill every slot, the
candidate block selector at ``topk_blocks == 2048``, and CUDA-graph replay.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import amax_topk_blocks
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or is_hip(),
    reason="topk_transform_paged uses the JIT v1 kernel on CUDA only",
)

# kSMEM / (2 * sizeof(int32_t)) in topk_v1.cuh: entries per stash round.
STASH_ENTRIES = 8192
PAGE_SIZE = 64
TOPKS = (512, 1024, 300)


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _coarse_keys(values: torch.Tensor) -> torch.Tensor:
    """Mirror the kernel's high-byte ordered-FP16 key on the CPU."""
    bits = values.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    ordered = torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits | 0x8000)
    return ordered >> 8


def _tight_cluster(length: int, base: float = 1.0) -> torch.Tensor:
    # Increasing values in one coarse bin, best candidates last, so an
    # arrival-order stash prefix is wrong.
    return base + torch.arange(length, dtype=torch.float32) * (1e-3 / length)


def _byte_depth_clusters(depth: int, per_side: int, gen) -> torch.Tensor:
    # Two clusters sharing the coarse bin whose keys first differ at bit
    # ``depth``; the cluster with that bit set holds the true top-k.
    keys = torch.full((2 * per_side,), 0x3F000000, dtype=torch.int64)
    keys += torch.randint(0, 1 << depth, (2 * per_side,), generator=gen)
    keys[:per_side] += 1 << depth
    keys = keys[torch.randperm(2 * per_side, generator=gen)]
    return keys.to(torch.int32).view(torch.float32)


def _run_and_check(
    rows: list[torch.Tensor], topk: int, stride_pad: int = 0, view_pad: int = 0
):
    gen = _generator(topk)
    batch = len(rows)
    stride = max(row.numel() for row in rows) + stride_pad
    # Pad with +inf so any read past ``seq_lens`` would corrupt the selection.
    # ``view_pad`` passes a column slice whose row stride exceeds its width.
    storage = torch.full(
        (batch, stride + view_pad), float("inf"), dtype=torch.float32, device="cuda"
    )
    scores = torch.full((batch, stride), float("inf"), dtype=torch.float32)
    for i, row in enumerate(rows):
        scores[i, : row.numel()] = row
    seq_lens = torch.tensor([row.numel() for row in rows], dtype=torch.int32)
    num_pages = (stride + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = torch.stack(
        [torch.randperm(4 * num_pages, generator=gen)[:num_pages] for _ in range(batch)]
    ).to(torch.int32)

    out_page = torch.full((batch, topk), -7, dtype=torch.int32, device="cuda")
    out_raw = torch.full((batch, topk), -7, dtype=torch.int32, device="cuda")
    scores_view = storage[:, :stride]
    scores_view.copy_(scores)
    assert scores_view.is_contiguous() == (view_pad == 0)
    topk_transform_paged(
        scores_view,
        seq_lens.cuda(),
        page_table.cuda(),
        out_page,
        PAGE_SIZE,
        out_raw,
    )
    out_page = out_page.cpu().to(torch.int64)
    out_raw = out_raw.cpu().to(torch.int64)

    for i, row in enumerate(rows):
        length = row.numel()
        raw = out_raw[i]
        if length <= topk:
            expected = torch.full((topk,), -1, dtype=torch.int64)
            expected[:length] = torch.arange(length)
            assert torch.equal(raw, expected), f"row {i}: trivial raw indices"
        else:
            assert bool((raw >= 0).all()) and bool((raw < length).all()), f"row {i}"
            assert torch.unique(raw).numel() == topk, f"row {i}: duplicate index"
            selected = row[raw].sort().values
            reference = torch.topk(row, topk).values.sort().values
            assert torch.equal(selected, reference), f"row {i}: wrong selection"
        valid = raw >= 0
        mapped = torch.full_like(raw, -1)
        mapped[valid] = (
            page_table[i, raw[valid] // PAGE_SIZE].to(torch.int64) * PAGE_SIZE
            + raw[valid] % PAGE_SIZE
        )
        assert torch.equal(out_page[i], mapped), f"row {i}: page mapping"


@pytest.mark.parametrize("topk", TOPKS)
def test_threshold_bin_around_stash_capacity(topk):
    gen = _generator(1)
    below = _tight_cluster(STASH_ENTRIES - 1)
    exact = _tight_cluster(STASH_ENTRIES)
    above = _tight_cluster(STASH_ENTRIES + 1)
    for row in (below, exact, above):
        assert torch.unique(_coarse_keys(row)).numel() == 1
    rows = [
        below,
        exact,
        exact.flip(0),
        exact[torch.randperm(exact.numel(), generator=gen)],
        _tight_cluster(STASH_ENTRIES, base=-0.9),
        above,
        _tight_cluster(3 * STASH_ENTRIES + 17),
    ]
    _run_and_check(rows, topk)


@pytest.mark.parametrize("topk", TOPKS)
def test_oversized_bin_after_strictly_higher_values(topk):
    gen = _generator(2)
    high = 2.0 + torch.arange(37, dtype=torch.float32) * 1e-3
    low = torch.full((257,), -2.0)
    row = torch.cat([high, _tight_cluster(STASH_ENTRIES + 5311), low])
    exact_bin_row = torch.cat([high, _tight_cluster(STASH_ENTRIES), low])
    rows = [row, row[torch.randperm(row.numel(), generator=gen)], exact_bin_row]
    _run_and_check(rows, topk)


@pytest.mark.parametrize("topk", TOPKS)
def test_cutoff_ties(topk):
    gen = _generator(3)
    top = torch.linspace(2.0, 3.0, topk)
    tie_row = torch.cat([top, torch.full((65536,), 1.0)])
    rows = [
        torch.full((STASH_ENTRIES + 1,), 0.75),
        torch.full((STASH_ENTRIES + 1,), 0.0),
        torch.full((STASH_ENTRIES + 1,), -0.0),
        torch.full((topk + 1,), 0.5),
        tie_row[torch.randperm(tie_row.numel(), generator=gen)],
        torch.cat([top[: topk // 2], torch.full((2 * STASH_ENTRIES,), 1.0)]),
    ]
    _run_and_check(rows, topk)


@pytest.mark.parametrize("topk", TOPKS)
def test_clusters_separating_at_lower_key_bytes(topk):
    gen = _generator(4)
    rows = [_byte_depth_clusters(depth, STASH_ENTRIES + 1, gen) for depth in (16, 8, 0)]
    _run_and_check(rows, topk)


@pytest.mark.parametrize("topk", TOPKS)
def test_short_and_random_rows_with_padded_and_strided_scores(topk):
    gen = _generator(5)
    rows = [
        torch.rand(1, generator=gen),
        torch.rand(topk - 1, generator=gen),
        torch.rand(topk, generator=gen),
        torch.rand(topk + 1, generator=gen),
        torch.rand(3000, generator=gen),
        torch.randn(40000, generator=gen),
    ]
    _run_and_check(rows, topk, stride_pad=PAGE_SIZE + 3)
    _run_and_check(rows, topk, view_pad=PAGE_SIZE + 5)


def _check_row(scores, indices, length, k):
    indices = indices.long()
    count = min(length, k)
    valid = indices[:count]
    assert ((valid >= 0) & (valid < length)).all()
    assert valid.unique().numel() == count
    assert (indices[count:] == -1).all()
    selected = scores[valid].sort().values
    expected = scores[:length].topk(count).values.sort().values
    assert torch.equal(selected, expected)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@pytest.mark.parametrize("length", [2048, 4096, 16384, 16385, 32768, 32769])
@pytest.mark.parametrize("mode", ["raw", "paged", "dual"])
def test_v1_runtime_topk(k, length, mode):
    torch.manual_seed(0)
    rows, page_size = 4, 64
    # Preserve vector alignment while exercising noncontiguous row strides.
    stride = (length + 3) // 4 * 4 + 8
    scores = torch.randn(rows, stride, device="cuda")[:, :length]
    lens = [0, k - 1, min(length, k + 1), length]
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    pages = (length + page_size - 1) // page_size
    table = torch.stack(
        [torch.randperm(pages, device="cuda", dtype=torch.int32) for _ in range(rows)]
    )
    output = torch.full((rows, k), -99, dtype=torch.int32, device="cuda")
    raw = torch.full_like(output, -99) if mode == "dual" else None
    topk_transform_paged(
        scores,
        lengths,
        None if mode == "raw" else table,
        output,
        page_size,
        raw,
    )
    torch.cuda.synchronize()
    if mode == "raw":
        selected = output
    elif mode == "dual":
        selected = raw
    else:
        inverse = table.argsort(dim=1)
        slots = output.long().clamp_min(0)
        selected = (
            inverse.gather(1, slots // page_size) * page_size + slots % page_size
        ).masked_fill(output < 0, -1)
    for row, size in enumerate(lens):
        _check_row(scores[row], selected[row], size, k)
    if mode == "dual":
        positions = raw.long().clamp_min(0)
        expected_slots = (
            table.gather(1, positions // page_size) * page_size + positions % page_size
        ).masked_fill(raw < 0, -1)
        assert torch.equal(output, expected_slots)


@pytest.mark.parametrize("length", [4096, 20000, 40000])
def test_v1_top2048_ties(length):
    # Requires a tie buffer larger than 1024 and writes all 2048 output slots.
    scores = torch.ones((1, length), device="cuda")
    scores[0, -1] = torch.inf
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    output = torch.full((1, 2048), -99, dtype=torch.int32, device="cuda")
    topk_transform_paged(scores, lengths, None, output, 1)
    torch.cuda.synchronize()
    _check_row(scores[0], output[0], length, 2048)


def test_v1_top2048_clustered_scores_are_exact():
    # 4096 distinct FP32 scores inside one coarse FP16 bin overflow the tie
    # buffer; the radix fallback must still select the exact top-k.
    length, k = 4096, 2048
    scores = (
        1.0 + torch.arange(length, dtype=torch.float32, device="cuda") * 1e-6
    ).unsqueeze(0)
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    output = torch.full((1, k), -99, dtype=torch.int32, device="cuda")
    topk_transform_paged(scores, lengths, None, output, 1)
    torch.cuda.synchronize()
    _check_row(scores[0], output[0], length, k)


def test_v1_top2048_clustered_scores_above_radix_capacity():
    # 16384 distinct FP32 scores inside one coarse bin overflow both the tie
    # buffer and radix's 8192-candidate refill; the exact fallback must still
    # select the whole row's top-k.
    length, k = 16384, 2048
    scores = (
        1.0 + torch.arange(length, dtype=torch.float32, device="cuda") * 2**-22
    ).unsqueeze(0)
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    output = torch.full((1, k), -99, dtype=torch.int32, device="cuda")
    topk_transform_paged(scores, lengths, None, output, 1)
    torch.cuda.synchronize()
    _check_row(scores[0], output[0], length, k)


def test_candidate_top2048_blocks():
    torch.manual_seed(0)
    width, k = 40000, 2048
    logits = torch.randn(4, width, device="cuda")
    lens = [0, 8193, 16384, width - 3]
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    nblocks = (lengths + 7) // 8
    blocks = amax_topk_blocks(logits, lengths, nblocks, k)
    torch.cuda.synchronize()
    for row, length in enumerate(lens):
        count = (length + 7) // 8
        padded = logits[row].clone()
        padded[length:] = -torch.inf
        keys = padded.reshape(-1, 8).amax(dim=1)
        if count:
            keys[count - 1] = torch.inf
        _check_row(keys, blocks[row], count, k)
        if count:
            assert (blocks[row] == count - 1).any(), "newest block must be retained"


def test_v1_top2048_cuda_graph_replay():
    torch.manual_seed(0)
    scores = torch.randn(2, 4096, device="cuda")
    lengths = torch.tensor([4096, 4096], dtype=torch.int32, device="cuda")
    output = torch.empty(2, 2048, dtype=torch.int32, device="cuda")
    # Compile and initialize the module before capture.
    topk_transform_paged(scores, lengths, None, output, 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        topk_transform_paged(scores, lengths, None, output, 1)
    for lens in ([4096, 4096], [0, 1500], [2048, 2049]):
        scores.normal_()
        lengths.copy_(torch.tensor(lens, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        for row, size in enumerate(lens):
            _check_row(scores[row], output[row], size, 2048)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
