"""V1 runtime top-k: cover top-2048 output, raw mode, and all length paths."""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    amax_topk_blocks,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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
