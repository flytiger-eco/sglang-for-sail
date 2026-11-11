"""PPU wheel integration: BF16 row windows, offsets, alignment and padding."""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.deep_select import topk_transform_ragged
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not is_sail(), reason="Requires the PPU deep_select wheel"
)


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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
