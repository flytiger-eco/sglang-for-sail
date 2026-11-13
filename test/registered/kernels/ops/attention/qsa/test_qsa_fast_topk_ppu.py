import pytest
import torch

from sglang.kernels.ops.attention.qsa_fast_topk import (
    fast_topk,
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_DTYPES = (torch.float32, torch.bfloat16)


def _check_topk_values(score, lengths, indices, topk, row_starts=None):
    for row_index in range(score.shape[0]):
        start = int(row_starts[row_index]) if row_starts is not None else 0
        length = int(lengths[row_index])
        row = indices[row_index]
        if length <= topk:
            assert torch.equal(
                row[:length].cpu(), torch.arange(length, dtype=torch.int32)
            )
            assert (row[length:] == -1).all()
            continue
        assert (row >= 0).all()
        section = score[row_index, start : start + length]
        picked = section[row.long()].float().sort().values
        expected = torch.topk(section.float(), topk).values.sort().values
        torch.testing.assert_close(picked, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("batch,length", [(1, 4096), (7, 3000), (33, 32768)])
def test_fast_topk_long_rows(dtype, topk, batch, length):
    torch.manual_seed(0)
    score = torch.randn(batch, length, dtype=dtype, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    indices = fast_topk_v2(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_short_mixed_rows_and_stride(dtype, topk):
    torch.manual_seed(0)
    batch = 6
    width = topk + 129
    base = torch.randn(batch, 2 * (width + 8), dtype=dtype, device="cuda")
    score = base[:, : width + 8]
    row_starts = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(
        [0, 1, topk - 1, topk, topk + 1, width],
        dtype=torch.int32,
        device="cuda",
    )
    indices = fast_topk(score, lengths, topk, row_starts=row_starts)
    _check_topk_values(score, lengths, indices, topk, row_starts)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("fill", ["binary", "few_levels", "constant"])
def test_fast_topk_duplicate_heavy(dtype, topk, fill):
    torch.manual_seed(0)
    batch, length = 4, 8192
    if fill == "binary":
        score = torch.randint(0, 2, (batch, length), device="cuda").to(dtype)
    elif fill == "few_levels":
        levels = torch.tensor([-5.0, -1.0, 0.0, 0.5, 2.0], device="cuda")
        score = levels[torch.randint(0, 5, (batch, length), device="cuda")].to(dtype)
    else:
        score = torch.full((batch, length), 3.25, dtype=dtype, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("mode", ["decode", "prefill", "target_verify"])
def test_fast_topk_paged_transform(dtype, topk, mode):
    torch.manual_seed(0)
    batch = 6
    width = topk + 129
    score = torch.randn(batch, width + 8, dtype=dtype, device="cuda")
    lengths = torch.tensor(
        [0, 1, topk - 1, topk, topk + 1, width],
        dtype=torch.int32,
        device="cuda",
    )
    if mode == "decode":
        row_starts = None
        cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
        request_rows = list(range(batch))
    else:
        row_starts = (
            torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device="cuda")
            if mode == "prefill"
            else None
        )
        cu_seqlens_q = torch.tensor([0, 2, batch], dtype=torch.int32, device="cuda")
        request_rows = [0, 0, 1, 1, 1, 1]

    page_rows = cu_seqlens_q.shape[0] - 1
    page_table = torch.arange(width, dtype=torch.int32, device="cuda").repeat(
        page_rows, 1
    )
    page_table += (
        torch.arange(page_rows, dtype=torch.int32, device="cuda").unsqueeze(1) * 100000
    )
    transformed = fast_topk_transform_fused(
        score,
        lengths,
        page_table,
        cu_seqlens_q,
        topk,
        row_starts=row_starts,
    )

    raw = transformed.clone()
    for row_index, request_row in enumerate(request_rows):
        valid = raw[row_index] >= 0
        raw[row_index, valid] -= request_row * 100000
    _check_topk_values(score, lengths, raw, topk, row_starts)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_ragged_transform(dtype, topk):
    torch.manual_seed(0)
    batch = 6
    width = topk + 129
    score = torch.randn(batch, width + 8, dtype=dtype, device="cuda")
    row_starts = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(
        [0, 1, topk - 1, topk, topk + 1, width],
        dtype=torch.int32,
        device="cuda",
    )
    offsets = torch.arange(batch, dtype=torch.int32, device="cuda") * 100000
    transformed = fast_topk_transform_ragged_fused(
        score, lengths, offsets, topk, row_starts=row_starts
    )
    raw = transformed.clone()
    valid = raw >= 0
    raw[valid] -= offsets[:, None].expand_as(raw)[valid]
    _check_topk_values(score, lengths, raw, topk, row_starts)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_empty_batch(dtype, topk):
    score = torch.empty((0, topk + 1), dtype=dtype, device="cuda")
    lengths = torch.empty((0,), dtype=torch.int32, device="cuda")
    offsets = torch.empty((0,), dtype=torch.int32, device="cuda")
    page_table = torch.empty((0, topk + 1), dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.zeros((1,), dtype=torch.int32, device="cuda")

    assert fast_topk(score, lengths, topk).shape == (0, topk)
    assert fast_topk_transform_fused(
        score, lengths, page_table, cu_seqlens_q, topk
    ).shape == (0, topk)
    assert fast_topk_transform_ragged_fused(score, lengths, offsets, topk).shape == (
        0,
        topk,
    )


def test_fast_topk_invalid_inputs():
    score = torch.randn(2, 4096, dtype=torch.float32, device="cuda")
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="topk"):
        fast_topk(score, lengths, 1024)
    with pytest.raises(RuntimeError, match="dtype"):
        fast_topk(score.to(torch.float16), lengths, 512)
    with pytest.raises(RuntimeError, match="last dimension"):
        fast_topk(score[:, ::2], lengths, 512)
    with pytest.raises(RuntimeError, match="lengths"):
        fast_topk(score, lengths.to(torch.int64), 512)
    with pytest.raises(RuntimeError, match="row_starts"):
        fast_topk(score, lengths, 512, torch.zeros(1, dtype=torch.int32, device="cuda"))

    page_table = torch.zeros((2, 4096), dtype=torch.int32, device="cuda")
    bad_cu_seqlens = torch.zeros((2,), dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="cu_seqlens_q"):
        fast_topk_transform_fused(score, lengths, page_table, bad_cu_seqlens, 512)
    with pytest.raises(RuntimeError, match="topk_indices_offset"):
        fast_topk_transform_ragged_fused(
            score,
            lengths,
            torch.zeros(2, dtype=torch.int64, device="cuda"),
            512,
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
