import pytest
import torch

from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="engram gather needs a CUDA device"
)

DIM = 256
BLOCK = 32
ROWS = 64
_E8M0_ZERO = 2.0**-127


def _make_table(rows=ROWS, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(rows, DIM, generator=gen, device="cuda").to(
        torch.float8_e4m3fn
    )
    # Exponents kept away from the e8m0 zero byte so the reference is exact.
    scales = torch.randint(
        120,
        141,
        (rows, DIM // BLOCK),
        generator=gen,
        device="cuda",
        dtype=torch.int32,
    ).to(torch.uint8)
    return weight, scales


def _reference(weight, scales, ids, *, row_lo=0, row_hi=2**62):
    owned = (ids >= row_lo) & (ids < row_hi)
    rows = torch.where(owned, ids - row_lo, torch.zeros_like(ids))
    vals = weight[rows].to(torch.float32)
    exps = scales[rows].to(torch.int32)
    scale = (exps << 23).view(torch.float32)
    scale = torch.where(exps == 0, torch.full_like(scale, _E8M0_ZERO), scale)
    scale = scale.repeat_interleave(BLOCK, dim=1)
    out = (vals * scale).to(torch.bfloat16)
    return torch.where(owned[:, None], out, torch.zeros_like(out))


def _gather(weight, scales, ids, *, sort_ids=False, row_lo=0, row_hi=2**62):
    out = torch.empty((ids.numel(), DIM), dtype=torch.bfloat16, device="cuda")
    return engram_gather(
        weight.data_ptr(),
        scales.data_ptr(),
        ids,
        out,
        DIM,
        BLOCK,
        row_lo=row_lo,
        row_hi=row_hi,
        sort_ids=sort_ids,
    )


def test_sorted_reads_match_unsorted_and_reference():
    weight, scales = _make_table()
    ids = torch.randint(0, ROWS, (512,), device="cuda", dtype=torch.int64)

    unsorted = _gather(weight, scales, ids)
    sorted_out = _gather(weight, scales, ids, sort_ids=True)

    torch.testing.assert_close(sorted_out, unsorted, rtol=0, atol=0)
    torch.testing.assert_close(
        unsorted, _reference(weight, scales, ids), rtol=0, atol=0
    )


def test_sorted_path_keeps_duplicate_ids_in_place():
    weight, scales = _make_table()
    ids = torch.tensor([7, 7, 3, 7, 0, 12, 3], device="cuda", dtype=torch.int64)

    out = _gather(weight, scales, ids, sort_ids=True)

    torch.testing.assert_close(
        out, _reference(weight, scales, ids), rtol=0, atol=0
    )


def test_ids_outside_the_owned_range_are_zero():
    weight, scales = _make_table()
    ids = torch.tensor([8, 5, 31, 32, -1, 39, 40, 12], device="cuda", dtype=torch.int64)

    out = _gather(weight, scales, ids, sort_ids=True, row_lo=8, row_hi=40)

    torch.testing.assert_close(
        out,
        _reference(weight, scales, ids, row_lo=8, row_hi=40),
        rtol=0,
        atol=0,
    )
    for row, value in enumerate(ids.tolist()):
        if not 8 <= value < 40:
            assert torch.count_nonzero(out[row]) == 0, (row, value)


def test_empty_batch_is_a_no_op():
    weight, scales = _make_table()
    ids = torch.empty((0,), device="cuda", dtype=torch.int64)

    assert _gather(weight, scales, ids).shape == (0, DIM)
    assert _gather(weight, scales, ids, sort_ids=True).shape == (0, DIM)
