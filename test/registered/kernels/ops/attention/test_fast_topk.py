import sys

import pytest
import torch

from sglang.kernels.ops.attention.fast_topk import (
    fast_topk,
    fast_topk_transform_fused,
    fast_topk_transform_ragged_fused,
    fast_topk_v2,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _check_topk_values(score, lengths, indices, topk, row_starts):
    """fast_topk leaves order and tie-breaking unspecified,
    so compare the sorted top-k values rather than index sets."""
    for b in range(score.shape[0]):
        start = int(row_starts[b]) if row_starts is not None else 0
        length = int(lengths[b])
        section = score[b, start : start + length]
        row = indices[b]
        if length <= topk:
            # naive path: identity indices, then -1 fill
            assert torch.equal(
                row[:length].cpu(), torch.arange(length, dtype=torch.int32)
            )
            assert (row[length:] == -1).all()
            continue
        assert (row >= 0).all(), "long rows must fill every slot"
        assert (row < length).all(), f"row {b}: index must stay within row length"
        assert row.unique().numel() == topk, f"row {b}: duplicate indices"
        picked = section[row.long()]
        expected = torch.topk(section, topk).values
        assert torch.equal(
            picked.sort(descending=True).values, expected.sort(descending=True).values
        ), f"row {b}: top-{topk} value multiset mismatch"


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "batch,length",
    [
        (1, 4096),
        (7, 3000),
        (33, 32768),
        (128, 2050),
    ],
)
def test_fast_topk_long_rows(topk, batch, length):
    torch.manual_seed(0)
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_short_and_mixed_rows(topk):
    torch.manual_seed(0)
    max_len = topk + 128
    batch = 8
    score = torch.randn(batch, max_len, dtype=torch.float32, device="cuda")
    # rows shorter than k (naive path), exactly k, and longer than k
    lens = [1, topk // 3, topk - 1, topk, topk + 1, topk + 7, 17, max_len]
    lengths = torch.tensor(lens[:batch], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_ragged_with_row_starts(topk):
    torch.manual_seed(0)
    batch, width = 16, 8192
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    row_starts = torch.randint(0, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.randint(1, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.minimum(lengths, width - row_starts).to(torch.int32)
    # ensure some rows are longer than k
    lengths[0] = min(width - int(row_starts[0]), topk + 100)

    indices = fast_topk(score, lengths, topk, row_starts=row_starts)
    _check_topk_values(score, lengths, indices, topk, row_starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_row_stride(topk):
    torch.manual_seed(0)
    batch, length = 8, 4096
    base = torch.randn(batch, 2 * length, dtype=torch.float32, device="cuda")
    score = base[:, :length]  # stride(0) == 2*length, stride(1) == 1
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "fill",
    [
        "binary",  # only 0s and 1s: extreme duplication at the threshold bin
        "few_levels",  # a handful of distinct levels incl. negatives
        "constant",  # whole rows of one value
    ],
)
def test_fast_topk_duplicate_heavy(topk, fill):
    torch.manual_seed(0)
    batch, length = 16, 8192
    if fill == "binary":
        score = torch.randint(0, 2, (batch, length), dtype=torch.float32, device="cuda")
    elif fill == "few_levels":
        levels = torch.tensor([-5.0, -1.0, 0.0, 0.5, 2.0], device="cuda")
        score = levels[torch.randint(0, 5, (batch, length), device="cuda")]
    else:
        score = torch.full((batch, length), 3.25, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_negative_and_zero(topk):
    torch.manual_seed(0)
    batch, length = 8, 16384
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda") * 100
    score[:, : length // 3] = 0.0  # long zero prefix
    score[:, length // 3 : length // 2] = -1e30  # very negative block
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


def test_fast_topk_unsupported_k():
    score = torch.randn(2, 4096, dtype=torch.float32, device="cuda")
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="topk"):
        fast_topk(score, lengths, 1024)


def _locate(src: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Positions of each value within a unique 1-D src; -1 where absent."""
    sorted_src, order = torch.sort(src.long())
    loc = torch.searchsorted(sorted_src, values.long())
    loc = loc.clamp(max=sorted_src.numel() - 1)
    found = sorted_src[loc] == values.long()
    return torch.where(found, order[loc], torch.full_like(order[loc], -1))


def _make_unique_page(rows, width, seed, stride_pad=0):
    """Unique shuffled int32 page table; stride_pad embeds it in wider rows."""
    gen = torch.Generator().manual_seed(seed)
    vals = torch.randperm(rows * width, generator=gen).to(torch.int32).cuda()
    page = vals.reshape(rows, width)
    if stride_pad:
        base = torch.full(
            (rows, width + stride_pad), -1, dtype=torch.int32, device="cuda"
        )
        base[:, :width] = page
        return base
    return page


def _check_transform_output(score, lengths, page, cu_seqlens_q, topk, row_starts, out):
    """Short rows copy the page prefix and pad -1; long rows must gather page
    entries whose positions select exactly the top-k value multiset."""
    batch = score.shape[0]
    rows = torch.arange(batch, device=cu_seqlens_q.device)
    seq = torch.searchsorted(cu_seqlens_q, rows, right=True) - 1
    for b in range(batch):
        length = int(lengths[b])
        start = int(row_starts[b]) if row_starts is not None else 0
        src = page[int(seq[b])]
        row = out[b]
        if length <= topk:
            expected = torch.full((topk,), -1, dtype=torch.int32, device=row.device)
            expected[:length] = src[:length]
            assert torch.equal(row, expected), f"row {b}: naive page mismatch"
            continue
        assert (row >= 0).all(), f"row {b}: long rows must fill every slot"
        pos = _locate(src, row)
        assert (pos >= 0).all(), f"row {b}: entries must come from own page row"
        assert (pos < length).all(), f"row {b}: gathered positions must be < length"
        assert pos.unique().numel() == topk, f"row {b}: duplicate positions"
        section = score[b, start : start + length].float()
        picked = section[pos].sort(descending=True).values
        expected_vals = torch.topk(section, topk).values.sort(descending=True).values
        assert torch.equal(picked, expected_vals), f"row {b}: top-k value mismatch"


def _check_ragged_output(score, lengths, offsets, topk, row_starts, out):
    for b in range(score.shape[0]):
        length = int(lengths[b])
        start = int(row_starts[b]) if row_starts is not None else 0
        offset = int(offsets[b])
        row = out[b]
        if length <= topk:
            expected = torch.full((topk,), -1, dtype=torch.int32, device=row.device)
            expected[:length] = (
                torch.arange(length, dtype=torch.int32, device=row.device) + offset
            )
            assert torch.equal(row, expected), f"row {b}: naive ragged mismatch"
            continue
        assert (row >= 0).all(), f"row {b}: long rows must fill every slot"
        pos = row.long() - offset
        assert (pos >= 0).all() and (pos < length).all(), f"row {b}: out of range"
        assert pos.unique().numel() == topk, f"row {b}: duplicate indices"
        section = score[b, start : start + length].float()
        picked = section[pos].sort(descending=True).values
        expected_vals = torch.topk(section, topk).values.sort(descending=True).values
        assert torch.equal(picked, expected_vals), f"row {b}: top-k value mismatch"


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("batch,length", [(7, 4096), (33, 32768)])
def test_fast_topk_bf16(topk, batch, length):
    torch.manual_seed(0)
    score = torch.randn(batch, length, dtype=torch.bfloat16, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    _check_topk_values(score, lengths, fast_topk(score, lengths, topk), topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fast_topk_length_boundaries(topk, dtype):
    torch.manual_seed(0)
    lens = [0, 1, topk - 1, topk, topk + 1, 4096]
    starts = [0, 0, 7, 512, 1234, 0]
    batch, width = len(lens), 8192
    base = torch.randn(batch, 2 * width, dtype=dtype, device="cuda")
    score = base[:, :width]
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    row_starts = torch.tensor(starts, dtype=torch.int32, device="cuda")
    indices = fast_topk(score, lengths, topk, row_starts=row_starts)
    _check_topk_values(score, lengths, indices, topk, row_starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_v2_entry(topk):
    torch.manual_seed(0)
    score = torch.randn(8, 4096, dtype=torch.float32, device="cuda")
    lengths = torch.full((8,), 4096, dtype=torch.int32, device="cuda")
    _check_topk_values(score, lengths, fast_topk_v2(score, lengths, topk), topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_empty_batch(topk):
    score = torch.randn(0, 4096, dtype=torch.float32, device="cuda")
    lengths = torch.zeros(0, dtype=torch.int32, device="cuda")
    assert fast_topk(score, lengths, topk).shape == (0, topk)

    page = torch.zeros(0, 8, dtype=torch.int32, device="cuda")
    cu = torch.zeros(1, dtype=torch.int32, device="cuda")
    assert fast_topk_transform_fused(score, lengths, page, cu, topk).shape == (0, topk)

    offsets = torch.zeros(0, dtype=torch.int32, device="cuda")
    out = fast_topk_transform_ragged_fused(score, lengths, offsets, topk)
    assert out.shape == (0, topk)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fast_topk_transform_decode(topk, dtype):
    # page rows == batch and no row_starts -> decode kernel
    torch.manual_seed(0)
    lens = [0, 1, topk - 1, topk, topk + 1, 17, 4096, 5000]
    batch, width = len(lens), 8192
    score = torch.randn(batch, width, dtype=dtype, device="cuda")
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    page = _make_unique_page(batch, width, seed=1234 + topk, stride_pad=64)
    cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    out = fast_topk_transform_fused(score, lengths, page, cu, topk)
    _check_transform_output(score, lengths, page, cu, topk, None, out)


def test_fast_topk_transform_decode_score_stride():
    torch.manual_seed(0)
    batch, width = 8, 4096
    base = torch.randn(batch, 2 * width, dtype=torch.float32, device="cuda")
    score = base[:, :width]
    lengths = torch.full((batch,), width, dtype=torch.int32, device="cuda")
    page = _make_unique_page(batch, width, seed=99)
    cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    out = fast_topk_transform_fused(score, lengths, page, cu, 2048)
    _check_transform_output(score, lengths, page, cu, 2048, None, out)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fast_topk_transform_prefill_mixed(topk, dtype):
    # page rows < batch -> prefill kernel
    torch.manual_seed(0)
    batch, width = 8, 8192
    cu_list = [0, 3, 4, 8]
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    lens = [topk - 1, 4096, 17, topk + 1, 1, 3000, topk, 2]
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    score = torch.randn(batch, width, dtype=dtype, device="cuda")
    page = _make_unique_page(len(cu_list) - 1, width, seed=77 + topk)
    out = fast_topk_transform_fused(score, lengths, page, cu, topk)
    _check_transform_output(score, lengths, page, cu, topk, None, out)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fast_topk_transform_prefill_row_starts(topk, dtype):
    # page rows == batch but row_starts given -> still the prefill kernel
    torch.manual_seed(0)
    batch, width = 8, 16384
    cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    lengths = torch.full((batch,), 4096, dtype=torch.int32, device="cuda")
    row_starts = torch.randint(0, 4096, (batch,), dtype=torch.int32, device="cuda")
    score = torch.randn(batch, width, dtype=dtype, device="cuda")
    page = _make_unique_page(batch, width, seed=55 + topk)
    out = fast_topk_transform_fused(
        score, lengths, page, cu, topk, row_starts=row_starts
    )
    _check_transform_output(score, lengths, page, cu, topk, row_starts, out)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_transform_target_verify(topk):
    torch.manual_seed(0)
    batch, width = 12, 8192
    cu_list = [0, 4, 6, 11, 12]
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    lens = [topk + 64, topk + 1, width, topk + 7, 4096, topk * 2]
    lens += [topk + 3, 2048, topk + 2, 4096, topk + 5, 3000]
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    page = _make_unique_page(len(cu_list) - 1, width, seed=31 + topk)
    out = fast_topk_transform_fused(score, lengths, page, cu, topk)
    _check_transform_output(score, lengths, page, cu, topk, None, out)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fast_topk_transform_prefill_batch_over_block(topk, dtype):
    # 1025 page rows > threads per block -> strided sequence lookup; the last
    # row maps to sequence 1024 and is long enough to take the radix path.
    torch.manual_seed(0)
    batch, width = 1026, 4096
    cu_list = [0, 2] + list(range(3, batch + 1))
    cu = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    lens = [width, 2048] + [128] * (batch - 3) + [topk + 64]
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    score = torch.randn(batch, width, dtype=dtype, device="cuda")
    page = _make_unique_page(len(cu_list) - 1, width, seed=15 + topk)
    out = fast_topk_transform_fused(score, lengths, page, cu, topk)
    _check_transform_output(score, lengths, page, cu, topk, None, out)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_starts", [False, True])
def test_fast_topk_transform_ragged(topk, dtype, use_starts):
    torch.manual_seed(0)
    batch, width = 8, 8192
    lens = [0, 1, topk - 1, topk, topk + 1, 17, 4096, 5000]
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    offsets = torch.randint(0, 100000, (batch,), dtype=torch.int32, device="cuda")
    row_starts = (
        torch.randint(0, 2048, (batch,), dtype=torch.int32, device="cuda")
        if use_starts
        else None
    )
    score = torch.randn(batch, width, dtype=dtype, device="cuda")
    out = fast_topk_transform_ragged_fused(
        score, lengths, offsets, topk, row_starts=row_starts
    )
    _check_ragged_output(score, lengths, offsets, topk, row_starts, out)


def test_fast_topk_input_validation():
    good = torch.randn(4, 4096, dtype=torch.float32, device="cuda")
    lengths = torch.full((4,), 4096, dtype=torch.int32, device="cuda")
    starts = torch.zeros(4, dtype=torch.int32, device="cuda")
    page = torch.zeros(4, 4096, dtype=torch.int32, device="cuda")
    cu = torch.arange(5, dtype=torch.int32, device="cuda")
    offsets = torch.zeros(4, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="CUDA"):
        fast_topk(good.cpu(), lengths, 512)
    with pytest.raises(RuntimeError, match="dtype"):
        fast_topk(good.half(), lengths, 512)
    with pytest.raises(RuntimeError, match="2D"):
        fast_topk(good[0], lengths[:1], 512)
    with pytest.raises(RuntimeError, match="last dimension"):
        fast_topk(torch.randn(4096, 4, device="cuda").t(), lengths, 512)
    with pytest.raises(RuntimeError, match="lengths"):
        fast_topk(good, lengths.long(), 512)
    with pytest.raises(RuntimeError, match="lengths"):
        fast_topk(good, lengths[:3], 512)
    with pytest.raises(RuntimeError, match="row_starts"):
        fast_topk(good, lengths, 512, row_starts=starts.float())

    with pytest.raises(RuntimeError, match="must not exceed"):
        fast_topk_transform_fused(
            good,
            lengths,
            torch.zeros(5, 4096, dtype=torch.int32, device="cuda"),
            torch.arange(6, dtype=torch.int32, device="cuda"),
            512,
        )
    with pytest.raises(RuntimeError, match="cu_seqlens_q"):
        fast_topk_transform_fused(
            good, lengths, page, torch.arange(3, dtype=torch.int32, device="cuda"), 512
        )
    with pytest.raises(RuntimeError, match="page_table_size_1"):
        fast_topk_transform_fused(good, lengths, page.long(), cu, 512)
    with pytest.raises(RuntimeError, match="page-table row"):
        fast_topk_transform_fused(
            good,
            lengths,
            torch.zeros(0, 8, dtype=torch.int32, device="cuda"),
            torch.zeros(1, dtype=torch.int32, device="cuda"),
            512,
        )

    with pytest.raises(RuntimeError, match="topk_indices_offset"):
        fast_topk_transform_ragged_fused(good, lengths, offsets[:3], 512)
    with pytest.raises(RuntimeError, match="topk_indices_offset"):
        fast_topk_transform_ragged_fused(good, lengths, offsets.long(), 512)


@pytest.mark.parametrize("topk", [512, 2048])
def test_call_sites_use_jit_without_sgl_kernel(monkeypatch, topk):
    # DSA and QSA CUDA top-k must not import sgl_kernel anymore.
    monkeypatch.setitem(sys.modules, "sgl_kernel", None)
    torch.manual_seed(0)
    batch, width = 6, 8192
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    row_starts = torch.tensor(
        [0, 3, 100, 512, 1000, 0], dtype=torch.int32, device="cuda"
    )
    lengths = torch.tensor(
        [0, 1, topk - 1, topk + 1, 4096, width], dtype=torch.int32, device="cuda"
    )

    out = DSATopKBackend.SGL_KERNEL.topk_func(
        score, lengths, topk, row_starts=row_starts
    )
    _check_topk_values(score, lengths, out, topk, row_starts)

    out = qsa_fast_topk(score, row_starts, row_starts + lengths, topk=topk)
    _check_topk_values(score, lengths, out, topk, row_starts)
    ref = qsa_fast_topk(
        score.cpu(), row_starts.cpu(), (row_starts + lengths).cpu(), topk
    )
    for b in range(batch):
        start, length = int(row_starts[b]), int(lengths[b])
        section = score[b, start : start + length]
        width_b = min(length, topk)
        assert torch.equal(
            section[out[b, :width_b].long()].sort().values.cpu(),
            section.cpu()[ref[b, :width_b].long()].sort().values,
        ), f"row {b}: CUDA and CPU reference select different values"


try:
    import sgl_kernel as _aot
except ImportError:
    _aot = None


@pytest.mark.skipif(_aot is None, reason="sgl_kernel is not installed")
@pytest.mark.parametrize("use_starts", [False, True])
def test_aot_parity_fast_topk(use_starts):
    torch.manual_seed(0)
    batch, length, pad = 33, 32768, 512
    score = torch.randn(batch, length + pad, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    row_starts = (
        torch.randint(0, pad, (batch,), dtype=torch.int32, device="cuda")
        if use_starts
        else None
    )
    jit_out = fast_topk_v2(score, lengths, 2048, row_starts=row_starts)
    aot_out = _aot.fast_topk_v2(score, lengths, 2048, row_starts=row_starts)
    _check_topk_values(score, lengths, jit_out, 2048, row_starts)
    _check_topk_values(score, lengths, aot_out, 2048, row_starts)


@pytest.mark.skipif(_aot is None, reason="sgl_kernel is not installed")
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_aot_parity_transform(mode):
    torch.manual_seed(0)
    batch, width = 12, 16384
    if mode == "decode":
        cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    else:
        cu = torch.tensor([0, 4, 6, 11, 12], dtype=torch.int32, device="cuda")
    lens = [100, 2048, 2049, 17, 4096, 3000, 5000, 16384, 2050, 4096, 2048, 8192]
    lengths = torch.tensor(lens, dtype=torch.int32, device="cuda")
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    page = _make_unique_page(cu.shape[0] - 1, width, seed=2024)
    jit_out = fast_topk_transform_fused(score, lengths, page, cu, 2048)
    aot_out = _aot.fast_topk_transform_fused(score, lengths, page, cu, 2048)
    _check_transform_output(score, lengths, page, cu, 2048, None, jit_out)
    _check_transform_output(score, lengths, page, cu, 2048, None, aot_out)

    offsets = torch.randint(0, 100000, (batch,), dtype=torch.int32, device="cuda")
    jit_out = fast_topk_transform_ragged_fused(score, lengths, offsets, 2048)
    aot_out = _aot.fast_topk_transform_ragged_fused(score, lengths, offsets, 2048)
    _check_ragged_output(score, lengths, offsets, 2048, None, jit_out)
    _check_ragged_output(score, lengths, offsets, 2048, None, aot_out)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
