"""Prefill score ranges and caller-owned selection buffers."""

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.kernel import (
    expand_map_qsa_physical_slots,
    expand_qsa_block_indices,
    qsa_fast_topk,
    torch_expand_qsa_block_indices,
)
from sglang.srt.layers.attention.qsa.mqa import tilelang_qsa_mqa_prefill
from sglang.srt.layers.attention.qsa.qsa_indexer import _qsa_prefill_row_chunk_size
from sglang.srt.utils.common import is_ppu


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("rows", [0, 1, 33])
def test_selection_writes_only_its_output_slice(device, rows):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required")
    ratio, topk, width = 4, 2048, 2051
    positions = torch.arange(rows, device=device, dtype=torch.int32) * 11 + 3
    lengths = torch.full((rows,), 4096, device=device, dtype=torch.int32)
    blocks = torch.full((rows, 512), -1, device=device, dtype=torch.int32)
    for row in range(rows):
        count = min(int((positions[row] + 1) // ratio), 512)
        blocks[row, :count] = torch.arange(count, device=device, dtype=torch.int32)
    expected = torch_expand_qsa_block_indices(blocks, positions, lengths, ratio, topk)
    parent = torch.full((rows + 2, width), -99, device=device, dtype=torch.int32)
    out = parent[1:-1]
    actual = expand_qsa_block_indices(blocks, positions, lengths, ratio, topk, out=out)
    assert actual.data_ptr() == out.data_ptr()
    assert torch.equal(actual, expected)
    batch = torch.arange(rows, device=device, dtype=torch.int32) % 2
    table = torch.arange(8192, device=device, dtype=torch.int32).reshape(2, 4096)
    mapped = torch.where(
        expected >= 0,
        table[batch.long()[:, None], expected.clamp_min(0).long()],
        -1,
    )
    actual = expand_map_qsa_physical_slots(
        blocks,
        positions,
        lengths,
        batch,
        table,
        compress_ratio=ratio,
        token_topk=topk,
        out=out,
    )
    assert actual.data_ptr() == out.data_ptr()
    assert torch.equal(actual, mapped)
    assert bool((parent[0] == -99).all() and (parent[-1] == -99).all())
    with pytest.raises(ValueError, match="contiguous int32"):
        expand_qsa_block_indices(
            blocks, positions, lengths, ratio, topk, out=out.float()
        )


@pytest.mark.skipif(not is_ppu(), reason="PPU native BF16 TopK contract")
@pytest.mark.parametrize("keys", [4097, 8193])
@pytest.mark.parametrize("topk", [512, 2048])
def test_native_topk_ignores_poisoned_invalid_scores(keys, topk):
    torch.manual_seed(113)
    rows = 33
    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(keys, 1, 128, device="cuda", dtype=torch.bfloat16)
    starts = torch.arange(rows, device="cuda", dtype=torch.int32) % 7
    ends = (
        starts + torch.arange(rows, device="cuda", dtype=torch.int32) * 263
    ).clamp_max(keys)
    valid = (torch.arange(keys, device="cuda")[None, :] >= starts[:, None]) & (
        torch.arange(keys, device="cuda")[None, :] < ends[:, None]
    )
    with envs.SGLANG_QSA_PREFILL_BF16_LOGITS.override(True):
        with envs.SGLANG_QSA_PREFILL_BLOCK_N.override(64):
            baseline = tilelang_qsa_mqa_prefill(q, k, starts, ends)
        optimized = tilelang_qsa_mqa_prefill(q, k, starts, ends)
        assert torch.equal(optimized, baseline)
        scores = tilelang_qsa_mqa_prefill(q, k, starts, ends, mask_logits=False)
    assert torch.equal(scores[valid], baseline[valid])
    scores.masked_fill_(~valid, torch.finfo(scores.dtype).max)
    selected = qsa_fast_topk(scores, starts, ends, topk)
    for row in range(rows):
        start, end = int(starts[row]), int(ends[row])
        indices = selected[row][selected[row] >= 0].long()
        assert indices.numel() == min(topk, end - start)
        assert bool((indices < end - start).all())
        actual = baseline[row, start + indices].sort().values
        expected = (
            torch.topk(baseline[row, start:end], indices.numel()).values.sort().values
        )
        assert torch.equal(actual, expected)


def test_logits_chunk_budget_uses_storage_dtype():
    assert _qsa_prefill_row_chunk_size(10000, 32768, 4) == 1024
    assert _qsa_prefill_row_chunk_size(10000, 32768, 4, torch.bfloat16) == 2048
