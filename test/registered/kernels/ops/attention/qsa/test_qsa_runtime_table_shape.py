"""Physical QSA mapping with changing table widths and noncompact row strides."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

from sglang.srt.layers.attention.qsa.kernel import (
    expand_map_qsa_physical_slots,
    torch_expand_qsa_block_indices,
)


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("stride_padding", [0, 7])
def test_changing_table_shape_preserves_physical_mapping(position_dtype, stride_padding):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    rows, ratio, topk = 8, 4, 2048
    for width in [1, 3, 2047, 2048, 2049, 4097, 16387, 75417]:
        table_storage = torch.arange(3 * (width + stride_padding), dtype=torch.int32)
        # Preserve the padded row stride when transferring the backing storage.
        backing = table_storage.reshape(3, -1) + 1000
        table = backing[:, :width]
        owners = torch.arange(rows, dtype=torch.int32) % 3
        positions = torch.tensor(
            [max(0, width - 1 - i % ratio) for i in range(rows)], dtype=position_dtype
        )
        lengths = torch.full((rows,), width, dtype=torch.int32)
        blocks = torch.full((rows, topk // ratio), -1, dtype=torch.int32)
        for row, position in enumerate(positions.tolist()):
            complete_blocks = (position + 1) // ratio
            count = min(topk // ratio, complete_blocks)
            if count:
                blocks[row, :count] = torch.arange(
                    complete_blocks - count, complete_blocks, dtype=torch.int32
                )
        # One empty selection also covers the partially filled tail group.
        blocks[0].fill_(-1)
        logical = torch_expand_qsa_block_indices(blocks, positions, lengths, ratio, topk)
        valid = (logical >= 0) & (logical < width)
        expected = table[owners.long()[:, None], logical.clamp(0, width - 1).long()]
        expected = torch.where(valid, expected, -1)
        gpu_table = backing.cuda()[:, :width]
        actual = expand_map_qsa_physical_slots(
            blocks.cuda(),
            positions.cuda(),
            lengths.cuda(),
            owners.cuda(),
            gpu_table,
            compress_ratio=ratio,
            token_topk=topk,
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
