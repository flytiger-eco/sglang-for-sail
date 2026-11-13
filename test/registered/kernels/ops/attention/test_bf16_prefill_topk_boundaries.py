"""BF16 prefill top-k storage alignment and row-boundary regressions."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.topk import top_k_per_row_prefill_bf16
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA-compatible device required")
class TestBf16PrefillTopKBoundaries(CustomTestCase):
    def _check(self, width, topk, *, storage_offset, tied):
        rows, page_size = 3, 64
        storage = torch.randn(
            rows * width + storage_offset, device="cuda", dtype=torch.bfloat16
        )
        scores = storage[storage_offset:].reshape(rows, width)
        if tied:
            scores.zero_()
        starts = torch.tensor([0, 1, 7], device="cuda", dtype=torch.int32)
        ends = torch.tensor(
            [width - 7, width - 3, width], device="cuda", dtype=torch.int32
        )
        pages = (width + page_size - 1) // page_size
        page_table = (
            torch.arange(rows * pages, device="cuda", dtype=torch.int32).reshape(
                rows, pages
            )
            * 3
            + 1
        )
        row_map = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
        physical = torch.full((rows, topk), -99, device="cuda", dtype=torch.int32)
        raw = torch.full_like(physical, -99)
        top_k_per_row_prefill_bf16(
            scores, starts, ends, page_table, physical, page_size, raw, row_map
        )
        for row in range(rows):
            length = int((ends[row] - starts[row]).item())
            count = min(length, topk)
            selected = raw[row].long()
            self.assertEqual(int((selected >= 0).sum()), count)
            if count == 0:
                continue
            selected = selected[selected >= 0]
            self.assertTrue(bool((selected < length).all()))
            self.assertEqual(selected.unique().numel(), count)
            logical = scores[row, int(starts[row]) : int(ends[row])].float()
            threshold = torch.topk(logical, count).values.min()
            self.assertTrue(bool((logical[selected] >= threshold).all()))
            table_row = int(row_map[row])
            expected = (
                page_table[table_row, selected // page_size] * page_size
                + selected % page_size
            )
            torch.testing.assert_close(
                physical[row][raw[row] >= 0].long(), expected.long()
            )
            self.assertTrue(bool((physical[row][raw[row] < 0] == -1).all()))

    def test_unaligned_storage_views_across_register_and_stream_boundaries(self):
        for width in (2047, 8191, 16383, 32761, 32768, 65535):
            for topk in (512, 2048):
                with self.subTest(width=width, topk=topk):
                    self._check(width, topk, storage_offset=1, tied=False)

    def test_tied_rows_initialize_all_2048_output_slots(self):
        for width in (2047, 16383, 32768, 65535):
            with self.subTest(width=width):
                self._check(width, 2048, storage_offset=0, tied=True)


if __name__ == "__main__":
    unittest.main()
