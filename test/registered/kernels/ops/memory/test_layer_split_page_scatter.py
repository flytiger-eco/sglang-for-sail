"""Byte-exact page scatter across dtype, alignment, layout and stream boundaries."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=12, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_ppu_ci(est_time=12, suite="stage-b-test-1-gpu-ppu")


class TestLayerSplitPageScatter(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_bit_patterns_and_alignment(self):
        from sglang.kernels.ops.memory.layer_split_page_scatter import (
            scatter_unique_pages,
        )

        for dtype in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn):
            for page_bytes in (6, 4352, 41984):
                for offset in (0, 1):
                    with self.subTest(
                        dtype=dtype, page_bytes=page_bytes, offset=offset
                    ):
                        item_size = torch.empty((), dtype=dtype).element_size()
                        skip = offset * item_size
                        target = (
                            torch.randint(
                                0,
                                256,
                                (9 * page_bytes + skip,),
                                dtype=torch.uint8,
                                device="cuda",
                            )[skip:]
                            .view(dtype)
                            .reshape(9, -1)
                        )
                        source = (
                            torch.randint(
                                0,
                                256,
                                (3 * page_bytes + skip,),
                                dtype=torch.uint8,
                                device="cuda",
                            )[skip:]
                            .view(dtype)
                            .reshape(3, -1)
                        )
                        pages = torch.tensor([8, 0, 3], device="cuda")
                        reference = target.view(torch.uint8).clone()
                        reference.index_copy_(0, pages, source.view(torch.uint8))
                        address = target.data_ptr()
                        # Producer, refill and consumer share a non-default stream.
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            scatter_unique_pages(target, pages, source)
                            observed = target.view(torch.uint8).clone()
                        torch.cuda.current_stream().wait_stream(stream)
                        self.assertTrue(torch.equal(observed, reference))
                        self.assertEqual(target.data_ptr(), address)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_empty_and_strided_pages(self):
        from sglang.kernels.ops.memory.layer_split_page_scatter import (
            scatter_unique_pages,
        )

        target = torch.full((9, 34), 17, device="cuda", dtype=torch.uint8)[:, ::2]
        source = torch.full((3, 34), 99, device="cuda", dtype=torch.uint8)[:, ::2]
        pages = torch.tensor([8, 0, 3], device="cuda")
        reference = target.clone()
        reference.index_copy_(0, pages, source)
        scatter_unique_pages(target, pages, source)
        self.assertTrue(torch.equal(target, reference))
        scatter_unique_pages(target, pages[:0], source[:0])
        self.assertTrue(torch.equal(target, reference))


if __name__ == "__main__":
    unittest.main()
