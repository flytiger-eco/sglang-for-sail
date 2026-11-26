"""Compare page-refill latency across page sizes and batch page counts.

Run this file directly to print JSON rows. Triton's do_bench clears L2 before
each measured call; JIT compilation is excluded from the measurements.
"""

import json
import unittest

import torch
import triton.testing

from sglang.kernels.ops.memory.layer_split_page_scatter import scatter_unique_pages
from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=30, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)
register_ppu_ci(est_time=30, suite="stage-b-test-1-gpu-ppu")


class TestLayerSplitPageScatterBenchmark(CustomTestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-compatible device required")
    def test_page_refill_latency(self):
        torch.manual_seed(0)
        print(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "triton": triton.__version__,
                    "cache": "L2 cleared before each measured call",
                    "timing": "median CUDA event latency; warmup=25ms, rep=100ms",
                    "dtype": "uint8 (raw page representations)",
                }
            ),
            flush=True,
        )
        for page_bytes in (6, 4352, 41984):
            for page_count in (1, 32, 512):
                with self.subTest(page_bytes=page_bytes, page_count=page_count):
                    destination = torch.randint(
                        0,
                        256,
                        (2 * page_count, page_bytes),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    packed = torch.randint(
                        0,
                        256,
                        (page_count, page_bytes),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    pages = torch.randperm(2 * page_count, device="cuda")[:page_count]
                    reference = destination.clone()
                    reference.index_copy_(0, pages, packed)
                    scatter_unique_pages(destination, pages, packed)
                    self.assertTrue(torch.equal(destination, reference))
                    before_ms = triton.testing.do_bench(
                        lambda: destination.index_copy_(0, pages, packed),
                        return_mode="median",
                    )
                    after_ms = triton.testing.do_bench(
                        lambda: scatter_unique_pages(destination, pages, packed),
                        return_mode="median",
                    )
                    print(
                        json.dumps(
                            {
                                "page_bytes": page_bytes,
                                "page_count": page_count,
                                "index_copy_us": before_ms * 1000,
                                "vector_scatter_us": after_ms * 1000,
                                "speedup": before_ms / after_ms,
                                "byte_exact": True,
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main()
