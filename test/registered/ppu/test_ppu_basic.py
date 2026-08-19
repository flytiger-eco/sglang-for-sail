"""
Basic PPU test: verifies the server starts and produces a non-empty
response on T-HEAD PPU with the default attention backend.

Assigned to stage-a so it gates stage-b before the heavier tests run.

Usage:
python3 -m unittest test_ppu_basic.TestPPUBasic.test_basic_generation

Dry-run change to validate the run-ci label gate (no functional change).
"""

import unittest

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    is_in_ci,
    run_bench_one_batch,
)

register_ppu_ci(est_time=300, suite="per-commit-1-ppu")

PPU_CI_MODEL_PATH = "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-0.5B-Instruct/"


class TestPPUBasic(CustomTestCase):

    def test_basic_generation(self):
        """Server starts on PPU and completes at least one decode step."""
        args = [
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.6",
            "--batch-size",
            "1",
        ]
        if is_in_ci():
            args += ["--input", "64", "--output", "4"]

        _, decode_throughput, _ = run_bench_one_batch(
            PPU_CI_MODEL_PATH, args
        )
        self.assertGreater(decode_throughput, 0, "PPU decode throughput must be > 0")


if __name__ == "__main__":
    unittest.main()