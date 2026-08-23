"""
Basic PPU test: verifies the engine initializes on T-HEAD PPU and
completes at least one decode step. It runs run_bench_one_batch
(``python3 -m sglang.bench_one_batch``), the OFFLINE engine path:
no HTTP server is launched, so launch_server / HTTP routing are not
covered (test_anthropic_server.py covers that surface in stage-a).

Registered in suite stage-a-test-1-gpu-ppu so it gates stage-b before
the heavier tests run.

Usage:
python3 -m unittest test_ppu_basic.TestPPUBasic.test_basic_generation
"""

import unittest

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import (
    CustomTestCase,
    is_in_ci,
    run_bench_one_batch,
)

register_ppu_ci(est_time=120, suite="stage-a-test-1-gpu-ppu")

PPU_CI_MODEL_PATH = "/nas_aisw/datasets/checkpoints/LLM/qwen/v2.5/Qwen2.5-0.5B-Instruct/"


class TestPPUBasic(CustomTestCase):

    def test_basic_generation(self):
        """Engine initializes on PPU and completes at least one decode step."""
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