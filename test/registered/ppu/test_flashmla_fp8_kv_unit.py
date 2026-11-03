"""Hardware-free FlashMLA contracts, loader policy and legacy-env regressions."""

import sys
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from registered.ppu import load_legacy_ppu_module

legacy = load_legacy_ppu_module("unit/test_flashmla_fp8_kv.py")
TestPpuFlashMlaFp8 = legacy.TestPpuFlashMlaFp8
TestFlashMlaReviewRegressions = legacy.TestFlashMlaReviewRegressions

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=10, suite="stage-a-test-cpu-ppu")

if __name__ == "__main__":
    unittest.main()
