"""Collect only the four CPU pool cases and the real mocked-boundary hook test."""

import sys
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ppu.module.test_ppu_deepep_hooks import TestPPUDeepEPHooks  # noqa: E402,F401
from ppu.test_kda_replayssm_pool import (  # noqa: E402,F401
    test_replayssm_pool_and_budget,
)

register_cpu_ci(est_time=10, suite="stage-a-test-cpu-ppu")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
