"""Run BF16 on all PPU workers; select FP8 probes only on eligible hardware.

The current 810E lane exercises the BF16 regressions. FP8 numerical/graph
coverage additionally requires an SM89-or-newer worker and a compatible
FlashMLA binary. An unsupported operator binary must fail these probes.
"""

import sys
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_ppu_ci

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from registered.ppu import load_legacy_ppu_module

legacy = load_legacy_ppu_module("unit/test_flashmla_fp8_kv_device.py")
TestFlashMlaBf16Device = legacy.TestFlashMlaBf16Device

# Do not weaken the strict PR runner's skip policy. On pre-FP8 hardware only
# BF16 cases belong to this lane; they still execute real numerical assertions.
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9):
    TestFlashMlaFp8Device = legacy.TestFlashMlaFp8Device

register_ppu_ci(est_time=90, suite="stage-b-kernel-test-1-gpu-ppu")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
