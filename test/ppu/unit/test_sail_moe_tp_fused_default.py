"""Default-on UT for SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED (fused MoE hybrid path).

Why: 810E FAB105/EXP07 A/B measured TTFT -3.7~-4.0% (P2/P3/P4, VERIFIED) and
Native companion-kernel share 22.90%->18.27% when the fused path is enabled;
the flag is promoted to default-on so every PPU deep_gemm MoE run gets the gain
without opt-in. Explicit SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED=0 must still disable
it (escape hatch preserved).

Run: python -m pytest -q test/ppu/unit/test_sail_moe_tp_fused_default.py
(no PPU device needed; env-field semantics only)
"""

import os

from sglang.srt.environ import envs

NAME = "SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED"


def test_unset_uses_new_default_on():
    os.environ.pop(NAME, None)
    assert envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.get() is True


def test_explicit_zero_still_disables():
    os.environ[NAME] = "0"
    try:
        assert envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.get() is False
    finally:
        os.environ.pop(NAME, None)


def test_explicit_one_enables():
    os.environ[NAME] = "1"
    try:
        assert envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.get() is True
    finally:
        os.environ.pop(NAME, None)
