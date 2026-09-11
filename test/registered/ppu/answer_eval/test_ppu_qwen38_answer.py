"""PPU nightly Answer MVP for Qwen3.8-27B (BF16, single device).

Ported from the internal `qwen3.8-27b-bf16_3001` case, whose test plan pins it
to ZW810E with one device.  It runs the same ten public Answer prompts and the
same deterministic evaluator as the 397B suite, against a model small enough to
serve on a single card, so a regression in the shared serving path is separable
from one that only appears at TP=8.

It is a second file rather than a second class because `register_ppu_ci`
registers a suite per file, and the two cases claim different device counts.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.answer_suite_kit import AnswerSuiteMixin

DATA_ROOT = Path(__file__).parent

register_ppu_ci(est_time=1200, suite="nightly-answer-1-ppu", nightly=True)


class TestPPUQwen38Answer(AnswerSuiteMixin, unittest.TestCase):
    data_root = DATA_ROOT
    default_test_config_path = DATA_ROOT / "configs" / "qwen3.8" / "27b-bf16.json"


if __name__ == "__main__":
    unittest.main()
