"""PPU nightly Answer MVP for Qwen3.5-397B-A17B-W8A8-INT8.

This ports the ten public Answer prompts into the GitHub-native CI path.  The
current gate is deliberately limited to L0 request integrity and L1
deterministic facts/quality checks.  LLM-as-Judge is deferred and its absence is
recorded explicitly in every result.

The executable body lives in `AnswerSuiteMixin`; this file only names the
reviewed configuration and the suite that owns the eight devices it needs.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.answer_suite_kit import AnswerSuiteMixin

DATA_DIR = Path(__file__).with_name("answer_eval") / "data"

register_ppu_ci(est_time=3600, suite="nightly-answer-8-ppu", nightly=True)


class TestPPUQwen35Answer(AnswerSuiteMixin, unittest.TestCase):
    default_test_config_path = DATA_DIR / "qwen3_5_397b_a17b_w8a8_int8_test_config.json"


if __name__ == "__main__":
    unittest.main()
