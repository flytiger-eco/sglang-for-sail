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

DATA_ROOT = Path(__file__).parent

register_ppu_ci(est_time=3600, suite="nightly-answer-8-ppu", nightly=True)


class TestPPUQwen35Answer(AnswerSuiteMixin, unittest.TestCase):
    data_root = DATA_ROOT
    default_test_config_path = (
        DATA_ROOT / "configs" / "qwen3.5" / "397b-a17b-w8a8-int8.json"
    )


if __name__ == "__main__":
    unittest.main()
