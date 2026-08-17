"""PPU FlashMLA wrapper verification.

Launches a server with an MLA-architecture model on PPU (TP=2) and verifies
FlashMLA correctness via GSM8K evaluation.
"""

import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_ppu_ci(est_time=480, suite="nightly-2-ppu", nightly=True)

PPU_MLA_MODEL = os.environ.get(
    "PPU_MLA_MODEL_PATH",
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
)


class TestPPUFlashMLA(CustomTestCase, GSM8KMixin):
    gsm8k_accuracy_thres = 0.30
    gsm8k_num_questions = 200
    gsm8k_num_threads = 64

    @classmethod
    def setUpClass(cls):
        cls.model = PPU_MLA_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 2,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main(verbosity=3)
