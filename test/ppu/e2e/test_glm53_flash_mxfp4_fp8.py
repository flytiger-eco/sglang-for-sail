import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# E2E test for GLM-5.3-Flash MXFP4/FP8 on PPU with NSA attention
# and flashmla_sparse/tilelang DSA backends.

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelscope.cn/organization/T-HEAD/GLM-5.3-Flash-MXFP4-FP8",
)
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 24000

GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestGlm53FlashMxfp4Fp8DsaCpFlashmla(GSM8KMixin, CustomTestCase):
    """E2E: GLM-5.3-Flash MXFP4/FP8 on PPU with NSA attention + flashmla backends."""

    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "8",
            "--attention-backend",
            "nsa",
            "--mem-fraction-static",
            "0.9",
            "--disable-radix-cache",
            "--trust-remote-code",
            "--watchdog-timeout",
            "3600",
            "--reasoning-parser",
            "glm45",
            "--tool-call-parser",
            "glm47",
            "--enable-metrics",
            "--dsa-prefill-backend",
            "flashmla_sparse",
            "--dsa-decode-backend",
            "tilelang",
        ]
        cls.process = popen_launch_server(
            MODEL_PATH,
            BASE_URL,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=os.environ,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @property
    def base_url(self):
        return BASE_URL


if __name__ == "__main__":
    unittest.main()
