import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

# This test verifies that the weight-loading guard in
# DeepseekV4ForCausalLM.load_weights accepts int8 wo_a.weight tensors
# when SGLANG_OPT_FP8_WO_A_GEMM=1 on PPU, where deep_gemm.int8_einsum
# is used for W8A8Int8LinearMethod wo_a projection.
#
# Without the fix, the guard raises:
#   ValueError: SGLANG_OPT_FP8_WO_A_GEMM is enabled but
#   layers.0.atttn.w_o.a.weight has dtype torch.int8, expected
#   torch.float8_e4m3fn.

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "modelscope.cn/organization/T-HEAD/DeepSeek-V4-Flash-W8A8-INT8",
)
SERVED_MODEL_NAME = "DeepSeek-V4"
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 60000  # match --dist-timeout

# Local GSM8K dataset path — set GSM8K_DATA_PATH env var to avoid
# re-downloading the dataset on every run.
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


class TestDeepseekV4PpuInt8WoAGemm(GSM8KMixin, CustomTestCase):
    """E2E: DeepSeek-V4 W8A8-INT8 on PPU with SGLANG_OPT_FP8_WO_A_GEMM=1.

    Verifies that the server boots successfully with int8 wo_a weights
    and produces correct GSM8K accuracy.
    """

    # --- GSM8KMixin config ---
    model = SERVED_MODEL_NAME
    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    # --- server launch config ---
    @classmethod
    def setUpClass(cls):
        other_args = [
            "--tp-size",
            "8",
            "--mem-fraction-static",
            "0.8",
            "--quantization",
            "w8a8_int8",
            "--trust-remote-code",
            "--dist-timeout",
            "60000",
            "--watchdog-timeout",
            "60000",
            "--soft-watchdog-timeout",
            "60000",
            "--served-model-name",
            SERVED_MODEL_NAME,
            "--disable-custom-all-reduce",
            "--disable-shared-experts-fusion",
            "--cuda-graph-max-bs",
            "64",
            "--disable-piecewise-cuda-graph",
        ]
        cls.process = popen_launch_server(
            MODEL_PATH,
            BASE_URL,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env={
                **os.environ,
                "SGLANG_OPT_FP8_WO_A_GEMM": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @property
    def base_url(self):
        return BASE_URL


if __name__ == "__main__":
    unittest.main()
