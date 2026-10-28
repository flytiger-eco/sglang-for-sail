"""Opt-in PPU FlashMLA FP8 KV inference and accuracy regressions.

Set PPU_FLASHMLA_MODEL_PATH for a dense MLA model (Q/K=576, V=512), or
PPU_DSA_MODEL_PATH for a DeepSeek V3.2-style DSA model. Select the actual
TP size with PPU_FP8_KV_TP_SIZE; no device count/model path is assumed.
Requires FlashMLA 2.1 and a verified PPU software environment.
"""

import math
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server


class _Fp8KvInference:
    gsm8k_score_threshold = float(os.environ.get("PPU_FP8_KV_GSM8K_THRESHOLD", "0.5"))
    gsm8k_num_examples = 200
    gsm8k_data_path = os.environ.get("GSM8K_DATA_PATH")
    model = "ppu-fp8-kv"
    backend_args = []

    @classmethod
    def setUpClass(cls):
        tp_size = os.environ.get("PPU_FP8_KV_TP_SIZE")
        if not tp_size:
            raise unittest.SkipTest(
                "Set PPU_FP8_KV_TP_SIZE to the verified device count"
            )
        cls.url = os.environ.get("PPU_FP8_KV_BASE_URL", "http://127.0.0.1:8999")
        cls.process = popen_launch_server(
            os.environ[cls.model_env],
            cls.url,
            timeout=24000,
            other_args=[
                "--trust-remote-code",
                "--served-model-name",
                cls.model,
                "--tp-size",
                tp_size,
                "--dtype",
                "bfloat16",
                "--kv-cache-dtype",
                "fp8_e4m3",
                "--page-size",
                "64",
                "--chunked-prefill-size",
                "256",
                "--cuda-graph-max-bs-decode",
                "4",
                "--disable-piecewise-cuda-graph",
                *cls.backend_args,
            ],
            env=os.environ,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    @property
    def base_url(self):
        return self.url

    def test_chunked_prefill_prefix_and_graph_decode(self):
        prefix = "The notebook contains the numbers one, two, three and four. " * 100
        # Repeated prefix across requests exercises cache reuse. Multiple new
        # tokens and batches exercise graph decode after chunked prefill.
        for texts in (
            [prefix + "What number comes after three?"],
            [
                prefix + "What number comes after two?",
                prefix + "What number comes before four?",
            ],
        ):
            response = requests.post(
                self.url + "/generate",
                json={
                    "text": texts,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                    "return_logprob": True,
                },
                timeout=600,
            )
            response.raise_for_status()
            outputs = response.json()
            self.assertEqual(len(outputs), len(texts))
            for output in outputs:
                self.assertTrue(output["text"].strip())
                self.assertGreater(output["meta_info"]["completion_tokens"], 0)
                logprobs = output["meta_info"]["output_token_logprobs"]
                self.assertTrue(logprobs)
                self.assertTrue(all(math.isfinite(entry[0]) for entry in logprobs))


@unittest.skipUnless(
    os.environ.get("PPU_FLASHMLA_MODEL_PATH"), "Set PPU_FLASHMLA_MODEL_PATH"
)
class TestPpuDenseFlashMlaFp8Kv(_Fp8KvInference, GSM8KMixin, CustomTestCase):
    model_env = "PPU_FLASHMLA_MODEL_PATH"
    backend_args = [
        "--prefill-attention-backend",
        "fa3",
        "--decode-attention-backend",
        "flashmla",
    ]


@unittest.skipUnless(os.environ.get("PPU_DSA_MODEL_PATH"), "Set PPU_DSA_MODEL_PATH")
class TestPpuDsaFlashMlaFp8Kv(_Fp8KvInference, GSM8KMixin, CustomTestCase):
    model_env = "PPU_DSA_MODEL_PATH"
    backend_args = [
        "--attention-backend",
        "dsa",
        "--dsa-prefill-backend",
        "flashmla_sparse",
        "--dsa-decode-backend",
        "flashmla_kv",
    ]


if __name__ == "__main__":
    unittest.main()
