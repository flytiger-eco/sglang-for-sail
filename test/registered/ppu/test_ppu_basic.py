"""PPU platform smoke test: platform detection, server startup, and basic inference."""

import concurrent.futures
import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_ppu_ci(est_time=180, suite="per-commit-1-ppu")

PPU_SMALL_MODEL = os.environ.get(
    "PPU_SMALL_MODEL_PATH",
    "Qwen/Qwen2.5-0.5B-Instruct",
)


class TestPPUPlatformDetection(CustomTestCase):
    def test_is_ppu_with_env(self):
        from sglang.srt.utils.common import is_ppu

        self.assertTrue(
            is_ppu(),
            "is_ppu() should return True when PPU_SDK is set (running on PPU hardware)",
        )

    def test_ppu_sdk_env_present(self):
        self.assertIn(
            "PPU_SDK",
            os.environ,
            "PPU_SDK environment variable must be set on PPU runners",
        )


class TestPPUServerStartup(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = PPU_SMALL_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_health(self):
        resp = requests.get(self.base_url + "/health")
        self.assertEqual(resp.status_code, 200)

    def test_generate(self):
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"max_new_tokens": 32, "temperature": 0},
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(len(resp.json()["text"]), 0)

    def test_generate_with_sampling(self):
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Once upon a time",
                "sampling_params": {
                    "max_new_tokens": 64,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(len(resp.json()["text"]), 0)

    def test_generate_long_output(self):
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": "List the first 10 prime numbers:",
                "sampling_params": {"max_new_tokens": 128, "temperature": 0},
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(len(resp.json()["text"]), 10)

    def test_concurrent_requests(self):
        prompts = [f"Count to {i}:" for i in range(1, 6)]

        def send_request(prompt):
            return requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
                },
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_request, p) for p in prompts]
            for future in concurrent.futures.as_completed(futures):
                resp = future.result()
                self.assertEqual(resp.status_code, 200)
                self.assertGreater(len(resp.json()["text"]), 0)

    def test_model_info(self):
        resp = requests.get(self.base_url + "/get_model_info")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("model_path", resp.json())


if __name__ == "__main__":
    unittest.main(verbosity=3)
