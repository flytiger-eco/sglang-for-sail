"""Logical-IDLE regression test for Kimi-K3 INT4/INT8 + DSPARK + DP4.

By default this launches node-rank 0 of the two-node TP32/EP32 topology;
the other node must be started separately with the same sglang serve
arguments and --node-rank 1. The test framework's HTTP service binds to
BASE_URL (default 127.0.0.1:8999) while distributed initialization still
goes through DIST_INIT_ADDR.

If both nodes are already running the reproduction command, run only the
test client against them:
  E2E_USE_EXISTING_SERVER=1 python test/ppu/e2e/test_kimi_k3_dspark_dp_attention.py

MODEL_PATH, DSPARK_PATH, DIST_INIT_ADDR, BASE_URL, and STARTUP_TIMEOUT
override the deployment paths and addresses. Attach mode never stops the
external server; keep the server exclusive to this test while it runs so
the single-request scenarios leave the other three DP ranks idle.
GSM8K_DATA_PATH may point at a local dataset copy.
"""

import concurrent.futures as cf
import json
import math
import os
import unittest
import urllib.request

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import CustomTestCase, popen_launch_server

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "moonshotai/Kimi-K3/",
)
DSPARK_PATH = os.environ.get("DSPARK_PATH", "RadixArk/K3/Kimi-K3-DSpark")
DIST_INIT_ADDR = os.environ.get("DIST_INIT_ADDR", "")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8999").rstrip("/")
SERVER_LAUNCH_TIMEOUT = int(os.environ.get("STARTUP_TIMEOUT", "24000"))
DP_SIZE = 4
BLOCK_SIZE = 7
CONCURRENT_REQUESTS = 16
# Only require that the accept length has not collapsed to pure target
# decoding (1.0); no unmeasured K3 performance baseline is assumed.
MIN_MEAN_ACCEPT_LENGTH = 1.0


def server_args():
    return [
        "--dist-init-addr",
        DIST_INIT_ADDR,
        "--nnodes",
        "2",
        "--node-rank",
        "0",
        "--tp-size",
        "32",
        "--ep-size",
        "32",
        "--mem-fraction-static",
        "0.85",
        "--speculative-algorithm",
        "DSPARK",
        "--speculative-draft-model-path",
        DSPARK_PATH,
        "--disable-radix-cache",
        "--enable-dp-attention",
        "--dp-size",
        str(DP_SIZE),
        "--trust-remote-code",
        "--max-running-requests",
        "128",
        "--enable-dp-lm-head",
        "--watchdog-timeout",
        "600",
        "--decode-attention-backend",
        "flashmla",
        "--prefill-attention-backend",
        "fa3",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--speculative-dspark-block-size",
        str(BLOCK_SIZE),
        "--speculative-attention-mode",
        "decode",
        "--moe-runner-backend",
        "deep_gemm",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--linear-attn-backend",
        "flashkda",
        "--enable-linear-replayssm-spec",
    ]


def server_info():
    with urllib.request.urlopen(f"{BASE_URL}/server_info", timeout=60) as response:
        return json.loads(response.read())


class TestKimiK3DsparkDpAttention(GSM8KMixin, CustomTestCase):
    model = MODEL_PATH
    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    # Single-threaded evaluation runs every DSPARK verify while the other
    # DP ranks are idle.
    gsm8k_num_threads = 1
    gsm8k_accept_length_thres = MIN_MEAN_ACCEPT_LENGTH
    gsm8k_data_path = os.environ.get("GSM8K_DATA_PATH")
    process = None

    @classmethod
    def setUpClass(cls):
        mode = os.environ.get("E2E_USE_EXISTING_SERVER", "0")
        if mode not in ("0", "1"):
            raise ValueError("E2E_USE_EXISTING_SERVER must be 0 or 1")
        if SERVER_LAUNCH_TIMEOUT <= 0:
            raise ValueError("STARTUP_TIMEOUT must be positive")
        if mode == "0":
            cls.process = popen_launch_server(
                MODEL_PATH,
                BASE_URL,
                timeout=SERVER_LAUNCH_TIMEOUT,
                other_args=server_args(),
                env=dict(os.environ),
            )
        info = server_info()
        cls("runTest")._assert_configuration(info)
        # An attached server may not set served-model-name; use the model
        # name it actually exposes.
        cls.model = info.get("served_model_name") or info["model_path"]

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
            cls.process = None

    @property
    def base_url(self):
        return BASE_URL

    def _assert_configuration(self, info):
        expected = {
            "tp_size": 32,
            "ep_size": 32,
            "nnodes": 2,
            "node_rank": 0,
            "dp_size": DP_SIZE,
            "max_running_requests": 128,
            "speculative_algorithm": "DSPARK",
            "speculative_dspark_block_size": BLOCK_SIZE,
            "speculative_num_draft_tokens": BLOCK_SIZE + 1,
            "enable_dp_attention": True,
            "enable_dp_lm_head": True,
            "disable_radix_cache": True,
            "enable_linear_replayssm_spec": True,
            "linear_attn_backend": "flashkda",
            "mamba_radix_cache_strategy": "extra_buffer",
            "moe_runner_backend": "deep_gemm",
            "decode_attention_backend": "flashmla",
            "speculative_attention_mode": "decode",
        }
        for key, value in expected.items():
            self.assertEqual(info.get(key), value, f"server config mismatch: {key}")
        self.assertEqual(
            os.path.normpath(info["model_path"]), os.path.normpath(MODEL_PATH)
        )
        self.assertEqual(
            os.path.normpath(info["speculative_draft_model_path"]),
            os.path.normpath(DSPARK_PATH),
        )
        # The SAIL q_v safety fallback rewrites the requested fa3 prefill
        # backend to flashmla.
        self.assertIn(info.get("prefill_attention_backend"), ("fa3", "flashmla"))

    def _generate(self, prompt, dp_rank):
        body = json.dumps(
            {
                "text": prompt,
                "routed_dp_rank": dp_rank,
                "sampling_params": {"temperature": 0, "max_new_tokens": 512},
            }
        ).encode()
        request = urllib.request.Request(
            f"{BASE_URL}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1200) as response:
            return json.loads(response.read())

    def _assert_output(self, output, expected, dp_rank):
        text = output.get("text", "")
        self.assertTrue(text.strip(), f"DP rank {dp_rank} returned empty output")
        self.assertNotIn(
            "\ufffd", text, f"DP rank {dp_rank} returned corrupted text: {text!r}"
        )
        self.assertRegex(
            text,
            rf"(?<!\d){expected}(?!\d)",
            f"DP rank {dp_rank} wrong answer, expected {expected}: {text!r}",
        )
        meta = output["meta_info"]
        self.assertEqual(meta.get("dp_rank"), dp_rank)
        self.assertGreater(meta.get("completion_tokens", 0), 0)
        self.assertGreater(meta.get("spec_verify_ct", 0), 1, meta)
        self.assertIn((meta.get("finish_reason") or {}).get("type"), ("stop", "length"))
        accept_length = meta.get("spec_accept_length")
        self.assertIsInstance(accept_length, (int, float), meta)
        self.assertTrue(math.isfinite(accept_length), meta)
        self.assertGreaterEqual(accept_length, 1.0, meta)
        return accept_length

    def _assert_mean_accept_length(self, lengths):
        self.assertTrue(lengths)
        mean = sum(lengths) / len(lengths)
        print(f"Kimi-K3 DSPARK mean spec_accept_length={mean:.4f}")
        self.assertGreater(
            mean,
            MIN_MEAN_ACCEPT_LENGTH,
            "DSPARK accept length collapsed; idle-rank output may be corrupted",
        )

    def _idle_rank_probes(self):
        lengths = []
        # Route one request at a time to each rank in turn; the other three
        # ranks see no user request each round. The second round also reuses
        # recycled request/state slots.
        for question, answer in (
            ("17 multiplied by 23", 391),
            ("144 divided by 12", 12),
        ):
            for rank in range(DP_SIZE):
                with self.subTest(question=question, dp_rank=rank):
                    output = self._generate(
                        f"Q: What is {question}? Explain the calculation briefly "
                        "and give the final numerical answer.\nA:",
                        rank,
                    )
                    lengths.append(self._assert_output(output, answer, rank))
        self._assert_mean_accept_length(lengths)

    def test_single_request_with_three_idle_dp_ranks(self):
        self._idle_rank_probes()

    def test_concurrent_requests_then_idle_dp_ranks(self):
        def one(index):
            rank = index % DP_SIZE
            output = self._generate(
                f"Q: A store has {100 + index} widgets. It sells 37 and then "
                "receives 55. How many widgets are there now? Explain briefly "
                "and give the final numerical answer.\nA:",
                rank,
            )
            return output, 100 + index - 37 + 55, rank

        with cf.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
            results = list(executor.map(one, range(CONCURRENT_REQUESTS)))
        self.assertEqual(len(results), CONCURRENT_REQUESTS)
        ranks = {output["meta_info"].get("dp_rank") for output, _, _ in results}
        self.assertEqual(ranks, set(range(DP_SIZE)))
        lengths = [
            self._assert_output(output, answer, rank)
            for output, answer, rank in results
        ]
        self._assert_mean_accept_length(lengths)
        # After the concurrent load, go back to single requests to cover the
        # active -> logical-IDLE transition.
        self._idle_rank_probes()


if __name__ == "__main__":
    unittest.main()
