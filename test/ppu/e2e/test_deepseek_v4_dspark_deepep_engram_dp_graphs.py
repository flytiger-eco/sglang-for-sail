"""E2E coverage for DeepEP DSpark routing and sharded Engram DP graphs on PPU.

One 8x PPU server runs DSpark + DeepEP low-latency dispatch with attention DP.
In this layout a single run exercises:

  * the target/draft low-latency layouts registered before buffer allocation,
    sized together and validated on every dispatch;
  * the CUDA-graph capture/replay path that restores the captured layout;
  * sharded Engram under attention DP (the host-table env stays unset), which
    must keep the MLP TP gather and DP-sized capture metadata working;
  * V4.1 vision with DeepEP when the served checkpoint carries vision layers,
    hence the separate shared experts.

Attention-TP verify padding needs attention TP > 1, which the DSpark PD
validation forbids together with decode DP; it is covered by TP>1 verify runs
outside this configuration.
"""

import concurrent.futures as cf
import json
import os
import unittest
import urllib.request

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "DeepSeek-V4.1-Flash",
)
BASE_URL = "http://127.0.0.1:8999"
SERVER_LAUNCH_TIMEOUT = 60000

# DSpark pushes bs * (gamma + 1) rows per rank into DeepEP low-latency
# dispatch; 64 * 6 = 384 stays under the hard 1024 ceiling.
DISPATCH_TOKENS_PER_RANK = "512"
MAX_RUNNING_REQUESTS = 64
GAMMA = 5
MIN_MEAN_ACCEPT_LENGTH = 2.0
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


def _sm89_or_newer() -> bool:
    try:
        import torch

        from sglang.srt.utils import get_device_sm

        return torch.cuda.is_available() and get_device_sm() >= 89
    except Exception:
        return False


@unittest.skipUnless(_sm89_or_newer(), "requires a device with SM89 or newer")
class TestDeepSeekV4DsparkDeepEpEngramDpGraphs(GSM8KMixin, CustomTestCase):
    """DSpark + DeepEP low-latency + attention DP + sharded Engram + graphs."""

    gsm8k_score_threshold = 0.90
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0
    gsm8k_data_path = GSM8K_DATA_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--served-model-name",
            "DeepSeek-V4-Flash",
            "--tp-size",
            "8",
            "--ep-size",
            "8",
            "--enable-dp-attention",
            "--dp-size",
            "8",
            "--enable-dp-lm-head",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "low_latency",
            "--moe-dense-tp-size",
            "1",
            # V4.1 vision with DeepEP needs separate shared experts.
            "--disable-shared-experts-fusion",
            "--attention-backend",
            "dsv4",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--mem-fraction-static",
            "0.7",
            "--max-running-requests",
            str(MAX_RUNNING_REQUESTS),
            "--cuda-graph-max-bs-decode",
            str(MAX_RUNNING_REQUESTS),
            "--chunked-prefill-size",
            "4096",
            "--page-size",
            "256",
            "--disable-radix-cache",
            "--disable-custom-all-reduce",
            "--schedule-conservativeness",
            "0.3",
            "--num-continuous-decode-steps",
            "1",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            str(GAMMA),
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            str(GAMMA + 1),
            "--reasoning-parser",
            "auto",
            "--tool-call-parser",
            "auto",
            "--enable-cache-report",
            "--decode-log-interval",
            "1",
            "--watchdog-timeout",
            "60000",
            "--soft-watchdog-timeout",
            "60000",
            "--dist-timeout",
            "60000",
        ]

        env = dict(os.environ)
        env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = DISPATCH_TOKENS_PER_RANK
        env["SGLANG_RAGGED_VERIFY_MODE"] = "static"
        env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
        env.setdefault("SGLANG_DG_CACHE_DIR", "/tmp/dg_cache0")
        # Keep the Engram host table disabled so the sharded-table path (MLP TP
        # gather + DP-sized graph metadata) is the one under test.
        env["SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"] = "0"

        cls.process = popen_launch_server(
            MODEL_PATH,
            BASE_URL,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        process = getattr(cls, "process", None)
        if process is not None:
            terminate_and_kill_process_tree(process)

    @property
    def base_url(self):
        return BASE_URL

    def _server_info(self):
        with urllib.request.urlopen(f"{BASE_URL}/server_info", timeout=60) as r:
            return json.loads(r.read())

    def _generate(self, text, max_new_tokens):
        body = json.dumps(
            {
                "text": text,
                "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
            }
        ).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.loads(r.read())

    def _chat(self, user_text, max_new_tokens):
        body = json.dumps(
            {
                "model": "DeepSeek-V4-Flash",
                "messages": [{"role": "user", "content": user_text}],
                "max_tokens": max_new_tokens,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            resp = json.loads(r.read())
        return resp["choices"][0]["message"].get("content") or ""

    def test_server_resolves_to_dspark_deepep_dp8(self):
        info = self._server_info()
        self.assertEqual(info["speculative_algorithm"], "DSPARK")
        self.assertEqual(info["moe_a2a_backend"], "deepep")
        self.assertEqual(info["tp_size"], 8)
        self.assertEqual(info["dp_size"], 8)
        self.assertEqual(info["ep_size"], 8)
        self.assertTrue(info["enable_dp_attention"])
        self.assertEqual(info["speculative_num_draft_tokens"], GAMMA + 1)

    def test_concurrent_requests_use_every_dp_rank_under_graph_replay(self):
        """64 concurrent requests must complete across all 8 DP ranks while
        decode replays the captured DeepEP layout."""
        # Trigger graph capture and layout selection with one request before
        # the burst so the burst itself runs on an established layout.
        self._generate("Hello", 8)

        prompt = (
            "Q: A store has {} widgets. It sells 37 and then receives a shipment "
            "of 55. How many widgets does it have now? Think step by step.\nA:"
        )

        def one(i):
            out = self._generate(prompt.format(100 + i), 1024)
            mi = out["meta_info"]
            return {
                "dp_rank": mi.get("dp_rank"),
                "accept_length": mi.get("spec_accept_length"),
                "retractions": mi.get("num_retractions"),
            }

        with cf.ThreadPoolExecutor(max_workers=MAX_RUNNING_REQUESTS) as ex:
            results = list(ex.map(one, range(MAX_RUNNING_REQUESTS)))

        self.assertEqual(len(results), MAX_RUNNING_REQUESTS)
        self.assertEqual(sum(r["retractions"] or 0 for r in results), 0)

        ranks = {r["dp_rank"] for r in results}
        self.assertEqual(
            ranks,
            set(range(8)),
            f"requests did not reach every DP rank: {sorted(ranks)}",
        )

        accept = [r["accept_length"] for r in results if r["accept_length"]]
        self.assertTrue(accept, "no spec_accept_length reported; draft not running?")
        mean_accept = sum(accept) / len(accept)
        self.assertGreater(
            mean_accept,
            MIN_MEAN_ACCEPT_LENGTH,
            f"mean accept length {mean_accept:.2f} suggests a broken draft",
        )

    def test_concurrent_chat_answers_are_accurate(self):
        """64 concurrent chat requests, the production templated path, must all
        answer correctly while every DP rank participates."""
        prompt = (
            "A store has {} widgets. It sells 37 and then receives a shipment "
            "of 55. How many widgets does it have now?"
        )

        def one(i):
            text = self._chat(prompt.format(100 + i), 4096)
            return {"expected": 100 + i - 37 + 55, "text": text}

        with cf.ThreadPoolExecutor(max_workers=MAX_RUNNING_REQUESTS) as ex:
            results = list(ex.map(one, range(MAX_RUNNING_REQUESTS)))

        wrong = [r for r in results if str(r["expected"]) not in r["text"]]
        if wrong:
            self.fail(
                f"{len(wrong)} wrong answers, e.g. expected {wrong[0]['expected']}: "
                f"{wrong[0]['text'][:200]!r}"
            )

    def test_single_request_on_idle_ranks_keeps_the_handshake(self):
        """One request leaves 7 DP ranks idle; they must still join the DeepEP
        low-latency handshake, and repeated decode steps must not deadlock."""
        self.assertIn("Paris", self._generate("The capital of France is", 32)["text"])
        for _ in range(3):
            out = self._generate("Q: What is 17 multiplied by 23?\nA:", 64)
            self.assertIn("391", out["text"])


if __name__ == "__main__":
    unittest.main()
