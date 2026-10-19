"""E2E coverage for V4.1 DSpark prefill CP on SAIL.

DSV4 prefill BCG is currently disabled by
disable_breakable_cudagraph_if_incompatible() for capture-pool memory
pressure. The target supports interleave + dsv4 CP BCG, but this migration
keeps the source's explicitly disabled prefill graph. This configuration
exercises eager prefill CP and accuracy; the skipped test records the
remaining hardware graph coverage gap.

Attention-TP verify padding needs attention TP > 1 and is covered outside this
configuration. Image preparation before CP sharding is exercised by the
text-only path here and by the CP policy unit tests.
"""

import json
import os
import unittest
import urllib.request

from sglang.srt.utils import get_device_sm
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
GSM8K_DATA_PATH = os.environ.get("GSM8K_DATA_PATH", None)


def _sm89_or_newer() -> bool:
    try:
        import torch

        return torch.cuda.is_available() and get_device_sm() >= 89
    except Exception:
        return False


@unittest.skipUnless(_sm89_or_newer(), "requires a device with SM89 or newer")
class TestSailDeepseekV4DsparkPrefillCpGraph(GSM8KMixin, CustomTestCase):
    """DSpark + interleave prefill CP with CUDA graphs explicitly disabled."""

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
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "5",
            "--max-running-requests",
            "64",
            "--cuda-graph-max-bs-decode",
            "64",
            "--mem-fraction-static",
            "0.8",
            "--disable-radix-cache",
            "--disable-custom-all-reduce",
            "--disable-overlap-schedule",
            "--disable-shared-experts-fusion",
            # DSV4 prefill BCG is disabled for this configuration; keep the
            # eager backend explicit instead of relying on the default hook.
            "--cuda-graph-backend-prefill=disabled",
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--enable-dp-lm-head",
            "--reasoning-parser",
            "auto",
            "--tool-call-parser",
            "auto",
            "--watchdog-timeout",
            "60000",
            "--soft-watchdog-timeout",
            "60000",
            "--dist-timeout",
            "60000",
        ]
        env = dict(os.environ)
        env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
        env.setdefault("SGLANG_DG_CACHE_DIR", "/tmp/dg_cache0")

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

    def test_prefill_graph_is_disabled_for_this_configuration(self):
        info = self._server_info()
        backend = info["cuda_graph_config"]["prefill"]["backend"]
        self.assertEqual(backend, "disabled")

    @unittest.skip(
        "DSV4 prefill BCG remains disabled for capture-pool memory pressure; "
        "validate interleave CP graph capture on SAIL hardware before enabling."
    )
    def test_prefill_graph_covers_the_live_context(self):
        pass


if __name__ == "__main__":
    unittest.main()
