import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_pd_server,
)
from sglang.utils import wait_for_http_ready

# This test reproduces the mooncake segment-descriptor lookup failure:
#   ImportError: Failed to get segment descriptor for segment <ip:port> address
# which occurs when:
#   - PD disaggregation with mooncake transfer backend
#   - P side uses NSA attention backend (dsa_seed_backend_enabled = True)
#   - D side uses flashmla/fa3 attention backend (dsa_seed_backend_enabled = False)
#   - Model has index_share_for_mtp_iteration=True (e.g. GLM-5.2 with EAGLE MTP)
#
# Root cause:
#   get_dsa_seed_metadata_dim() checked the *local* attention backend and
#   returned 0 on the D side (flashmla), so D did not allocate the
#   output_dsa_topk_indices buffer.  P side (NSA) allocated it, inserting an
#   extra pointer into aux_data_ptrs.  The resulting list-length mismatch
#   shifted bootstrap_room's index, causing mooncake to look up a segment
#   descriptor that was never registered.
#
# Fix:
#   get_dsa_seed_metadata_dim() now returns get_dsa_index_topk(hf_config)
#   whenever index_share_for_mtp_iteration=True, regardless of the local
#   attention backend.  The D side allocates the buffer and fills it with
#   -1 sentinel.  Runtime guards (dsa_seed_backend_enabled in
#   eagle_disaggregation.py and index_share_for_mtp_iteration in
#   eagle_worker_v2.py) still prevent the data from reaching non-DSA
#   attention backends.
#
# ---------------------------------------------------------------------------
# Usage:
#
#   # ---- Cross-machine: P on machine A, D on machine B ----
#
#   # Machine A (P side):
#   CUDA_VISIBLE_DEVICES=0,1,...,15 PD_ROLE=prefill \
#       python3 -m pytest test/ppu_e2e/test_glm52_ppu_pd_disagg.py -s
#   # → starts P server, blocks until you press Enter or Ctrl+C
#
#   # Machine B (D side):
#   CUDA_VISIBLE_DEVICES=0,1,...,15 PD_ROLE=decode \
#       python3 -m pytest test/ppu_e2e/test_glm52_ppu_pd_disagg.py -s
#   # → starts D server, blocks until you press Enter or Ctrl+C
#
#   # Then start LB on either machine:
#   python3 -m sglang_router.launch_router --pd-disaggregation \
#       --prefill http://<P_IP>:8100 --decode http://<D_IP>:12100 \
#       --host 0.0.0.0 --port 8000
#
#   # Then run GSM8K from any machine:
#   python3 -m sglang.test.few_shot_gsm8k --num-questions 200 \
#       --host http://<LB_IP> --port 8000
#
#   # ---- Single-machine (both P and D as local subprocesses) ----
#   # Requires enough GPUs to split between P and D.
#   python3 -m pytest test/ppu_e2e/test_glm52_ppu_pd_disagg.py -s
#   # → starts P + D + LB, runs GSM8K automatically
# ---------------------------------------------------------------------------

PD_ROLE = os.environ.get("PD_ROLE", "both")  # "prefill" | "decode" | "both"

MODEL_PATH = os.environ.get("MODEL_PATH", "GLM-5.2-int8")
SERVED_MODEL_NAME = "GLM-5.2"

PD_BASE_ENV = {
    "PYTHONUNBUFFERED": "1",
    "MC_NUM_QP_PER_EP": "4",
    "SGL_ENABLE_JIT_DEEPGEMM": "1",
    "MOONCAKE_PROTOCOL": "rdma",
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    "SGLANG_NSA_FLASHMLA_BACKEND_DECODE_COMPUTE_FP8": "0",
    "MC_TE_METRIC": "1",
    "SGLANG_NSA_DUAL_STREAM": "0",
    "MC_LOG_LEVEL": "TRACE",
    "SGLANG_WARMUP_TIMEOUT": "3600",
}

IB_DEVICES = os.environ.get(
    "SGLANG_TEST_PD_DISAGG_DEVICES",
    "mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4",
)

SERVER_LAUNCH_TIMEOUT = 3600

# Port / host config for single-role mode
PD_HOST = os.environ.get("PD_HOST", "0.0.0.0")
PD_PREFILL_PORT = os.environ.get("PD_PREFILL_PORT", "8100")
PD_DECODE_PORT = os.environ.get("PD_DECODE_PORT", "12100")
PD_BOOTSTRAP_PORT = os.environ.get("PD_BOOTSTRAP_PORT", "8998")

_SHARED_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "16",
    "--mem-fraction-static",
    "0.85",
    "--page-size",
    "64",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "2",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "3",
    "--disable-radix-cache",
    "--quantization",
    "w8a8_int8",
    "--trust-remote-code",
    "--dist-timeout",
    "60000",
    "--watchdog-timeout",
    "60000",
    "--enable-cache-report",
    "--log-level",
    "info",
    "--enable-metrics",
    "--disable-custom-all-reduce",
    "--speculative-attention-mode",
    "decode",
    "--disable-shared-experts-fusion",
    "--disaggregation-transfer-backend",
    "mooncake",
    "--disaggregation-ib-device",
    IB_DEVICES,
    "--disable-piecewise-cuda-graph",
    "--served-model-name",
    SERVED_MODEL_NAME,
]

PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-bootstrap-port",
    PD_BOOTSTRAP_PORT,
    "--attention-backend",
    "nsa",
    "--nsa-decode-backend",
    "flashmla_kv",
    "--nsa-prefill-cp-mode",
    "round-robin-split",
    "--nsa-prefill-backend",
    "flashmla_sparse",
    "--attn-cp-size",
    "16",
    "--enable-nsa-prefill-context-parallel",
    *_SHARED_ARGS,
]

DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--disaggregation-bootstrap-port",
    PD_BOOTSTRAP_PORT,
    "--enable-dp-lm-head",
    "--cuda-graph-max-bs",
    "256",
    "--moe-dense-tp-size",
    "1",
    "--load-balance-method",
    "auto",
    "--dp-size",
    "16",
    "--prefill-round-robin-balance",
    "--kv-cache-dtype",
    "bf16",
    "--decode-attention-backend",
    "flashmla",
    "--decode-log-interval",
    "50",
    "--prefill-attention-backend",
    "fa3",
    "--max-running-requests",
    "256",
    "--enable-dp-attention",
    "--ep",
    "16",
    *_SHARED_ARGS,
]

DECODE_ENV = {
    **PD_BASE_ENV,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
    "SGL_DEEP_EP_RECV_HOOK": "False",
}


# ---------------------------------------------------------------------------
# Single-role mode: launch only P or only D, block to keep server alive.
# ---------------------------------------------------------------------------


class _SingleSideBase(CustomTestCase):
    """Launch a single P or D server, verify health, then block.

    The server stays alive until the user presses Enter or sends Ctrl+C,
    so the other side (running on a different machine) can connect.
    """

    role: str = ""  # "prefill" or "decode", set by subclass

    @classmethod
    def setUpClass(cls):
        port = PD_PREFILL_PORT if cls.role == "prefill" else PD_DECODE_PORT
        launch_url = f"http://{PD_HOST}:{port}"
        cls.health_url = f"http://127.0.0.1:{port}"
        args = PREFILL_ARGS if cls.role == "prefill" else DECODE_ARGS
        env = dict(PD_BASE_ENV) if cls.role == "prefill" else dict(DECODE_ENV)

        print(f"\n[{cls.role.upper()}] Launching server on {PD_HOST}:{port} ...")
        cls.process = popen_launch_pd_server(
            MODEL_PATH,
            launch_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=args,
            env=env,
        )
        wait_for_http_ready(
            url=cls.health_url + "/health",
            timeout=SERVER_LAUNCH_TIMEOUT,
            process=cls.process,
        )
        print(f"[{cls.role.upper()}] Server is ready at {launch_url}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid, wait_timeout=60)

    def test_server_running(self):
        """Verify server is healthy, then block to keep it alive.

        Press Enter or Ctrl+C to shut down the server.
        """
        import requests

        resp = requests.get(self.health_url + "/health", timeout=10)
        self.assertEqual(resp.status_code, 200)
        print(f"\n[{self.role.upper()}] Health check passed. Server is running.")
        print(
            f"[{self.role.upper()}] The other side can now connect to "
            f"http://<this_host>:{PD_PREFILL_PORT if self.role == 'prefill' else PD_DECODE_PORT}"
        )
        print(f"[{self.role.upper()}] Press Enter to stop the server ...")
        try:
            input()
        except (KeyboardInterrupt, EOFError):
            pass


@unittest.skipUnless(PD_ROLE == "prefill", "Only runs when PD_ROLE=prefill")
class TestPrefillSide(_SingleSideBase):
    """P side: NSA attention backend (dsa_seed_backend_enabled = True)."""

    role = "prefill"


@unittest.skipUnless(PD_ROLE == "decode", "Only runs when PD_ROLE=decode")
class TestDecodeSide(_SingleSideBase):
    """D side: flashmla/fa3 (dsa_seed_backend_enabled = False).

    Before the fix this side did not allocate output_dsa_topk_indices,
    causing the aux_data_ptrs list-length mismatch.
    """

    role = "decode"


# ---------------------------------------------------------------------------
# Both mode: launch P + D + LB locally, run GSM8K accuracy test.
# ---------------------------------------------------------------------------


@unittest.skipUnless(PD_ROLE == "both", "Only runs when PD_ROLE=both (default)")
class TestGlm52PpuPdDisaggDsaSeedWireSchema(GSM8KMixin, PDDisaggregationServerBase):
    """E2E: GLM-5.2 PD-disagg with NSA prefill + flashmla/fa3 decode.

    Launches P, D, and LB as local subprocesses, then runs GSM8K.
    Use this mode when a single machine has enough GPUs for both sides.
    """

    model = SERVED_MODEL_NAME
    gsm8k_score_threshold = 0.50
    gsm8k_num_examples = 200
    gsm8k_accept_length_thres = 1.0

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = []
        cls.rdma_devices = []
        cls.model = MODEL_PATH

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=PREFILL_ARGS,
            env=dict(PD_BASE_ENV),
        )

    @classmethod
    def start_decode(cls):
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=DECODE_ARGS,
            env=dict(DECODE_ENV),
        )


if __name__ == "__main__":
    unittest.main()
