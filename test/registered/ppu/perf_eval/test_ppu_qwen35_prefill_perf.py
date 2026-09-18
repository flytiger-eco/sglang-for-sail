"""PPU nightly serving performance entries for Qwen3.5-397B-A17B (four devices).

Two reviewed configs share this file, one per checkpoint the btv1.5 prefill plan
measures at tp 4:

  configs/qwen3.5/397b-a17b-mxfp4-fp8-144g-prefill.json        the default below
  configs/qwen3.5/397b-a17b-fp8-channelwise-144g-prefill.json

Their server arguments are identical -- the plan's two cases differ in nothing but
the weights they read -- and each carries a 4k-token and a 64k-token prefill of
ten prompts at concurrency one, sharing one launch with the KV cache flushed
between them.

`visible_devices` names four devices rather than eight because tp 4 is what the
plan asks for; the job still holds the whole board, and the board-side script
narrows `CUDA_VISIBLE_DEVICES` to the configured set.

Only one of the two runs per job: the workflow names the config in
`SGLANG_PPU_PERF_TEST_CONFIG` and the class falls back to the default when it is
unset.

Nothing here judges a number.  The suite is red only when a measurement produced
no numbers at all, never for being slow.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.perf_suite_kit import PerfSuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate covers a cold load plus both prefill sweeps, and the first
# measured run replaces it.
register_ppu_ci(est_time=4800, suite="nightly-perf-4-qwen35-ppu", nightly=True)


class TestPPUQwen35Perf(PerfSuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "qwen3.5" / "397b-a17b-mxfp4-fp8-144g-prefill.json"
    )


if __name__ == "__main__":
    unittest.main()
