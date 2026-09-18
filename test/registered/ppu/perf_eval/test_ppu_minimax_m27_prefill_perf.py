"""PPU nightly serving performance entries for MiniMax-M2.7 (ZW-M890P, two devices).

Two reviewed configs share this file, one per checkpoint the btv1.5 prefill plan
measures at tp 2:

  configs/minimax2.7/mxfp4-fp8-144g-prefill.json        the default below
  configs/minimax2.7/fp8-channelwise-144g-prefill.json

Each carries a 4k-token and a 64k-token prefill of ten prompts at concurrency
one, sharing one launch with the KV cache flushed between them.  The two differ
in one argument: the MXFP4 case turns off the custom all-reduce and the
channelwise one does not, which is how the plan states them and is therefore
carried across as stated.

`visible_devices` names two devices rather than eight because tp 2 is what the
plan asks for and validation ties `tp_size` times `pp_size` to the declared
device count; the job still holds the whole board, and the board-side script
narrows `CUDA_VISIBLE_DEVICES` to the configured set so the preflight means
something.

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
register_ppu_ci(est_time=4800, suite="nightly-perf-2-minimax27-ppu", nightly=True)


class TestPPUMinimaxM27Perf(PerfSuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "minimax2.7" / "mxfp4-fp8-144g-prefill.json"
    )


if __name__ == "__main__":
    unittest.main()
