"""PPU nightly serving performance entries for Kimi-K2.6 (ZW-M890P, eight devices).

Two reviewed configs share this file, one per checkpoint the btv1.5 prefill plan
measures on a single board at tp 8:

  configs/kimi2.6/mxfp4-fp8-144g-prefill.json   the default below
  configs/kimi2.6/w4a8-int8-144g-prefill.json

Their server arguments are identical -- the plan's INT4 and MXFP4 cases differ in
nothing but the weights they read -- and each config carries two measurements, a
4k-token and a 64k-token prefill of ten prompts at concurrency one, which share
one launch with the KV cache flushed between them.

Only one of the two runs per job: the workflow names the config in
`SGLANG_PPU_PERF_TEST_CONFIG` and the class falls back to the default when it is
unset.  They are configs rather than two files because they claim the same eight
devices and differ only in the weights being read.

Nothing here judges a number.  The suite is red only when a measurement produced
no numbers at all, never for being slow.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.perf_suite_kit import PerfSuiteMixin

DATA_ROOT = Path(__file__).parent

# A suite of its own, not a shared one: a suite is what `--suite` selects, and
# every file in one runs in the same job off the same config variable, so two
# models under one name could not be given different weights.  The estimate
# covers a cold load of the larger of the two checkpoints plus both prefill
# sweeps, and the first measured run replaces it.
register_ppu_ci(est_time=5400, suite="nightly-perf-8-kimi26-ppu", nightly=True)


class TestPPUKimiK26Perf(PerfSuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "kimi2.6" / "mxfp4-fp8-144g-prefill.json"
    )


if __name__ == "__main__":
    unittest.main()
