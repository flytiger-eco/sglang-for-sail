"""PPU nightly serving performance entries for GLM-5.2-MXFP4-FP8 (tp 4).

The same two lengths and the same two shapes of server as the channelwise entry,
on a quarter of the board rather than all of it, which is what the btv1.5 prefill
plan asks for this checkpoint:

  configs/glm5.2/mxfp4-fp8-144g-prefill-4k.json    the default below
  configs/glm5.2/mxfp4-fp8-144g-prefill-64k.json

The 4k case runs dense attention with the piecewise CUDA graph off; the 64k case
runs `dsa` with expert parallelism, prefill context parallelism at `attn_cp_size`
4, a 32768-token chunked prefill and the overlap schedule off.  A single server
cannot be both, so each config carries exactly one measurement.

A file of its own rather than two more configs of `test_ppu_glm52_perf.py`
because these claim four devices and those claim eight, and the suite name is
what tells the workflow which to ask the cluster for.

Nothing here judges a number.  The suite is red only when a measurement produced
no numbers at all, never for being slow.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.perf_suite_kit import PerfSuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate covers a cold load plus one prefill sweep, and the first measured
# run replaces it.
register_ppu_ci(est_time=5400, suite="nightly-perf-4-glm52-ppu", nightly=True)


class TestPPUGlm52Mxfp4Perf(PerfSuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "glm5.2" / "mxfp4-fp8-144g-prefill-4k.json"
    )


if __name__ == "__main__":
    unittest.main()
