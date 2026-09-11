"""PPU nightly serving performance entries for GLM-5.2-FP8-Channelwise (tp 8).

Two reviewed configs share this file, and they are two configs rather than two
measurements of one because the btv1.5 prefill plan serves this checkpoint with
different server arguments at each length:

  configs/glm5.2/fp8-channelwise-144g-prefill-4k.json    the default below
  configs/glm5.2/fp8-channelwise-144g-prefill-64k.json

The 4k case runs dense attention -- `fa3` for prefill, `flashmla` for decode --
with the piecewise CUDA graph off.  The 64k case runs the sparse attention path
instead: `dsa` with `flashmla_sparse`/`flashmla_kv`, expert parallelism over the
same eight devices, prefill context parallelism at `attn_cp_size` 8, a 32768-token
chunked prefill and the overlap schedule off.  A single server cannot be both, so
each config carries exactly one measurement and the workflow runs them as two
jobs.

Only one of the two runs per job: the workflow names the config in
`SGLANG_PPU_PERF_TEST_CONFIG` and the class falls back to the default when it is
unset.  The MXFP4 checkpoint of the same model is a separate file, not a config
here, because it is served at tp 4 and a suite has to state how many devices it
claims; see `test_ppu_glm52_mxfp4_prefill_perf.py`.

Nothing here judges a number.  The suite is red only when a measurement produced
no numbers at all, never for being slow.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.perf_suite_kit import PerfSuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate covers a cold load plus one prefill sweep, and its server also
# waits out a 3600-second warmup; the first measured run replaces it.
register_ppu_ci(est_time=6600, suite="nightly-perf-8-glm52-ppu", nightly=True)


class TestPPUGlm52Perf(PerfSuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "glm5.2" / "fp8-channelwise-144g-prefill-4k.json"
    )


if __name__ == "__main__":
    unittest.main()
