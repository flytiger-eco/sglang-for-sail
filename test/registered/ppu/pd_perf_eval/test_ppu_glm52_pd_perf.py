"""GLM-5.2 served prefill/decode-disaggregated, 1p1d.

Two reviewed configs share this file, one per checkpoint the btv1.5
PD-Disaggregation plan measures at 1p1d:

  configs/glm5.2/fp8-channelwise-144g-pd-1p1d-4k-1500.json        the default below
  configs/glm5.2/mxfp4-fp8-144g-pd-1p1d-4k-1500.json

The fp8-channelwise case was the first ported of the plan, because its checkpoint
is the one the colocated ``nightly-perf-8-glm52-ppu`` suite already serves.  The
mxfp4-fp8 case is the second: its weights already exist on the NAS this suite
reads (recorded in ``model_weight_path.csv``), so it too waits on evidence rather
than on work -- see ``README.md`` next to this file for which of the plan's
remaining cases waits on what.

The two cases are not one server with the weights swapped.  The mxfp4-fp8 case
decodes with Multi-Token-Prediction: the source names ``speculative_algorithm
EAGLE`` on both roles with ``speculative_draft_model_path: ""``, which SGLang
resolves to EAGLE over the base checkpoint's own MTP layers, so no separate draft
checkpoint is needed and this file omits the empty draft key -- an absent draft
path and an empty one both fall back to the model path.  The fp8-channelwise case
names no speculative algorithm and decodes without it.

Two boards, one checkpoint, two servers.  Every node of the group runs this same
file and is told by the launcher which rank it is; rank 0 serves prefill, runs the
router the benchmark posts to, and owns the numbers, while rank 1 serves decode
and holds its half of the KV path until rank 0 is done.  What each role is launched
with is the reviewed config, not this file.

Only one of the two runs per job: the workflow names the config in
``SGLANG_PPU_PD_PERF_TEST_CONFIG`` and the class falls back to the default when it
is unset.

Nothing here judges a number.  The suite is red only when a measurement produced
no numbers, never for being slow.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.pd_perf_suite_kit import PDPerfSuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate covers two cold loads that proceed in parallel, the endpoint
# exchange between the boards, and one measurement of 80 requests at 4096 in /
# 1500 out; the first measured run replaces it.
register_ppu_ci(est_time=7200, suite="nightly-pd-perf-16-glm52-ppu", nightly=True)


class TestPPUGlm52PdPerf(PDPerfSuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "glm5.2" / "fp8-channelwise-144g-pd-1p1d-4k-1500.json"
    )


if __name__ == "__main__":
    unittest.main()
