"""GLM-5.2 FP8-channelwise served prefill/decode-disaggregated, 1p1d.

The first ported case of the btv1.5 PD-Disaggregation plan, and the only one in
this batch, because it is the only 1p1d case whose weights this repository can
show already exist: the checkpoint below is the one the colocated
``nightly-perf-8-glm52-ppu`` suite already serves.  The rest of that plan waits on
evidence rather than on work -- see ``README.md`` next to this file for which case
waits on what.

Two boards, one checkpoint, two servers.  Every node of the group runs this same
file and is told by the launcher which rank it is; rank 0 serves prefill, runs the
router the benchmark posts to, and owns the numbers, while rank 1 serves decode
and holds its half of the KV path until rank 0 is done.  What each role is launched
with is the reviewed config, not this file.

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
