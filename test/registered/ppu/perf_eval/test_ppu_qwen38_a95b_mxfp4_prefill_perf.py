"""PPU nightly serving performance entry for Qwen3.8-2.4T-A95B-MXFP4-FP8 (two nodes).

The btv1.5 prefill plan measures this checkpoint across two ZW-M890P boards at
tp 8 with pp 2, and this file is the two-node topology of that plan:

  configs/qwen3.8/2.4t-a95b-mxfp4-fp8-144g-prefill-2n.json

Both measurements the config names -- a 4k-token and a 64k-token prefill, ten
prompts each at concurrency one -- share one launch of the server, with the KV
cache flushed between them, because loading a 2.4T checkpoint costs far more
than the measurements do.

A file of its own rather than a config of the four-node entry because
`register_ppu_ci` registers a suite per file and the workflow dispatching a suite
has to know how many whole boards to gang-schedule; a two-node entry cannot share
a name with a four-node one.

Nothing here judges a number.  The suite is red only when a measurement produced
no numbers at all -- the server never came up, the benchmark crashed, every
request failed, a metric was missing, or the flush between measurements did not
take -- never for being slow; see the docstring of `perf_eval_kit`.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.perf_suite_kit import PerfSuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate covers a cold load of the whole checkpoint from the NAS plus the
# two prefill sweeps, and the first measured run replaces it.
register_ppu_ci(est_time=7200, suite="nightly-perf-16-ppu", nightly=True)


class TestPPUQwen38A95BMxfp4Perf(PerfSuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "qwen3.8" / "2.4t-a95b-mxfp4-fp8-144g-prefill-2n.json"
    )


if __name__ == "__main__":
    unittest.main()
