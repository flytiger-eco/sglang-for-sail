"""PPU nightly serving performance entry for Qwen3.8-2.4T-A95B-FP8 (four nodes).

The FP8 weights of this checkpoint do not fit on two boards, so the btv1.5
prefill plan serves them across four ZW-M890P boards at tp 8 with pp 4:

  configs/qwen3.8/2.4t-a95b-fp8-144g-prefill-4n.json

Every node runs this same registered file; the launcher tells each which rank it
is, rank 0 owns the HTTP API and the numbers, and the other three hold their
devices in the group until rank 0 publishes its completion.  See
`PerfSuiteMixin._hold_until_rank_zero_completes`.

The two measurements -- 4k and 64k prefill, ten prompts each at concurrency one
-- share one launch of the server, with the KV cache flushed between them.
Nothing here judges a number: the suite is red only when a measurement produced
no numbers at all, never for being slow.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.perf_suite_kit import PerfSuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate covers a cold load of the whole checkpoint across four nodes plus
# the two prefill sweeps, and the first measured run replaces it.
register_ppu_ci(est_time=7200, suite="nightly-perf-32-ppu", nightly=True)


class TestPPUQwen38A95BPerf(PerfSuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "qwen3.8" / "2.4t-a95b-fp8-144g-prefill-4n.json"
    )


if __name__ == "__main__":
    unittest.main()
