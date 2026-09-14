"""PPU nightly Answer MVP for Qwen3.8-2.4T-A95B-FP8 across four boards.

Ported from the internal `qwen3.8-2.4t-a95b-fp8_3001` case, whose test plan
pins it to four ZW-M890P nodes at TP=32.  It runs the same ten public Answer
prompts and the same deterministic evaluator as the single-node suites; what is
new is the topology, and the checkpoint, at 2324.7 GiB over 213 shards, does not
fit on any smaller one.

Every node of the group executes this same file.  Rank 0 owns the HTTP API and
the verdict; the other three hold their eight devices in the group until rank 0
is done.  Which node a given process is comes from the environment the launcher
injects, not from the config, so nothing here distinguishes them -- see
`resolve_distributed_runtime`.

It is a third file rather than a third class because `register_ppu_ci` registers
a suite per file, and this case claims four whole boards where the others claim
one.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.answer_suite_kit import AnswerSuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate covers a cold load of the whole checkpoint from the NAS: the
# node's 2266 GiB of memory is smaller than the checkpoint tree, so the page
# cache warm the single-node entries rely on cannot help here, and the first
# measured run is what will replace this number.
register_ppu_ci(est_time=7200, suite="nightly-answer-32-ppu", nightly=True)


class TestPPUQwen38A95BAnswer(AnswerSuiteMixin, unittest.TestCase):
    data_root = DATA_ROOT
    # No ZW810E variant: 96 GiB boards cannot hold this checkpoint at any node
    # count this cluster can gang-schedule, so the 144 GiB config is the default
    # rather than an override the workflow selects.
    default_test_config_path = (
        DATA_ROOT / "configs" / "qwen3.8" / "2.4t-a95b-fp8-144g.json"
    )


if __name__ == "__main__":
    unittest.main()
