"""PPU nightly accuracy entries for GLM-5.2-FP8-Channelwise (tp 8).

Two reviewed configs share this file:

  configs/glm5.2/fp8-channelwise-144g-gsm8k.json        the default below
  configs/glm5.2/fp8-channelwise-144g-gsm8k-smoke.json  the same run, 20 samples

The smoke config exists because the full split is 1319 samples against a
32768-token generation budget, and the first thing worth knowing about a new
entry is whether the tool, the staged dataset and the server talk to each other
at all -- which twenty samples establish in minutes rather than hours.  It is a
config and not a flag so that what was evaluated is recorded in the report like
everything else; a smoke report carries `limit: 20` and its own `test_id`, and
cannot be mistaken for a full run.

Only one of the two runs per job: the workflow names the config in
`SGLANG_PPU_ACCURACY_TEST_CONFIG` and the class falls back to the default when
it is unset.

The serving configuration is this repository's, not the internal case's, and the
difference is deliberate: the btv1.5 case serves GLM-5.2 with dense attention
and an EAGLE draft attached, and neither has been brought up on this board,
while the sparse-attention arguments below are the ones the Answer and perf lines
already start this checkpoint with.  A score is only comparable to a baseline
measured on the same arguments, which is why `evaluation.baseline` is null here
and the first green run is what fills it in.  The MTP variant is a config to add
once it starts, not a change to this one -- see the suite README.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.accuracy_suite_kit import AccuracySuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate is the full config's, built out of the smoke run that measured its
# parts (run 34376918707): 1160s to bring the server up, of which 1050s was CUDA
# graph capture, then 278s for one batch of the split.  The full split is 33 such
# batches, which lands a little over eleven thousand seconds; the number below
# leaves room for batches of forty running longer than the batch of twenty that
# was timed.  The smoke config finishes far inside it -- 1454s, measured.
register_ppu_ci(est_time=12000, suite="nightly-accuracy-8-glm52-ppu", nightly=True)


class TestPPUGlm52AccuracyGSM8K(AccuracySuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "glm5.2" / "fp8-channelwise-144g-gsm8k.json"
    )


if __name__ == "__main__":
    unittest.main()
