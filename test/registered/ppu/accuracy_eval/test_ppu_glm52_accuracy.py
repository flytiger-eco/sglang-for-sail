"""PPU nightly accuracy entries for GLM-5.2 (tp 8).

Eight reviewed configs share this file, two checkpoints across three datasets:

  configs/glm5.2/fp8-channelwise-144g-gsm8k.json        the default below
  configs/glm5.2/fp8-channelwise-144g-gsm8k-smoke.json  the same run, 20 samples
  configs/glm5.2/fp8-channelwise-144g-ceval.json
  configs/glm5.2/fp8-channelwise-144g-ifeval.json
  configs/glm5.2/mxfp4-fp8-144g-gsm8k.json
  configs/glm5.2/mxfp4-fp8-144g-gsm8k-smoke.json
  configs/glm5.2/mxfp4-fp8-144g-ceval.json
  configs/glm5.2/mxfp4-fp8-144g-ifeval.json

They are configs rather than eight files for the reason the Answer line's header
gives: they claim the same eight devices and differ only in the weights being
read and the split being scored.  Only one runs per job -- the workflow names it
in `SGLANG_PPU_ACCURACY_TEST_CONFIG` and the class falls back to the default when
it is unset.

A smoke config per checkpoint, GSM8K only.  The first thing worth knowing about a
checkpoint on this line is whether the tool, the staged dataset and the server
talk to each other at all, which twenty samples establish in minutes rather than
hours; it is per checkpoint and not per dataset because what it exercises is the
weight load and the report round trip, and the cheapest split proves both.  It is
a config and not a flag so that what was evaluated is recorded in the report like
everything else: a smoke report carries `limit: 20` and its own `test_id`, and
cannot be mistaken for a full run.

The serving configuration is this repository's, not the internal case's, and the
difference is deliberate: the btv1.5 cases serve GLM-5.2 with dense attention and
an EAGLE draft attached, and neither has been brought up on this board, while the
sparse-attention arguments below are the ones the Answer and perf lines already
start these checkpoints with.  A score is only comparable to a baseline measured
on the same arguments, which is why `evaluation.baseline` is null throughout and
the first green run of each is what fills it in.  The MTP variant is a config to
add once it starts, not a change to these -- see the suite README.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.accuracy_suite_kit import AccuracySuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate is the slowest config's, built out of the smoke run that measured
# its parts (run 34376918707): 1160s to bring the channelwise server up, of which
# 1050s was CUDA graph capture, then 278s for one batch of the split.  The full
# GSM8K split is 33 such batches and C-Eval's 1346 samples are 34, which lands a
# little over eleven thousand seconds; the number below leaves room for batches of
# forty running longer than the batch of twenty that was timed.  The smoke configs
# finish far inside it -- 1454s, measured.
register_ppu_ci(est_time=12000, suite="nightly-accuracy-8-glm52-ppu", nightly=True)


class TestPPUGlm52Accuracy(AccuracySuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "glm5.2" / "fp8-channelwise-144g-gsm8k.json"
    )


if __name__ == "__main__":
    unittest.main()
