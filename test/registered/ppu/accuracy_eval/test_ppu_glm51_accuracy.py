"""PPU nightly accuracy entries for GLM-5.1 (tp 8).

Four reviewed configs share this file, two checkpoints across two splits of one
dataset:

  configs/glm5.1/fp8-channelwise-144g-gsm8k.json        the default below
  configs/glm5.1/fp8-channelwise-144g-gsm8k-smoke.json  the same run, 20 samples
  configs/glm5.1/mxfp4-fp8-144g-gsm8k.json
  configs/glm5.1/mxfp4-fp8-144g-gsm8k-smoke.json

They are configs rather than four files for the reason the Answer line's header
gives: they claim the same eight devices and differ only in the weights being
read and the split being scored.  Only one runs per job -- the workflow names it
in `SGLANG_PPU_ACCURACY_TEST_CONFIG` and the class falls back to the default when
it is unset.

A smoke config per checkpoint, GSM8K only, for the reason every other file on
this line states: twenty samples prove the weight load and the report round
trip in minutes rather than hours, and a smoke report carries `limit: 20` and
its own `test_id`, so it cannot be mistaken for a full run.

The serving configuration is this repository's, not the internal case's, and
the difference is deliberate: the btv1.5 cases serve GLM-5.1 with dense
attention (`prefill fa3` / `decode flashmla`, or the renamed sparse family on
the mxfp4 case) and an EAGLE draft attached, and neither has been brought up on
this board, while the sparse-attention arguments below are the ones this line
already starts GLM-5.2 with -- the fp8-channelwise checkpoint with context
parallelism over all eight devices, the mxfp4 checkpoint with the plain sparse
pair.  A score is only comparable to a baseline measured on the same arguments,
which is why `evaluation.baseline` is null throughout and the first green run
of each is what fills it in.

One further deviation from the internal cases, inherited from the GLM-5.2
entries: the mxfp4 case serves at tp 4 and this config serves it at tp 8.  The
line's own convention is eight declared devices throughout -- a full split is
generation-bound and the job holds the whole board either way -- and every
config of this suite says so in its `hardware` block.

Both checkpoints score GSM8K four-shot at the dataset contract's default, with
the generation keys the cases state (`max_tokens 32768`, `temperature 1.0`,
`top_p 0.95`) and no thinking switch, exactly as the cases' commands do.

The estimate is a placeholder in the spirit of the line's first entries: nothing
of these checkpoints has been measured on this board yet, so it borrows
GLM-5.2's reviewed number until the first run replaces it with one.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.accuracy_suite_kit import AccuracySuiteMixin

DATA_ROOT = Path(__file__).parent

register_ppu_ci(est_time=12000, suite="nightly-accuracy-8-glm51-ppu", nightly=True)


class TestPPUGlm51Accuracy(AccuracySuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "glm5.1" / "fp8-channelwise-144g-gsm8k.json"
    )


if __name__ == "__main__":
    unittest.main()
