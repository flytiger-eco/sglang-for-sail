"""PPU nightly accuracy entries for MiniMax-M2.7 (tp 8).

Eight reviewed configs share this file, two checkpoints across three datasets:

  configs/minimax2.7/fp8-channelwise-144g-gsm8k.json        the default below
  configs/minimax2.7/fp8-channelwise-144g-gsm8k-smoke.json  the same run, 20 samples
  configs/minimax2.7/fp8-channelwise-144g-ceval.json
  configs/minimax2.7/fp8-channelwise-144g-ifeval.json
  configs/minimax2.7/mxfp4-fp8-144g-gsm8k.json
  configs/minimax2.7/mxfp4-fp8-144g-gsm8k-smoke.json
  configs/minimax2.7/mxfp4-fp8-144g-ceval.json
  configs/minimax2.7/mxfp4-fp8-144g-ifeval.json

Served at tp 8, which is the Answer line's arrangement for these checkpoints and
not the source cases' -- the internal plan runs this model at tp 4 channelwise and
tp 2 for MXFP4, and the perf line follows it at tp 2.  Eight because a job on this
line holds a whole board for as long as the checkpoint takes to load either way,
and a full split of 1346 samples at 32768 tokens is generation-bound: tp 2 would
leave six devices idle and turn hours into most of a night.  The suite name states
the devices its configs declare, so this is the 8 suite.

`reasoning_parser: minimax`, from the Answer line, rather than the source cases'
`minimax-append-think`.  The parser decides what part of a completion is scored,
so this is worth knowing when a number here is compared to an internal one.

`top_k: 40` travels from the source cases, which is why the schema models the
truncation keys at all: it narrows the distribution these configs sample from, and
a score is a score of a distribution.

The rest of the serving configuration is the Answer line's `fa3` arrangement,
already started on this board for both checkpoints.  The source cases add an
EAGLE/EAGLE3 draft model, which has not been; that is a config to add once it
starts.  `evaluation.baseline` is null throughout until a green full run on this
hardware fills it in.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.accuracy_suite_kit import AccuracySuiteMixin

DATA_ROOT = Path(__file__).parent

# Drawn to the one measured accuracy run on this line (run 34376918707, GLM-5.2
# channelwise): 1160s of server startup and 278s per batch of twenty.  These
# checkpoints load faster -- the Answer line gives them 3600s of startup budget
# against GLM-5.2 channelwise's 5400s -- so the estimate is GLM-5.2's, which is
# generous here rather than tight.  The first green run replaces it.
register_ppu_ci(est_time=12000, suite="nightly-accuracy-8-minimax27-ppu", nightly=True)


class TestPPUMiniMaxM27Accuracy(AccuracySuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "minimax2.7" / "fp8-channelwise-144g-gsm8k.json"
    )


if __name__ == "__main__":
    unittest.main()
