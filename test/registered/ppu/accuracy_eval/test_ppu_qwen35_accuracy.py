"""PPU nightly accuracy entries for Qwen3.5-397B-A17B (tp 8).

Eight reviewed configs share this file, two checkpoints across three datasets:

  configs/qwen3.5/397b-a17b-fp8-channelwise-144g-gsm8k.json        the default below
  configs/qwen3.5/397b-a17b-fp8-channelwise-144g-gsm8k-smoke.json  20 samples
  configs/qwen3.5/397b-a17b-fp8-channelwise-144g-ceval.json
  configs/qwen3.5/397b-a17b-fp8-channelwise-144g-ifeval.json
  configs/qwen3.5/397b-a17b-mxfp4-fp8-144g-gsm8k.json
  configs/qwen3.5/397b-a17b-mxfp4-fp8-144g-gsm8k-smoke.json        20 samples
  configs/qwen3.5/397b-a17b-mxfp4-fp8-144g-ceval.json
  configs/qwen3.5/397b-a17b-mxfp4-fp8-144g-ifeval.json

Served at tp 8, the Answer line's arrangement, rather than the tp 4 the source
cases and the perf line use, for the reason the MiniMax file's header gives: a job
holds a whole board either way and a full split is generation-bound.

This model is the reason the schema models the penalty keys.  Its source cases
pin the widest sampling configuration of the seven -- `temperature` 0.6 rather
than 1.0, with `top_k` 20, `min_p`, `presence_penalty` and `repetition_penalty` --
and all of it travels into `evaluation.generation`.  Two of those three penalties
are at their neutral values and could have been dropped without changing a token;
they are stated anyway, because the config is then readable against the source
case it names instead of readable only by someone who knows which defaults the
server happens to hold.

`mamba_scheduler_strategy: extra_buffer`, which the source cases set and the
schema models, is deliberately **not** set here: the Answer line starts these
checkpoints without it, and this line's job is to score a serving configuration
this board has already stood up.  The MXFP4 case's `page_size: 64` and
`disable_shared_experts_fusion` are left off for the same reason.

`evaluation.baseline` is null throughout until a green full run on this hardware
fills it in.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.accuracy_suite_kit import AccuracySuiteMixin

DATA_ROOT = Path(__file__).parent

# Drawn to the one measured accuracy run on this line (run 34376918707, GLM-5.2
# channelwise): 1160s of server startup and 278s per batch of twenty.  These
# checkpoints load faster -- the Answer line gives them 3600s of startup budget
# against GLM-5.2 channelwise's 5400s -- and `temperature` 0.6 with `top_k` 20
# generates shorter completions than GLM-5.2's 1.0, so the estimate below is
# generous rather than tight.  The first green run replaces it.
register_ppu_ci(est_time=12000, suite="nightly-accuracy-8-qwen35-ppu", nightly=True)


class TestPPUQwen35Accuracy(AccuracySuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "qwen3.5" / "397b-a17b-fp8-channelwise-144g-gsm8k.json"
    )


if __name__ == "__main__":
    unittest.main()
