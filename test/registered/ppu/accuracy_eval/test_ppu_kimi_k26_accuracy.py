"""PPU nightly accuracy entries for Kimi-K2.6-MXFP4-FP8 (tp 8).

Four reviewed configs share this file:

  configs/kimi2.6/mxfp4-fp8-144g-gsm8k.json        the default below
  configs/kimi2.6/mxfp4-fp8-144g-gsm8k-smoke.json  the same run, 20 samples
  configs/kimi2.6/mxfp4-fp8-144g-ceval.json
  configs/kimi2.6/mxfp4-fp8-144g-ifeval.json

A suite of its own rather than a shared one: every file in a suite runs in the
same job off the same `SGLANG_PPU_ACCURACY_TEST_CONFIG`, so two checkpoints under
one name could not be given different weights.  Only one config runs per job, the
workflow naming which.

The one checkpoint of this model on the accuracy line, although the Answer line
carries three: the internal plan states a single Kimi-K2.6 quantisation for
evalscope, and a config here without a source case would be a measurement nobody
asked for.

Concurrency is 32 rather than the 40 the other suites use, because that is what
the source cases state for this model -- `eval_batch_size` 32 where GLM-5.2 and
the rest say 128.  It is the largest checkpoint of the seven, and the number the
plan chose for it is the one worth honouring; `eval_batch_size` then matches the
server, as it must in `openai_api` mode where it is the client's concurrency.

The serving configuration is the Answer line's, which is the split
`prefill_attention_backend: fa3` / `decode_attention_backend: flashmla` pair this
board already starts this checkpoint with.  The source case adds an EAGLE3 draft
model on top of it; that has not been brought up here, and a score measured with
a draft attached is not comparable to one measured without, so it is a config to
add once it starts rather than a change to these.  `evaluation.baseline` is null
throughout until a green full run on this hardware fills it in.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.accuracy_suite_kit import AccuracySuiteMixin

DATA_ROOT = Path(__file__).parent

# Drawn to the one measured accuracy run on this line (run 34376918707, GLM-5.2
# channelwise): 1160s of server startup and 278s per batch of twenty.  This
# checkpoint is larger and its batches are 32 rather than 40, so the estimate is
# GLM-5.2's with room for a slower load -- the Answer line gives this checkpoint a
# 4200s startup budget against GLM-5.2 channelwise's 5400s, so the load is the
# part that is unlikely to be worse.  The first green run replaces it.
register_ppu_ci(est_time=12000, suite="nightly-accuracy-8-kimi26-ppu", nightly=True)


class TestPPUKimiK26Accuracy(AccuracySuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "kimi2.6" / "mxfp4-fp8-144g-gsm8k.json"
    )


if __name__ == "__main__":
    unittest.main()
