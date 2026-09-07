"""PPU nightly Answer entries for GLM-5.2 (ZW-M890P, eight devices).

Three reviewed configs share this file, one per checkpoint the cluster holds for
this model:

  configs/glm5.2/w8a8-int8-144g.json     the default below
  configs/glm5.2/mxfp4-fp8-144g.json
  configs/glm5.2/fp8-channelwise-144g.json

Their server arguments come from the internal btv1.5 answer_144g plan:
`glm-5.1-w8a8-int8_3001` for the INT8 entry (the plan has no GLM-5.2 INT8 case
and the GLM-5.x INT8 cases are argument-identical), `glm-5.2-fp8_3001` for the
MXFP4 entry, and `glm-5.2-fp8_channel_cp_3001` for the channelwise one, which is
the only entry here that turns on prefill context parallelism.  The deviations
from those cases are listed in README.md; the substantive one is that
`quantization` is left unset so each checkpoint's own `quantization_config`
decides, which is what lets one set of arguments serve three different weight
formats.

Only one of the three runs per job: the workflow names the config in
`SGLANG_PPU_ANSWER_TEST_CONFIG` and the class falls back to the default when it
is unset.  They are configs rather than three files because they claim the same
eight devices and differ only in the weights being read.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.answer_suite_kit import AnswerSuiteMixin

DATA_ROOT = Path(__file__).parent

# A suite of its own, not `nightly-answer-8-ppu`: a suite is what `--suite`
# selects, and every file in one runs in the same job off the same config
# variable, so two models under one name could not be given different weights.
# The estimate covers the slowest of the three, the channelwise entry, whose
# server also waits out a 3600-second warmup; the first measured run replaces it.
register_ppu_ci(est_time=7200, suite="nightly-answer-8-glm52-ppu", nightly=True)


class TestPPUGlm52Answer(AnswerSuiteMixin, unittest.TestCase):
    data_root = DATA_ROOT
    # No ZW810E variant: the plan schedules GLM-5.2 on 144 GiB boards only.
    default_test_config_path = DATA_ROOT / "configs" / "glm5.2" / "w8a8-int8-144g.json"


if __name__ == "__main__":
    unittest.main()
