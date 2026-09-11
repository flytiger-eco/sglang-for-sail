"""PPU nightly Answer entry for Qwen3.8-2.4T-A95B-MXFP4-FP8 (two ZW-M890P nodes).

The same model and the same ten prompts as `test_ppu_qwen38_a95b_answer.py`, but
a different checkpoint and a different topology: the MXFP4 weights the cluster
holds are small enough for two boards, so this entry asks for two rather than
four, at tp 8 with pp 2.

It is a fourth file rather than a config of the four-node one because
`register_ppu_ci` registers a suite per file and the workflow that dispatches a
suite has to know how many whole boards to gang-schedule; a two-node entry cannot
share a name with a four-node one.

`quantization` is left unset so the checkpoint's own `quantization_config` -- mxfp4
for the experts, per-channel FP8 for the dense layers -- decides, and
`reasoning_parser` is `qwen3-thinking` because this template refuses to disable
thinking and grades `reasoning_effort` instead; the config asks for `low`.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.answer_suite_kit import AnswerSuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate covers a cold load of the whole checkpoint from the NAS plus ten
# thinking answers at `reasoning_effort` low, and the first measured run replaces
# it.
register_ppu_ci(est_time=7200, suite="nightly-answer-16-ppu", nightly=True)


class TestPPUQwen38A95BMxfp4Answer(AnswerSuiteMixin, unittest.TestCase):
    data_root = DATA_ROOT
    default_test_config_path = (
        DATA_ROOT / "configs" / "qwen3.8" / "2.4t-a95b-mxfp4-fp8-144g.json"
    )


if __name__ == "__main__":
    unittest.main()
