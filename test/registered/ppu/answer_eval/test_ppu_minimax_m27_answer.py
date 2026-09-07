"""PPU nightly Answer entries for MiniMax-M2.7 (ZW-M890P, eight devices).

Three reviewed configs share this file:

  configs/minimax2.7/fp8-channelwise-144g.json   the default below
  configs/minimax2.7/mxfp4-fp8-144g.json
  configs/minimax2.7/w8a8-int8-144g.json

The server arguments come from the internal btv1.5 answer_144g case
`minimax-m3-bf16_3001`, the only MiniMax entry the sglang plan carries: tp 8, the
fa3 backend, `mem_fraction_static` 0.8 and a 600-second watchdog.  The
checkpoints here are quantized rather than BF16, so `quantization` is left unset
and each one's own `quantization_config` decides.  All three fit one board group
comfortably: the largest holds 214.6 GiB against a 921 GiB static pool.

This model's chat template reads no reasoning switch at all -- its generation
prompt opens `<think>` unconditionally -- so the reasoning pass cannot be turned
off and `chat_template_kwargs` is empty rather than absent, stating that shape
explicitly.  Both configs therefore budget a thinking answer: 16384 output tokens
and a 900-second per-request timeout, against 2048 and 300 for the entries that
can be asked not to think.  That budget is not measured; it is the assumption
these two entries carry into their first run.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.answer_suite_kit import AnswerSuiteMixin

DATA_ROOT = Path(__file__).parent

# Its own suite for the same reason as the GLM file: one config variable per job.
# The estimate is dominated by ten forced-thinking answers rather than by the
# load, and the first measured run replaces it.
register_ppu_ci(est_time=7200, suite="nightly-answer-8-minimax27-ppu", nightly=True)


class TestPPUMinimaxM27Answer(AnswerSuiteMixin, unittest.TestCase):
    data_root = DATA_ROOT
    default_test_config_path = (
        DATA_ROOT / "configs" / "minimax2.7" / "fp8-channelwise-144g.json"
    )


if __name__ == "__main__":
    unittest.main()
