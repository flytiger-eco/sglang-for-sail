"""PPU nightly Answer entry for Kimi-K2.6-W8A8-INT8 (ZW-M890P, two nodes).

  configs/kimi2.6/w8a8-int8-144g-2n.json

This is the format the internal btv1.5 case `kimi-k2.6-w8a8-int8_3001` actually
names, so it is the closest comparison against that case this repository has --
and it is the only Kimi config here that can pass `quantization` through
verbatim instead of leaving the checkpoint's own declaration to decide.

It is a two-node entry, and separate from the eight-device Kimi file, because it
does not fit one board group.  The checkpoint holds 968.3 GiB of INT8 tensors
against 1152 GiB of device memory on a node, which is 84 per cent of it: the
0.8 static fraction its source case names would reserve 921 GiB and fail during
the weight load, and even 0.9 would leave about 8 GiB per device for the KV pool
with nothing measured to say that is enough.  Across two nodes at tp 8 x pp 2 the
same 0.8 fraction reserves 1843 GiB, which the weights occupy just over half of.
The internal 144GiB answer plan schedules its cases as `1node8ppu` and does not
schedule this one at all, which is consistent with that arithmetic.

The rest follows the eight-device Kimi file: no unified attention backend but a
prefill and a decode backend (fa3 and flashmla), and `chat_template_kwargs
{"thinking": false}`, the switch this model's template and SGLang's `kimi_k2`
detector both spell that way.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.answer_suite_kit import AnswerSuiteMixin

DATA_ROOT = Path(__file__).parent

# Its own suite because a suite is what a workflow runs, and this one needs the
# two-node workflow rather than the single-board matrix.  The estimate is a cold
# 968.3 GiB load split over two nodes plus ten non-thinking answers, and the
# first measured run replaces it.
register_ppu_ci(est_time=7200, suite="nightly-answer-16-kimi26-ppu", nightly=True)


class TestPPUKimiK26W8A8Answer(AnswerSuiteMixin, unittest.TestCase):
    data_root = DATA_ROOT
    default_test_config_path = (
        DATA_ROOT / "configs" / "kimi2.6" / "w8a8-int8-144g-2n.json"
    )


if __name__ == "__main__":
    unittest.main()
