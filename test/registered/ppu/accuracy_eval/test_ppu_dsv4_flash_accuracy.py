"""PPU nightly accuracy entries for DeepSeek-V4-Flash W8A8 (tp 8).

Two reviewed configs share this file, one checkpoint across two splits of one
dataset:

  configs/dsv4-flash/w8a8-int8-144g-gsm8k.json        the default below
  configs/dsv4-flash/w8a8-int8-144g-gsm8k-smoke.json  the same run, 20 samples

They are configs rather than two files for the reason the Answer line's header
gives: they claim the same eight devices and differ only in the split being
scored.  Only one runs per job -- the workflow names it in
`SGLANG_PPU_ACCURACY_TEST_CONFIG` and the class falls back to the default when
it is unset.

A smoke config per checkpoint, GSM8K only.  The first thing worth knowing about
a checkpoint on this line is whether the tool, the staged dataset and the server
talk to each other at all, which twenty samples establish in minutes rather than
hours; it is per checkpoint and not per dataset because what it exercises is the
weight load and the report round trip, and the cheapest split proves both.  It
is a config and not a flag so that what was evaluated is recorded in the report
like everything else: a smoke report carries `limit: 20` and its own `test_id`,
and cannot be mistaken for a full run.

The serving configuration is this repository's, not the internal case's, and the
difference is deliberate, in three places:

  * The btv1.1 case (deepseek-v4-flash-a8w8_3003.json) serves with EAGLE
    speculation and the DP/deepep expert layout (`enable_dp_lm_head`,
    `enable_dp_attention`, `dp_size 8`, `moe_a2a_backend deepep`).  The reviewed
    schema models neither a draft checkpoint nor the DP/deepep family, so this
    config serves plain tp 8 with `attention_backend dsv4` -- the backend the
    AMD DeepSeek-V4-Flash suite and the PPU attention unittest both exercise --
    and no speculation.  A score is only comparable to a baseline measured on
    the same arguments, which is why `baseline` is null throughout and the first
    green run of each config is what fills it in.

  * The btv1.1 case asks for thinking through the request body
    (`chat_template_kwargs.thinking`, `reasoning_effort: high`), which the
    reviewed generation schema does not carry.  The workflow jobs export
    `SGLANG_DEFAULT_THINKING=1` and `SGLANG_DSV4_REASONING_EFFORT=high`
    instead, so every request thinks by server default.

  * The btv1.1 case exports `SGLANG_OPT_USE_TOPK_V2=0` and
    `SGLANG_DSV4_FP4_EXPERTS=0` around the server; both travel in the workflow
    jobs' `extra_env` because the reviewed server-env schema accepts only the
    variables this tree documents.  `SGLANG_SAIL_DSV4_USE_FLASH_MLA_SPARSE_FWD`
    is left off entirely: it is a variable of the internal platform's sglang
    build, and nothing in this tree reads it.

Both GSM8K configs follow the internal btv1.1 EvalScope client command for this
checkpoint rather than the dataset contract's few-shot default: a thinking model
scores best zero-shot with `filters: {remove_until: "</think>"}`, which the
internal command states and the schema's `filters` key carries.  The generation
budget is the source case's `max_tokens: 128000` -- the internal command's
384000 is a ceiling for interactive serving, not for a scored split -- except in
the smoke, whose 32768 keeps twenty samples to minutes.

The estimate is a placeholder in the spirit of the line's first entries: nothing
of this checkpoint has been measured on this board yet, so it borrows GLM-5.2's
reviewed number until the first run replaces it with one.  The full GSM8K budget
is likewise the line's standard full-entry one; the first green run is what says
whether a 128000-token thinking budget on 1319 samples fits inside it.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.accuracy_suite_kit import AccuracySuiteMixin

DATA_ROOT = Path(__file__).parent

register_ppu_ci(est_time=12000, suite="nightly-accuracy-8-dsv4flash-ppu", nightly=True)


class TestPPUDsv4FlashAccuracy(AccuracySuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT / "configs" / "dsv4-flash" / "w8a8-int8-144g-gsm8k.json"
    )


if __name__ == "__main__":
    unittest.main()
