"""Qwen3.5-397B-A17B served prefill/decode-disaggregated with native MTP, 1p1d.

Two reviewed configs share this file, one per checkpoint the btv1.5
PD-Disaggregation plan measures at 1p1d:

  configs/qwen3.5/397b-a17b-mxfp4-fp8-144g-pd-1p1d-4k-1500.json        the default below
  configs/qwen3.5/397b-a17b-fp8-channelwise-144g-pd-1p1d-4k-1500.json

Both are portable because their weights already exist on the NAS this suite reads
and because their decode is Multi-Token-Prediction rather than an external draft:
the case names ``NEXTN``, which SGLang resolves to EAGLE over the base
checkpoint's own MTP layers, so no separate draft checkpoint is needed.  The
source cases state ``speculative_draft_model_path: ""``; this file omits the key,
which is the same thing to the server -- an empty or absent draft path both fall
back to the model path, which is where the Qwen3.5 MTP layers are read from.  The
two cases differ only in their weights and, on fp8-channelwise, a triton draft
attention backend the mxfp4 case does not name.

Two boards, one checkpoint, two servers.  Every node of the group runs this same
file and is told by the launcher which rank it is; rank 0 serves prefill, runs the
router the benchmark posts to, and owns the numbers, while rank 1 serves decode
and holds its half of the KV path until rank 0 is done.  What each role is launched
with is the reviewed config, not this file.

Only one of the two runs per job: the workflow names the config in
``SGLANG_PPU_PD_PERF_TEST_CONFIG`` and the class falls back to the default when it
is unset.

Nothing here judges a number.  The suite is red only when a measurement produced
no numbers at all, never for being slow.
"""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.kits.pd_perf_suite_kit import PDPerfSuiteMixin

DATA_ROOT = Path(__file__).parent

# The estimate covers two cold loads that proceed in parallel, the endpoint
# exchange between the boards, and one measurement of 80 requests at 4096 in /
# 1500 out; the first measured run replaces it.
register_ppu_ci(est_time=7200, suite="nightly-pd-perf-16-qwen35-ppu", nightly=True)


class TestPPUQwen35PdPerf(PDPerfSuiteMixin, unittest.TestCase):
    default_test_config_path = (
        DATA_ROOT
        / "configs"
        / "qwen3.5"
        / "397b-a17b-mxfp4-fp8-144g-pd-1p1d-4k-1500.json"
    )


if __name__ == "__main__":
    unittest.main()
