import unittest

from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.ci.skip_utils import skip_if_model_missing
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

MODEL = "openai/gpt-oss-20b"

register_cuda_ci(est_time=151, stage="base-b", runner_config="1-gpu-large")
register_ppu_ci(est_time=151, suite="nightly-1-ppu", nightly=True)


@skip_if_model_missing("openai/gpt-oss-20b")
class TestSWARadixCacheKL(KLDivergenceMixin, DefaultServerBase):
    model = MODEL
    kl_div_thres = 0.02  # it was 0.002
    kl_div_decode_max_new_tokens = 2048
    other_args = [
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.70",
        "--disable-piecewise-cuda-graph",
    ]


if __name__ == "__main__":
    unittest.main()
