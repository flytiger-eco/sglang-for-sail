import unittest

from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.server_fixtures.standalone_fixture import StandaloneServerBase
from sglang.test.test_utils import CustomTestCase

# V2 standalone speculative decoding tests (FA3, Triton, FlashInfer backends).
# Non-V2 backends moved to test_spec_standalone_extra.py.
register_cuda_ci(est_time=406, stage="base-b", runner_config="1-gpu-large")
register_ppu_ci(est_time=600, suite="nightly-1-ppu", nightly=True, disabled="flashinfer 0.6.7.post2 < required 0.6.11.post1")


class TestStandaloneV2SpeculativeDecodingBase(StandaloneServerBase, CustomTestCase):
    attention_backend = "fa3"


class TestStandaloneV2SpeculativeDecodingTriton(StandaloneServerBase, CustomTestCase):
    attention_backend = "triton"


class TestStandaloneV2SpeculativeDecodingFlashinfer(
    StandaloneServerBase, CustomTestCase
):
    attention_backend = "flashinfer"


if __name__ == "__main__":
    unittest.main()
