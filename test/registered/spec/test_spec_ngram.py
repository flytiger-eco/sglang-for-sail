import unittest

from sglang.test.ci.ci_register import register_cuda_ci, register_ppu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase

# Per-commit: Paged backend only.
# - FA3 base test archived to test/manual/spec/test_spec_ngram_fa3.py
# - Triton + Flashinfer moved to test_spec_ngram_extra.py
register_cuda_ci(est_time=254, stage="base-b", runner_config="1-gpu-large")
register_ppu_ci(est_time=254, suite="nightly-1-ppu", nightly=True)


class TestNgramSpeculativeDecodingPaged(NgramServerBase, GSM8KMixin):
    attention_backend = "flashinfer"
    extra_args = ["--page-size", "64"]


if __name__ == "__main__":
    unittest.main()
