"""Regression tests for native GLM model registration."""

import unittest

from sglang.srt.models.registry import ModelRegistry
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGlmModelRegistry(unittest.TestCase):
    def test_resolves_glm_dsa_to_native_model(self):
        model_cls, architecture = ModelRegistry.resolve_model_cls(
            "GlmMoeDsaForCausalLM"
        )

        self.assertEqual(architecture, "GlmMoeDsaForCausalLM")
        self.assertEqual(model_cls.__module__, "sglang.srt.models.glm4_moe")

    def test_resolves_glm_dsa_nextn_once(self):
        model_cls, architecture = ModelRegistry.resolve_model_cls(
            "GlmMoeDsaForCausalLMNextN"
        )

        self.assertEqual(architecture, "GlmMoeDsaForCausalLMNextN")
        self.assertEqual(model_cls.__module__, "sglang.srt.models.glm4_moe_nextn")


if __name__ == "__main__":
    unittest.main()
