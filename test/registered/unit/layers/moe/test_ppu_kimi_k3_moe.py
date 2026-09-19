"""CPU checks for PPU Kimi-K3 MoE compatibility paths."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.moe.moe_runner import acext as acext_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPpuKimiK3Moe(CustomTestCase):
    def test_acext_situ_falls_back_before_importing_acext(self):
        sentinel = object()
        runner_config = SimpleNamespace(activation="situ")
        with (
            patch.object(acext_module, "is_sail", return_value=True),
            patch.object(
                acext_module,
                "fused_experts_none_to_triton",
                return_value=sentinel,
            ) as fallback,
        ):
            result = acext_module.fused_experts_none_to_acext(
                "dispatch", "quant", runner_config
            )

        self.assertIs(result, sentinel)
        fallback.assert_called_once_with("dispatch", "quant", runner_config)


if __name__ == "__main__":
    unittest.main()
