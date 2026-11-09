"""W4AFP8 checkpoint exclusions must survive runtime names and fused linears."""

import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization import w4afp8
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestW4AFP8Config(CustomTestCase):
    def setUp(self):
        self.linear = Mock(spec=LinearBase)
        self.mapping = {"in_proj_ba": ["in_proj_b", "in_proj_a"]}

    def test_exclusion_aliases_and_checkpoint_names(self):
        for key in ["ignored_layers", "modules_to_not_convert", "ignore"]:
            with self.subTest(key=key):
                raw = {
                    "quant_method": "w4afp8",
                    key: [
                        "model.language_model.layers.0.self_attn.in_proj_b",
                        "model.language_model.layers.0.self_attn.in_proj_a",
                    ],
                    "packed_modules_mapping": self.mapping,
                }
                original = deepcopy(raw)
                config = w4afp8.W4AFp8Config.from_config(raw)
                for prefix in [
                    "model.language_model.layers.0.self_attn.in_proj_ba",
                    "model.layers.0.self_attn.in_proj_ba",
                    "model.layers.0.in_proj_ba",
                ]:
                    self.assertIsInstance(
                        config.get_quant_method(self.linear, prefix),
                        UnquantizedLinearMethod,
                    )
                self.assertEqual(raw, original)

    def test_model_supplied_mapping_is_used(self):
        config = w4afp8.W4AFp8Config.from_config(
            {
                "quant_method": "w4afp8",
                "modules_to_not_convert": ["in_proj_b", "in_proj_a"],
            }
        )
        config.update_packed_modules_mapping(self.mapping)
        self.assertIsInstance(
            config.get_quant_method(self.linear, "model.layers.0.in_proj_ba"),
            UnquantizedLinearMethod,
        )

    def test_mixed_precision_in_fused_shards_is_rejected(self):
        config = w4afp8.W4AFp8Config.from_config(
            {
                "quant_method": "w4afp8",
                "ignore": ["in_proj_b"],
                "packed_modules_mapping": self.mapping,
            }
        )
        with self.assertRaisesRegex(ValueError, "some but not all shards"):
            config.get_quant_method(self.linear, "model.layers.0.in_proj_ba")

    def test_gate_exclusion_does_not_skip_gate_up_projection(self):
        config = w4afp8.W4AFp8Config.from_config(
            {"quant_method": "w4afp8", "modules_to_not_convert": ["mlp.gate"]}
        )
        self.assertIsInstance(
            config.get_quant_method(self.linear, "model.layers.0.mlp.gate"),
            UnquantizedLinearMethod,
        )
        # Selection is the subject of this CPU test, not GPU backend discovery.
        with patch.object(w4afp8, "Fp8LinearMethod") as fp8:
            self.assertIs(
                config.get_quant_method(self.linear, "model.layers.0.mlp.gate_up_proj"),
                fp8.return_value,
            )
            fp8.assert_called_once_with(config)

    def test_missing_or_null_exclusions_preserve_fp8_linears(self):
        for extra in [{}, {"ignored_layers": None}, {"modules_to_not_convert": []}]:
            with self.subTest(extra=extra):
                config = w4afp8.W4AFp8Config.from_config(
                    {"quant_method": "w4afp8", **extra}
                )
                with patch.object(w4afp8, "Fp8LinearMethod") as fp8:
                    self.assertIs(
                        config.get_quant_method(self.linear, "model.layers.0.qkv_proj"),
                        fp8.return_value,
                    )
                    fp8.assert_called_once_with(config)

    def test_expert_quantization_is_preserved(self):
        config = w4afp8.W4AFp8Config.from_config(
            {"quant_method": "w4afp8", "modules_to_not_convert": ["mlp.gate"]}
        )
        self.assertIsInstance(
            config.get_quant_method(Mock(spec=FusedMoE), "model.layers.0.mlp.experts"),
            w4afp8.W4AFp8MoEMethod,
        )


if __name__ == "__main__":
    unittest.main()
