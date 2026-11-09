"""W4AFP8 checkpoint exclusions must survive runtime names and fused linears."""

import shutil
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

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


class TestW4AFP8RuntimeSafety(CustomTestCase):
    def test_derived_parameter_replacement_keeps_loader_metadata(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        loader = Mock()
        layer.weight.weight_loader = loader
        layer.weight.output_dim = 0
        w4afp8._replace_weight_parameter(layer, "weight", torch.zeros(3))
        self.assertIs(layer.weight.weight_loader, loader)
        self.assertEqual(layer.weight.output_dim, 0)
        self.assertEqual(tuple(layer.weight.shape), (3,))

    def test_processed_w4afp8_publishes_update_guard_for_upstream_fallback(self):
        layer = torch.nn.Module()
        for name in ("w13_weight", "w2_weight"):
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.zeros(1, 8, 8, dtype=torch.int8), requires_grad=False
                ),
            )
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
            layer.register_parameter(
                name, torch.nn.Parameter(torch.ones(1, 8, 4), requires_grad=False)
            )
        for name in ("w13_input_scale", "w2_input_scale"):
            layer.register_parameter(
                name, torch.nn.Parameter(torch.ones(1), requires_grad=False)
            )
        method = w4afp8.W4AFp8MoEMethod.__new__(w4afp8.W4AFp8MoEMethod)
        method.use_sm = False
        method._weights_processed = False
        method.quant_config = SimpleNamespace(
            group_size=128, moe_activation_scheme="static"
        )
        with w4afp8.envs.SGLANG_W4AFP8_MOE_BACKEND.override("upstream"):
            method.process_weights_after_loading(layer)
        self.assertIn("Online weight updates", layer._derived_weight_cache_error)
        first = layer.w13_weight_scale_inv
        method.process_weights_after_loading(layer)
        self.assertIs(layer.w13_weight_scale_inv, first)

    def test_replicated_shared_scale_matches_output_reduction_semantics(self):
        from sglang.srt.models import qwen2_moe

        for complete, expected in ((False, 0.25), (True, 1.0)):
            with (
                self.subTest(complete=complete),
                patch.object(
                    qwen2_moe, "post_experts_output_is_complete", return_value=complete
                ),
            ):
                self.assertEqual(qwen2_moe._replicated_shared_output_scale(4), expected)

    @unittest.skipUnless(shutil.which("cmake"), "cmake required")
    def test_w4a8_build_gate_checks_compiler_and_toolkit(self):
        source = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/kernels/aot/CMakeLists.txt"
        ).read_text()
        gate = source[
            source.index("set(DEFAULT_SGL_KERNEL_BUILD_W4A8_SM OFF)") : source.index(
                "if(SGL_KERNEL_W4A8_SM_ONLY)"
            )
        ]
        gate = gate.replace(
            "include(${CMAKE_CURRENT_LIST_DIR}/cmake/w4a8_sm.cmake)", ""
        )
        for compiler, toolkit, explicit, success, enabled in (
            ("11.8", "11.8", None, True, "OFF"),
            ("12.1", "12.1", None, True, "OFF"),
            ("12.4", "12.4", None, True, "ON"),
            ("13.0", "13.0", None, True, "ON"),
            ("11.8", "13.0", "ON", False, None),
            ("13.0", "11.8", "ON", False, None),
            ("13.0", "13.0", "OFF", True, "OFF"),
        ):
            with (
                self.subTest(compiler=compiler, toolkit=toolkit, explicit=explicit),
                tempfile.TemporaryDirectory() as tmp,
            ):
                script = Path(tmp) / "gate.cmake"
                code = f"set(CMAKE_CUDA_COMPILER_VERSION {compiler})\nset(CUDAToolkit_VERSION {toolkit})\n"
                if explicit:
                    code += f'set(SGL_KERNEL_BUILD_W4A8_SM {explicit} CACHE BOOL "")\n'
                script.write_text(
                    code
                    + gate
                    + '\nmessage(STATUS "W4A8=${SGL_KERNEL_BUILD_W4A8_SM}")\n'
                )
                result = subprocess.run(
                    ["cmake", "-P", str(script)], capture_output=True, text=True
                )
                self.assertEqual(
                    result.returncode == 0, success, result.stdout + result.stderr
                )
                if success:
                    self.assertIn(f"W4A8={enabled}", result.stdout)


if __name__ == "__main__":
    unittest.main()
