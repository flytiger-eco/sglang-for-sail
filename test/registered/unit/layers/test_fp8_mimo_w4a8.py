"""CPU contracts for MiMo's PPU W4A8 selection and weight preparation."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.model_config import MIMO_V2_MODEL_ARCHS
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.fused_moe_triton import layer as moe_layer
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import fp8
from sglang.srt.runtime_context import get_flags, publish, reset_context
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFp8MimoW4A8(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(
            ServerArgs(
                model_path="dummy", device="cpu", moe_runner_backend="deep_gemm"
            ),
            role="test",
        )
        get_flags().moe.runner_backend = MoeRunnerBackend.DEEP_GEMM
        self.config = fp8.Fp8Config(
            is_checkpoint_fp8_serialized=True,
            is_fp4_experts=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )

    def tearDown(self):
        reset_context()
        super().tearDown()

    def test_ppu_mimo_selection(self):
        layer = object.__new__(FusedMoE)
        for arch in MIMO_V2_MODEL_ARCHS:
            model = SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))
            with (
                self.subTest(arch=arch),
                patch.object(fp8, "_is_ppu", True),
                patch(
                    "sglang.srt.runtime_context.process_model_config",
                    return_value=model,
                ),
            ):
                method = self.config.get_quant_method(
                    layer, "model.layers.0.mlp.experts"
                )
                self.assertIsInstance(method, fp8.Fp8MoEMethod)
                self.assertTrue(method.use_ppu_w4a8)

    def test_other_ppu_models_keep_existing_method(self):
        layer = object.__new__(FusedMoE)
        model = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        )
        original = object()
        with (
            patch.object(fp8, "_is_ppu", True),
            patch(
                "sglang.srt.runtime_context.process_model_config", return_value=model
            ),
            patch(
                "sglang.srt.layers.quantization.mxfp4.Mxfp4MoEMethod",
                return_value=original,
            ),
        ):
            self.assertIs(self.config.get_quant_method(layer, "experts"), original)

    def test_non_ppu_keeps_existing_method(self):
        with patch.object(fp8, "_is_ppu", False):
            method = self.config.get_quant_method(object.__new__(FusedMoE), "experts")
        self.assertIsInstance(method, fp8.Fp8MoEMethod)
        self.assertFalse(method.use_ppu_w4a8)

    def make_layer(self):
        method = fp8.Fp8MoEMethod(self.config, use_ppu_w4a8=True)
        layer = torch.nn.Module()
        layer.moe_runner_config = MoeRunnerConfig(
            num_experts=2,
            num_local_experts=2,
            hidden_size=128,
            intermediate_size_per_partition=128,
            top_k=1,
            params_dtype=torch.bfloat16,
        )
        method.create_weights(layer, 2, 128, 128, torch.bfloat16)
        return layer, method

    def test_raw_checkpoint_storage(self):
        layer, _ = self.make_layer()
        self.assertEqual(layer.w13_weight.dtype, torch.int8)
        self.assertEqual(layer.w13_weight.shape, (2, 256, 64))
        self.assertEqual(layer.w13_weight_scale_inv.dtype, torch.uint8)
        self.assertEqual(layer.w13_weight_scale_inv.shape, (2, 256, 4))
        self.assertTrue(torch.all(layer.w13_weight_scale_inv == 127))
        self.assertTrue(torch.all(layer.w2_weight_scale_inv == 127))

    def test_preparation_preserves_bytes_and_configures_per_token_dispatch(self):
        layer, method = self.make_layer()
        expected = []
        for name in ("w13", "w2"):
            weight = getattr(layer, f"{name}_weight")
            scale = getattr(layer, f"{name}_weight_scale_inv")
            packed = torch.randint(256, weight.shape, dtype=torch.uint8)
            raw = torch.randint(119, 124, scale.shape, dtype=torch.uint8)
            weight.data.copy_(packed.view(torch.int8))
            scale.data.copy_(raw)
            expected.append((packed, raw))
        calls = []

        def prepare(mode, inplace, weight, scale):
            packed, raw = expected[len(calls)]
            torch.testing.assert_close(weight, packed, rtol=0, atol=0)
            torch.testing.assert_close(scale, raw, rtol=0, atol=0)
            self.assertFalse(inplace)
            calls.append(mode)
            return weight.clone(), scale.clone(), 0.125 * len(calls)

        library = SimpleNamespace(
            PreprocessMode=SimpleNamespace(W4A8_MXFP4_MMA=object()),
            preprocess_weights_and_scales=prepare,
        )
        layer.dispatcher = Mock()
        with patch.dict(sys.modules, deep_gemm=library):
            method.process_weights_after_loading(layer)
        self.assertEqual(len(calls), 2)
        self.assertEqual(layer.w13_w4a8_inv_scale, 0.125)
        self.assertEqual(layer.w2_w4a8_inv_scale, 0.25)
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(
            layer.dispatcher.set_quant_config.call_args.args[0],
            {"dispatcher_output_dtype": "fp8", "is_channel_quant": True},
        )

    def test_deepep_dispatch_dtype_contract(self):
        get_flags().moe.a2a_backend = MoeA2ABackend.DEEPEP
        for dtype in ("auto", "fp8", "bf16", "uint8"):
            for legacy_bf16 in (False, True):
                config = SimpleNamespace(
                    moe=SimpleNamespace(deepep_dispatcher_output_dtype=dtype)
                )
                with (
                    self.subTest(dtype=dtype, legacy_bf16=legacy_bf16),
                    patch("sglang.srt.runtime_context.get_exec", return_value=config),
                    patch.object(
                        fp8.envs.SGLANG_DEEPEP_BF16_DISPATCH,
                        "get",
                        return_value=legacy_bf16,
                    ),
                ):
                    if dtype in ("auto", "fp8") and not legacy_bf16:
                        method = fp8.Fp8MoEMethod(self.config, use_ppu_w4a8=True)
                        self.assertTrue(method.use_ppu_w4a8)
                    else:
                        with self.assertRaisesRegex(ValueError, "FP8 DeepEP"):
                            fp8.Fp8MoEMethod(self.config, use_ppu_w4a8=True)

    def test_rejects_requantized_weights(self):
        self.config.dequant_fp4_to_fp8 = True
        with self.assertRaisesRegex(ValueError, "native MXFP4"):
            fp8.Fp8MoEMethod(self.config, use_ppu_w4a8=True)

    def test_rejects_missing_block_quantization(self):
        self.config.weight_block_size = None
        with self.assertRaisesRegex(ValueError, "weight_block_size"):
            fp8.Fp8MoEMethod(self.config, use_ppu_w4a8=True)
        # The restriction is specific to the PPU W4A8 path.
        fp8.Fp8MoEMethod(self.config)

    def test_weight_loader_preserves_scale_bytes(self):
        for dtype in (torch.uint8, torch.float8_e8m0fnu):
            for padded in (False, True):
                for tp_rank in (0, 1):
                    with self.subTest(dtype=dtype, padded=padded, tp_rank=tp_rank):
                        method = fp8.Fp8MoEMethod(self.config, use_ppu_w4a8=True)
                        layer = FusedMoE.__new__(FusedMoE)
                        torch.nn.Module.__init__(layer)
                        layer.quant_method = method
                        layer.quant_config = self.config
                        layer.scheme = None
                        layer.moe_tp_rank = tp_rank
                        layer.moe_tp_size = 2
                        layer.use_padded_loading = padded
                        layer.use_presharded_weights = False
                        layer.use_triton_kernels = False
                        layer.use_flashinfer_trtllm_moe = False
                        layer._has_fused_shared = False
                        layer._expert_storage_rank = 0
                        layer._num_local_routed = 2
                        layer.moe_runner_config = MoeRunnerConfig(
                            num_experts=2,
                            num_local_experts=2,
                            hidden_size=128,
                            intermediate_size_per_partition=128,
                            top_k=1,
                            params_dtype=torch.bfloat16,
                        )
                        method.create_weights(
                            layer,
                            2,
                            128,
                            128,
                            torch.bfloat16,
                            weight_loader=layer.weight_loader,
                        )
                        expected_w13 = layer.w13_weight_scale_inv.detach().clone()
                        expected_w2 = layer.w2_weight_scale_inv.detach().clone()
                        for expert_id in (0, 1):
                            for shard_id in ("w1", "w3", "w2"):
                                is_down = shard_id == "w2"
                                shape = (128, 8) if is_down else (256, 4)
                                # Exponent bytes include 127 (1.0), not byte 1.
                                indices = torch.arange(1024).reshape(shape)
                                raw = (
                                    (indices // 32 + indices % 16) % 32
                                    + 112
                                    + expert_id
                                ).to(torch.uint8)
                                param = (
                                    layer.w2_weight_scale_inv
                                    if is_down
                                    else layer.w13_weight_scale_inv
                                )
                                with (
                                    patch.object(moe_layer, "_is_ppu", True),
                                    patch.object(
                                        moe_layer,
                                        "get_global_expert_location_metadata",
                                        return_value=None,
                                    ),
                                ):
                                    param.weight_loader(
                                        param,
                                        raw.view(dtype),
                                        "weight_scale_inv",
                                        shard_id,
                                        expert_id,
                                    )
                                if is_down:
                                    expected_w2[expert_id] = raw[
                                        :, tp_rank * 4 : (tp_rank + 1) * 4
                                    ]
                                else:
                                    start = 128 if shard_id == "w3" else 0
                                    expected_w13[expert_id, start : start + 128] = raw[
                                        tp_rank * 128 : (tp_rank + 1) * 128
                                    ]
                                torch.testing.assert_close(
                                    layer.w13_weight_scale_inv, expected_w13
                                )
                                torch.testing.assert_close(
                                    layer.w2_weight_scale_inv, expected_w2
                                )


if __name__ == "__main__":
    unittest.main()
