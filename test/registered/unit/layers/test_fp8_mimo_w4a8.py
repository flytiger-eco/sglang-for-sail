"""CPU contracts for MiMo's PPU W4A8 selection and weight preparation."""

import os
import sys
import unittest
from contextlib import ExitStack, nullcontext
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
        self.enterContext(patch.dict(os.environ, SGLANG_PPU_W4A8_A_PER_GROUP_QUANT="1"))
        publish(
            ServerArgs(
                model_path="dummy", device="cpu", moe_runner_backend="deep_gemm"
            ),
            role="test",
        )
        get_flags().moe.runner_backend = MoeRunnerBackend.DEEP_GEMM
        get_flags().moe.a2a_backend = MoeA2ABackend.DEEPEP
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

    def test_activation_quantization_flag_is_read_once(self):
        flag = fp8.envs.SGLANG_PPU_W4A8_A_PER_GROUP_QUANT
        with patch.dict(os.environ):
            os.environ.pop(flag.name, None)
            self.assertFalse(flag.get())
            self.assertTrue(
                fp8.Fp8MoEMethod(
                    self.config, use_ppu_w4a8=True
                ).ppu_w4a8_per_token_quant
            )
        for per_token in (False, True):
            with self.subTest(per_token=per_token), flag.override(not per_token):
                method = fp8.Fp8MoEMethod(self.config, use_ppu_w4a8=True)
                with flag.override(per_token):
                    self.assertEqual(method.ppu_w4a8_per_token_quant, per_token)
                # The override must not affect other model/platform methods.
                self.assertFalse(fp8.Fp8MoEMethod(self.config).ppu_w4a8_per_token_quant)

    def test_group128_requires_matching_deepgemm_before_weight_preparation(self):
        layer, method = self.make_layer()
        prepare = Mock()
        for supported in (None, [0]):
            library = SimpleNamespace(preprocess_weights_and_scales=prepare)
            if supported is not None:
                library.get_w4a8_activation_group_sizes = lambda: supported
            with (
                self.subTest(supported=supported),
                patch.dict(sys.modules, deep_gemm=library),
            ):
                with self.assertRaisesRegex(RuntimeError, "activation group size 128"):
                    method.process_weights_after_loading(layer)
        prepare.assert_not_called()

    def test_preparation_preserves_bytes_and_configures_selected_dispatch(self):
        for per_token in (False, True):
            with (
                self.subTest(per_token=per_token),
                fp8.envs.SGLANG_PPU_W4A8_A_PER_GROUP_QUANT.override(not per_token),
            ):
                self._check_preparation(per_token)

    def _check_preparation(self, per_token):
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
            get_w4a8_activation_group_sizes=lambda: [0, 128],
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
            {"dispatcher_output_dtype": "fp8", "is_channel_quant": per_token},
        )

    def test_selected_granularity_reaches_runner_metadata(self):
        for per_token in (False, True):
            with (
                self.subTest(per_token=per_token),
                fp8.envs.SGLANG_PPU_W4A8_A_PER_GROUP_QUANT.override(not per_token),
            ):
                layer, method = self.make_layer()
                method.moe_runner_config = layer.moe_runner_config
                method.runner = Mock(runner_backend=MoeRunnerBackend.DEEP_GEMM)
                method.runner.run.side_effect = lambda dispatch, quant: quant
                dispatch = SimpleNamespace(hidden_states=torch.zeros(1, 128))
                with patch.object(fp8, "use_intel_amx_backend", return_value=False):
                    quant = method.apply(layer, dispatch)
                self.assertTrue(quant.use_mxfp4_w4a8)
                self.assertEqual(quant.per_channel_quant, per_token)
                self.assertEqual(quant.block_shape, None if per_token else [128, 128])
                self.assertIs(quant.w13_weight, layer.w13_weight)
                self.assertIs(quant.w13_scale, layer.w13_weight_scale_inv)

    def test_both_deepep_modes_select_the_same_activation_group(self):
        from sglang.srt.layers.moe.token_dispatcher import deepep

        def quantize(x, group_size, **kwargs):
            groups = x.float().reshape(x.shape[0], -1, group_size)
            scales = groups.abs().amax(-1).clamp_min(1e-10) / 448
            values = (groups / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
            return values.reshape(x.shape), scales

        x = torch.ones(2, 256, dtype=torch.bfloat16)
        x[:, 128:] = 16
        topk = SimpleNamespace(
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            topk_weights=torch.ones(2, 1),
        )
        for per_token in (False, True):
            with self.subTest(per_token=per_token):
                expected_groups = 1 if per_token else 2
                normal = object.__new__(deepep._DeepEPDispatcherImplNormal)
                normal.use_fp8 = True
                normal.quant_config = {"is_channel_quant": per_token}
                normal.async_finish = False
                with (
                    patch.object(deepep.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
                    patch.object(
                        deepep, "sglang_per_token_group_quant_fp8", side_effect=quantize
                    ),
                ):
                    (normal_q, normal_s), *_ = normal.dispatch_a(x, topk)
                self.assertEqual(normal_s.shape, (2, expected_groups))
                low_latency = object.__new__(deepep._DeepEPDispatcherImplLowLatency)
                low_latency.use_int8 = False
                low_latency.use_mxfp4 = False
                low_latency.num_experts = 2
                low_latency.quant_config = normal.quant_config
                low_latency._get_buffer = lambda: SimpleNamespace(group_size=1)

                def dispatch(values, ids, weights, *, quant_size):
                    payload = quantize(values, quant_size)
                    return payload, torch.tensor([2, 0]), None, None

                low_latency._dispatch_core = dispatch
                (ll_q, ll_s), *_ = low_latency.dispatch_a(x, topk)
                torch.testing.assert_close(ll_q.float(), normal_q.float())
                torch.testing.assert_close(ll_s, normal_s)
                reconstructed = ll_q.float() * ll_s.repeat_interleave(
                    256 // expected_groups, dim=-1
                )
                torch.testing.assert_close(reconstructed, x.float())

    def test_native_runner_preserves_selected_scales_for_both_gemms(self):
        from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module

        def quantize(x, group_size, **kwargs):
            groups = x.float().reshape(*x.shape[:-1], -1, group_size)
            scales = groups.abs().amax(-1).clamp_min(1e-10) / 448
            q = (groups / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
            return q.reshape(x.shape), scales

        def dequantize(pair):
            q, scales = pair
            return q.float() * scales.repeat_interleave(
                q.shape[-1] // scales.shape[-1], -1
            )

        def activation(x):
            gate, up = x.float().chunk(2, -1)
            return (torch.nn.functional.silu(gate) * up).bfloat16()

        torch.manual_seed(20261109)
        w13 = torch.zeros(2, 512, 128, dtype=torch.uint8)
        w2 = torch.zeros(2, 256, 128, dtype=torch.uint8)
        dense13 = torch.randn(512, 256) * 0.03
        dense2 = torch.randn(256, 256) * 0.03
        for per_token in (False, True):
            for masked in (False, True):
                with (
                    self.subTest(per_token=per_token, masked=masked),
                    ExitStack() as stack,
                ):
                    group_size = 256 if per_token else 128
                    x = torch.randn((2, 3, 256) if masked else (3, 256)).bfloat16()
                    x[..., 128:] *= 8
                    lhs = quantize(x, group_size)
                    calls = []

                    def native_gemm(a, b, out, *args, inv_scale):
                        weight = b[0]
                        self.assertTrue(a[1].is_contiguous())
                        self.assertEqual(a[1].shape[-1], 1 if per_token else 2)
                        dense = dense13 if weight is w13 else dense2
                        self.assertEqual(inv_scale, 0.5 if weight is w13 else 0.25)
                        out.copy_((dequantize(a) @ dense.T * inv_scale).bfloat16())
                        calls.append(weight)

                    def masked_activation(
                        input, output, output_scale, quant_group_size, *args, **kwargs
                    ):
                        q, scales = quantize(activation(input), quant_group_size)
                        output.copy_(q)
                        output_scale.copy_(scales)

                    library = SimpleNamespace(
                        m_grouped_gemm_fp8_fp4_bf16_nt_nopad=native_gemm,
                        m_grouped_gemm_fp8_fp4_bf16_nt_masked=native_gemm,
                    )
                    stack.enter_context(patch.dict(sys.modules, deep_gemm=library))
                    for name in (
                        "DEEPGEMM_SCALE_UE8M0",
                        "DEEPGEMM_NEED_TMA_ALIGNED_SCALES",
                    ):
                        stack.enter_context(
                            patch.object(runner_module.deep_gemm_wrapper, name, False)
                        )
                    stack.enter_context(patch.object(runner_module, "dispose_tensor"))
                    stack.enter_context(
                        patch.object(runner_module, "get_tp_group", return_value=None)
                    )
                    stack.enter_context(
                        patch.object(
                            runner_module, "is_allocation_symmetric", return_value=False
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            runner_module,
                            "use_symmetric_memory",
                            side_effect=lambda *a, **k: nullcontext(),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            runner_module,
                            "_legacy_silu_and_mul",
                            side_effect=lambda x, out: out.copy_(activation(x)),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            runner_module,
                            "_varlen_deep_gemm_silu_mul_quant",
                            side_effect=lambda x, rows, group_size, **kw: quantize(
                                activation(x), group_size
                            ),
                        )
                    )
                    stack.enter_context(
                        patch(
                            "sglang.kernels.ops.quantization.fp8_kernel.sglang_per_token_group_quant_fp8",
                            side_effect=quantize,
                        )
                    )
                    stack.enter_context(
                        patch(
                            "sglang.kernels.ops.elementwise.silu_mul_quant.silu_and_mul_post_per_token_quant_fp8",
                            side_effect=lambda x, **kw: quantize(activation(x), 256),
                        )
                    )
                    stack.enter_context(
                        patch(
                            "sglang.kernels.ops.moe.ep_moe_kernels.silu_and_mul_masked_post_quant_fwd",
                            side_effect=masked_activation,
                        )
                    )
                    core = runner_module.DeepGemmRunnerCore(
                        MoeRunnerConfig(
                            num_experts=2,
                            num_local_experts=2,
                            hidden_size=256,
                            intermediate_size_per_partition=256,
                            top_k=1,
                            params_dtype=torch.bfloat16,
                            gate_up_interleaved=False,
                        )
                    )
                    quant = runner_module.DeepGemmMoeQuantInfo(
                        w13_weight=w13,
                        w2_weight=w2,
                        w13_scale=torch.ones(1),
                        w2_scale=torch.ones(1),
                        w13_w4a8_inv_scale=0.5,
                        w2_w4a8_inv_scale=0.25,
                        use_fp8=True,
                        is_fp4_experts=True,
                        use_mxfp4_w4a8=True,
                        per_channel_quant=per_token,
                        block_shape=None if per_token else [128, 128],
                    )
                    rows = torch.tensor([0, 3], dtype=torch.int32)
                    runner_input = runner_module.DeepGemmRunnerInput(
                        *lhs,
                        use_masked_gemm=masked,
                        m_indices=None if masked else torch.ones(3, dtype=torch.int32),
                        masked_m=rows if masked else None,
                        expected_m=3,
                    )
                    actual = core.run(
                        runner_input,
                        quant,
                        {
                            "all_tokens": 3,
                            "hidden_states_device": x.device,
                            "hidden_states_dtype": x.dtype,
                            "hidden_states_shape": x.shape,
                            "m_rows": rows,
                        },
                    ).hidden_states
                    gate_up = (dequantize(lhs) @ dense13.T * 0.5).bfloat16()
                    down = dequantize(quantize(activation(gate_up), group_size))
                    expected = (down @ dense2.T * 0.25).bfloat16()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(len(calls), 2)
                    self.assertIs(calls[0], w13)
                    self.assertIs(calls[1], w2)

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
