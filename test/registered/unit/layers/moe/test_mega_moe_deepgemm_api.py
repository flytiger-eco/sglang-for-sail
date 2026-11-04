"""Unit tests for the DeepGEMM MegaMoE interface."""

import sys
import unittest
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe import mega_moe
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestDeepGemmMegaMoeApi(CustomTestCase):
    def setUp(self):
        super().setUp()
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        self.deep_gemm = ModuleType("deep_gemm")
        self.deep_gemm.num_sms = 132
        self.deep_gemm.get_num_sms = MagicMock(
            side_effect=lambda: self.deep_gemm.num_sms
        )
        self.deep_gemm.set_num_sms = MagicMock(
            side_effect=lambda num_sms: setattr(self.deep_gemm, "num_sms", num_sms)
        )
        max_num_sms = patch.object(mega_moe, "_mega_moe_max_num_sms", return_value=130)
        self.max_num_sms = max_num_sms.start()
        self.addCleanup(max_num_sms.stop)

    def tearDown(self):
        mega_moe._MEGA_MOE_SYMM_BUFFER.clear()
        super().tearDown()

    def test_ppu_activation_separates_symmetric_buffers(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *args, **kwargs: SimpleNamespace(**kwargs)
        )
        group = object()
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):

            def allocate(activation):
                return mega_moe._get_mega_moe_symm_buffer(
                    group,
                    8,
                    64,
                    2,
                    512,
                    512,
                    mma_type="fp4xfp4",
                    activation=activation,
                )

            swiglu = allocate("swiglu")
            situ = allocate("situ")
            self.assertIs(situ, allocate("situ"))
        self.assertIsNot(swiglu, situ)
        self.assertEqual((swiglu.activation, situ.activation), ("swiglu", "situ"))
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)

    def test_ppu_routed_experts_forwards_situ_and_scales_once(self):
        deep_gemm = self.deep_gemm
        deep_gemm.mega_moe_pre_dispatch = MagicMock()
        deep_gemm.fp4_fp4_mega_moe = MagicMock(
            side_effect=lambda y, *_args, **_kwargs: y.fill_(2)
        )
        experts = SimpleNamespace(mega_l1_weights=object(), mega_l2_weights=object())
        hidden = torch.ones((2, 4), dtype=torch.bfloat16)
        ids = torch.zeros((2, 2), dtype=torch.int32)[:, :1]
        weights = torch.ones((2, 2))[:, :1]
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            result = mega_moe.run_ppu_sm89_mega_routed(
                experts,
                hidden,
                ids,
                weights,
                object(),
                2,
                hidden_size=4,
                mma_type="fp4xfp4",
                activation="situ",
                routed_scaling_factor=3.0,
            )
        self.assertTrue(torch.all(result == 6))
        self.assertEqual(
            deep_gemm.fp4_fp4_mega_moe.call_args.kwargs["activation"], "situ"
        )
        self.assertEqual(
            deep_gemm.mega_moe_pre_dispatch.call_args.args[1].dtype, torch.int64
        )
        self.assertTrue(
            deep_gemm.mega_moe_pre_dispatch.call_args.args[1].is_contiguous()
        )
        self.assertTrue(
            deep_gemm.mega_moe_pre_dispatch.call_args.args[2].is_contiguous()
        )

    def test_ppu_weight_transform_preserves_activation(self):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        method.use_deepgemm_mxfp4_w4a16_mma = False
        method.use_deepgemm_mxfp4_w4a16_valu = False
        method.with_bias = False
        for activation in ("situ", "silu"):
            with self.subTest(activation=activation):
                layer = SimpleNamespace(
                    moe_runner_config=SimpleNamespace(activation=activation)
                )
                for name in (
                    "w13_weight",
                    "w13_weight_scale",
                    "w2_weight",
                    "w2_weight_scale",
                ):
                    setattr(
                        layer,
                        name,
                        torch.nn.Parameter(
                            torch.zeros(1, dtype=torch.uint8), requires_grad=False
                        ),
                    )
                transform = MagicMock(side_effect=lambda l1, l2, **kwargs: (l1, l2))
                with (
                    patch("sglang.srt.layers.quantization.mxfp4._is_ppu", True),
                    patch.dict(
                        sys.modules,
                        {
                            "deep_gemm": SimpleNamespace(
                                transform_weights_for_mega_moe=transform
                            )
                        },
                    ),
                    patch.object(torch.cuda, "empty_cache"),
                ):
                    method.process_weights_after_loading(layer)
                self.assertEqual(
                    transform.call_args.kwargs,
                    {"activation": "situ"} if activation == "situ" else {},
                )
                self.assertTrue(layer._mega_moe_weights_built)
                self.assertTrue(layer._mega_moe_requires_only)
                self.assertEqual(layer._mega_moe_mma_type, "fp4xfp4")

    def test_repacked_experts_reject_over_cap_instead_of_falling_back(self):
        moe = SimpleNamespace(experts=SimpleNamespace(_mega_moe_requires_only=True))
        with (
            patch.object(mega_moe, "_use_amd_flydsl_mega_moe", return_value=False),
            patch.object(mega_moe, "get_moe_a2a_backend") as backend,
            patch.object(mega_moe, "is_mega_moe_experts_ready", return_value=True),
            patch.object(mega_moe, "get_is_capture_mode") as capture,
            patch.object(
                mega_moe, "get_dp_global_num_tokens", return_value=None
            ) as tokens,
            patch.object(mega_moe, "is_dsa_enable_prefill_cp", return_value=False),
            mega_moe.envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.override(
                4
            ),
        ):
            backend.return_value.is_megamoe.return_value = True
            for capturing in (False, True):
                capture.return_value = capturing
                with self.subTest(capturing=capturing):
                    self.assertTrue(
                        mega_moe.should_use_mega_moe(moe, torch.empty(4, 1))
                    )
                    with self.assertRaisesRegex(ValueError, "cannot fall back"):
                        mega_moe.should_use_mega_moe(moe, torch.empty(5, 1))
            capture.return_value = False
            tokens.return_value = [4, 5]
            with self.assertRaisesRegex(ValueError, "required_per_rank=5"):
                mega_moe.should_use_mega_moe(moe, torch.empty(4, 1))
            # The existing fallback remains valid for non-repacked experts.
            moe.experts._mega_moe_requires_only = False
            self.assertFalse(mega_moe.should_use_mega_moe(moe, torch.empty(4, 1)))

    def test_repacked_experts_cannot_use_missing_or_disabled_megamoe(self):
        moe = SimpleNamespace(experts=SimpleNamespace(_mega_moe_requires_only=True))
        with (
            patch.object(mega_moe, "_use_amd_flydsl_mega_moe", return_value=False),
            patch.object(mega_moe, "get_moe_a2a_backend") as backend,
            patch.object(mega_moe, "is_mega_moe_experts_ready", return_value=False),
        ):
            for enabled in (False, True):
                backend.return_value.is_megamoe.return_value = enabled
                with self.subTest(enabled=enabled):
                    with self.assertRaisesRegex(RuntimeError, "Repacked PPU MXFP4"):
                        mega_moe.should_use_mega_moe(moe, torch.empty(1, 1))

    def test_kimi_k3_ppu_routes_situ_and_scaling(self):
        from sglang.srt.models import kimi_k3

        experts = SimpleNamespace(should_fuse_routed_scaling_factor_in_topk=False)
        model = SimpleNamespace(
            experts=experts,
            routed_scaling_factor=2.5,
            moe_hidden_size=4,
            _mega_intermediate_size=8,
            _mega_top_k=2,
        )
        x = torch.ones((3, 4), dtype=torch.bfloat16)
        topk = SimpleNamespace(
            topk_ids=torch.zeros((3, 2)), topk_weights=torch.ones((3, 2))
        )
        output = torch.ones_like(x)
        with (
            patch.object(kimi_k3, "is_ppu", return_value=True),
            patch.object(
                mega_moe, "run_mega_routed_experts", return_value=output
            ) as run,
        ):
            result = kimi_k3.KimiK3MoE._forward_mega_experts(model, x, topk)
        self.assertIs(result, output)
        self.assertEqual(run.call_args.kwargs["activation"], "situ")
        self.assertEqual(run.call_args.kwargs["num_tokens"], 3)
        self.assertEqual(run.call_args.kwargs["routed_scaling_factor"], 2.5)

    def test_mxf4_buffer_uses_typed_api(self):
        deep_gemm = self.deep_gemm
        expected_buffer = object()
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(return_value=expected_buffer)
        group = object()

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(
                mega_moe,
                "_mega_moe_mma_type",
                return_value="mxf4xmxf4",
                create=True,
            ),
        ):
            actual_buffer = mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
            )

        self.assertIs(actual_buffer, expected_buffer)
        call = deep_gemm.get_symm_buffer_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp8_dispatch", call.kwargs)

    def test_buffer_cache_separates_mma_types(self):
        deep_gemm = self.deep_gemm
        expected_buffers = (object(), object())
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(side_effect=expected_buffers)
        group = object()

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(
                mega_moe,
                "_mega_moe_mma_type",
                side_effect=("fp8xfp4", "mxf4xmxf4"),
            ),
        ):
            actual_buffers = tuple(
                mega_moe._get_mega_moe_symm_buffer(
                    group,
                    num_experts=8,
                    num_max_tokens_per_rank=64,
                    num_topk=2,
                    hidden=128,
                    intermediate_hidden=256,
                )
                for _ in range(2)
            )

        self.assertEqual(actual_buffers, expected_buffers)
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)
        self.assertEqual(
            [
                call.kwargs["mma_type"]
                for call in deep_gemm.get_symm_buffer_for_mega_moe.call_args_list
            ],
            ["fp8xfp4", "mxf4xmxf4"],
        )

    def test_buffer_cache_uses_effective_sm_budget(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: SimpleNamespace(
                num_sms=deep_gemm.get_num_sms()
            )
        )
        group = object()
        buffers = []

        for current_num_sms, expected_num_sms in (
            (132, 130),
            (131, 130),
            (96, 96),
            (97, 96),
            (132, 130),
        ):
            with self.subTest(current_num_sms=current_num_sms):
                deep_gemm.set_num_sms(current_num_sms)
                buf = self._get_test_buffer(group)
                buffers.append(buf)
                self.assertEqual(buf.num_sms, expected_num_sms)
                self.assertEqual(deep_gemm.get_num_sms(), current_num_sms)
                with mega_moe._configure_mega_moe_deep_gemm_num_sms(deep_gemm):
                    self.assertEqual(buf.num_sms, deep_gemm.get_num_sms())
                self.assertEqual(deep_gemm.get_num_sms(), current_num_sms)

        self.assertIs(buffers[0], buffers[1])
        self.assertIs(buffers[0], buffers[4])
        self.assertIs(buffers[2], buffers[3])
        self.assertIsNot(buffers[0], buffers[2])
        self.assertEqual(deep_gemm.get_symm_buffer_for_mega_moe.call_count, 2)

    def test_buffer_allocation_without_sm_override(self):
        deep_gemm = self.deep_gemm
        self.max_num_sms.return_value = None
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=lambda *_args, **_kwargs: deep_gemm.get_num_sms()
        )

        buf = self._get_test_buffer(object())
        self.assertEqual(buf, 132)
        deep_gemm.set_num_sms.assert_not_called()

    def test_buffer_allocation_failure_restores_sm_budget(self):
        deep_gemm = self.deep_gemm
        deep_gemm.get_symm_buffer_for_mega_moe = MagicMock(
            side_effect=RuntimeError("allocation failed")
        )

        with self.assertRaisesRegex(RuntimeError, "allocation failed"):
            self._get_test_buffer(object())

        self.assertEqual(deep_gemm.get_num_sms(), 132)
        self.assertEqual(deep_gemm.set_num_sms.call_args_list, [call(130), call(132)])
        self.assertEqual(mega_moe._MEGA_MOE_SYMM_BUFFER, {})

    def test_mxf4_weight_transform_uses_matching_mma_type(self):
        from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.transform_sf_into_required_layout = MagicMock(
            side_effect=lambda _sf, mn, k, recipe, num_groups, disable_ue8m0_cast: (
                torch.zeros((num_groups, mn, max(1, k // 32)), dtype=torch.int32)
            )
        )
        deep_gemm.transform_weights_for_mega_moe = MagicMock(
            side_effect=lambda l1, l2, **_kwargs: (l1, l2)
        )
        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = True
        layer = SimpleNamespace(
            w13_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w13_weight_scale=torch.nn.Parameter(
                torch.zeros((1, 32, 1), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight=torch.nn.Parameter(
                torch.zeros((1, 32, 16), dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale=torch.nn.Parameter(
                torch.zeros((1, 32, 1), dtype=torch.uint8), requires_grad=False
            ),
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch("sglang.srt.layers.quantization.mxfp4._is_ppu", False),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
        ):
            method.process_weights_after_loading(layer)

        call = deep_gemm.transform_weights_for_mega_moe.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")

    def test_mxf4_pre_dispatch_uses_typed_api(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.mega_moe_pre_dispatch = MagicMock()
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            should_fuse_routed_scaling_factor_in_topk=True,
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
        )
        topk_output = SimpleNamespace(
            topk_ids=torch.tensor([[0, 1]]),
            topk_weights=torch.tensor([[0.6, 0.4]]),
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=4,
                num_experts_per_tok=2,
                moe_intermediate_size=8,
                swiglu_limit=None,
            ),
            experts=experts,
            gate=MagicMock(return_value=torch.empty((1, 8))),
            topk=MagicMock(return_value=topk_output),
            is_hash=False,
            num_fused_shared_experts=0,
            layer_id=0,
            routed_scaling_factor=1.0,
        )

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="mxf4xmxf4"),
            patch.object(
                mega_moe,
                "_get_mega_moe_symm_buffer",
                return_value=buffer,
            ),
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
            patch.object(
                mega_moe.ExpertLocationDispatchInfo,
                "init_new",
                return_value=object(),
            ),
            get_parallel().override(
                moe_ep_group=SimpleNamespace(device_group=object())
            ),
        ):
            mega_moe._run_mega_routed(
                moe,
                torch.zeros((1, 4)),
                forward_batch=None,
                input_ids_global=None,
                num_tokens=1,
            )

        self.assertTrue(deep_gemm.mega_moe_pre_dispatch.called)
        call = deep_gemm.mega_moe_pre_dispatch.call_args
        self.assertEqual(call.kwargs.get("mma_type"), "mxf4xmxf4")
        self.assertNotIn("use_fp4_acts", call.kwargs)

    def test_run_mega_routed_experts_generic_entry(self):
        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.fp8_fp4_mega_moe = MagicMock()
        buffer = SimpleNamespace(
            x=object(),
            x_sf=object(),
            topk_idx=object(),
            topk_weights=object(),
        )
        experts = SimpleNamespace(
            num_experts=8,
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
        )
        hidden_states = torch.zeros((3, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int64)
        topk_weights = torch.full((3, 2), 0.5, dtype=torch.bfloat16)

        with (
            patch.dict(sys.modules, {"deep_gemm": deep_gemm}),
            patch.object(mega_moe, "_device_sm", 100),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
            patch.object(mega_moe, "mega_moe_pre_dispatch") as pre_dispatch,
            patch.object(
                mega_moe, "_get_mega_moe_symm_buffer", return_value=buffer
            ) as get_buffer,
            patch.object(
                mega_moe,
                "_configure_mega_moe_deep_gemm_num_sms",
                return_value=nullcontext(),
            ),
            # Toy shapes.
            patch.object(mega_moe, "check_mega_moe_shapes"),
            patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_group=SimpleNamespace(device_group=object())
                ),
            ),
        ):
            out = mega_moe.run_mega_routed_experts(
                experts,
                hidden_states,
                topk_ids,
                topk_weights,
                hidden_size=4,
                intermediate_size=8,
                top_k=2,
                num_tokens=3,
                activation_clamp=7.0,
                routed_scaling_factor=1.0,
            )

        self.assertEqual(out.shape, (3, 4))
        self.assertEqual(out.dtype, torch.bfloat16)
        buf_call = get_buffer.call_args
        self.assertEqual(buf_call.kwargs.get("num_topk"), 2)
        self.assertEqual(buf_call.kwargs.get("hidden"), 4)
        self.assertEqual(buf_call.kwargs.get("intermediate_hidden"), 8)
        # The kernel wants int32 ids and fp32 weights regardless of the router dtype.
        ids_arg, weights_arg = pre_dispatch.call_args.args[1:3]
        self.assertEqual(ids_arg.dtype, torch.int32)
        self.assertEqual(weights_arg.dtype, torch.float32)
        mega_call = deep_gemm.fp8_fp4_mega_moe.call_args
        self.assertIs(mega_call.args[1], experts.mega_l1_weights)
        self.assertIs(mega_call.args[2], experts.mega_l2_weights)
        self.assertEqual(mega_call.kwargs.get("activation_clamp"), 7.0)

    def test_shape_check_rejects_unaligned_intermediate(self):
        # Qwen3-30B-A3B: intermediate 768 leaves a 24-byte scale row.
        with self.assertRaisesRegex(ValueError, "multiples of 512"):
            mega_moe.check_mega_moe_shapes(2048, 768, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1536, "fp8xfp4")
        mega_moe.check_mega_moe_shapes(4096, 1024, "mxf4xmxf4")
        # 768 is a multiple of 256, so the NVFP4 (g16) rule accepts it.
        mega_moe.check_mega_moe_shapes(2048, 768, "nvfp4xnvfp4")
        with self.assertRaisesRegex(ValueError, "multiples of 256"):
            mega_moe.check_mega_moe_shapes(2048, 384, "nvfp4xnvfp4")

    def test_mxf4_l1_uses_packed_gate_up_interleave(self):
        source = torch.arange(32).reshape(1, 32)
        expected = torch.tensor(
            [
                0,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                1,
                3,
                5,
                7,
                9,
                11,
                13,
                15,
                17,
                19,
                21,
                23,
                25,
                27,
                29,
                31,
            ]
        ).reshape(1, 32)

        actual = mega_moe._interleave_mega_moe_gate_up(source, gran=16)

        torch.testing.assert_close(actual, expected)

    def _get_test_buffer(self, group):
        with (
            patch.dict(sys.modules, {"deep_gemm": self.deep_gemm}),
            patch.object(mega_moe, "_mega_moe_mma_type", return_value="fp8xfp4"),
        ):
            return mega_moe._get_mega_moe_symm_buffer(
                group,
                num_experts=8,
                num_max_tokens_per_rank=64,
                num_topk=2,
                hidden=128,
                intermediate_hidden=256,
            )


if __name__ == "__main__":
    unittest.main()
