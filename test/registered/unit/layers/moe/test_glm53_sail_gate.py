"""MegaGate must preserve routing modes implemented by the standard TopK."""

import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang.kernels import KernelBackend, PlatformInfo, fused_op, select_kernel
from sglang.kernels.ops.moe.mega_gate import bf16_mega_gate
from sglang.srt.environ import envs
from sglang.srt.layers.moe import glm53_sail_gate
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm53SailGate(CustomTestCase):
    def setUp(self):
        super().setUp()
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        contexts.enter_context(published_topology(device="cpu"))
        contexts.enter_context(envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True))
        self.device_name = contexts.enter_context(
            patch.object(torch.cuda, "get_device_name", return_value="ZW-M890P")
        )
        glm53_sail_gate._is_sail_ppu.cache_clear()
        self.addCleanup(glm53_sail_gate._is_sail_ppu.cache_clear)
        with FakeTensorMode():
            self.hidden = torch.empty((1, 6144), dtype=torch.bfloat16, device="cuda:0")
            weight = self.hidden.new_empty((256, 6144))
            bias = self.hidden.new_empty((256,), dtype=torch.float32)
        self.gate = SimpleNamespace(
            weight=weight, e_score_correction_bias=bias, e_score_correction_bias_vl=None
        )

    def make_moe(self, num_shared):
        topk = TopK(
            top_k=8 + num_shared,
            num_fused_shared_experts=num_shared,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=1,
            scoring_func="sigmoid",
            correction_bias=self.gate.e_score_correction_bias,
        )
        return SimpleNamespace(
            config=SimpleNamespace(model_type="glm_moe_dsa"),
            is_hash=False,
            gate=self.gate,
            topk=topk,
            num_fused_shared_experts=num_shared,
        )

    def test_waterfill_keeps_the_shared_expert_slot(self):
        get_context().override(source="test", enable_waterfill=True)
        moe = self.make_moe(num_shared=1)
        # Waterfill normalizes these fields before its balancer appends slot 9.
        self.assertEqual(moe.topk.topk_config.top_k, 8)
        self.assertEqual(moe.topk.topk_config.num_fused_shared_experts, 0)
        self.assertFalse(
            glm53_sail_gate.can_route(moe, self.hidden, dispatch_info=None)
        )

    def test_fake_dispatch_is_a_per_call_override(self):
        # A target layer receives fake dispatch_info even with EPLB disabled.
        # DeepEP NextN in the same process receives None and uses learned routing.
        get_context().override(
            source="test", enable_eplb=False, ep_dispatch_algorithm="fake"
        )
        moe = self.make_moe(num_shared=0)
        for info, eligible in (
            (SimpleNamespace(ep_dispatch_algorithm="fake"), False),
            (SimpleNamespace(ep_dispatch_algorithm="lp"), False),
            (None, True),
            (SimpleNamespace(ep_dispatch_algorithm="static"), True),
        ):
            with self.subTest(dispatch_info=info):
                self.assertEqual(
                    glm53_sail_gate.can_route(moe, self.hidden, info), eligible
                )

    def test_hardware_eligibility(self):
        spec = select_kernel("moe.bf16_mega_gate")
        moe = self.make_moe(num_shared=0)
        for sm, name, available, sail_route in (
            ((10, 0), "NVIDIA B200", True, False),
            ((8, 9), "ZW-M890P", True, True),
            ((8, 0), "ZW-810E", True, True),
            ((8, 9), "NVIDIA L40S", False, False),
            ((8, 0), "NVIDIA A100", False, False),
        ):
            with (
                self.subTest(sm=sm, name=name),
                patch.object(torch.cuda, "get_device_capability", return_value=sm),
            ):
                self.device_name.return_value = name
                glm53_sail_gate._is_sail_ppu.cache_clear()
                self.assertEqual(
                    spec.is_available(PlatformInfo("cuda", *sm, name)), available
                )
                self.assertEqual(
                    glm53_sail_gate.can_route(moe, self.hidden, None), sail_route
                )
                with envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(False):
                    self.assertFalse(glm53_sail_gate.can_route(moe, self.hidden, None))

    def test_fullgraph_compile_uses_standard_routing(self):
        moe = self.make_moe(num_shared=0)
        hidden = self.hidden

        def forward(x):
            if glm53_sail_gate.can_route(moe, hidden, None):
                return x + 1
            return x - 1

        x = torch.tensor([2.0])
        compiled = torch.compile(forward, backend="eager", fullgraph=True)
        torch.testing.assert_close(forward(x), x + 1)
        torch.testing.assert_close(compiled(x), x - 1)
        torch.testing.assert_close(forward(x), x + 1)

    def test_forced_backend_uses_standard_routing(self):
        moe = self.make_moe(num_shared=0)
        with (
            envs.SGLANG_FORCE_FUSED_OP_BACKEND.override("torch"),
            patch.object(fused_op, "_forced_backend", fused_op._UNRESOLVED),
        ):
            self.assertFalse(glm53_sail_gate.can_route(moe, self.hidden, None))
            for backend in (None, KernelBackend.TORCH, KernelBackend.JIT, None):
                with self.subTest(backend=backend):
                    fused_op.set_fused_op_backend(backend)
                    self.assertEqual(
                        glm53_sail_gate.can_route(moe, self.hidden, None),
                        backend is None,
                    )

    def test_nonstandard_output_uses_standard_routing(self):
        moe = self.make_moe(num_shared=0)
        for output_format in TopKOutputFormat:
            with self.subTest(output_format=output_format):
                moe.topk.topk_config.output_format = output_format
                self.assertEqual(
                    glm53_sail_gate.can_route(moe, self.hidden, None),
                    output_format == TopKOutputFormat.STANDARD,
                )

    def test_deterministic_and_simulated_routes_are_preserved(self):
        moe = self.make_moe(num_shared=0)
        for setting in ("enable_eplb", "enable_deterministic_inference"):
            with self.subTest(setting=setting):
                self.assertTrue(glm53_sail_gate.can_route(moe, self.hidden, None))
                get_context().override(source="test", **{setting: True})
                self.assertFalse(glm53_sail_gate.can_route(moe, self.hidden, None))
                get_context().override(source="test", **{setting: False})
        for setting in (
            envs.SGLANG_SIMULATE_UNIFORM_EXPERTS,
            envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS,
        ):
            with setting.override(True):
                self.assertFalse(glm53_sail_gate.can_route(moe, self.hidden, None))

    def test_row_limit(self):
        moe = self.make_moe(num_shared=0)
        for rows, eligible in [(0, False), (1 << 20, True), ((1 << 20) + 1, False)]:
            hidden = self.hidden.new_empty((rows, 6144))
            self.assertEqual(glm53_sail_gate.can_route(moe, hidden, None), eligible)

    def test_masked_input_is_forwarded_without_copy(self):
        hidden = torch.tensor([[1.0], [float("nan")], [float("inf")]])
        mask = torch.tensor([True, False, False])
        native = Mock()
        with patch.dict(
            sys.modules, {"deep_gemm": SimpleNamespace(bf16_mega_gate=native)}
        ):
            bf16_mega_gate(
                hidden,
                torch.ones(8, 1),
                8,
                routed_scaling_factor=1.0,
                ep_rank=0,
                valid_token_mask=mask,
            )
        self.assertIs(native.call_args.args[0], hidden)
        self.assertIs(native.call_args.kwargs["mask"], mask)

    def test_operator_contract_and_glm_routing(self):
        # Stand in only for the device extension. The real upstream wrapper owns
        # the output buffers and reorders the native (ids, weights) result.
        def native(
            x,
            weight,
            k,
            *,
            scoring_func,
            bias,
            routed_scaling_factor,
            mask,
            out,
            **kwargs,
        ):
            logits = x.float() @ weight.float().T
            scores = (
                logits.sigmoid()
                if scoring_func == "sigmoid"
                else torch.nn.functional.softplus(logits).sqrt()
            )
            ids = (scores + bias).topk(k).indices
            selected = scores.gather(1, ids)
            weights = selected / selected.sum(1, keepdim=True) * routed_scaling_factor
            if mask is not None:
                ids = ids.masked_fill(~mask[:, None], -1)
                weights = weights.masked_fill(~mask[:, None], 0)
            out[0].copy_(ids)
            out[1].copy_(weights)
            return out

        moe = self.make_moe(num_shared=0)
        moe.layer_id, moe.is_nextn = 0, False
        x = torch.tensor([[1.0], [float("nan")]], dtype=torch.bfloat16)
        weight = torch.arange(9, dtype=torch.bfloat16)[:, None]
        bias = torch.tensor([10.0, *([0.0] * 8)])
        moe.gate = SimpleNamespace(weight=weight, e_score_correction_bias=bias)
        cfg = moe.topk.topk_config
        cfg.routed_scaling_factor = 2.5
        batch = SimpleNamespace(moe_num_token_non_padded=lambda: torch.tensor([1]))
        expert_ids = torch.tensor([0, 8, 7, 6, 5, 4, 3, 2])
        with (
            patch.dict(
                sys.modules, {"deep_gemm": SimpleNamespace(bf16_mega_gate=native)}
            ),
            get_parallel().override(moe_ep_rank=0),
        ):
            weights, ids = bf16_mega_gate(
                x[:1], weight, 8, bias=bias, routed_scaling_factor=1.0, ep_rank=0
            )
            torch.testing.assert_close(ids[0], expert_ids)
            expected = torch.nn.functional.softplus(expert_ids.float()).sqrt()
            torch.testing.assert_close(weights[0], expected / expected.sum())
            # Reuse the public return value with a different input.
            kwargs = dict(bias=bias, routed_scaling_factor=1.0, ep_rank=0)
            expected_weights, expected_ids = bf16_mega_gate(-x[:1], weight, 8, **kwargs)
            reused_weights, reused_ids = bf16_mega_gate(
                -x[:1], weight, 8, out=(weights, ids), **kwargs
            )
            self.assertIs(reused_weights, weights)
            self.assertIs(reused_ids, ids)
            torch.testing.assert_close(weights, expected_weights)
            torch.testing.assert_close(ids, expected_ids)
            for fuse_scale in (False, True):
                cfg.apply_routed_scaling_factor_on_output = fuse_scale
                output = glm53_sail_gate.route(moe, x, batch, dispatch_info=None)
                expected = expert_ids.float().sigmoid()
                expected = expected / expected.sum() * (2.5 if fuse_scale else 1.0)
                torch.testing.assert_close(output.topk_ids[0], expert_ids)
                torch.testing.assert_close(output.topk_weights[0], expected)
                self.assertTrue((output.topk_ids[1] == -1).all())
                self.assertTrue((output.topk_weights[1] == 0).all())


if __name__ == "__main__":
    unittest.main()
