import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.moe.mega_gate import bf16_mega_gate
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-b-test-1-gpu-ppu-m890")


class TestGlmSigmoidMegaGate(CustomTestCase):
    def _prepare_expert_location(self):
        from sglang.srt.eplb.expert_location import (
            ExpertLocationMetadata,
            set_global_expert_location_metadata,
        )
        from sglang.srt.runtime_context import get_context

        # Unlike 0.5.23, this branch resolves metadata at each call site.
        # Install the same identity placement ModelRunner prepares for EP1.
        get_context().override(source="test", ep_dispatch_algorithm="static")
        ids = torch.arange(256, device="cuda").expand(4, -1).contiguous()
        metadata = ExpertLocationMetadata._init_raw(
            ep_size=1,
            physical_to_logical_map=ids,
            logical_to_all_physical_map=ids[..., None],
            moe_ep_rank=0,
        )
        set_global_expert_location_metadata(metadata)
        self.addCleanup(set_global_expert_location_metadata, None, True)

    def _glm_fixture(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
        from sglang.test.test_utils import published_topology

        self.enterContext(published_topology(device="cuda"))
        self._prepare_expert_location()
        self.enterContext(envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True))
        weight = torch.zeros((256, 6144), dtype=torch.bfloat16, device="cuda")
        weight[:8, 0] = torch.arange(-4, 4, device="cuda")
        hidden = torch.ones((2, 6144), dtype=torch.bfloat16, device="cuda")
        bias = torch.zeros(256, dtype=torch.float32, device="cuda")
        bias[:8] = 2
        topk = TopK(
            top_k=8,
            layer_id=0,
            num_expert_group=1,
            topk_group=1,
            use_grouped_topk=True,
            scoring_func="sigmoid",
            correction_bias=bias,
            routed_scaling_factor=1.5,
            apply_routed_scaling_factor_on_output=True,
            output_format=TopKOutputFormat.STANDARD,
        )
        gate = Mock(
            weight=weight, e_score_correction_bias=bias, e_score_correction_bias_vl=None
        )
        experts = Mock(
            return_value=hidden, should_fuse_routed_scaling_factor_in_topk=True
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(
                model_type="glm_moe_dsa",
                num_experts_per_tok=8,
                hidden_size=6144,
                moe_intermediate_size=2048,
            ),
            layer_id=0,
            is_nextn=False,
            is_hash=False,
            gate=gate,
            topk=topk,
            experts=experts,
            alt_stream=None,
            _fuse_shared_experts_inside_sbo=False,
            num_fused_shared_experts=0,
            mega_shared_l1_weights=None,
            mega_shared_l2_weights=None,
            _forward_shared_experts=lambda _: None,
            routed_scaling_factor=1.5,
            routed_scaling_factor_fused_in_topk=True,
        )
        bound = torch.tensor([1], dtype=torch.int32, device="cuda")
        batch = SimpleNamespace(moe_num_token_non_padded=lambda: bound)
        return moe, hidden, batch

    def _assert_routes(self, moe, ids, weights):
        ordered_ids, order = ids[0].sort()
        torch.testing.assert_close(ordered_ids, torch.arange(8, device="cuda"))
        scores = moe.gate.weight[:8, 0].float().sigmoid()
        torch.testing.assert_close(weights[0, order], scores / scores.sum() * 1.5)
        self.assertTrue((ids[1] == -1).all())
        self.assertTrue((weights[1] == 0).all())

    def test_deepep_forward_passes_sigmoid_routes_to_experts(self):
        from sglang.srt.layers.moe.utils import MoeA2ABackend
        from sglang.srt.models import deepseek_v2

        moe, hidden, batch = self._glm_fixture()
        with patch.object(
            deepseek_v2, "get_moe_a2a_backend", return_value=MoeA2ABackend.DEEPEP
        ):
            deepseek_v2.DeepseekV2MoE.forward_deepep(moe, hidden, batch)
        moe.gate.assert_not_called()
        output = moe.experts.call_args.kwargs["topk_output"]
        self._assert_routes(moe, output.topk_ids, output.topk_weights)

    def test_megamoe_passes_gate_outputs_and_applies_scale_once(self):
        from sglang.srt.layers.moe import mega_moe

        moe, hidden, batch = self._glm_fixture()
        for fused_scale in (True, False):
            with (
                self.subTest(fused_scale=fused_scale),
                patch.object(
                    mega_moe, "run_mega_routed_experts", return_value=hidden
                ) as experts,
            ):
                moe.topk.topk_config.apply_routed_scaling_factor_on_output = fused_scale
                moe.experts.should_fuse_routed_scaling_factor_in_topk = fused_scale
                mega_moe._run_mega_routed(moe, hidden, batch, None, len(hidden))
                moe.gate.assert_not_called()
                _, _, ids, weights = experts.call_args.args
                self._assert_routes(
                    moe,
                    ids,
                    weights * experts.call_args.kwargs["routed_scaling_factor"],
                )

    def test_masked_graph_replay(self):
        weight = torch.zeros((256, 6144), device="cuda", dtype=torch.bfloat16)
        weight[:, 0] = 1
        offsets = torch.arange(-3, 5, dtype=torch.float32)
        weight[:8, 1] = offsets
        bias = torch.zeros(256, device="cuda")
        bias[:8] = 2  # Select experts 0..7, even when all sigmoid scores are zero.
        bases = torch.tensor([0, -64, -128], dtype=torch.float32)
        rows = torch.zeros((3, 6144), dtype=torch.bfloat16)
        rows[:, 0], rows[:, 1] = bases, 1
        hidden = rows.cuda()
        mask = torch.ones(3, device="cuda", dtype=torch.bool)

        scores = (bases[:, None] + offsets).sigmoid()
        sums = scores.sum(1, keepdim=True)
        self.assertTrue(0 < sums[1].item() < 1e-20)
        self.assertEqual(sums[2].item(), 0)
        expected = scores / torch.where(sums > 0, sums, 1) * 2.5

        def route():
            return bf16_mega_gate(
                hidden,
                weight,
                8,
                routed_scaling_factor=2.5,
                ep_rank=0,
                scoring_func="sigmoid",
                bias=bias,
                valid_token_mask=mask,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            route()
            with torch.cuda.graph(graph, stream=stream):
                weights, ids = route()
        torch.cuda.current_stream().wait_stream(stream)

        for validity in (
            [True, True, True],
            [True, False, True],
            [False, False, False],
            [True, True, True],
        ):
            with self.subTest(validity=validity):
                valid = torch.tensor(validity)
                batch = rows.clone()
                batch[~valid, 0] = float("nan")
                batch[~valid, 1] = float("inf")
                hidden.copy_(batch)
                mask.copy_(valid)
                graph.replay()
                row_ids, row_weights = ids.cpu(), weights.cpu()
                self.assertTrue((row_ids[~valid] == -1).all())
                self.assertTrue((row_weights[~valid] == 0).all())
                ordered_ids, order = row_ids[valid].sort(dim=1)
                self.assertTrue(
                    torch.equal(ordered_ids, torch.arange(8).expand_as(ordered_ids))
                )
                torch.testing.assert_close(
                    row_weights[valid].gather(1, order),
                    expected[valid],
                    rtol=3e-4,
                    atol=0,
                )

    def test_megamoe_expert_outputs_match_standard_routing(self):
        """Compare live W4A4 experts after both complete routing paths."""
        import deep_gemm
        import torch.distributed as dist

        from sglang.srt.environ import envs
        from sglang.srt.layers.moe import mega_moe
        from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
        from sglang.srt.layers.moe.utils import MoeA2ABackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE, MoEGate
        from sglang.srt.runtime_context import get_flags, get_parallel
        from sglang.test.test_utils import published_topology

        self.enterContext(published_topology(device="cuda"))
        self._prepare_expert_location()
        self.enterContext(get_flags().moe.override(a2a_backend=MoeA2ABackend.MEGAMOE))
        self.enterContext(
            envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK.override(32)
        )
        rendezvous = self.enterContext(tempfile.TemporaryDirectory())
        dist.init_process_group(
            "nccl",
            init_method=(Path(rendezvous) / "process-group").as_uri(),
            rank=0,
            world_size=1,
        )
        self.addCleanup(dist.destroy_process_group)
        self.addCleanup(mega_moe._MEGA_MOE_SYMM_BUFFER.clear)
        self.enterContext(
            get_parallel().override(
                moe_ep_group=SimpleNamespace(device_group=dist.group.WORLD),
                tp_group=None,
            )
        )
        self.enterContext(torch.inference_mode())
        generator = torch.Generator(device="cuda").manual_seed(20261114)
        config = SimpleNamespace(
            model_type="glm_moe_dsa",
            hidden_size=6144,
            moe_intermediate_size=2048,
            num_experts_per_tok=8,
            n_routed_experts=256,
            topk_method="noaux_tc",
        )
        gate = MoEGate(config, quant_config=None).cuda()
        gate.weight = torch.nn.Parameter(
            torch.randn(
                (256, 6144), device="cuda", dtype=torch.bfloat16, generator=generator
            )
            * 0.01,
            requires_grad=False,
        )
        # Keep the selected experts away from ties while exercising dense GEMM.
        gate.e_score_correction_bias.data.zero_()
        gate.e_score_correction_bias.data[:8].fill_(2.0)
        topk = TopK(
            8,
            layer_id=3,
            is_nextn=False,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=1,
            scoring_func="sigmoid",
            correction_bias=gate.e_score_correction_bias,
            routed_scaling_factor=1.5,
            output_format=TopKOutputFormat.STANDARD,
        )

        def packed_weights(rows, cols):
            packed = torch.randint(
                0,
                256,
                (256, rows, cols // 2),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            )
            scales = torch.full(
                (256, rows, cols // 32), 120, device="cuda", dtype=torch.uint8
            )
            return packed, scales

        l1, l2 = deep_gemm.transform_weights_for_mega_moe(
            packed_weights(4096, 6144), packed_weights(6144, 2048)
        )
        experts = SimpleNamespace(
            _mega_moe_weights_built=True,
            _mega_moe_nvfp4=False,
            _mega_moe_mma_type="mxf4xmxf4",
            num_experts=256,
            mega_l1_weights=l1,
            mega_l2_weights=l2,
            moe_runner_config=SimpleNamespace(swiglu_limit=None),
            should_fuse_routed_scaling_factor_in_topk=False,
        )
        moe = SimpleNamespace(
            config=config,
            gate=gate,
            topk=topk,
            experts=experts,
            is_hash=False,
            is_nextn=False,
            layer_id=3,
            num_fused_shared_experts=0,
            mega_shared_l1_weights=None,
            mega_shared_l2_weights=None,
            alt_stream=None,
            _forward_shared_experts=lambda _: None,
            routed_scaling_factor=1.5,
            routed_scaling_factor_fused_in_topk=False,
        )
        for mode, rows in (
            (ForwardMode.EXTEND, 16),
            (ForwardMode.DECODE, 4),
            (ForwardMode.TARGET_VERIFY, 8),
        ):
            hidden = torch.randn(
                (rows, 6144),
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            outputs, routes = [], []
            for enabled in (False, True):
                with (
                    envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(enabled),
                    patch.object(
                        deep_gemm, "bf16_mega_gate", wraps=deep_gemm.bf16_mega_gate
                    ) as fused,
                    patch.object(
                        deep_gemm,
                        "fp4_fp4_mega_moe",
                        wraps=deep_gemm.fp4_fp4_mega_moe,
                    ) as kernel,
                ):
                    output = DeepseekV2MoE.forward(
                        moe,
                        hidden,
                        SimpleNamespace(
                            forward_mode=mode, moe_num_token_non_padded=lambda: None
                        ),
                    )
                    outputs.append(output.clone())
                    buffer = kernel.call_args.args[3]
                    routes.append(
                        (
                            buffer.topk_idx[:rows].clone(),
                            buffer.topk_weights[:rows].clone(),
                        )
                    )
                    self.assertEqual(fused.call_count, int(enabled))
                    self.assertEqual(kernel.call_count, 1)
            (ids_a, weights_a), (ids_b, weights_b) = routes
            sorted_a, order_a = ids_a.sort(1)
            sorted_b, order_b = ids_b.sort(1)
            torch.testing.assert_close(sorted_a, sorted_b, rtol=0, atol=0)
            torch.testing.assert_close(
                weights_a.gather(1, order_a),
                weights_b.gather(1, order_b),
                rtol=3e-4,
                atol=2e-6,
            )
            a, b = (value.float() for value in outputs)
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue(torch.isfinite(b).all())
            self.assertGreater(a.norm().item(), 0)
            relative_l2 = (a - b).norm() / a.norm().clamp(min=1e-10)
            self.assertLess(relative_l2.item(), 0.003, mode.name)

        hidden = hidden[:4]
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE, moe_num_token_non_padded=lambda: None
        )
        graph_outputs = []
        for enabled in (False, True):
            with envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(enabled):
                eager = DeepseekV2MoE.forward(moe, hidden, batch).clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    DeepseekV2MoE.forward(moe, hidden, batch)
                    with torch.cuda.graph(graph, stream=stream):
                        replayed = DeepseekV2MoE.forward(moe, hidden, batch)
                torch.cuda.current_stream().wait_stream(stream)
                graph.replay()
                torch.testing.assert_close(replayed, eager)
                graph_outputs.append(replayed.clone().float())
        a, b = graph_outputs
        self.assertLess(((a - b).norm() / a.norm()).item(), 0.003)


if __name__ == "__main__":
    unittest.main()
