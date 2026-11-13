import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.moe.mega_gate import bf16_mega_gate
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=30, suite="stage-b-test-1-gpu-ppu-m890")


class TestGlmSigmoidMegaGate(CustomTestCase):
    def test_deepep_forward_passes_sigmoid_routes_to_experts(self):
        from sglang.srt.configs.shared_experts import SharedExpertsConfig
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
        from sglang.srt.layers.moe.utils import MoeA2ABackend
        from sglang.srt.models import deepseek_v2
        from sglang.srt.runtime_context import get_model
        from sglang.test.test_utils import published_topology

        self.enterContext(published_topology(device="cuda"))
        self.enterContext(get_model().override(shared_experts=SharedExpertsConfig()))
        self.enterContext(envs.SGLANG_OPT_DEEPGEMM_MEGA_GATE.override(True))
        self.enterContext(
            patch.object(
                deepseek_v2, "get_moe_a2a_backend", return_value=MoeA2ABackend.DEEPEP
            )
        )
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
        experts = Mock(return_value=hidden)
        moe = SimpleNamespace(
            config=SimpleNamespace(model_type="glm_moe_dsa"),
            layer_id=0,
            is_nextn=False,
            is_hash=False,
            gate=gate,
            topk=topk,
            experts=experts,
            alt_stream=None,
            _fuse_shared_experts_inside_sbo=False,
            num_fused_shared_experts=0,
            _forward_shared_experts=lambda _: None,
            routed_scaling_factor=1.5,
            routed_scaling_factor_fused_in_topk=True,
        )
        bound = torch.tensor([1], dtype=torch.int32, device="cuda")
        with patch.object(topk, "routing_inputs", return_value=(bound, None)):
            deepseek_v2.DeepseekV2MoE.forward_deepep(moe, hidden, None)
        gate.assert_not_called()
        output = experts.call_args.kwargs["topk_output"]
        ordered_ids, order = output.topk_ids[0].sort()
        torch.testing.assert_close(ordered_ids, torch.arange(8, device="cuda"))
        scores = weight[:8, 0].float().sigmoid()
        torch.testing.assert_close(
            output.topk_weights[0, order], scores / scores.sum() * 1.5
        )
        self.assertTrue((output.topk_ids[1] == -1).all())
        self.assertTrue((output.topk_weights[1] == 0).all())

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


if __name__ == "__main__":
    unittest.main()
