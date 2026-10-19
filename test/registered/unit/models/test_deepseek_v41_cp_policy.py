"""CPU regressions for V4.1 CP configuration and model-boundary inputs."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.deepseek_v4_hook import (
    _dsv41_dspark_pd_parallelism_supported,
    validate_deepseek_v41_features,
)
from sglang.srt.arg_groups.model_hook import handle_language_model_only
from sglang.srt.arg_groups.overrides import resolution_result, resolving_view
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    with_phase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.models.deepseek_v4 import (
    MM_PAD_SHIFT_VALUE,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
)
from sglang.srt.runtime_context import get_context, get_flags, get_forward, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV41CPPolicy(CustomTestCase):
    def parallel_config(self, **kwargs):
        fields = dict(
            disaggregation_mode="prefill",
            dp_size=1,
            attn_dp_size=1,
            tp_size=4,
            attn_cp_size=1,
            dcp_size=1,
            ep_size=1,
            moe_dp_size=1,
            enable_prefill_cp=True,
            cp_strategy="interleave",
        )
        fields.update(kwargs)
        return resolving_view(ServerArgs(model_path="dummy", **fields))

    def test_prefill_cp_before_and_after_topology_resolution(self):
        for cp_size in (1, 4):
            with self.subTest(cp_size=cp_size):
                self.assertTrue(
                    _dsv41_dspark_pd_parallelism_supported(
                        self.parallel_config(attn_cp_size=cp_size)
                    )
                )

    def test_rejects_unsupported_prefill_topologies(self):
        for override in (
            dict(cp_strategy="zigzag", attn_cp_size=4),
            dict(enable_prefill_cp=False, attn_cp_size=4),
            dict(disaggregation_mode="null"),
            dict(attn_dp_size=2),
            dict(dp_size=2),
            dict(dcp_size=2),
        ):
            with self.subTest(override=override):
                self.assertFalse(
                    _dsv41_dspark_pd_parallelism_supported(
                        self.parallel_config(**override)
                    )
                )

    def test_decode_attention_dp_is_independent_of_system_dp(self):
        for attn_dp_size in (1, 4):
            base = dict(
                disaggregation_mode="decode",
                enable_prefill_cp=False,
                attn_dp_size=attn_dp_size,
            )
            self.assertTrue(
                _dsv41_dspark_pd_parallelism_supported(self.parallel_config(**base))
            )
            for override in (
                dict(enable_prefill_cp=True),
                dict(attn_cp_size=4),
                dict(dp_size=2),
            ):
                with self.subTest(attn_dp_size=attn_dp_size, override=override):
                    self.assertFalse(
                        _dsv41_dspark_pd_parallelism_supported(
                            self.parallel_config(**(base | override))
                        )
                    )
        self.assertFalse(
            _dsv41_dspark_pd_parallelism_supported(
                self.parallel_config(
                    disaggregation_mode="decode",
                    enable_prefill_cp=False,
                    attn_dp_size=2,
                )
            )
        )

    def test_cp_breakable_graph_preserves_live_context_budget(self):
        for cp, backend, limit, expected in (
            (True, Backend.BREAKABLE, None, None),
            (False, Backend.BREAKABLE, None, 16384),
            (True, Backend.FULL, None, 16384),
            (True, Backend.DISABLED, None, None),
            (True, Backend.BREAKABLE, 32768, 32768),
        ):
            with self.subTest(cp=cp, backend=backend, limit=limit):
                graph = with_phase(
                    CudaGraphConfig(), Phase.PREFILL, backend=backend, max_seq_len=limit
                )
                args = ServerArgs(
                    model_path="dummy",
                    tp_size=4,
                    enable_prefill_cp=cp,
                    cp_strategy="interleave",
                    cuda_graph_config=graph,
                    dsv4_attn_backend="flashmla",
                )
                with (
                    patch(
                        "sglang.srt.arg_groups.deepseek_v4_hook.model_config_of",
                        return_value=NS(hf_config=NS(model_type="deepseek_v41")),
                    ),
                    patch(
                        "sglang.srt.layers.attention.dsv4.indexer_quant.is_sail",
                        return_value=False,
                    ),
                    patch(
                        "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                        return_value=False,
                    ),
                ):
                    validate_deepseek_v41_features(args)
                self.assertEqual(
                    resolution_result(args, "cuda_graph_config").prefill.max_seq_len,
                    expected,
                )
                self.assertEqual(args.cuda_graph_config.prefill.max_seq_len, limit)

    def test_image_inputs_prepare_without_mutating_scheduler_ids(self):
        ids = torch.tensor([10, MM_PAD_SHIFT_VALUE + 7, 20])
        original = ids.clone()
        embeds = torch.randn(3, 4)

        def prepare(full_ids, batch):
            torch.testing.assert_close(full_ids, original)
            self.assertIs(batch.input_ids, ids)
            return embeds

        model = NS(
            vision=object(),
            config=NS(image_token_id=42),
            _prepare_mm_embeddings=prepare,
        )
        batch = NS(
            forward_mode=ForwardMode.EXTEND,
            contains_mm_inputs=lambda: True,
            input_ids=ids,
        )
        output_ids, output_embeds = DeepseekV4ForCausalLM.prepare_language_model_inputs(
            model, ids, batch
        )
        torch.testing.assert_close(ids, original)
        torch.testing.assert_close(output_ids, torch.tensor([10, 42, 20]))
        self.assertIs(output_embeds, embeds)
        with self.assertRaisesRegex(ValueError, "Cannot combine"):
            DeepseekV4ForCausalLM.prepare_language_model_inputs(
                model, ids, batch, embeds
            )

    def test_decode_verify_idle_and_text_only_skip_image_preparation(self):
        ids = torch.tensor([10, 20])
        embeds = torch.randn(2, 4)
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.IDLE):
            with self.subTest(mode=mode):
                model = NS(
                    vision=object(),
                    _prepare_mm_embeddings=Mock(
                        side_effect=AssertionError("must not embed images")
                    ),
                )
                batch = NS(forward_mode=mode, contains_mm_inputs=lambda: True)
                output = DeepseekV4ForCausalLM.prepare_language_model_inputs(
                    model, ids, batch, embeds
                )
                self.assertIs(output[0], ids)
                self.assertIs(output[1], embeds)
        output = DeepseekV4ForCausalLM.prepare_language_model_inputs(
            NS(vision=None), ids, NS(forward_mode=ForwardMode.EXTEND), embeds
        )
        self.assertIs(output[0], ids)
        self.assertIs(output[1], embeds)

    def test_language_model_only_pd_allowed_only_for_v41(self):
        for mode in ("prefill", "decode"):
            args = ServerArgs(
                model_path="dummy", language_model_only=True, disaggregation_mode=mode
            )
            hf = NS(model_type="deepseek_v41", architectures=["DeepseekV4ForCausalLM"])
            with patch(
                "sglang.srt.arg_groups.model_hook.model_config_of",
                return_value=NS(hf_config=hf),
            ):
                handle_language_model_only(args)
                hf.model_type = "other"
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    handle_language_model_only(args)

    def test_engram_image_mask_is_disabled_for_language_model_only(self):
        """Text-only workers must apply Engram even to vocabulary image IDs."""
        ids = torch.tensor([10, 42])
        hidden = torch.zeros(2, 2, 4)

        def engram(residual, hash_ids, batch, **kwargs):
            return residual + 1

        engram.layer_hash_index = 0
        layer = NS(
            engram=engram,
            hc_cfg=None,
            hc_boundary_fused=False,
            forward_hc_pre_from_prev=lambda **kwargs: kwargs["state"],
        )
        with (
            get_context().override_server_args(),
            get_forward().scoped(sp_active=False),
        ):
            for language_model_only, has_mm, expected_image in (
                (False, True, 0),
                (True, True, 1),
                (False, False, 1),
            ):
                with self.subTest(
                    language_model_only=language_model_only, has_mm=has_mm
                ):
                    model = NS(
                        pp_group=NS(world_size=1),
                        config=NS(
                            model_type="deepseek_v41",
                            vision_n_layers=1,
                            image_token_id=42,
                            language_model_only=language_model_only,
                        ),
                        engram_hasher=lambda *args: torch.zeros(
                            2, 1, dtype=torch.int64
                        ),
                        late_layer_start=None,
                        start_layer=0,
                        end_layer=1,
                        layers=[layer],
                    )
                    batch = NS(
                        input_ids=ids,
                        forward_mode=ForwardMode.DECODE,
                        contains_mm_inputs=lambda: has_mm,
                    )
                    result, _, tail = DeepseekV4Model._forward_layers_hc_pre_from_prev(
                        model, torch.arange(2), hidden, batch, ids, ids, False, []
                    )
                    torch.testing.assert_close(result[0], torch.ones_like(hidden[0]))
                    torch.testing.assert_close(
                        result[1], torch.full_like(hidden[1], expected_image)
                    )
                    self.assertIsNone(tail)

    def test_deepep_text_only_does_not_disable_shared_expert_fusion(self):
        with (
            get_context().override_server_args(enforce_shared_experts_fusion=True),
            get_flags().moe.override(a2a_backend=MoeA2ABackend.DEEPEP),
        ):
            for language_model_only in (False, True):
                hf = NS(
                    model_type="deepseek_v41",
                    vision_n_layers=1,
                    language_model_only=language_model_only,
                    n_shared_experts=1,
                )
                reason = DeepseekV4ForCausalLM.shared_experts_fusion_disable_reason(
                    hf, None
                )
                if language_model_only:
                    self.assertIsNone(reason)
                else:
                    self.assertIn("vision routing with DeepEP", reason)

    def test_low_ratio_captured_metadata_excludes_cp(self):
        module = "sglang.srt.layers.attention.deepseek_v4_backend"
        with (
            get_context().override_server_args(tp_size=4),
            patch(module + ".has_dense_fp4_indexer", return_value=True),
            patch(module + ".is_sm100_or_newer", return_value=True),
            patch(
                "sglang.srt.layers.attention.dsv4.indexer_quant.is_sail",
                return_value=False,
            ),
        ):
            for cp_size, expected in ((1, True), (4, False)):
                with get_parallel().override(
                    attn_cp_size=cp_size, attn_tp_size=4 // cp_size
                ):
                    self.assertEqual(
                        DeepseekV4AttnBackend.low_ratio_prefill_graph.fget(
                            NS(low_ratios=[1, 2])
                        ),
                        expected,
                    )

    def test_cp_graph_rejects_non_cp_batch_before_bucket_selection(self):
        runner = NS(
            _has_inactive_dp_rank=lambda batch: False,
            enable_cp_bcg_capture=True,
            prefill_cp_bcg_input=Mock(),
        )
        batch = NS(
            global_num_tokens_cpu=None,
            input_ids=torch.tensor([1]),
            forward_mode=ForwardMode.EXTEND,
        )
        with get_context().override_server_args(enable_prefill_cp=False):
            self.assertFalse(PrefillCudaGraphRunner.can_run_graph(runner, batch))
        runner.prefill_cp_bcg_input.select_replay_bucket_for_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
