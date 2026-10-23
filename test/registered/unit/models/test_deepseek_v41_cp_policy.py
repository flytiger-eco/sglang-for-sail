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
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.srt.models.deepseek_v4 import MM_PAD_SHIFT_VALUE, DeepseekV4ForCausalLM
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseekV41CPPolicy(unittest.TestCase):
    def parallel_config(self, **kwargs):
        fields = dict(
            disaggregation_mode="prefill",
            dp_size=1,
            tp_size=4,
            attn_cp_size=1,
            dcp_size=1,
            enable_dp_attention=False,
            enable_prefill_cp=True,
            cp_strategy="interleave",
        )
        fields.update(kwargs)
        return NS(**fields)

    def test_prefill_cp_before_and_after_topology_resolution(self):
        for cp_size in (1, 4):
            for dp_attention in (False, True):
                with self.subTest(cp_size=cp_size, dp_attention=dp_attention):
                    self.assertTrue(
                        _dsv41_dspark_pd_parallelism_supported(
                            self.parallel_config(
                                attn_cp_size=cp_size, enable_dp_attention=dp_attention
                            )
                        )
                    )

    def test_rejects_unsupported_prefill_topologies(self):
        for override in (
            dict(cp_strategy="zigzag", attn_cp_size=4),
            dict(enable_prefill_cp=False, attn_cp_size=4),
            dict(dp_size=2),
            dict(dcp_size=2),
        ):
            with self.subTest(override=override):
                self.assertFalse(
                    _dsv41_dspark_pd_parallelism_supported(
                        self.parallel_config(**override)
                    )
                )

    def test_decode_dp_and_cp_restrictions(self):
        for dp_size in (1, 4):
            cfg = self.parallel_config(
                disaggregation_mode="decode",
                enable_prefill_cp=False,
                dp_size=dp_size,
                enable_dp_attention=dp_size > 1,
                tp_size=4,
            )
            self.assertTrue(_dsv41_dspark_pd_parallelism_supported(cfg))
            cfg.enable_prefill_cp = True
            self.assertFalse(_dsv41_dspark_pd_parallelism_supported(cfg))
            cfg.enable_prefill_cp = False
            cfg.attn_cp_size = 4
            self.assertFalse(_dsv41_dspark_pd_parallelism_supported(cfg))
        cfg = self.parallel_config(
            disaggregation_mode="decode",
            enable_prefill_cp=False,
            dp_size=2,
            tp_size=4,
            enable_dp_attention=True,
        )
        self.assertFalse(_dsv41_dspark_pd_parallelism_supported(cfg))

    def test_cp_breakable_graph_preserves_live_context_budget(self):
        for cp, backend, limit, expect_default in (
            (True, Backend.BREAKABLE, None, False),
            (False, Backend.BREAKABLE, None, True),
            (True, Backend.DISABLED, None, False),
            (True, Backend.BREAKABLE, 32768, False),
        ):
            with self.subTest(cp=cp, backend=backend, limit=limit):
                cfg = self.parallel_config(
                    enable_prefill_cp=cp,
                    disaggregation_mode="null",
                    enable_encoder_swa_bounded_replay=False,
                    enable_decoder_swa_bounded_replay=False,
                    speculative_algorithm=None,
                    enable_hisparse=False,
                    dsv4_attn_backend="flashmla",
                    enable_two_batch_overlap=False,
                    pp_size=1,
                    cuda_graph_config=NS(
                        prefill=NS(backend=backend, max_seq_len=limit)
                    ),
                )
                module = "sglang.srt.arg_groups.deepseek_v4_hook"
                with (
                    patch(module + ".resolving_view", return_value=cfg),
                    patch(
                        module + ".model_config_of",
                        return_value=NS(hf_config=NS(model_type="deepseek_v41")),
                    ),
                    patch(
                        "sglang.srt.layers.attention.dsv4.indexer_quant.dsv41_requires_int8_indexer",
                        return_value=False,
                    ),
                    patch(
                        "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                        return_value=False,
                    ),
                    patch(module + ".declare_resolution") as declare,
                    patch(
                        "sglang.srt.model_executor.cuda_graph_config.with_phase"
                    ) as update,
                ):
                    validate_deepseek_v41_features(cfg)
                    self.assertEqual(declare.call_count, int(expect_default))
                    self.assertEqual(update.call_count, int(expect_default))
                    if expect_default:
                        self.assertEqual(update.call_args.kwargs["max_seq_len"], 16384)

    def test_image_inputs_prepare_before_sharding_without_mutating_scheduler_ids(self):
        ids = torch.tensor([10, MM_PAD_SHIFT_VALUE + 7, 20])
        original = ids.clone()
        embeds = torch.randn(3, 4)
        prepare = Mock(return_value=embeds)
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
        prepare.assert_called_once_with(ids, batch)
        with self.assertRaisesRegex(ValueError, "Cannot combine"):
            DeepseekV4ForCausalLM.prepare_language_model_inputs(
                model, ids, batch, embeds
            )

    def test_decode_verify_idle_and_text_only_skip_image_preparation(self):
        ids = torch.tensor([10, 20])
        embeds = torch.randn(2, 4)
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.IDLE):
            prepare = Mock(side_effect=AssertionError("must not embed images"))
            model = NS(vision=object(), _prepare_mm_embeddings=prepare)
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
        cfg = NS(
            language_model_only=True,
            encoder_only=False,
            language_only=False,
            enable_prefix_mm_cache=False,
            enable_broadcast_mm_inputs_process=False,
            mm_enable_dp_encoder=False,
            disaggregation_mode="prefill",
        )
        args = NS(
            LANGUAGE_MODEL_ONLY_ARCHITECTURES=ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES
        )
        hf = NS(model_type="deepseek_v41", architectures=["DeepseekV4ForCausalLM"])
        with (
            patch("sglang.srt.arg_groups.model_hook.resolving_view", return_value=cfg),
            patch(
                "sglang.srt.arg_groups.model_hook.model_config_of",
                return_value=NS(hf_config=hf),
            ),
        ):
            handle_language_model_only(args)
            hf.model_type = "other"
            with self.assertRaisesRegex(ValueError, "incompatible"):
                handle_language_model_only(args)

    def test_low_ratio_captured_metadata_excludes_cp(self):
        module = "sglang.srt.layers.attention.deepseek_v4_backend"
        for cp_size, expected in ((1, True), (4, False)):
            with (
                patch(module + ".has_dense_fp4_indexer", return_value=True),
                patch(module + ".is_sm100_or_newer", return_value=True),
                patch(module + ".get_parallel", return_value=NS(attn_cp_size=cp_size)),
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
            max_context_size=None,
            can_replay_locally=lambda **kwargs: True,
            enable_lora=False,
            enable_cp_bcg_capture=True,
            prefill_cp_bcg_input=Mock(),
        )
        batch = NS(
            global_num_tokens_cpu=None,
            batch_size=1,
            input_ids=[1],
            input_embeds=None,
            replace_embeds=None,
            extend_prefix_lens_cpu=[0],
            forward_mode=ForwardMode.EXTEND,
            capture_hidden_mode=None,
            return_logprob=False,
        )
        with patch(
            "sglang.srt.model_executor.runner.prefill_cuda_graph_runner.is_cp_active",
            return_value=False,
        ):
            self.assertFalse(PrefillCudaGraphRunner.can_run_graph(runner, batch))
        runner.prefill_cp_bcg_input.select_replay_bucket_for_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
