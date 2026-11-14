import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import dsa_backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSAPagedMQAMetadata(unittest.TestCase):
    def make_backend(self, pool):
        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.num_q_heads = 32
        backend.indexer_n_heads = 32
        backend.indexer_head_dim = 128
        backend.token_to_kv_pool = pool
        return backend

    def test_fp4_uses_packed_width_without_changing_logical_dimension(self):
        backend = self.make_backend(SimpleNamespace(use_fp4_indexer=True))
        with (
            patch.object(dsa_backend, "get_device_sm", return_value=90),
            envs.SGLANG_SAIL_BF16_INDEXER.override(False),
        ):
            self.assertEqual(
                backend._paged_mqa_schedule_metadata_extra(),
                (1, 32, 64, torch.float8_e4m3fn),
            )
        self.assertEqual(backend.indexer_head_dim, 128)

    def test_non_fp4_pool_keeps_width_when_global_fp4_is_requested(self):
        # KPool and other pool fallbacks can keep FP8/INT8 despite the flag.
        backend = self.make_backend(SimpleNamespace(use_fp4_indexer=False))
        with (
            patch.object(dsa_backend, "get_device_sm", return_value=90),
            envs.SGLANG_OPT_USE_FP4_INDEXER_CACHE.override(True),
            envs.SGLANG_SAIL_BF16_INDEXER.override(False),
        ):
            self.assertEqual(
                backend._paged_mqa_schedule_metadata_extra(),
                (1, 32, 128, torch.float8_e4m3fn),
            )

    def test_actual_fp4_pool_does_not_depend_on_a_later_flag_change(self):
        backend = self.make_backend(SimpleNamespace(use_fp4_indexer=True))
        with (
            patch.object(dsa_backend, "get_device_sm", return_value=90),
            envs.SGLANG_OPT_USE_FP4_INDEXER_CACHE.override(False),
            envs.SGLANG_SAIL_BF16_INDEXER.override(False),
        ):
            self.assertEqual(
                backend._paged_mqa_schedule_metadata_extra(),
                (1, 32, 64, torch.float8_e4m3fn),
            )

    def test_bf16_bypass_has_no_legal_dtype_hint_and_keeps_logical_width(self):
        # The realigned indexer contract only accepts torch.int8 /
        # torch.float8_e4m3fn as the qk_dtype item (docs/api/indexer.md), so a
        # BF16-indexer bypass cannot be expressed in the hint; the hint stays
        # contract-legal and only the width semantics are exercised here.
        # BF16-indexer configurations remain out of scope for this path.
        backend = self.make_backend(SimpleNamespace(use_fp4_indexer=False))
        with (
            patch.object(dsa_backend, "get_device_sm", return_value=90),
            envs.SGLANG_SAIL_BF16_INDEXER.override(True),
        ):
            self.assertEqual(
                backend._paged_mqa_schedule_metadata_extra(),
                (1, 32, 128, torch.float8_e4m3fn),
            )

    def test_pool_without_fp4_attribute_keeps_existing_layout(self):
        backend = self.make_backend(SimpleNamespace())
        with (
            patch.object(dsa_backend, "get_device_sm", return_value=80),
            envs.SGLANG_SAIL_BF16_INDEXER.override(False),
        ):
            self.assertEqual(
                backend._paged_mqa_schedule_metadata_extra(), (1, 32, 128, torch.int8)
            )

    def test_ppu_builder_forwards_packed_width_to_legacy_planner(self):
        backend = self.make_backend(SimpleNamespace(use_fp4_indexer=True))
        context_lens = torch.tensor([[128], [256]], dtype=torch.int32)
        schedule = object()
        planner = Mock(return_value=schedule)
        deep_gemm = SimpleNamespace(
            get_num_sms=Mock(return_value=84),
            get_paged_mqa_logits_metadata=planner,
        )
        with (
            patch.object(dsa_backend, "_is_ppu", True),
            patch.object(dsa_backend, "get_device_sm", return_value=90),
            patch.object(dsa_backend, "deep_gemm", deep_gemm, create=True),
            envs.SGLANG_SAIL_BF16_INDEXER.override(False),
        ):
            result = backend._build_paged_mqa_schedule_metadata(context_lens)
        self.assertIs(result, schedule)
        planner.assert_called_once_with(
            context_lens,
            64,
            84,
            metadata_extra=(1, 32, 64, torch.float8_e4m3fn),
        )

    def test_non_ppu_builder_does_not_add_legacy_metadata_extra(self):
        backend = self.make_backend(SimpleNamespace(use_fp4_indexer=False))
        context_lens = torch.tensor([[128]], dtype=torch.int32)
        planner = Mock()
        deep_gemm = SimpleNamespace(
            get_num_sms=Mock(return_value=84),
            get_paged_mqa_logits_metadata=planner,
        )
        with (
            patch.object(dsa_backend, "_is_ppu", False),
            patch.object(dsa_backend, "deep_gemm", deep_gemm, create=True),
        ):
            backend._build_paged_mqa_schedule_metadata(context_lens)
        planner.assert_called_once_with(context_lens, 64, 84)

    def test_graph_refresh_uses_packed_width_and_keeps_schedule_storage(self):
        backend = self.make_backend(SimpleNamespace(use_fp4_indexer=True))
        context_lens = torch.tensor([[128], [256]], dtype=torch.int32)
        captured_schedule = torch.full((4, 4), 9, dtype=torch.int32)
        original_ptr = captured_schedule.data_ptr()
        schedule = torch.full((2, 4), 3, dtype=torch.int32)
        metadata = SimpleNamespace(paged_mqa_schedule_metadata=captured_schedule)
        planner = Mock(return_value=schedule)
        deep_gemm = SimpleNamespace(
            get_num_sms=Mock(return_value=84),
            get_paged_mqa_logits_metadata=planner,
        )
        with (
            patch.object(dsa_backend, "_is_ppu", True),
            patch.object(dsa_backend, "get_device_sm", return_value=90),
            patch.object(dsa_backend, "deep_gemm", deep_gemm, create=True),
            envs.SGLANG_SAIL_BF16_INDEXER.override(False),
        ):
            backend._refresh_paged_mqa_schedule_metadata(metadata, context_lens)
        planner.assert_called_once_with(
            context_lens,
            64,
            84,
            metadata_extra=(1, 32, 64, torch.float8_e4m3fn),
        )
        self.assertIs(metadata.paged_mqa_schedule_metadata, captured_schedule)
        self.assertEqual(captured_schedule.data_ptr(), original_ptr)
        self.assertTrue(torch.equal(captured_schedule[:2], schedule))
        self.assertEqual(torch.count_nonzero(captured_schedule[2:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
