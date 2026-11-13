import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.qsa import mqa
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestQSADecodeHeadPadding(unittest.TestCase):
    def test_platform_head_alignment(self):
        cases = (
            (False, False, {1: 8, 8: 8, 9: 16, 16: 16, 24: 24, 32: 32}),
            (True, False, {1: 16, 8: 16, 9: 16, 16: 16, 24: 32, 32: 32}),
            (False, True, {1: 16, 8: 16, 9: 16, 16: 16, 24: 32, 32: 32}),
        )
        for hip, ppu, expected in cases:
            with (
                self.subTest(hip=hip, ppu=ppu),
                patch.object(mqa, "is_hip", return_value=hip),
                patch.object(mqa, "is_sail", return_value=ppu),
            ):
                for heads, padded in expected.items():
                    self.assertEqual(mqa._qsa_decode_padded_heads(heads), padded)

    def check_decode_wrapper(self, heads, expected_heads, *, ppu, reuse=False):
        q = (torch.arange(2 * heads * 128) % 7 - 3).reshape(2, heads, 128)
        q = q.to(torch.bfloat16)
        cache = (torch.arange(4 * 8 * 128) % 11 - 5).reshape(4, 8, 1, 128)
        cache = cache.to(torch.bfloat16)
        page_table = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
        lengths = torch.tensor([5, 13], dtype=torch.int32)
        scale = 3.0

        def kernel(q_kernel, k_kernel, pages, context_lens, logits, score_scale):
            self.assertEqual(q_kernel.shape, (2, 1, expected_heads, 128))
            self.assertEqual(score_scale, scale)
            torch.testing.assert_close(q_kernel[:, 0, :heads], q)
            self.assertEqual(torch.count_nonzero(q_kernel[:, 0, heads:]).item(), 0)
            if reuse:
                self.assertEqual(q_kernel.data_ptr(), q.data_ptr())
            logits.copy_(
                mqa.torch_qsa_mqa_decode(
                    q_kernel[:, 0], k_kernel, pages, context_lens, 20, score_scale
                )
            )

        factory = Mock(return_value=kernel)
        with (
            patch.object(mqa, "is_hip", return_value=False),
            patch.object(mqa, "is_sail", return_value=ppu),
            patch.object(mqa, "HAS_TILELANG", True),
            patch.object(mqa, "_decode_ctas_per_row", return_value=1),
            patch.object(mqa, "_tilelang_qsa_mqa_decode_kernel", factory, create=True),
        ):
            actual = mqa.tilelang_qsa_mqa_decode(
                q, cache, page_table, lengths, 20, scale
            )
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["heads"], expected_heads)
        self.assertEqual(factory.call_args.kwargs["head_dim"], 128)
        self.assertEqual(factory.call_args.kwargs["page_size"], 8)
        reference = mqa.torch_qsa_mqa_decode(q, cache, page_table, lengths, 20, scale)
        torch.testing.assert_close(actual, reference)

    def test_ppu_wrapper_zero_padding_preserves_head_sum(self):
        for heads, expected in ((8, 16), (24, 32)):
            with self.subTest(heads=heads):
                self.check_decode_wrapper(heads, expected, ppu=True)

    def test_cuda_keeps_eight_head_kernel(self):
        self.check_decode_wrapper(8, 8, ppu=False, reuse=True)

    def test_prepared_ppu_query_needs_no_second_padding_copy(self):
        self.check_decode_wrapper(16, 16, ppu=True, reuse=True)

    def run_indexer(self, mode):
        decode, verify, draft = mode
        forward_mode = SimpleNamespace(
            is_decode=lambda: decode,
            is_target_verify=lambda: verify,
            is_draft_extend_v2=lambda: draft,
        )
        positions = torch.arange(2)
        lengths = torch.tensor([8, 16], dtype=torch.int32)
        expected = torch.zeros((2, 4), dtype=torch.int32)

        def project(hidden, rope_positions, *, q_heads_padded, **kwargs):
            return (
                torch.zeros((2, q_heads_padded or 8, 128)),
                torch.zeros((2, 1, 128)),
                True,
            )

        indexer = SimpleNamespace(
            layer_id=0,
            index_n_heads=8,
            project_qk=Mock(side_effect=project),
            update_key_state_and_compress=Mock(),
            select_decode_tokens=Mock(return_value=expected),
            select_prefill_tokens=Mock(return_value=expected),
        )
        metadata = SimpleNamespace(
            decode_logical_positions=positions,
            get_token_to_batch_idx=lambda: torch.arange(2),
            pending_ring_slots=torch.zeros(2, dtype=torch.int64),
            token_to_kv_pool=object(),
            get_decode_mqa_inputs=lambda _: (object(), object(), lengths, 20),
            get_seqlens_int32=lambda: lengths,
            get_prefill_mqa_inputs=lambda *_: (
                object(),
                object(),
                object(),
                lengths,
            ),
            prefill_max_row_keys=lambda _: 20,
        )
        batch = SimpleNamespace(
            forward_mode=forward_mode, positions=positions, seq_lens_cpu=None
        )
        result = QSAIndexer._forward_impl(
            indexer, torch.zeros((2, 4)), positions, batch, metadata
        )
        self.assertIs(result, expected)
        return indexer

    def test_decode_verify_and_draft_use_same_alignment_as_kernel(self):
        for hip, ppu, expected in (
            (False, False, 8),
            (True, False, 16),
            (False, True, 16),
        ):
            for mode in (
                (True, False, False),
                (False, True, False),
                (False, False, True),
            ):
                with (
                    self.subTest(hip=hip, ppu=ppu, mode=mode),
                    patch.object(mqa, "is_hip", return_value=hip),
                    patch.object(mqa, "is_sail", return_value=ppu),
                ):
                    indexer = self.run_indexer(mode)
                    self.assertEqual(
                        indexer.project_qk.call_args.kwargs["q_heads_padded"], expected
                    )
                    self.assertEqual(
                        indexer.select_decode_tokens.call_args.args[0].shape[1],
                        expected,
                    )

    def test_prefill_does_not_request_decode_head_padding(self):
        with (
            patch.object(mqa, "is_hip", return_value=False),
            patch.object(mqa, "is_sail", return_value=True),
        ):
            indexer = self.run_indexer((False, False, False))
        self.assertIsNone(indexer.project_qk.call_args.kwargs["q_heads_padded"])
        indexer.select_decode_tokens.assert_not_called()


if __name__ == "__main__":
    unittest.main()
