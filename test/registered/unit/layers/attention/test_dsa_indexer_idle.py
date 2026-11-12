import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _make_eager_idle_topk_result,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsa.dsa_indexer"


class TestDSAIndexerIdle(unittest.TestCase):
    def test_builds_invalid_topk_rows_for_padded_idle_tokens(self):
        result = _make_eager_idle_topk_result(
            torch.empty((8, 16)),
            index_topk=4,
            return_indices=True,
        )

        self.assertEqual(result.shape, (8, 4))
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.all(result == -1))

    def test_returns_none_when_indices_are_not_consumed(self):
        self.assertIsNone(
            _make_eager_idle_topk_result(
                torch.empty((8, 16)),
                index_topk=4,
                return_indices=False,
            )
        )

    def test_eager_idle_short_circuits_before_paged_mqa(self):
        indexer = SimpleNamespace(index_topk=4)
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)

        with (
            patch(f"{_INDEXER}._is_cuda", True),
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=False),
            patch(
                f"{_INDEXER}._broadcast_indexer_topk_from_rank0",
                side_effect=lambda result: result,
            ),
            patch(
                f"{_INDEXER}.maybe_capture_indexer_topk",
                side_effect=lambda _layer_id, result: result,
            ),
        ):
            result = Indexer.forward_cuda(
                indexer,
                x=torch.empty((8, 16)),
                q_lora=torch.empty((8, 16)),
                positions=torch.empty((8,), dtype=torch.int64),
                forward_batch=batch,
                layer_id=3,
            )

        self.assertEqual(result.shape, (8, 4))
        self.assertTrue(torch.all(result == -1))


class TestDSAIndexerDCPGather(CustomTestCase):
    def test_empty_and_nonempty_shards_keep_byte_width(self):
        """Empty shards still join the gather with full key/scale byte columns."""
        for rows, global_rows in [(0, 3), (2, 16)]:
            with self.subTest(rows=rows):
                keys = torch.arange(rows * 128).to(torch.int8).view(rows, 128)
                scales = torch.arange(rows, dtype=torch.float32).view(rows, 1)
                local_bytes = torch.cat(
                    [keys.view(torch.uint8), scales.view(torch.uint8)], dim=1
                )
                expected = torch.arange(global_rows * 132).to(torch.uint8)
                expected = expected.view(global_rows, 132)
                expected[7::8] = local_bytes
                seq_lens = torch.tensor([global_rows], dtype=torch.int64)
                pool = SimpleNamespace(
                    get_index_k_scale_buffer=lambda *args: (keys, scales)
                )
                metadata = SimpleNamespace(get_indexer_seq_len=lambda: seq_lens)

                def gather(payload, lengths):
                    torch.testing.assert_close(payload, local_bytes)
                    torch.testing.assert_close(lengths, seq_lens)
                    return expected

                with (
                    patch(
                        f"{_INDEXER}.get_parallel",
                        return_value=SimpleNamespace(attn_dcp_size=8, attn_dcp_rank=7),
                    ),
                    patch(f"{_INDEXER}.get_token_to_kv_pool", return_value=pool),
                    patch(f"{_INDEXER}.dcp_gather_request_rows", side_effect=gather),
                ):
                    got_keys, got_scales = Indexer._dcp_gather_index_k(
                        0, metadata, torch.zeros((1, 1), dtype=torch.int32), seq_lens
                    )

                torch.testing.assert_close(got_keys, expected[:, :128])
                torch.testing.assert_close(got_scales, expected[:, 128:])


if __name__ == "__main__":
    unittest.main()
