"""Regression for valid DFLASH verify rows being mistaken for graph padding.

The real attention boundary is replaced with a row-preserving CPU kernel so
that this checks query preservation without loading weights or requiring GPU
attention numerics.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import flashmla_backend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.test_utils import CustomTestCase


class TestFlashMLADFlashBlockWidth(CustomTestCase):
    def test_valid_verify_rows_survive_padding_removal(self):
        cases = (
            (SpeculativeAlgorithm.DFLASH, 1, 8, 0),
            (SpeculativeAlgorithm.DFLASH, 3, 8, 8),
            (SpeculativeAlgorithm.DFLASH, 2, 4, 0),
            (SpeculativeAlgorithm.DSPARK, 1, 8, 0),
            (SpeculativeAlgorithm.EAGLE, 2, 2, 0),
        )
        for algorithm, batch_size, block_size, padding in cases:
            with self.subTest(algorithm=algorithm, batch=batch_size, padding=padding):
                backend = object.__new__(flashmla_backend.FlashMLABackend)
                backend.speculative_num_steps = 1
                backend.num_draft_tokens = block_size
                backend.kv_lora_rank = 8
                backend.is_fp8_kvcache = False
                backend.token_to_kv_pool = SimpleNamespace(
                    get_key_buffer=lambda _: torch.zeros((64, 1, 16))
                )
                backend.forward_metadata = SimpleNamespace(
                    block_kv_indices=torch.zeros((batch_size, 1), dtype=torch.int32),
                    flashmla_metadata=None,
                    num_splits=None,
                )
                layer = SimpleNamespace(
                    layer_id=0,
                    tp_q_head_num=2,
                    head_dim=16,
                    v_head_dim=8,
                    scaling=0.25,
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    spec_algorithm=algorithm,
                    spec_info=SimpleNamespace(draft_token_num=block_size),
                    batch_size=batch_size,
                    seq_lens=torch.full((batch_size,), 16, dtype=torch.int32),
                )
                valid = batch_size * block_size
                rows = valid + padding
                marker = torch.arange(1, rows + 1, dtype=torch.float32)
                query = marker[:, None, None].expand(rows, 2, 16).contiguous()

                def attention_boundary(**kwargs):
                    q = kwargs["q"]
                    return q[..., :1].expand(*q.shape[:-1], 8).contiguous(), None

                with patch.object(
                    flashmla_backend,
                    "flash_mla_with_kvcache",
                    side_effect=attention_boundary,
                ):
                    output = backend.forward_extend(
                        query, None, None, layer, batch, save_kv_cache=False
                    )
                expected = torch.zeros((rows, 16))
                expected[:valid] = marker[:valid, None]
                torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
