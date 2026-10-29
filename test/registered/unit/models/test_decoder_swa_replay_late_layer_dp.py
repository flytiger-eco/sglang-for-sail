import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models.deepseek_common import decoder_swa_bounded_replay as replay
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MODULE = "sglang.srt.models.deepseek_common.decoder_swa_bounded_replay"


def _make_batch(counts_len=2):
    return SimpleNamespace(
        global_num_tokens_cpu=[0] * counts_len,
        global_num_tokens_gpu=torch.zeros(counts_len, dtype=torch.int64),
        global_num_tokens_padded_cpu=None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_tokens_for_logprob_gpu=None,
        global_dp_buffer_len=0,
        dp_padding_mode=None,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        global_num_token_non_padded=None,
        global_num_token_non_padded_cpu=7,
        num_token_non_padded=None,
    )


class TestEnterLateLayerDpCounts(CustomTestCase):
    def _enter(self, counts):
        batch = _make_batch(len(counts))
        with patch.object(replay, "set_dp_buffer_len_from_batch"), patch.object(
            replay, "get_parallel", lambda: SimpleNamespace(attn_dp_rank=0)
        ):
            saved = replay.enter_late_layer_dp(batch, counts)
        return batch, saved

    def test_uniform_counts_fill_the_device_tensor_in_place(self):
        batch, _ = self._enter([5, 5])

        self.assertEqual(batch.global_num_tokens_gpu.tolist(), [5, 5])
        self.assertEqual(batch.global_num_tokens_gpu.dtype, torch.int64)
        self.assertEqual(batch.global_dp_buffer_len, 10)
        self.assertEqual(batch.global_num_tokens_for_logprob_gpu.data_ptr(),
                         batch.global_num_tokens_gpu.data_ptr())

    def test_mixed_counts_are_staged_without_a_host_copy(self):
        staged = Mock(
            side_effect=lambda counts, *, dtype, device: torch.tensor(
                counts, dtype=dtype, device=device
            )
        )
        with patch.object(replay, "async_h2d", staged):
            batch, _ = self._enter([3, 5])

        self.assertEqual(staged.call_count, 1)
        self.assertEqual(batch.global_num_tokens_gpu.tolist(), [3, 5])

    def test_exit_restores_the_original_counts(self):
        batch = _make_batch(2)
        original_gpu = batch.global_num_tokens_gpu
        with patch.object(replay, "set_dp_buffer_len_from_batch"), patch.object(
            replay, "get_parallel", lambda: SimpleNamespace(attn_dp_rank=0)
        ):
            saved = replay.enter_late_layer_dp(batch, [4, 4])
            replay.exit_late_layer_dp(batch, saved)

        self.assertIs(batch.global_num_tokens_gpu, original_gpu)
        self.assertEqual(batch.global_num_tokens_gpu.tolist(), [0, 0])
        self.assertEqual(batch.global_dp_buffer_len, 0)


if __name__ == "__main__":
    unittest.main()
