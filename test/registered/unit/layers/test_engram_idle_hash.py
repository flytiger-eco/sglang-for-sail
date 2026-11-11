"""The engram hasher skips DP-attention pad rows instead of rejecting idle batches."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.engram import EngramHasher
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

NGRAM, N_PRIMES, N_HEADS = 4, 3, 2


def _hasher() -> EngramHasher:
    # The idle path only reads the n-gram width, the hash descriptors and the
    # history table, so the tokenizer-backed init can be skipped.
    hasher = EngramHasher.__new__(EngramHasher)
    torch.nn.Module.__init__(hasher)
    hasher.history = torch.zeros(2, NGRAM - 1, dtype=torch.int64)
    hasher.max_ngram_size = NGRAM
    hasher.primes = torch.arange(1, N_PRIMES + 1, dtype=torch.int64)
    hasher.offsets = torch.zeros(N_PRIMES, N_HEADS, dtype=torch.int64)
    return hasher


class TestEngramIdleHash(unittest.TestCase):
    def test_idle_pad_rows_hash_to_zeros_without_committing(self):
        """An idle attention-DP rank is padded to the global token count; those rows
        are masked downstream, so they hash to zeros and leave the history alone."""
        hasher = _hasher()
        history_before = hasher.history.clone()

        out = hasher.forward(
            torch.zeros(7, dtype=torch.int64),
            SimpleNamespace(forward_mode=ForwardMode.IDLE),
        )

        self.assertEqual(out.shape, (7, N_PRIMES, N_HEADS))
        self.assertEqual(int(out.abs().sum()), 0)
        torch.testing.assert_close(hasher.history, history_before)

    def test_zero_token_batch_hashes_to_an_empty_tensor(self):
        hasher = _hasher()

        out = hasher.forward(
            torch.zeros(0, dtype=torch.int64),
            SimpleNamespace(forward_mode=ForwardMode.IDLE),
        )

        self.assertEqual(out.shape, (0, N_PRIMES, N_HEADS))


if __name__ == "__main__":
    unittest.main()
