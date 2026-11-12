"""Interleave prefill CP row order around linear attention: the boundary gathers
CP shards rank-major (token i in rank i % size's block at row i // size, the
last group padded), the recurrence needs sequence order, and the output goes
back to the padded rank-major rows."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    cp_interleave_to_sequence_order,
    cp_sequence_to_interleave_order,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(tokens):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
        attn_cp_metadata=InterleaveContextParallelMetadata(total_seq_lens=tokens),
    )


def _rank_major(seq, size):
    """The boundary's gathered rows for sequence rows `seq`, zero padded."""
    tokens = seq.shape[0]
    rows = -(-tokens // size)
    out = seq.new_zeros((size * rows, *seq.shape[1:]))
    for i in range(tokens):
        out[(i % size) * rows + i // size] = seq[i]
    return out


class TestCpInterleaveOrder(CustomTestCase):
    def test_round_trip(self):
        for size in (2, 4, 8, 16):
            for tokens in (1, 7, 8, 63, 64, 1000):
                with self.subTest(size=size, tokens=tokens):
                    seq = torch.randn(tokens, 3, 5)
                    gathered = _rank_major(seq, size)
                    batch = _batch(tokens)
                    with get_parallel().override(attn_cp_size=size):
                        restored = cp_interleave_to_sequence_order(gathered, batch)
                        back = cp_sequence_to_interleave_order(
                            restored, batch, gathered.shape[0]
                        )
                    torch.testing.assert_close(restored, seq, rtol=0, atol=0)
                    # Padding rows come back zero, not stale.
                    torch.testing.assert_close(back, gathered, rtol=0, atol=0)

    def test_non_interleave_batch_passes_through(self):
        rows = torch.randn(6, 4)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: False),
            attn_cp_metadata=None,
        )
        self.assertIs(cp_interleave_to_sequence_order(rows, batch), rows)
        self.assertIs(cp_sequence_to_interleave_order(rows, batch, 6), rows)

    def test_kimi_k31_cp_has_no_bcg_adapter(self):
        """Removing K3.1 BCG must retain eager CP and the GLM graph adapter."""
        from unittest.mock import patch

        from sglang.srt.layers.cp import bcg
        from sglang.srt.layers.cp.bcg import interleave_bcg

        for arch, pp_size, ep_size, expected in (
            ("KimiK31ForConditionalGeneration", 1, 1, False),
            ("KimiK31ForConditionalGeneration", 1, 8, False),
            ("KimiK31ForConditionalGeneration", 2, 8, False),
            ("Glm5NextForConditionalGeneration", 1, 1, True),
            ("Glm5NextForConditionalGeneration", 2, 1, False),
        ):
            cfg = SimpleNamespace(
                enable_prefill_cp=True,
                pp_size=pp_size,
                tp_size=8,
                cp_strategy="interleave",
                dp_size=1,
                ep_size=ep_size,
                moe_dp_size=1,
            )
            view = SimpleNamespace(
                attn_cp_size=8, ep_size=ep_size, moe_a2a_backend="megamoe"
            )
            model = SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch]))
            with (
                self.subTest(arch=arch, pp_size=pp_size, ep_size=ep_size),
                patch.multiple(
                    bcg,
                    resolving_view=lambda _: cfg,
                    resolved_view=lambda _: view,
                    attention_backends_of=lambda _: ("dsa", "dsa"),
                ),
                patch.multiple(
                    interleave_bcg,
                    resolving_view=lambda _: cfg,
                    resolved_view=lambda _: view,
                    model_config_of=lambda _: model,
                ),
            ):
                self.assertEqual(bcg.supports_prefill_cp_bcg(cfg), expected)


if __name__ == "__main__":
    unittest.main()
