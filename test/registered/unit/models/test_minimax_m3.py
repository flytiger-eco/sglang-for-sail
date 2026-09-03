"""Unit tests for MiniMax-M3 decoder FFN argument routing."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from sglang.srt.models.minimax_m3_vl import MiniMaxM3SparseForConditionalGeneration


class TestMiniMaxM3DecoderLayer(CustomTestCase):
    def _run_decoder_layer(self, *, sparse, rows):
        hidden = torch.ones(rows, 4)
        batch = SimpleNamespace(num_token_non_padded=rows, multiplier=3)
        identity = lambda x, forward_batch, **kwargs: x
        boundary = SimpleNamespace(prepare=identity, finish=identity)

        def sparse_mlp(*, hidden_states, forward_batch):
            return hidden_states * forward_batch.multiplier

        def dense_mlp(x, forward_batch=None):
            if not rows:
                raise AssertionError("empty dense batches must skip the MLP")
            return x * forward_batch.multiplier

        ffn_exit = SimpleNamespace(finish=lambda x: x + 2)
        layer = SimpleNamespace(
            is_layer_sparse=sparse,
            attn_boundary=boundary,
            ffn_boundary=SimpleNamespace(
                prepare=identity, exit=lambda forward_batch: nullcontext(ffn_exit)
            ),
            self_attn=lambda positions, hidden_states, forward_batch: hidden_states,
            mlp=sparse_mlp if sparse else dense_mlp,
        )
        output = MiniMaxM3DecoderLayer.forward(
            layer,
            positions=torch.arange(rows),
            hidden_states=hidden,
            forward_batch=batch,
        )
        torch.testing.assert_close(output, hidden * batch.multiplier + 2)

    def test_sparse_mlp_receives_forward_batch(self):
        self._run_decoder_layer(sparse=True, rows=1)

    def test_dense_mlp_receives_forward_batch(self):
        self._run_decoder_layer(sparse=False, rows=1)

    def test_empty_batches_preserve_boundary_completion(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                self._run_decoder_layer(sparse=sparse, rows=0)


class TestMiniMaxM3VlEagle3Capture(CustomTestCase):
    @staticmethod
    def _model(num_layers=60):
        layers = [SimpleNamespace() for _ in range(num_layers)]
        return SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            capture_aux_hidden_states=False,
            config=SimpleNamespace(
                text_config=SimpleNamespace(num_hidden_layers=num_layers)
            ),
            model=SimpleNamespace(layers=layers, layers_to_capture=[]),
        )

    def test_default_capture_layers_apply_output_offset(self):
        model = self._model()

        MiniMaxM3SparseForConditionalGeneration.set_eagle3_layers_to_capture(model)

        self.assertEqual(model.model.layers_to_capture, [3, 31, 58])
        self.assertEqual(
            [
                i
                for i, layer in enumerate(model.model.layers)
                if getattr(layer, "_is_layer_to_capture", False)
            ],
            [3, 31, 58],
        )

    def test_explicit_capture_layers_apply_output_offset(self):
        model = self._model()

        MiniMaxM3SparseForConditionalGeneration.set_eagle3_layers_to_capture(
            model, [1, 29, 56]
        )

        self.assertEqual(model.model.layers_to_capture, [2, 30, 57])


if __name__ == "__main__":
    unittest.main()
