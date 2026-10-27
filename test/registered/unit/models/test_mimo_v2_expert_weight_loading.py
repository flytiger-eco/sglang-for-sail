import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.mimo_v2 import MiMoV2ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiMoV2ExpertWeightLoading(unittest.TestCase):
    def load(self, checkpoint, *, suffix, destination_suffix, destination_dtype):
        param_name = f"model.layers.1.mlp.experts.w2_{destination_suffix}"
        param = torch.nn.Parameter(
            torch.empty_like(checkpoint, dtype=destination_dtype), requires_grad=False
        )
        observed = []

        def loader(target, weight, name, *, shard_id, expert_id):
            self.assertIs(target, param)
            self.assertEqual((name, shard_id, expert_id), (param_name, "w2", 0))
            observed.append(weight)
            target.data.copy_(weight)

        param.weight_loader = loader
        model = SimpleNamespace(
            config=SimpleNamespace(
                n_routed_experts=1, encoder_only=False, tie_word_embeddings=False
            ),
            model=SimpleNamespace(),
            _is_multimodal=False,
            _VISION_WEIGHT_PREFIXES=MiMoV2ForCausalLM._VISION_WEIGHT_PREFIXES,
            _AUDIO_WEIGHT_PREFIXES=MiMoV2ForCausalLM._AUDIO_WEIGHT_PREFIXES,
            _AUDIO_WEIGHT_SUBSTRING=MiMoV2ForCausalLM._AUDIO_WEIGHT_SUBSTRING,
            named_parameters=lambda: [(param_name, param)],
        )
        name = f"model.layers.1.mlp.experts.0.down_proj.{suffix}"
        MiMoV2ForCausalLM.load_weights(model, [(name, checkpoint)])
        self.assertEqual(len(observed), 1)
        return param, observed[0]

    def test_native_uint8_scales_are_not_renamed_or_converted(self):
        scales = torch.tensor([0, 1, 126, 127, 128, 254], dtype=torch.uint8)
        param, loaded = self.load(
            scales,
            suffix="weight_scale",
            destination_suffix="weight_scale",
            destination_dtype=torch.uint8,
        )
        self.assertIs(loaded, scales)
        torch.testing.assert_close(param, scales, rtol=0, atol=0)

    def test_native_packed_weights_preserve_all_byte_values(self):
        weights = torch.arange(256, dtype=torch.uint8)
        param, loaded = self.load(
            weights,
            suffix="weight",
            destination_suffix="weight",
            destination_dtype=torch.uint8,
        )
        self.assertIs(loaded, weights)
        torch.testing.assert_close(param, weights, rtol=0, atol=0)

    def test_upstream_fp32_scales_decode_e8m0_once(self):
        scales = torch.tensor([1, 126, 127, 128, 254], dtype=torch.uint8)
        param, loaded = self.load(
            scales,
            suffix="weight_scale",
            destination_suffix="weight_scale_inv",
            destination_dtype=torch.float32,
        )
        self.assertEqual(loaded.dtype, torch.float32)
        torch.testing.assert_close(
            param, torch.exp2(scales.float() - 127), rtol=0, atol=0
        )

    def test_upstream_int8_weights_keep_packed_bits(self):
        weights = torch.arange(256, dtype=torch.uint8)
        param, loaded = self.load(
            weights,
            suffix="weight",
            destination_suffix="weight",
            destination_dtype=torch.int8,
        )
        self.assertEqual(loaded.dtype, torch.int8)
        torch.testing.assert_close(param.view(torch.uint8), weights, rtol=0, atol=0)

    def test_already_decoded_scales_are_unchanged(self):
        scales = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        param, loaded = self.load(
            scales,
            suffix="weight_scale",
            destination_suffix="weight_scale_inv",
            destination_dtype=torch.float32,
        )
        self.assertIs(loaded, scales)
        torch.testing.assert_close(param, scales, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
