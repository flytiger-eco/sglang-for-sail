"""CPU regressions for backend-specific sink allocation and checkpoint loading."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.model_loader import weight_utils
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models import dflash
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashAttentionSink(unittest.TestCase):
    def make_attention(
        self,
        *,
        ppu=False,
        npu=False,
        default_dtype=torch.bfloat16,
        backend="fa3",
        enabled=True,
        nemotron=False,
    ):
        config = SimpleNamespace(
            hidden_size=2048,
            num_attention_heads=128,
            num_key_value_heads=8,
            head_dim=128,
            attention_sink_bias=enabled,
        )
        draft = SimpleNamespace(attention_value_scale=None, attention_sink_bias=enabled)
        with ExitStack() as stack:
            for name, value in (
                ("_is_ppu", ppu),
                ("_is_npu", npu),
            ):
                stack.enter_context(patch.object(dflash, name, value))
            stack.enter_context(set_default_torch_dtype(default_dtype))
            for name, value in (
                ("get_parallel", SimpleNamespace(tp_size=8)),
                (
                    "get_spec",
                    SimpleNamespace(speculative_draft_attention_backend=backend),
                ),
                ("parse_dflash_draft_config", draft),
                ("is_nemotron_35_draft_config", nemotron),
                ("get_rope_config", (10000, None)),
                (
                    "_get_dflash_layer_attention_params",
                    (-1, dflash.AttentionType.ENCODER_ONLY),
                ),
            ):
                stack.enter_context(patch.object(dflash, name, return_value=value))
            # Only stub unrelated layers; exercise the real sink allocation and
            # real sharded weight loader without allocating model weights.
            for name in (
                "QKVParallelLinear",
                "RowParallelLinear",
                "RMSNorm",
                "get_rope",
                "RadixAttention",
            ):
                stack.enter_context(
                    patch.object(
                        dflash, name, side_effect=lambda *a, **kw: nn.Identity()
                    )
                )
            return dflash.DFlashAttention(config, layer_id=0)

    def check_dtypes(self, expected, **kwargs):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(default_dtype=dtype, **kwargs):
                sink = self.make_attention(
                    default_dtype=dtype, **kwargs
                ).attention_sink_bias
                self.assertEqual(sink.dtype, expected)
                self.assertEqual(sink.shape, (16,))
                self.assertTrue(sink.is_contiguous())
                self.assertFalse(sink.requires_grad)

    def test_ppu_fa3_requires_bf16_independent_of_model_dtype(self):
        self.check_dtypes(torch.bfloat16, ppu=True, backend="fa3")

    def test_ppu_other_backends_retain_fp32(self):
        for backend in ("fa4", "trtllm_mha"):
            self.check_dtypes(torch.float32, ppu=True, backend=backend)

    def test_non_ppu_backends_retain_fp32(self):
        for backend in ("fa3", "fa4", "trtllm_mha"):
            self.check_dtypes(torch.float32, backend=backend)

    def test_npu_retains_fp32(self):
        self.check_dtypes(torch.float32, npu=True, backend="ascend")

    def test_disabled_sinks_are_not_allocated(self):
        self.assertIsNone(
            self.make_attention(ppu=True, enabled=False).attention_sink_bias
        )

    def test_nemotron_retains_fp32_and_backend_restriction(self):
        self.check_dtypes(torch.float32, nemotron=True, backend="trtllm_mha")
        with self.assertRaisesRegex(ValueError, "attention sinks require"):
            self.make_attention(ppu=True, nemotron=True, backend="fa3")

    def test_checkpoint_load_shards_casts_and_preserves_parameter_storage(self):
        checkpoint = torch.linspace(-3.13, 2.71, 256)[::2]
        original = checkpoint.clone()
        for ppu in (False, True):
            for rank in range(8):
                with self.subTest(ppu=ppu, rank=rank):
                    model = nn.Module()
                    model.projector_type = "linear"
                    model.attention = self.make_attention(ppu=ppu)
                    sink = model.attention.attention_sink_bias
                    pointer = sink.data_ptr()
                    with patch.object(
                        weight_utils,
                        "get_parallel",
                        return_value=SimpleNamespace(attn_tp_rank=rank, tp_size=8),
                    ):
                        for weight in (checkpoint, checkpoint + 0.123):
                            dflash.DFlashDraftModel.load_weights(
                                model, [("attention.attention_sink_bias", weight)]
                            )
                            expected = weight[rank * 16 : (rank + 1) * 16].to(
                                sink.dtype
                            )
                            torch.testing.assert_close(sink, expected, rtol=0, atol=0)
                            self.assertEqual(sink.data_ptr(), pointer)
        torch.testing.assert_close(checkpoint, original, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
