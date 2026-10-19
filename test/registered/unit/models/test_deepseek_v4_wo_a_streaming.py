"""Unit tests for the DSv4 wo_a streaming-dequant scale-name handling.

``_dequant_fp8_wo_a_streaming`` pairs ``.wo_a`` weights with their block
scales as checkpoints stream past. Beyond the legacy ``.scale`` name, W8A8
INT8 checkpoints carry ``.weight_scale``/``.weight_scale_inv``; a quantized
(fp8/int8) wo_a weight that streams without its scale must fail loudly
instead of silently staying quantized, while a pre-dequantized bf16 weight
still passes through. CPU-only: tensors are tiny and the pairing logic does
not touch a device.
"""

import unittest

import torch

from sglang.srt.models.deepseek_v4 import _dequant_fp8_wo_a_streaming
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def fp8_weight(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(128, 128, generator=g) * 0.1).to(torch.float8_e4m3fn)


def fp32_scale(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, 1, generator=g) + 0.5


def expected_bf16(weight, scale):
    return (weight.float() * scale).to(torch.bfloat16)


class TestWoAStreamingScaleNames(CustomTestCase):
    def run_stream(self, entries):
        return list(_dequant_fp8_wo_a_streaming(iter(entries)))

    def assert_dequantized(self, out, name, weight, scale):
        self.assertEqual([n for n, _ in out], [name])
        self.assertEqual(out[0][1].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out[0][1], expected_bf16(weight, scale)))

    def test_weight_then_scale_legacy_suffix(self):
        w, s = fp8_weight(), fp32_scale()
        out = self.run_stream(
            [("m.wo_a.weight", w), ("m.wo_a.scale", s), ("m.other", torch.ones(2))]
        )
        self.assert_dequantized(out[:1], "m.wo_a.weight", w, s)
        self.assertEqual(out[1][0], "m.other")

    def test_scale_then_weight_weight_scale_suffix(self):
        w, s = fp8_weight(3), fp32_scale(4)
        out = self.run_stream([("m.wo_a.weight_scale", s), ("m.wo_a.weight", w)])
        self.assert_dequantized(out, "m.wo_a.weight", w, s)

    def test_weight_scale_inv_suffix(self):
        w, s = fp8_weight(5), fp32_scale(6)
        out = self.run_stream([("m.wo_a.weight", w), ("m.wo_a.weight_scale_inv", s)])
        self.assert_dequantized(out, "m.wo_a.weight", w, s)

    def test_int8_weight_without_scale_raises(self):
        entries = [("m.wo_a.weight", torch.zeros(128, 128, dtype=torch.int8))]
        with self.assertRaisesRegex(ValueError, "missing its scale"):
            self.run_stream(entries)

    def test_bf16_weight_without_scale_passes_through(self):
        w = torch.randn(8, 8, dtype=torch.bfloat16)
        out = self.run_stream([("m.wo_a.weight", w)])
        self.assertEqual([n for n, _ in out], ["m.wo_a.weight"])
        self.assertTrue(torch.equal(out[0][1], w))

    def test_orphan_scale_raises(self):
        entries = [("m.wo_a.weight_scale", fp32_scale())]
        with self.assertRaisesRegex(ValueError, "missing its weight"):
            self.run_stream(entries)


if __name__ == "__main__":
    unittest.main()
