"""Unit tests: PPU dense-INT8 dispatch in the compressed-tensors W8A8 scheme.

Background: ``sgl_kernel.int8_scaled_mm`` is a CUTLASS kernel that PPU builds do not
register (the module-level python symbol exists, the ``torch.ops`` op does not), so an
INT8 compressed-tensors checkpoint fails at the first prefill with ``AttributeError``.
The fix routes PPU dense INT8 through the PPU DeepGEMM int8 NT kernel
(``gemm_nt_i8i8bf16``) behind ``SGLANG_SAIL_DEEPGEMM_DENSE``, default off.

These tests pin the dispatch, the weight/scale layout that DeepGEMM NT needs, the
default (CUDA) path staying byte-for-byte unchanged, and the fail-fast behaviour. None
of them needs a GPU or the DeepGEMM extension.
"""

import unittest
from unittest import mock

import torch
from torch import nn

from sglang.srt.layers.parameter import ModelWeightParameter
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w8a8_int8 as mod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)


def _make_scheme(*, channel=True, static=False, symmetric=True, enabled=True):
    """Build a scheme without going through __init__'s env probing."""
    scheme = CompressedTensorsW8A8Int8.__new__(CompressedTensorsW8A8Int8)
    scheme.strategy = (
        mod.QuantizationStrategy.CHANNEL if channel else mod.QuantizationStrategy.TENSOR
    )
    scheme.is_static_input_scheme = static
    scheme.input_symmetric = symmetric
    scheme.use_deepgemm_int8 = enabled
    scheme.deepgemm_int8_ready = False
    return scheme


class _Layer(nn.Module):
    """Mirrors create_weights: the weight is a ModelWeightParameter subclass."""

    def __init__(self, out_features, in_features, *, scale_shape=None):
        super().__init__()
        self.weight = ModelWeightParameter(
            data=torch.zeros(out_features, in_features, dtype=torch.int8),
            input_dim=1,
            output_dim=0,
            weight_loader=lambda *a, **k: None,
        )
        if scale_shape is None:
            scale_shape = (out_features, 1)
        self.weight_scale = nn.Parameter(
            torch.full(scale_shape, 0.02, dtype=torch.float32), requires_grad=False
        )
        self.logical_widths = [out_features]


def _quant_stub(shape):
    return mock.patch.object(
        mod,
        "per_token_quant_int8",
        return_value=(
            torch.zeros(shape, dtype=torch.int8),
            torch.ones(*shape[:-1], 1, dtype=torch.float32),
        ),
    )


class _FakeGemm:
    """Stand-in for gemm_nt_i8i8bf16(A, As, B, Bs, C); records the call."""

    def __init__(self):
        self.calls = {}

    def __call__(self, a, scale_a, b, scale_b, c):
        self.calls.update(
            a_shape=tuple(a.shape),
            a_dtype=a.dtype,
            scale_a_shape=tuple(scale_a.shape),
            b_shape=tuple(b.shape),
            b_dtype=b.dtype,
            scale_b_shape=tuple(scale_b.shape),
            c_shape=tuple(c.shape),
            c_dtype=c.dtype,
        )
        return None


class TestDeepGemmDispatch(unittest.TestCase):
    def test_apply_weights_dispatches_to_deepgemm(self):
        scheme = _make_scheme()
        layer = _Layer(out_features=64, in_features=32)
        scheme.process_weights_after_loading(layer)
        fake = _FakeGemm()
        x = torch.zeros(2, 5, 32, dtype=torch.bfloat16)

        with mock.patch.object(mod, "_get_deepgemm_int8_gemm", return_value=fake):
            with _quant_stub((2, 5, 32)):
                out = scheme.apply_weights(layer, x, bias=None)

        # NT layout: A is [M, K], B stays row-major [N, K], C is bf16 [M, N].
        self.assertEqual(fake.calls["a_shape"], (10, 32))
        self.assertEqual(fake.calls["scale_a_shape"], (10, 1))
        self.assertEqual(fake.calls["b_shape"], (64, 32))
        self.assertEqual(fake.calls["scale_b_shape"], (64, 1))
        self.assertEqual(fake.calls["c_shape"], (10, 64))
        self.assertEqual(fake.calls["c_dtype"], torch.bfloat16)
        self.assertEqual(tuple(out.shape), (2, 5, 64))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_apply_weights_2d_activation_and_output_cast(self):
        scheme = _make_scheme()
        layer = _Layer(out_features=16, in_features=8)
        scheme.process_weights_after_loading(layer)
        fake = _FakeGemm()
        x = torch.zeros(7, 8, dtype=torch.float16)

        with mock.patch.object(mod, "_get_deepgemm_int8_gemm", return_value=fake):
            with _quant_stub((7, 8)):
                out = scheme.apply_weights(layer, x, bias=None)

        self.assertEqual(fake.calls["a_shape"], (7, 8))
        self.assertEqual(tuple(out.shape), (7, 16))
        # kernel writes bf16, the layer keeps the activation dtype
        self.assertEqual(out.dtype, torch.float16)

    def test_bias_is_rejected(self):
        scheme = _make_scheme()
        layer = _Layer(out_features=16, in_features=8)
        scheme.process_weights_after_loading(layer)
        x = torch.zeros(1, 8, dtype=torch.bfloat16)
        with _quant_stub((1, 8)):
            with self.assertRaisesRegex(NotImplementedError, "bias"):
                scheme.apply_weights(
                    layer, x, bias=torch.zeros(16, dtype=torch.bfloat16)
                )

    def test_dispatch_requires_load_time_validation(self):
        scheme = _make_scheme()  # process_weights_after_loading not called
        layer = _Layer(16, 8)
        x = torch.zeros(1, 8, dtype=torch.bfloat16)
        with _quant_stub((1, 8)):
            with self.assertRaisesRegex(RuntimeError, "not validated"):
                scheme.apply_weights(layer, x, bias=None)

    def test_unavailable_deepgemm_raises_clear_error(self):
        scheme = _make_scheme()
        layer = _Layer(16, 8)
        scheme.process_weights_after_loading(layer)
        x = torch.zeros(1, 8, dtype=torch.bfloat16)
        with (
            mock.patch.object(
                mod, "_get_deepgemm_int8_gemm", side_effect=ImportError("no deep_gemm")
            ),
            _quant_stub((1, 8)),
        ):
            with self.assertRaisesRegex(RuntimeError, "DeepGEMM"):
                scheme.apply_weights(layer, x, bias=None)

    def test_disabled_dispatch_falls_back_to_cutlass(self):
        scheme = _make_scheme(enabled=False)
        layer = _Layer(8, 4)
        seen = {}

        def fake_scaled_mm(a, b, scale_a, scale_b, out_dtype=None, bias=None):
            seen["b_shape"] = tuple(b.shape)
            return torch.zeros(a.shape[0], b.shape[1], dtype=out_dtype)

        # mirror the real sequence: weights are transposed at load time
        scheme.process_weights_after_loading(layer)
        x = torch.zeros(1, 4, dtype=torch.bfloat16)
        with (
            mock.patch.object(mod, "int8_scaled_mm", fake_scaled_mm, create=True),
            _quant_stub((1, 4)),
        ):
            out = scheme.apply_weights(layer, x, bias=None)

        # Non-square so a missing/extra transpose is visible: cutlass keeps [K, N].
        self.assertEqual(seen["b_shape"], (4, 8))
        self.assertEqual(tuple(out.shape), (1, 8))


class TestWeightAndScaleLayout(unittest.TestCase):
    def test_deepgemm_keeps_row_major_weight(self):
        scheme = _make_scheme()
        layer = _Layer(out_features=8, in_features=4)
        scheme.process_weights_after_loading(layer)
        self.assertEqual(tuple(layer.weight.shape), (8, 4))
        self.assertEqual(tuple(layer.weight_scale.shape), (8, 1))
        self.assertTrue(scheme.deepgemm_int8_ready)

    def test_default_path_transposes_weight(self):
        scheme = _make_scheme(enabled=False)
        layer = _Layer(out_features=8, in_features=4)
        scheme.process_weights_after_loading(layer)
        self.assertEqual(tuple(layer.weight.shape), (4, 8))
        self.assertFalse(scheme.deepgemm_int8_ready)

    def test_tensor_strategy_expands_scale_for_deepgemm(self):
        scheme = _make_scheme(channel=False)
        layer = _Layer(out_features=8, in_features=4, scale_shape=(1,))
        # per-tensor requantization needs fp8 kernels that are absent on some builds;
        # stub it so the test only covers the scale expansion.
        with mock.patch.object(
            mod,
            "requantize_with_max_scale",
            return_value=(torch.tensor(0.02), layer.weight),
        ):
            scheme.process_weights_after_loading(layer)
        # per-tensor scalar becomes one scale per output channel, weight stays [N, K]
        self.assertEqual(tuple(layer.weight.shape), (8, 4))
        self.assertEqual(tuple(layer.weight_scale.shape), (8, 1))
        self.assertEqual(layer.weight_scale.numel(), 8)

    def test_load_time_layout_validation(self):
        scheme = _make_scheme()
        layer = _Layer(out_features=8, in_features=4)
        layer.weight = nn.Parameter(
            torch.zeros(8, 4, dtype=torch.float32), requires_grad=False
        )
        with self.assertRaisesRegex(RuntimeError, "int8"):
            scheme.process_weights_after_loading(layer)

    def test_load_time_scale_count_validation(self):
        scheme = _make_scheme()
        layer = _Layer(out_features=8, in_features=4, scale_shape=(1,))
        with self.assertRaisesRegex(RuntimeError, "one weight scale"):
            scheme.process_weights_after_loading(layer)


class TestEnablePredicate(unittest.TestCase):
    def test_enable_predicate_requires_ppu_and_switch(self):
        with (
            mock.patch.object(mod, "_is_ppu", False),
            mock.patch.object(
                mod.envs.SGLANG_SAIL_DEEPGEMM_DENSE, "get", return_value=True
            ),
        ):
            self.assertFalse(mod._ppu_deepgemm_int8_enabled())
        with (
            mock.patch.object(mod, "_is_ppu", True),
            mock.patch.object(
                mod.envs.SGLANG_SAIL_DEEPGEMM_DENSE, "get", return_value=False
            ),
        ):
            self.assertFalse(mod._ppu_deepgemm_int8_enabled())
        with (
            mock.patch.object(mod, "_is_ppu", True),
            mock.patch.object(
                mod.envs.SGLANG_SAIL_DEEPGEMM_DENSE, "get", return_value=True
            ),
        ):
            self.assertTrue(mod._ppu_deepgemm_int8_enabled())

    def test_no_capability_restriction(self):
        """DeepGEMM covers every PPU part, so load must not probe/limit capability."""
        scheme = _make_scheme()
        layer = _Layer(out_features=8, in_features=4)
        with mock.patch.object(
            mod.torch.cuda, "get_device_capability", return_value=(8, 9)
        ) as cap:
            scheme.process_weights_after_loading(layer)
        cap.assert_not_called()
        self.assertTrue(scheme.deepgemm_int8_ready)


class TestFailFast(unittest.TestCase):
    def test_asymmetric_activation_rejected_on_deepgemm(self):
        scheme = _make_scheme(symmetric=False)
        layer = _Layer(out_features=8, in_features=4)
        layer.input_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )
        layer.input_zero_point = nn.Parameter(
            torch.zeros(1, dtype=torch.int8), requires_grad=False
        )
        with self.assertRaisesRegex(NotImplementedError, "azp"):
            scheme.process_weights_after_loading(layer)

    def test_asymmetric_activation_allowed_by_default_path(self):
        scheme = _make_scheme(symmetric=False, enabled=False)
        layer = _Layer(out_features=8, in_features=4)
        layer.input_scale = nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )
        layer.input_zero_point = nn.Parameter(
            torch.zeros(1, dtype=torch.int8), requires_grad=False
        )
        scheme.process_weights_after_loading(layer)  # no raise: old behaviour intact


if __name__ == "__main__":
    unittest.main()
