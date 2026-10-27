import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import unquant
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPpuBf16Linear(unittest.TestCase):
    def setUp(self):
        self.method = unquant.UnquantizedLinearMethod()

    def test_ppu_bf16_uses_deepgemm(self):
        x = (
            torch.arange(24, dtype=torch.float32)
            .reshape(2, 3, 4)
            .transpose(0, 1)
            .contiguous()
            .to(torch.bfloat16)
        )
        weight = torch.nn.Parameter(
            torch.arange(20, dtype=torch.float32).reshape(5, 4).to(torch.bfloat16),
            requires_grad=False,
        )
        bias = torch.arange(5, dtype=torch.bfloat16)
        layer = SimpleNamespace(weight=weight)
        expected = torch.nn.functional.linear(x, weight, bias)

        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "is_ppu", return_value=True),
            patch.object(
                unquant.envs.SGLANG_SAIL_DEEPGEMM_DENSE_BF16,
                "get",
                return_value=True,
            ),
            patch.object(
                unquant.envs.SGLANG_SAIL_TO_KMAJOR_DENSE_BF16,
                "get",
                return_value=False,
            ),
            patch.object(unquant, "deep_gemm_wrapper") as wrapper,
        ):

            def run_gemm(lhs, rhs, out):
                self.assertTrue(lhs.is_contiguous())
                self.assertTrue(rhs.is_contiguous())
                self.assertEqual(lhs.dtype, torch.bfloat16)
                self.assertEqual(rhs.dtype, torch.bfloat16)
                out.copy_(lhs @ rhs.T)

            wrapper.gemm_nt_bf16bf16bf16.side_effect = run_gemm
            output = self.method.apply(layer, x, bias)

        self.assertEqual(output.shape, (3, 2, 5))
        torch.testing.assert_close(output, expected)
        wrapper.gemm_nt_bf16bf16bf16.assert_called_once()

    def test_non_ppu_keeps_torch_fallback(self):
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        weight = torch.nn.Parameter(
            torch.ones(3, 4, dtype=torch.bfloat16), requires_grad=False
        )
        layer = SimpleNamespace(weight=weight)

        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "is_ppu", return_value=False),
            patch.object(unquant, "deep_gemm_wrapper") as wrapper,
            patch.object(unquant.F, "linear", wraps=unquant.F.linear) as torch_linear,
        ):
            self.method.apply(layer, x)

        torch_linear.assert_called_once()
        wrapper.gemm_nt_bf16bf16bf16.assert_not_called()

    def test_ppu_non_bf16_keeps_torch_fallback(self):
        x = torch.ones(2, 4, dtype=torch.float32)
        weight = torch.nn.Parameter(
            torch.ones(3, 4, dtype=torch.float32), requires_grad=False
        )
        layer = SimpleNamespace(weight=weight)

        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "is_ppu", return_value=True),
            patch.object(
                unquant.envs.SGLANG_SAIL_DEEPGEMM_DENSE_BF16,
                "get",
                return_value=True,
            ),
            patch.object(unquant, "deep_gemm_wrapper") as wrapper,
            patch.object(unquant.F, "linear", wraps=unquant.F.linear) as torch_linear,
        ):
            self.method.apply(layer, x)

        torch_linear.assert_called_once()
        wrapper.gemm_nt_bf16bf16bf16.assert_not_called()

    def test_ppu_without_deepgemm_keeps_torch_fallback(self):
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        weight = torch.nn.Parameter(
            torch.ones(3, 4, dtype=torch.bfloat16), requires_grad=False
        )
        layer = SimpleNamespace(weight=weight)

        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "is_ppu", return_value=True),
            patch.object(
                unquant.envs.SGLANG_SAIL_DEEPGEMM_DENSE_BF16,
                "get",
                return_value=True,
            ),
            patch.object(unquant, "deep_gemm_wrapper") as wrapper,
            patch.object(unquant.F, "linear", wraps=unquant.F.linear) as torch_linear,
        ):
            wrapper.ENABLE_JIT_DEEPGEMM = False
            self.method.apply(layer, x)

        torch_linear.assert_called_once()
        wrapper.gemm_nt_bf16bf16bf16.assert_not_called()

    def test_ppu_torch_backend_keeps_torch_selection(self):
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        weight = torch.nn.Parameter(
            torch.ones(3, 4, dtype=torch.bfloat16), requires_grad=False
        )
        layer = SimpleNamespace(weight=weight)

        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "is_ppu", return_value=True),
            patch.object(
                unquant.envs.SGLANG_SAIL_DEEPGEMM_DENSE_BF16,
                "get",
                return_value=True,
            ),
            patch.object(unquant, "deep_gemm_wrapper") as wrapper,
            patch.object(
                unquant,
                "get_bf16_gemm_backend",
                return_value=unquant.Bf16GemmBackend.TORCH,
            ),
            patch.object(unquant.F, "linear", wraps=unquant.F.linear) as torch_linear,
        ):
            wrapper.ENABLE_JIT_DEEPGEMM = True
            self.method.apply(layer, x)

        torch_linear.assert_called_once()
        wrapper.gemm_nt_bf16bf16bf16.assert_not_called()

    def test_ppu_dense_bf16_disabled_keeps_torch_fallback(self):
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        weight = torch.nn.Parameter(
            torch.ones(3, 4, dtype=torch.bfloat16), requires_grad=False
        )
        layer = SimpleNamespace(weight=weight)

        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "is_ppu", return_value=True),
            patch.object(
                unquant.envs.SGLANG_SAIL_DEEPGEMM_DENSE_BF16,
                "get",
                return_value=False,
            ),
            patch.object(
                unquant.envs.SGLANG_SAIL_TO_KMAJOR_DENSE_BF16,
                "get",
                return_value=True,
            ),
            patch.object(unquant, "deep_gemm_wrapper") as wrapper,
            patch.object(unquant.F, "linear", wraps=unquant.F.linear) as torch_linear,
        ):
            self.method.apply(layer, x)

        torch_linear.assert_called_once()
        wrapper.gemm_nt_bf16bf16bf16.assert_not_called()

    def test_ppu_non_k_major_without_conversion_keeps_torch_fallback(self):
        x = torch.ones(2, 3, 4, dtype=torch.bfloat16).transpose(0, 1)
        weight = torch.nn.Parameter(
            torch.ones(4, 5, dtype=torch.bfloat16).T, requires_grad=False
        )
        layer = SimpleNamespace(weight=weight)

        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "is_ppu", return_value=True),
            patch.object(
                unquant.envs.SGLANG_SAIL_DEEPGEMM_DENSE_BF16,
                "get",
                return_value=True,
            ),
            patch.object(
                unquant.envs.SGLANG_SAIL_TO_KMAJOR_DENSE_BF16,
                "get",
                return_value=False,
            ),
            patch.object(unquant, "deep_gemm_wrapper") as wrapper,
            patch.object(unquant.F, "linear", wraps=unquant.F.linear) as torch_linear,
        ):
            self.method.apply(layer, x)

        torch_linear.assert_called_once()
        wrapper.gemm_nt_bf16bf16bf16.assert_not_called()

    def test_ppu_non_k_major_with_conversion_uses_deepgemm(self):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 1)
        x = x.to(torch.bfloat16)
        weight = torch.nn.Parameter(
            torch.arange(20, dtype=torch.float32).reshape(4, 5).T.to(torch.bfloat16),
            requires_grad=False,
        )
        layer = SimpleNamespace(weight=weight)
        expected = torch.nn.functional.linear(x, weight)

        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "is_ppu", return_value=True),
            patch.object(
                unquant.envs.SGLANG_SAIL_DEEPGEMM_DENSE_BF16,
                "get",
                return_value=True,
            ),
            patch.object(
                unquant.envs.SGLANG_SAIL_TO_KMAJOR_DENSE_BF16,
                "get",
                return_value=True,
            ),
            patch.object(unquant, "deep_gemm_wrapper") as wrapper,
        ):

            def run_gemm(lhs, rhs, out):
                self.assertTrue(lhs.is_contiguous())
                self.assertTrue(rhs.is_contiguous())
                out.copy_(lhs @ rhs.T)

            wrapper.gemm_nt_bf16bf16bf16.side_effect = run_gemm
            output = self.method.apply(layer, x)

        self.assertEqual(output.shape, (3, 2, 5))
        torch.testing.assert_close(output, expected)
        wrapper.gemm_nt_bf16bf16bf16.assert_called_once()

    def test_zero_inner_dimension_keeps_torch_fallback(self):
        x = torch.ones(2, 0, dtype=torch.bfloat16)
        weight = torch.nn.Parameter(
            torch.ones(16, 0, dtype=torch.bfloat16), requires_grad=False
        )
        layer = SimpleNamespace(weight=weight)

        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "is_ppu", return_value=True),
            patch.object(unquant, "deep_gemm_wrapper") as wrapper,
            patch.object(unquant.F, "linear", wraps=unquant.F.linear) as torch_linear,
        ):
            apply_output = self.method.apply(layer, x)
            dispatch_output = unquant.bf16_gemm_dispatch(
                torch.ones(2, 0, dtype=torch.bfloat16, device="meta"),
                torch.ones(16, 0, dtype=torch.bfloat16, device="meta"),
                None,
            )

        self.assertEqual(torch_linear.call_count, 1)
        wrapper.gemm_nt_bf16bf16bf16.assert_not_called()
        self.assertEqual(apply_output.shape, (2, 16))
        self.assertEqual(dispatch_output.shape, (2, 16))
        self.assertTrue(torch.all(apply_output == 0))


if __name__ == "__main__":
    unittest.main()
