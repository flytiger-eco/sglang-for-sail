"""Unit tests for the env-gated PPU DeepGEMM route of unquantized BF16 dense GEMM.

The routing itself is pure Python (dtype/shape/bias gate, switch check, warmup
skip condition), so it is covered here without a PPU device. The numerical
equivalence check against the acblas path needs the real kernel and lives in
test/registered/kernels/ops/gemm/test_ppu_bf16_dense_gemm.py, which is
registered for the PPU suite instead.
"""

import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.deep_gemm_wrapper.compile_utils as compile_utils
import sglang.srt.layers.quantization.unquant as unquant
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _bf16(*shape):
    return torch.empty(shape, dtype=torch.bfloat16)


class TestPpuBf16DenseGate(CustomTestCase):
    """Only bias-free, K-major, aligned BF16 layers may take the DeepGEMM path."""

    def test_accepts_contiguous_k_major_without_bias(self):
        self.assertTrue(
            unquant._is_k_major_bf16_dense(_bf16(8, 64), _bf16(32, 64), None)
        )

    def test_accepts_1d_activation(self):
        self.assertTrue(unquant._is_k_major_bf16_dense(_bf16(64), _bf16(32, 64), None))

    def test_rejects_any_bias(self):
        # `out + bias` rounds twice (bf16 GEMM result, then bf16 add) where
        # F.linear adds the bias in the fp32 accumulator, so bias-carrying
        # layers must keep falling back.
        for bias in (_bf16(32), torch.empty(32, dtype=torch.float32)):
            with self.subTest(bias_dtype=bias.dtype):
                self.assertFalse(
                    unquant._is_k_major_bf16_dense(_bf16(8, 64), _bf16(32, 64), bias)
                )

    def test_rejects_zero_sized_feature_dim(self):
        # Guards the `x.numel() // x.shape[-1]` in _ppu_bf16_dense_linear.
        self.assertFalse(
            unquant._is_k_major_bf16_dense(_bf16(8, 0), _bf16(32, 0), None)
        )

    def test_rejects_unaligned_shapes(self):
        self.assertFalse(
            unquant._is_k_major_bf16_dense(_bf16(8, 60), _bf16(32, 60), None)
        )
        self.assertFalse(
            unquant._is_k_major_bf16_dense(_bf16(8, 64), _bf16(33, 64), None)
        )

    def test_rejects_non_k_major_and_foreign_dtypes(self):
        self.assertFalse(
            unquant._is_k_major_bf16_dense(_bf16(64, 8).t(), _bf16(32, 64), None)
        )
        self.assertFalse(
            unquant._is_k_major_bf16_dense(_bf16(8, 64), _bf16(64, 32).t(), None)
        )
        self.assertFalse(
            unquant._is_k_major_bf16_dense(
                torch.empty((8, 64), dtype=torch.float16), _bf16(32, 64), None
            )
        )
        self.assertFalse(
            unquant._is_k_major_bf16_dense(
                _bf16(8, 64), torch.empty((32, 64), dtype=torch.float32), None
            )
        )


class TestPpuBf16DenseSwitch(CustomTestCase):
    def test_switch_declares_default_off(self):
        # Assert the declared default rather than the resolved value: this file
        # must stay valid when the suite runs with the switch exported as 1.
        self.assertIs(
            unquant.envs.SGLANG_SAIL_DEEPGEMM_DENSE_BF16.default,
            False,
        )

    @staticmethod
    def _enabled(*, ppu, env_on, jit, sm, compiling, batch_invariant=False):
        with (
            patch.object(unquant, "_is_ppu", ppu),
            patch.object(unquant, "_PPU_BF16_DENSE_DEVICE_SM", sm),
            patch.object(
                unquant.envs.SGLANG_SAIL_DEEPGEMM_DENSE_BF16,
                "get",
                return_value=env_on,
            ),
            patch.object(unquant.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", jit),
            patch.object(
                unquant,
                "is_batch_invariant_mode_enabled",
                return_value=batch_invariant,
            ),
            patch.object(torch.compiler, "is_compiling", return_value=compiling),
        ):
            return unquant._ppu_bf16_dense_enabled()

    def test_enabled_when_every_condition_holds(self):
        self.assertTrue(
            self._enabled(ppu=True, env_on=True, jit=True, sm=80, compiling=False)
        )

    def test_disabled_when_any_condition_fails(self):
        for case in (
            dict(ppu=False, env_on=True, jit=True, sm=80, compiling=False),
            dict(ppu=True, env_on=False, jit=True, sm=80, compiling=False),
            dict(ppu=True, env_on=True, jit=False, sm=80, compiling=False),
            dict(ppu=True, env_on=True, jit=True, sm=70, compiling=False),
            dict(ppu=True, env_on=True, jit=True, sm=80, compiling=True),
            # --enable-deterministic-inference must keep winning over this
            # opt-in: its aten::mm/addmm overrides cannot see this kernel.
            dict(
                ppu=True,
                env_on=True,
                jit=True,
                sm=80,
                compiling=False,
                batch_invariant=True,
            ),
        ):
            with self.subTest(**case):
                self.assertFalse(self._enabled(**case))


class TestPpuBf16DenseLinear(CustomTestCase):
    def setUp(self):
        self._saved_stats = dict(unquant._PPU_BF16_DENSE_STATS)
        unquant._PPU_BF16_DENSE_STATS.update(hit=0, fallback=0, error=0)

    def tearDown(self):
        unquant._PPU_BF16_DENSE_STATS.update(self._saved_stats)

    def test_empty_batch_returns_empty_output_without_the_kernel(self):
        kernel = Mock()
        with patch.object(unquant, "_ppu_gemm_nt_bf16", kernel):
            out = unquant._ppu_bf16_dense_linear(_bf16(0, 64), _bf16(32, 64))

        kernel.assert_not_called()
        self.assertEqual(out.shape, (0, 32))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(unquant._PPU_BF16_DENSE_STATS["hit"], 0)
        self.assertEqual(unquant._PPU_BF16_DENSE_STATS["error"], 0)

    def test_flattens_leading_dims_and_counts_a_hit(self):
        kernel = Mock()

        def _fake_kernel(lhs, rhs, out):
            self.assertEqual(lhs.shape, (6, 64))
            self.assertEqual(rhs.shape, (32, 64))
            out.fill_(1.0)

        kernel.side_effect = _fake_kernel
        with patch.object(unquant, "_ppu_gemm_nt_bf16", kernel):
            out = unquant._ppu_bf16_dense_linear(_bf16(2, 3, 64), _bf16(32, 64))

        kernel.assert_called_once()
        self.assertEqual(out.shape, (2, 3, 32))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.all(out == 1.0))
        self.assertEqual(unquant._PPU_BF16_DENSE_STATS["hit"], 1)

    def test_kernel_error_falls_back_to_acblas(self):
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        weight = torch.randn(32, 64, dtype=torch.bfloat16)

        with patch.object(
            unquant, "_ppu_gemm_nt_bf16", side_effect=RuntimeError("boom")
        ):
            with self.assertLogs(unquant.logger, level="WARNING"):
                out = unquant._ppu_bf16_dense_linear(x, weight)

        self.assertTrue(torch.equal(out, F.linear(x, weight)))
        self.assertEqual(unquant._PPU_BF16_DENSE_STATS["error"], 1)
        self.assertEqual(unquant._PPU_BF16_DENSE_STATS["hit"], 0)

    def test_output_buffer_dies_before_the_fallback_allocates(self):
        # Recovery must not hold the routed (M, N) buffer while F.linear
        # allocates its own, or a large prefill needs two output buffers. Only a
        # lifetime check can see this: the buffer survives `del out` through the
        # exception traceback's callee frame.
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        weight = torch.randn(32, 64, dtype=torch.bfloat16)
        probe = {}

        def _raising_kernel(lhs, rhs, out):
            probe["buffer"] = weakref.ref(out)
            raise RuntimeError("boom")

        def _fallback_probe(x, weight):
            probe["alive_at_fallback"] = probe["buffer"]() is not None
            return F.linear(x, weight)

        with (
            patch.object(unquant, "_ppu_gemm_nt_bf16", _raising_kernel),
            patch.object(unquant, "F", SimpleNamespace(linear=_fallback_probe)),
            self.assertLogs(unquant.logger, level="WARNING"),
        ):
            out = unquant._ppu_bf16_dense_linear(x, weight)

        self.assertFalse(probe["alive_at_fallback"])
        self.assertEqual(out.shape, (4, 32))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, F.linear(x, weight)))
        self.assertEqual(unquant._PPU_BF16_DENSE_STATS["error"], 1)


class TestPpuBf16DenseWarmupExecutor(CustomTestCase):
    @staticmethod
    def _executor(max_m):
        # Bypass __init__: setup_tensors allocates on "cuda".
        executor = object.__new__(compile_utils._NormalWarmupExecutor_bf16)
        executor.max_m = max_m
        executor.lhs = executor.rhs = executor.out = [-1]
        return executor

    def test_registered_for_the_dense_bf16_kernel_type(self):
        executor = Mock()
        with patch.object(
            compile_utils, "_NormalWarmupExecutor_bf16", return_value=executor
        ) as factory:
            created = compile_utils._BaseWarmupExecutor.create(
                compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16BF16,
                max_m=16,
                n=256,
                k=256,
                num_groups=1,
            )

        self.assertIs(created, executor)
        factory.assert_called_once_with(max_m=16, n=256, k=256, num_groups=1)
        self.assertGreater(
            compile_utils._BaseWarmupExecutor.get_memory_requirement(
                compile_utils.DeepGemmKernelType.GEMM_NT_BF16BF16BF16,
                max_m=16,
                n=256,
                k=256,
                num_groups=1,
            ),
            0,
        )

    def test_warms_up_max_m_and_only_skips_beyond_it(self):
        deep_gemm = Mock()
        executor = self._executor(max_m=8)

        with (
            patch.object(compile_utils, "deep_gemm", deep_gemm, create=True),
            self.assertLogs(compile_utils.logger, level="WARNING"),
        ):
            executor.execute(m=4)
            executor.execute(m=8)  # m == max_m is a warmup shape and must run
            executor.execute(m=9)  # only m > max_m is skipped

        self.assertEqual(deep_gemm.bf16_gemm_nt.call_count, 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
