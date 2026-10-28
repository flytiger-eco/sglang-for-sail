"""Numerical tests for the PPU DeepGEMM BF16 dense GEMM used by unquantized layers.

Validates the routed kernel wrapper (`_ppu_bf16_dense_linear`) against the
acblas path on the shapes the 810E rollout exercises. Registered for the PPU
suite: the routing logic itself is covered by the CPU-side unit tests in
test/registered/unit/layers/quantization/test_ppu_bf16_dense_deepgemm.py.

Bulk precompilation is disabled for the whole fixture: the first call per
(N, K) pair would otherwise sweep every M in `_BUILTIN_M_LIST` (1..16384)
instead of compiling the tested shapes, which does not fit the suite budget.
"""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.deep_gemm_wrapper.compile_utils as compile_utils
import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.utils import is_ppu
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-b-test-1-gpu-ppu")

# (M, N, K): decode-sized, prefill-sized and the lm-head-like wide N.
_SHAPES = ((16, 8192, 4096), (2048, 8192, 4096), (256, 32768, 4096))


@unittest.skipUnless(
    torch.cuda.is_available()
    and is_ppu()
    and unquant.deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM,
    "requires a PPU device with the DeepGEMM JIT enabled",
)
class TestPpuBf16DenseGemmNumerics(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self._saved_stats = dict(unquant._PPU_BF16_DENSE_STATS)
        unquant._PPU_BF16_DENSE_STATS.update(hit=0, fallback=0, error=0)
        # `_maybe_compile_deep_gemm_one_type_all` starts a full precompile for
        # the first call of every (kernel_type, N, K, num_groups) key, sweeping
        # all 16384 Ms of `_BUILTIN_M_LIST`. compile_utils snapshots the env at
        # import time, so the cached flag is the thing to flip; the kernels for
        # the tested shapes still JIT-compile on demand.
        precompile = patch.object(
            compile_utils, "_ENABLE_JIT_DEEPGEMM_PRECOMPILE", False
        )
        precompile.start()
        self.addCleanup(precompile.stop)

    def tearDown(self):
        unquant._PPU_BF16_DENSE_STATS.update(self._saved_stats)

    def test_bulk_precompile_stays_disabled(self):
        # Regression guard: re-enabling it here means a 16k-shape sweep per
        # (N, K) pair on a cold cache inside a 120s-registered test.
        self.assertFalse(compile_utils._ENABLE_JIT_DEEPGEMM_PRECOMPILE)

    def test_matches_acblas_within_bf16_tolerance(self):
        for m, n, k in _SHAPES:
            with self.subTest(m=m, n=n, k=k):
                x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
                weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

                actual = unquant._ppu_bf16_dense_linear(x, weight)
                reference = F.linear(x, weight)

                self.assertEqual(actual.shape, reference.shape)
                self.assertEqual(actual.dtype, torch.bfloat16)
                torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-1)
                # A kernel error would have silently served this shape from
                # acblas and made the comparison above vacuous.
                self.assertEqual(unquant._PPU_BF16_DENSE_STATS["error"], 0)

        self.assertEqual(unquant._PPU_BF16_DENSE_STATS["hit"], len(_SHAPES))

    def test_leading_dims_are_preserved(self):
        m, n, k = 2, 1024, 4096
        x = torch.randn((3, m, k), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

        actual = unquant._ppu_bf16_dense_linear(x, weight)

        self.assertEqual(actual.shape, (3, m, n))
        torch.testing.assert_close(actual, F.linear(x, weight), rtol=2e-2, atol=2e-1)


if __name__ == "__main__":
    unittest.main()
