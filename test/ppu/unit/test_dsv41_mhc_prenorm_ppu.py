"""PPU acceptance for the DeepGEMM mHC prenorm projection.

The independent coefficient path (the boundaries MegaMHC cannot fuse) must run
its mix GEMM through ``tf32_hc_prenorm_gemm`` with a single K split, and must
reproduce the Triton ``hc_mix_stats`` + ``hc_split_sinkhorn`` coefficients.

Run on PPU SM89+ with a DeepGEMM build that exports ``tf32_hc_prenorm_gemm``.
"""

import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase

HIDDEN = 5120
HC = 4
ITERS = 10
RMS_EPS = 1e-6
EPS = 1e-6


class TestMhcPrenormPPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch is not installed")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("PPU is required")
        from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
        from sglang.srt.utils import get_device_sm, is_sail

        if not is_sail() or get_device_sm() < 89:
            raise unittest.SkipTest("SAIL SM89+ is required")
        if not ENABLE_JIT_DEEPGEMM:
            raise unittest.SkipTest("JIT DeepGEMM is required")
        import deep_gemm

        if not callable(getattr(deep_gemm, "tf32_hc_prenorm_gemm", None)):
            raise unittest.SkipTest("deep_gemm.tf32_hc_prenorm_gemm is required")
        cls.torch = torch

    def make_inputs(self, rows):
        torch = self.torch
        torch.manual_seed(9061 + rows)
        rand = lambda shape: torch.randn(shape, device="cuda", dtype=torch.float32)
        x = rand((rows, HC * HIDDEN)).bfloat16()
        fn = rand((24, HC * HIDDEN)) * 0.02
        scale = torch.tensor([0.5, 0.25, 0.25], device="cuda")
        base = rand((24,))
        return x, fn, scale, base

    def reference(self, x, fn, scale, base):
        from sglang.kernels.ops.layernorm.mhc import (
            hc_mix_stats,
            hc_split_sinkhorn,
        )

        mixes = hc_mix_stats(x, fn, RMS_EPS).unsqueeze(1)
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base, HC, ITERS, EPS)
        return pre.squeeze(1), post.squeeze(1), comb.squeeze(1)

    def assert_triplet_close(self, actual, expected):
        torch = self.torch
        for name, got, want in zip(("pre", "post", "comb"), actual, expected):
            with self.subTest(output=name):
                self.assertEqual(got.shape, want.shape)
                self.assertTrue(torch.isfinite(got).all().item())
                got, want = got.double(), want.double()
                diff = (
                    (got - want).square().sum()
                    / (got.square() + want.square()).sum().clamp_min(1e-30)
                ).item()
                self.assertLess(diff, 5e-5)

    def test_deepgemm_matches_triton_coefficients(self):
        from sglang.kernels.ops.layernorm.mhc import (
            hc_mix_stats_sinkhorn_deepgemm,
            split_tf32_hc_weight,
        )

        torch = self.torch
        with torch.no_grad():
            for rows in (1024, 4096, 16384):
                with self.subTest(rows=rows):
                    x, fn, scale, base = self.make_inputs(rows)
                    actual = hc_mix_stats_sinkhorn_deepgemm(
                        x,
                        split_tf32_hc_weight(fn),
                        scale,
                        base,
                        ITERS,
                        RMS_EPS,
                        EPS,
                    )
                    self.assert_triplet_close(
                        actual, self.reference(x, fn, scale, base)
                    )

    def test_build_tf32_parts(self):
        from sglang.srt.hardware_backend.ppu.attention import (
            ppu_mhc_prenorm_hooks as hooks,
        )

        torch = self.torch
        fn = torch.randn((24, HC * HIDDEN), device="cuda")
        layer = SimpleNamespace(hc_attn_fn=fn, hc_ffn_fn=fn)
        hooks._build_tf32_parts(layer)
        high, low = layer._hc_attn_tf32_parts
        self.assertTrue(torch.equal(high + low, fn))

    def test_dispatch_uses_deepgemm_on_ppu(self):
        from sglang.kernels.ops.layernorm import mhc as kernels
        from sglang.srt.environ import envs
        from sglang.srt.hardware_backend.ppu.attention import (  # noqa: F401
            ppu_mhc_prenorm_hooks,
        )
        from sglang.srt.models import deepseek_v4_mhc as model_mhc
        from sglang.srt.plugins.hook_registry import HookRegistry
        from sglang.srt.runtime_context import get_platform

        HookRegistry.apply_hooks()
        self.assertTrue(get_platform().is_ppu)
        torch = self.torch
        x, fn, scale, base = self.make_inputs(1024)
        hc = model_mhc.HcSubLayer(
            model_mhc.HcConfig(HC, ITERS, EPS, RMS_EPS, HIDDEN, True, False),
            fn,
            scale,
            base,
            SimpleNamespace(),
            kernels.split_tf32_hc_weight(fn),
            None,
        )

        calls = []
        original = kernels.hc_mix_stats_sinkhorn_deepgemm

        def spy(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        kernels.hc_mix_stats_sinkhorn_deepgemm = spy
        self.addCleanup(setattr, kernels, "hc_mix_stats_sinkhorn_deepgemm", original)
        with envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True), torch.no_grad():
            model_mhc._mix_stats_impl(hc, x.view(1024, HC, HIDDEN))
        self.assertEqual(len(calls), 1, "PPU must dispatch to the DeepGEMM path")


if __name__ == "__main__":
    unittest.main()
