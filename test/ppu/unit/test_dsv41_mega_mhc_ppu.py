"""PPU numerical and CUDA Graph acceptance tests for shifted MegaMHC.

Run on PPU SM89+ with a compatible DeepGEMM mega_mhc backend installed.
The FP32 reference follows DeepGEMM's third-party/tilelang_ops/ref_mhc.py.
"""

import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase


class TestMegaMHCPPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch is not installed")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("PPU is required")
        from sglang.srt.utils import get_device_sm, is_sail

        if not is_sail() or get_device_sm() < 89:
            raise unittest.SkipTest("SAIL SM89+ is required")
        from sglang.kernels.ops.layernorm.mhc_mega import can_use_mega_mhc_prefill
        from sglang.srt.environ import envs
        from sglang.srt.models import deepseek_v4_mhc as mhc

        cls.batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True)
        )
        with envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True):
            if not can_use_mega_mhc_prefill(cls.batch):
                raise unittest.SkipTest("compatible JIT DeepGEMM MegaMHC is required")
        cls.torch = torch
        cls.mhc = mhc
        cls.envs = envs

    def boundary(
        self,
        x,
        residual,
        pre,
        post,
        comb,
        fn,
        scale,
        base,
        weight,
        rms_eps,
        hc_eps,
        norm_eps,
        sinkhorn_iters,
    ):
        mhc = self.mhc
        cfg = mhc.HcConfig(4, sinkhorn_iters, hc_eps, rms_eps, x.shape[1], True, False)
        norm = SimpleNamespace(
            weight=weight,
            variance_epsilon=norm_eps,
            cast_x_before_out_mul=False,
            variance_size_override=None,
        )
        hc = mhc.HcSubLayer(cfg, fn, scale, base, norm)
        next_boundary = mhc.make_boundary(norm, accepts_mxfp8=False, hc=hc)
        with self.envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.override(True):
            state = mhc._post_fusion(
                hc, x, residual, (pre, post, comb), next_boundary, self.batch
            )
        self.assertIs(state.pre, pre)
        self.assertIsInstance(state.input, mhc.HcNormed, "valid inputs must fuse")
        self.assertIsNotNone(state.stats, "the next sublayer must receive its stats")
        return state

    def make_inputs(self, rows):
        torch = self.torch
        torch.manual_seed(42808)
        hidden = 5120
        rand = lambda shape: torch.randn(shape, device="cuda", dtype=torch.float32)
        return [
            rand((rows, hidden)).bfloat16(),
            rand((rows, 4, hidden)).bfloat16(),
            rand((rows, 4)).sigmoid(),
            rand((rows, 4)).sigmoid() * 2,
            rand((rows, 4, 4)).softmax(-1),
            rand((24, 4 * hidden)) * 0.02,
            rand((3,)) * 0.1,
            rand((24,)),
            rand((hidden,)).bfloat16(),
        ]

    def reference(self, args):
        torch = self.torch
        x, residual, pre, post, comb, fn, scale, base, weight = args
        updated = (
            x.float().unsqueeze(1) * post.unsqueeze(-1)
            + torch.einsum("tij,tih->tjh", comb, residual.float())
        ).bfloat16()
        collapsed = (updated.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            mixes = updated.flatten(1).float() @ fn.T
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        mixes *= torch.rsqrt(updated.float().square().mean((1, 2)) + 1e-6)[:, None]
        scales = torch.cat(
            [scale[0].expand(4), scale[1].expand(4), scale[2].expand(16)]
        )
        mixes = mixes * scales + base
        next_pre = mixes[:, :4].sigmoid() + 1e-6
        next_post = mixes[:, 4:8].sigmoid() * 2
        next_comb = mixes[:, 8:].reshape(-1, 4, 4).softmax(-1) + 1e-6
        next_comb /= next_comb.sum(-2, keepdim=True) + 1e-6
        for _ in range(9):
            next_comb /= next_comb.sum(-1, keepdim=True) + 1e-6
            next_comb /= next_comb.sum(-2, keepdim=True) + 1e-6
        normalized = (
            collapsed.float()
            * torch.rsqrt(collapsed.float().square().mean(1) + 1e-6)[:, None]
            * weight.float()
        ).bfloat16()
        return updated, normalized, next_pre, next_post, next_comb

    def flatten_result(self, state):
        self.assertIsInstance(state, self.mhc.HcState)
        return (state.residual, state.input.rows, *state.stats)

    def assert_numerics(self, actual, expected):
        torch = self.torch
        for name, got, want in zip(
            ("residual", "normalized", "pre", "post", "comb"), actual, expected
        ):
            with self.subTest(output=name):
                self.assertEqual(got.shape, want.shape)
                self.assertEqual(got.dtype, want.dtype)
                self.assertTrue(torch.isfinite(got).all().item())
                # DeepGEMM's symmetric relative squared error / PyTorch threshold.
                got, want = got.double(), want.double()
                diff = (
                    (got - want).square().sum()
                    / (got.square() + want.square()).sum().clamp_min(1e-30)
                ).item()
                self.assertLess(diff, 5e-5)

    def test_two_shifted_boundaries(self):
        with self.torch.no_grad():
            for rows in (1, 65, 2496, 4096):
                with self.subTest(rows=rows):
                    args = self.make_inputs(rows)
                    original = args[1].clone()
                    actual = self.flatten_result(
                        self.boundary(*args, 1e-6, 1e-6, 1e-6, 10)
                    )
                    self.assert_numerics(actual, self.reference(args))
                    self.assertTrue(self.torch.equal(original, args[1]))
                    second = self.make_inputs(rows)
                    second[1], second[2], second[3], second[4] = (
                        actual[0],
                        actual[2],
                        actual[3],
                        actual[4],
                    )
                    self.assert_numerics(
                        self.flatten_result(
                            self.boundary(*second, 1e-6, 1e-6, 1e-6, 10)
                        ),
                        self.reference(second),
                    )

    def test_capture_stream_warmup_and_replay_with_changed_input(self):
        torch = self.torch
        args = self.make_inputs(65)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # DeepGEMM initializes its barriers on this same stream before capture.
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(3):
                self.boundary(*args, 1e-6, 1e-6, 1e-6, 10)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
            result = self.boundary(*args, 1e-6, 1e-6, 1e-6, 10)
        for factor in (1.0, 0.5, -0.75):
            args[0].copy_(args[0] * factor)
            graph.replay()
            torch.cuda.synchronize()
            self.assert_numerics(self.flatten_result(result), self.reference(args))


if __name__ == "__main__":
    unittest.main()
