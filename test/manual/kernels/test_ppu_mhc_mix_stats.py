"""PPU row-tiled mHC reduction: numerical, batch and Graph regressions.

Run with python test/manual/kernels/test_ppu_mhc_mix_stats.py.
Kept manual until the PPU runner has a registered kernel-test suite.
"""

import unittest
from unittest.mock import patch

import torch
import triton

from sglang.test.test_utils import CustomTestCase


class TestMhcMixStatsPPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.utils import is_sail

        if not torch.cuda.is_available() or not is_sail():
            raise unittest.SkipTest("A PPU is required")
        from sglang.kernels.ops.layernorm import mhc

        cls.mhc = mhc

    def original(self, x, fn):
        """Run the real partial/reduce kernels with the pre-change row tile."""
        kernel = self.mhc._hc_mix_stats_reduce_kernel
        block_m = self.mhc._block_m_for(x.shape[0])

        class OriginalReduce:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    kwargs["BLOCK_M"] = block_m
                    return kernel[(triton.cdiv(x.shape[0], block_m),)](*args, **kwargs)

                return launch

        with patch.object(self.mhc, "_hc_mix_stats_reduce_kernel", OriginalReduce()):
            return self.mhc.hc_mix_stats(x, fn, 1e-6)

    def test_reduce_launch_tiles(self):
        kernel = self.mhc._hc_mix_stats_reduce_kernel
        calls = []

        class RecordingReduce:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    calls.append((grid, kwargs["BLOCK_M"]))
                    return kernel[grid](*args, **kwargs)

                return launch

        cases = (
            (8, 20480, 24, True),
            (9, 20480, 24, True),
            (2048, 20480, 24, True),
            (2049, 20480, 24, True),
            (32, 16384, 24, True),
            (32, 20480, 16, True),
            (32, 20480, 24, False),
        )
        for rows, k, mix, sail in cases:
            with self.subTest(rows=rows, k=k, mix=mix, sail=sail):
                x = torch.zeros(rows, k, device="cuda", dtype=torch.bfloat16)
                fn = torch.zeros(mix, k, device="cuda", dtype=torch.float32)
                calls.clear()
                with patch.object(self.mhc, "_is_sail", sail), patch.object(
                    self.mhc, "_hc_mix_stats_reduce_kernel", RecordingReduce()
                ):
                    self.mhc.hc_mix_stats(x, fn, 1e-6)
                expected = (
                    8
                    if sail and 8 < rows <= 2048 and k == 20480 and mix == 24
                    else self.mhc._block_m_for(rows)
                )
                self.assertEqual(calls, [((triton.cdiv(rows, expected),), expected)])

    def test_reference_batch_invariance_and_boundaries(self):
        shapes = (1, 2, 4, 8, 9, 16, 32, 64, 128, 256, 512, 1024, 2048, 2049)
        for seed in (17, 31, 59):
            torch.manual_seed(seed)
            cpu_x = torch.randn(max(shapes), 20480, dtype=torch.bfloat16)
            cpu_fn = torch.randn(24, 20480) / 20480**0.5
            reference = (cpu_x.float() @ cpu_fn.T) * torch.rsqrt(
                cpu_x.float().square().mean(-1, keepdim=True) + 1e-6
            )
            x, fn = cpu_x.cuda(), cpu_fn.cuda()
            single = self.mhc.hc_mix_stats(x[:1], fn, 1e-6)
            for rows in shapes:
                with self.subTest(seed=seed, rows=rows):
                    actual = self.mhc.hc_mix_stats(x[:rows], fn, 1e-6)
                    self.assertTrue(torch.equal(actual, self.original(x[:rows], fn)))
                    self.assertTrue(torch.equal(actual[:1], single))
                    torch.testing.assert_close(
                        actual.cpu(), reference[:rows], atol=1e-4, rtol=1e-4
                    )

    def test_graph_replay_with_changed_inputs(self):
        for seed in (17, 31, 59):
            for rows in (1, 32, 1024):
                with self.subTest(seed=seed, rows=rows):
                    torch.manual_seed(seed)
                    x = torch.randn(rows, 20480, device="cuda", dtype=torch.bfloat16)
                    fn = torch.randn(24, 20480, device="cuda") / 20480**0.5
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            self.mhc.hc_mix_stats(x, fn, 1e-6)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = self.mhc.hc_mix_stats(x, fn, 1e-6)
                    # Do not use uniform rescaling: normalization nearly cancels it.
                    x[:, ::2].neg_()
                    graph.replay()
                    self.assertTrue(torch.equal(output, self.original(x, fn)))


if __name__ == "__main__":
    unittest.main()
