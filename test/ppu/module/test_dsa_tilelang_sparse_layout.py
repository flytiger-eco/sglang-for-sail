"""PPU compile/numerics regression for sparse-MLA v1 fragment layouts."""

import math
import unittest

import torch

from sglang.srt.utils import is_sail


def _reference(q, kv, indices, scale, value_dim):
    rows = indices[:, 0].long()
    valid = rows >= 0
    selected = kv[rows.clamp_min(0), 0].float()
    scores = torch.einsum("thd,tkd->thk", q.float(), selected) * scale
    scores.masked_fill_(~valid[:, None, :], -float("inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("thk,tkd->thd", probabilities, selected[..., :value_dim]).to(
        q.dtype
    )


@unittest.skipUnless(torch.cuda.is_available() and is_sail(), "PPU device required")
class TestDSATileLangSparseLayout(unittest.TestCase):
    def test_v1_compile_and_numerics_for_head_and_tail_geometries(self):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            sparse_attention_fwd_kernel_v1,
        )

        torch.manual_seed(7)
        tokens, cache_rows, topk = 3, 256, 128
        for heads in (16, 32, 64, 128):
            for value_dim, tail_dim in ((256, 0), (512, 64)):
                with self.subTest(heads=heads, value_dim=value_dim, tail_dim=tail_dim):
                    dim = value_dim + tail_dim
                    q = torch.randn(
                        tokens, heads, dim, dtype=torch.bfloat16, device="cuda"
                    )
                    kv = torch.randn(
                        cache_rows, 1, dim, dtype=torch.bfloat16, device="cuda"
                    )
                    indices = torch.arange(topk, device="cuda", dtype=torch.int32)
                    indices = indices.view(1, 1, topk).expand(tokens, -1, -1).clone()
                    indices[..., 17::11] = -1  # Keep at least one valid key per row.
                    scale = 1.0 / math.sqrt(dim)
                    expected = _reference(q, kv, indices, scale, value_dim)
                    kernel = sparse_attention_fwd_kernel_v1(
                        heads,
                        value_dim,
                        tail_dim,
                        topk,
                        sm_scale=scale,
                        num_stages=1,
                    )
                    actual = kernel(
                        q.unsqueeze(0), kv.unsqueeze(0), indices.unsqueeze(0)
                    ).squeeze(0)
                    torch.cuda.synchronize()
                    self.assertTrue(torch.isfinite(actual).all().item())
                    torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)


if __name__ == "__main__":
    unittest.main()
