"""PPU compile/numerics regression for sparse-MLA v1 fragment layouts."""

import math
import unittest

import torch

from sglang.srt.utils import is_ppu

COMPARISON_RESULTS = []


def _check_result(actual, expected, **shape):
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all().item()
    # The kernel rounds its online softmax probabilities to BF16 before PV.
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=3e-3)
    COMPARISON_RESULTS.append(
        {
            **shape,
            "max_absolute_error": (actual.float() - expected.float())
            .abs()
            .max()
            .item(),
        }
    )


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


@unittest.skipUnless(torch.cuda.is_available() and is_ppu(), "PPU device required")
class TestDSATileLangSparseLayout(unittest.TestCase):
    def test_v1_compile_and_numerics_for_head_and_tail_geometries(self):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            sparse_attention_fwd_kernel_v1,
        )

        torch.manual_seed(7)
        tokens, cache_rows, topk = 3, 256, 128
        for heads in (8, 16, 32, 64, 128):
            for value_dim, tail_dim in ((256, 0), (512, 0), (512, 64)):
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
                    offsets = torch.arange(tokens, device="cuda", dtype=torch.int32)
                    offsets = offsets.view(tokens, 1, 1) * (cache_rows // 4 + 1)
                    indices = (indices + offsets) % cache_rows
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
                    _check_result(
                        actual,
                        expected,
                        heads=heads,
                        value_dim=value_dim,
                        tail_dim=tail_dim,
                        topk=topk,
                        mode="direct",
                    )

    def test_glm53_head8_no_rope_wrapper_and_cuda_graph(self):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

        torch.manual_seed(19)
        tokens, heads, value_dim, topk, cache_rows = 3, 8, 512, 2048, 4096
        q = torch.randn(tokens, heads, value_dim, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(cache_rows, 1, value_dim, dtype=torch.bfloat16, device="cuda")
        indices = torch.arange(topk, dtype=torch.int32, device="cuda")
        indices = indices.view(1, 1, topk).expand(tokens, -1, -1).clone()
        offsets = torch.arange(tokens, device="cuda", dtype=torch.int32)
        offsets = offsets.view(tokens, 1, 1) * (cache_rows // 4 + 1)
        indices = (indices + offsets) % cache_rows
        indices[..., 17::11] = -1
        scale = 1.0 / math.sqrt(value_dim)

        def forward():
            return tilelang_sparse_fwd(q, kv, indices, scale, d_v=value_dim).reshape(
                tokens, heads, value_dim
            )

        expected = _reference(q, kv, indices, scale, value_dim)
        actual = forward()  # Compile before entering graph capture.
        shape = dict(heads=heads, value_dim=value_dim, tail_dim=0, topk=topk)
        _check_result(actual, expected, **shape, mode="glm53-wrapper")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward()
        q.copy_(torch.randn_like(q))
        graph.replay()
        expected = _reference(q, kv, indices, scale, value_dim)
        _check_result(captured, expected, **shape, mode="glm53-graph-replay")


if __name__ == "__main__":
    unittest.main()
