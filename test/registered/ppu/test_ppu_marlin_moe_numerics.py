"""PPU Marlin MoE workspace-reduce numerics regression guard.

Guards against the silent NaN corruption found on PPU with sgl-kernel 0.4.3 +
SDK 2.1.1 (internal bug report, regression #2): fused_marlin_moe emitted NaN
on 34% of output elements (86008/251904) at larger element counts. PPU forces
``use_atomic_add=False`` (fused_marlin_moe.py), so only the workspace-reduce
path is exercised -- the exact path where the bug lives.

Design constraints (decision doc section 8.1, W6-1):

1. The NaN check is an INDEPENDENT assertion that runs BEFORE assert_close.
   The original investigation lost time because assert_close reported the
   precision drift of a small config first and masked the NaN of the larger
   one ("Raising the tolerance to 0.15 clears (1) and exposes (2)").
2. The configs cover the element scale where the bug appeared: output
   (m=123, k=2048) = 251904 elements, plus a larger m. Small configs only
   showed a 0.2% precision drift and would not have caught it.

The corresponding upstream test registered/quant/test_marlin_moe.py is
``disabled=`` on PPU. This native test turns the failure into a loud
assertion: it FAILS while the SDK bug is present (instead of corrupting
outputs silently) and keeps guarding against re-regression after the fix.

Usage:
python3 -m pytest test/registered/ppu/test_ppu_marlin_moe_numerics.py -v
"""

import unittest

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_marlin_utils import marlin_quantize
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=300, suite="nightly-1-ppu", nightly=True)

set_global_server_args_for_scheduler(object.__new__(ServerArgs))


def stack_and_dev(tensors: list[torch.Tensor]) -> torch.Tensor:
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)


def torch_experts(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Unquantized reference MoE (mirrors registered/quant/test_marlin_moe.py)."""
    M, K = a.shape
    topk = topk_ids.shape[1]
    a = a.view(M, -1, K).repeat(1, topk, 1).reshape(-1, K)

    out = torch.zeros(M * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    num_experts = w1.shape[0]
    topk_ids = topk_ids.view(-1)
    f32 = torch.float32

    for i in range(num_experts):
        mask = topk_ids == i
        if mask.sum():
            tmp1 = a[mask] @ w1[i].transpose(0, 1)
            tmp2 = SiluAndMul()(tmp1)
            out[mask] = tmp2 @ w2[i].transpose(0, 1)

    return (
        (out.view(M, -1, w2.shape[1]).to(f32) * topk_weight.view(M, -1, 1))
        .sum(dim=1)
        .to(out.dtype)
    )


def torch_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    score: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    return torch_experts(a, w1, w2, topk_weight, topk_ids)


def _quantize_experts(w: torch.Tensor, group_size: int, perm_dim: int):
    """Marlin-quantize each expert (uint4b8, no act_order, k full)."""
    w_ref_l, qweight_l, scales_l, g_idx_l, sort_indices_l = [], [], [], [], []
    for i in range(w.shape[0]):
        test_perm = torch.randperm(perm_dim)
        w_ref, qweight, scales, g_idx, sort_indices, _ = marlin_quantize(
            w[i].transpose(1, 0),
            scalar_types.uint4b8,
            group_size,
            False,
            test_perm,
        )
        w_ref_l.append(w_ref.T)
        qweight_l.append(qweight)
        scales_l.append(scales)
        g_idx_l.append(g_idx)
        sort_indices_l.append(sort_indices)
    return (
        stack_and_dev(w_ref_l),
        stack_and_dev(qweight_l).contiguous(),
        stack_and_dev(scales_l),
        stack_and_dev(g_idx_l),
        stack_and_dev(sort_indices_l),
    )


class TestPPUMarlinMoeNumerics(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("This test requires a CUDA device.")
        torch.set_default_device("cuda")

    def _run_once(self, m: int, n: int, k: int, e: int, topk: int, dtype):
        torch.manual_seed(0)
        group_size = 128

        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 20
        w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 20

        w_ref1, qweight1, scales1, g_idx1, sort_indices1 = _quantize_experts(
            w1, group_size, k
        )
        w_ref2, qweight2, scales2, g_idx2, sort_indices2 = _quantize_experts(
            w2, group_size, n
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        from sglang.srt.layers.moe.topk import fused_topk_torch_native

        topk_weights, topk_ids = fused_topk_torch_native(a, score, topk, False)

        torch_output = torch_moe(a, w_ref1, w_ref2, score, topk)

        marlin_output = fused_marlin_moe(
            a,
            qweight1,
            qweight2,
            scales1,
            scales2,
            score,
            topk_weights,
            topk_ids,
            g_idx1=g_idx1,
            g_idx2=g_idx2,
            sort_indices1=sort_indices1,
            sort_indices2=sort_indices2,
            num_bits=4,
            is_k_full=True,
        )

        # (1) NaN guard: independent and BEFORE assert_close. assert_close on a
        # NaN tensor only reports "greatest absolute difference: nan" -- that is
        # how the original silent corruption was first masked by a tolerance
        # question. This assertion fails with an unambiguous message instead.
        nan_count = int(torch.isnan(marlin_output).sum().item())
        self.assertEqual(
            nan_count,
            0,
            f"fused_marlin_moe produced {nan_count}/{marlin_output.numel()} NaN "
            f"elements (m={m}, n={n}, k={k}, dtype={dtype}) -- PPU "
            f"workspace-reduce regression",
        )

        # (2) Precision guard, decoupled from (1). atol=0.15 is the tolerance
        # that cleared the small-config drift in the original investigation.
        torch.testing.assert_close(marlin_output, torch_output, atol=0.15, rtol=0)

    def test_no_nan_at_bug_scale(self):
        """(m=123, k=2048) -> 251904 output elements: the exact scale where
        34% of elements were NaN in the original report."""
        for dtype in (torch.half, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._run_once(m=123, n=1024, k=2048, e=4, topk=2, dtype=dtype)

    def test_no_nan_at_larger_m(self):
        """Larger m at the same k, in case the workspace-buffer sizing bug is
        m-dependent (the original report suspected zero-init / sizing)."""
        for dtype in (torch.half, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._run_once(m=666, n=1024, k=2048, e=4, topk=2, dtype=dtype)


if __name__ == "__main__":
    unittest.main()
