"""PPU Marlin MoE workspace-reduce numerics regression guard.

Regression guard (NOT a bug sentinel -- see the history below). Asserts that
fused_marlin_moe produces no NaN and stays within atol=0.15 of the reference
MoE on the PPU workspace-reduce path at the element scales where silent NaN
corruption was first reported.

History:

- Original bug criterion (sgl-kernel 0.4.3 + SDK 2.1.1, internal bug report,
  regression #2): fused_marlin_moe emitted NaN on 34% of output elements
  (86008/251904) at m=123, on the workspace-reduce path. That criterion was
  recorded while PPU still forced ``use_atomic_add=False``, i.e. while EVERY
  dtype went through workspace-reduce.
- 5aa674aa05 (2026-08-19) reverted that forcing, so the claim "on PPU only
  the workspace-reduce path is exercised" is no longer true. There is no PPU
  branch left::

      use_atomic_add = (
          hidden_states.dtype == torch.half
          or torch.cuda.get_device_capability(...)[0] >= 9
      ) and (not is_mxfp4_marlin)

  PPU reports capability (8, 0), so within THIS file: the float16 subTests
  take atomic-add, and only the bfloat16 subTests still exercise the
  workspace-reduce path where the NaN lived. Do not read a green float16
  subTest as evidence about the NaN.
- This file was first written as an xfail(strict=True) sentinel: xfail while
  the bug reproduces, XPASS failure as the "SDK fix landed" signal.
- 2026-08-21 (three preflight rounds on ppu1) and 2026-08-22 (nightly run
  32578521498, image llm:v2.1.1-... / SDK 2.1.1-a5c56e unchanged): every
  config in this file PASSES. The sentinel premise -- "the bug reproduces on
  these configs" -- never held here, and XPASS proved unreliable as a signal
  (it appeared with no SDK/image change at all). A strict-xfail that always
  XPASSes makes nightly permanently red, so on 2026-08-23 the markers were
  removed and the file became a regression guard: green today, red only if
  the NaN / precision corruption comes back at these scales.

Design constraints (decision doc section 8.1, W6-1), still valid:

1. The NaN check is an INDEPENDENT assertion that runs BEFORE assert_close.
   The original investigation lost time because assert_close reported the
   precision drift of a small config first and masked the NaN of the larger
   one ("Raising the tolerance to 0.15 clears (1) and exposes (2)").
2. The configs cover the element scale where the bug appeared: output
   (m=123, k=2048) = 251904 elements, plus a larger m. Small configs only
   showed a 0.2% precision drift and would not have caught it.

The STILL-ALIVE marlin regression is NOT covered by this file: m=1 + float16
selects ``use_atomic_add=True``, which hangs on PPU (measured: EXIT=124,
1800s hard timeout, py-spy sampling pinned on config #1 of the upstream
matrix; the same config with use_atomic_add=False finishes in 0.3s). That
hang is what the ``disabled=`` on registered/quant/test_marlin_moe.py
actually guards against -- its stated reason ("produces NaN at larger m")
hits the hang first. When a future SDK upgrade claims a marlin fix, the
verification step is to run registered/quant/test_marlin_moe.py and check
whether config #1 still hangs -- NOT to look at this file.

Usage (CI runs it as ``python3 <file> -f``):
python3 test/registered/ppu/test_ppu_marlin_moe_numerics.py -f
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

register_ppu_ci(est_time=1100, suite="nightly-1-ppu", nightly=True)

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


# CI executes this file as ``python3 <file> -f`` (ci_utils.run_unittest_files),
# so the runner selected below decides what gets executed. The xfail markers
# are gone and no pytest feature remains (no parametrize / skipif / module-level
# test_* functions), so the stdlib unittest runner is correct again -- and it
# restores real self.subTest() semantics: under pytest, subTest degrades to a
# plain statement (result_supports_subtests is False), so the first failing
# dtype aborts the loop and the failing dtype is never reported. With
# unittest.main() every dtype is reported independently.
if __name__ == "__main__":
    unittest.main()
