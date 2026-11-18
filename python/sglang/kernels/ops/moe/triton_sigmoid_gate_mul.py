"""Fused sigmoid-gate-multiply Triton kernels.

Two variants:
- ``sigmoid_gate_mul``: element-wise ``x * sigmoid(gate)`` when x and gate
  have identical shapes.
- ``sigmoid_gate_mul_broadcast``: broadcast ``x * sigmoid(gate)`` when gate
  is ``(N, 1)`` and x is ``(N, D)``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def _sigmoid_gate_mul_kernel(
    x_ptr,
    gate_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    g = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    out = x * tl.sigmoid(g)
    tl.store(out_ptr + offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


def sigmoid_gate_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute ``x * sigmoid(gate)`` in a single fused kernel (same-shape)."""
    assert x.shape == gate.shape, f"shape mismatch: {x.shape=} {gate.shape=}"
    assert x.is_contiguous() and gate.is_contiguous(), "inputs must be contiguous"
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    _sigmoid_gate_mul_kernel[grid](x, gate, out, n, BLOCK_SIZE=1024)
    return out


@triton.jit
def _sigmoid_gate_mul_broadcast_kernel(
    out_ptr,
    gate_ptr,
    x_ptr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.load(gate_ptr + row).to(tl.float32)
    g = tl.sigmoid(g)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_dim
    x = tl.load(x_ptr + row * hidden_dim + offs, mask=mask).to(tl.float32)
    out = x * g
    tl.store(
        out_ptr + row * hidden_dim + offs,
        out.to(x_ptr.dtype.element_ty),
        mask=mask,
    )


def sigmoid_gate_mul_broadcast(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute ``x * sigmoid(gate)`` where gate is (N, 1) and x is (N, D)."""
    bs, hidden_dim = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    max_warps = 16 if _is_hip else 32
    num_warps = max(
        min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
    )
    _sigmoid_gate_mul_broadcast_kernel[(bs,)](
        out,
        gate,
        x,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _sigmoid_gate_mul_broadcast_bf16_kernel(X, G, Y, M, N: tl.constexpr):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    cols = tl.program_id(1) * 256 + tl.arange(0, 256)
    gate = tl.load(G + rows, rows < M, 0).to(tl.float32)
    # Eager sigmoid rounds to BF16 before multiplying; keep that boundary.
    gate = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    value = tl.load(X + rows[:, None] * N + cols[None, :], mask, 0).to(tl.float32)
    tl.store(Y + rows[:, None] * N + cols[None, :], gate[:, None] * value, mask)


def sigmoid_gate_mul_broadcast_bf16(
    x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """Fuse the eager BF16 sigmoid/broadcast product, retaining its rounding.

    Other dtypes/layouts preserve the eager path. Unlike the HIP helper above,
    the gate is rounded before multiplication rather than only at the output.
    """
    supported = (
        x.ndim == 2
        and gate.shape == (x.shape[0], 1)
        and x.is_cuda
        and gate.device == x.device
        and x.dtype == gate.dtype == torch.bfloat16
        and x.is_contiguous()
        and gate.is_contiguous()
    )
    if not supported:
        return torch.sigmoid(gate) * x
    out = torch.empty_like(x)
    if x.numel():
        _sigmoid_gate_mul_broadcast_bf16_kernel[
            (triton.cdiv(x.shape[0], 4), triton.cdiv(x.shape[1], 256))
        ](x, gate, out, x.shape[0], x.shape[1], num_warps=4, enable_fp_fusion=False)
    return out
