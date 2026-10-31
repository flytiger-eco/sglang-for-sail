# SPDX-License-Identifier: Apache-2.0
"""Triton bf16/fp16 -> fp32 widening cast for logits.

At large vocab widths the aten ``unrolled_elementwise direct_copy`` path stays
well below memcpy bandwidth. These vectorized widening-cast kernels run close
to the copy anchor while remaining bit-identical to ``.float()`` / ``copy_()``
(bf16 -> fp32 widening is exact), so the numerics contract is unchanged.

Deliberately outside the kernel registry/selector machinery: single-op JIT
with an aten fallback at the call site (logits_processor).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Tuned tile size / warp count for the logits width range.
BLOCK = 2048
NUM_WARPS = 4


@triton.jit
def bf16_to_fp32_cast_kernel(
    x_ptr,
    y_ptr,
    width,
    stride_x_row,
    stride_y_row,
    BLOCK: tl.constexpr,
):
    """Row-tiled widening cast: program (row, col-block) covers
    ``x[row, col] -> y[row, col].astype(fp32)``. Used for sliced (row-strided)
    sources; the contiguous hot path uses the flat 1D kernel below."""
    row = tl.program_id(0).to(tl.int64)
    cb = tl.program_id(1)
    offs = cb * BLOCK + tl.arange(0, BLOCK)
    mask = offs < width
    x = tl.load(x_ptr + row * stride_x_row + offs, mask=mask)
    tl.store(y_ptr + row * stride_y_row + offs, x.to(tl.float32), mask=mask)


@triton.jit
def bf16_to_fp32_cast_flat_kernel(x_ptr, y_ptr, numel, BLOCK: tl.constexpr):
    """Flat 1D widening cast for fully contiguous tensors (the hot path).
    Kept separate from the row-tiled kernel because the generalization costs
    measurable bandwidth on contiguous inputs."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x.to(tl.float32), mask=mask)


def _eligible(src: torch.Tensor) -> bool:
    return (
        src.is_cuda
        and src.dim() == 2
        and src.dtype in (torch.bfloat16, torch.float16)
        and src.stride(1) == 1
    )


def cast_bf16_to_fp32(src: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Widen ``src`` [M, W(any dtype width)] into fp32 ``out`` (same shape),
    in place. Falls back to aten ``copy_`` for layouts the tile rules exclude."""
    if not _eligible(src) or out.shape != src.shape or out.dtype != torch.float32:
        return out.copy_(src)
    numel = src.numel()
    if src.is_contiguous() and out.is_contiguous():
        grid = (triton.cdiv(numel, BLOCK),)
        bf16_to_fp32_cast_flat_kernel[grid](
            src, out, numel, BLOCK=BLOCK, num_warps=NUM_WARPS
        )
        return out
    m, w = src.shape
    grid = (m, triton.cdiv(w, BLOCK))
    bf16_to_fp32_cast_kernel[grid](
        src,
        out,
        w,
        src.stride(0),
        out.stride(0),
        BLOCK=BLOCK,
        num_warps=NUM_WARPS,
    )
    return out


def cast_bf16_to_fp32_new(src: torch.Tensor) -> torch.Tensor:
    """Widen ``src`` into a fresh fp32 tensor (allocation semantics mirror
    ``.float()``; caching allocator makes empty() ~free)."""
    if not _eligible(src):
        return src.float()
    out = torch.empty(src.shape, dtype=torch.float32, device=src.device)
    return cast_bf16_to_fp32(src, out)
