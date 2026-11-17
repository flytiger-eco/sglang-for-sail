"""Small-batch PPU HC kernels for Qwen's four 2560-wide BF16 branches.

Large prefill batches keep the existing GEMM and combine paths. The down
projection retains BF16 rounding at the linear and division boundaries; the
combine rounds the updated residual to BF16 before the following RMSNorm.
"""

import torch
import triton
import triton.language as tl

_HC = 4
_HIDDEN = 2560
_LOWRANK = 320
_SPLIT = 16
_MAX_ROWS = 32


@triton.jit
def _hc_down_split_kernel(
    x,
    weight,
    partials,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    rows = tl.arange(0, BLOCK_M)
    k = tl.arange(0, 128)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    # 10240 inputs / 16 splits = five 128-wide tiles per CTA.
    for tile in range(5):
        kk = split * 640 + tile * 128 + k
        a = tl.load(x + rows[:, None] * 10240 + kk[None, :], rows[:, None] < ROWS, 0)
        b = tl.load(
            weight + cols[None, :] * 10240 + kk[:, None], cols[None, :] < 320, 0
        )
        acc = tl.dot(a, b, acc)
    tl.store(
        partials + split * ROWS * 320 + rows[:, None] * 320 + cols[None, :],
        acc,
        (rows[:, None] < ROWS) & (cols[None, :] < 320),
    )


@triton.jit
def _hc_down_reduce_silu_kernel(partials, out, ROWS: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    splits = tl.arange(0, 16)
    values = tl.load(
        partials + splits[:, None] * ROWS * 320 + row * 320 + cols[None, :]
    )
    linear = tl.sum(values, axis=0).to(tl.bfloat16).to(tl.float32)
    scaled = (linear * 0.25).to(tl.bfloat16).to(tl.float32)
    result = scaled / (1.0 + tl.exp(-scaled))
    tl.store(out + row * 320 + cols, result)


def hc_mix_down_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.ndim == 2
        and 1 < x.shape[0] <= _MAX_ROWS
        and x.shape[1] == _HC * _HIDDEN
        and weight.shape == (_LOWRANK, _HC * _HIDDEN)
        and x.is_cuda
        and weight.device == x.device
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.is_contiguous()
    )


def hc_mix_down(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows = x.shape[0]
    partials = torch.empty(
        (_SPLIT, rows, _LOWRANK), device=x.device, dtype=torch.float32
    )
    out = torch.empty((rows, _LOWRANK), device=x.device, dtype=x.dtype)
    block_n = 16 if rows <= 16 else 32
    _hc_down_split_kernel[(triton.cdiv(_LOWRANK, block_n), _SPLIT)](
        x,
        weight,
        partials,
        rows,
        block_n,
        max(16, triton.next_power_of_2(rows)),
        num_warps=4,
        num_stages=2,
    )
    _hc_down_reduce_silu_kernel[(rows, 5)](partials, out, rows, num_warps=4)
    return out


@triton.jit
def _hc_combine_norm_small_kernel(
    normed,
    inject_weight,
    block,
    residual,
    norm_weight,
    updated,
    out,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    branch = tl.program_id(1)
    k = tl.arange(0, 16384)
    n = tl.load(normed + row * 10240 + k, k < 10240, 0).to(tl.float32)
    w = tl.load(inject_weight + branch * 10240 + k, k < 10240, 0).to(tl.float32)
    gate = 2.0 / (1.0 + tl.exp(-tl.sum(n * w, 0) * 0.25))
    j = tl.arange(0, 4096)
    index = (row * 4 + branch) * 2560 + j
    y = tl.load(block + row * 2560 + j, j < 2560, 0).to(tl.float32)
    x = tl.load(residual + index, j < 2560, 0).to(tl.float32)
    # The next branch consumes the stored BF16 residual, not its FP32 precursor.
    value = (x + gate * y).to(tl.bfloat16)
    rounded = value.to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(rounded * rounded, 0) / 2560 + EPS)
    weight = tl.load(norm_weight + branch * 2560 + j, j < 2560, 0).to(tl.float32)
    result = rounded * inv_rms * (1.0 + weight)
    tl.store(updated + index, value, j < 2560)
    tl.store(out + index, result, j < 2560)


def hc_combine_norm_supported(block, residual, normed, inject_weight, norm_weight):
    rows = residual.shape[0]
    return (
        residual.ndim == 2
        and 0 < rows <= _MAX_ROWS
        and residual.shape == normed.shape == (rows, _HC * _HIDDEN)
        and block.shape == (rows, _HIDDEN)
        and inject_weight.shape == (_HC, _HC * _HIDDEN)
        and norm_weight.numel() == _HC * _HIDDEN
        and all(
            t.is_cuda
            and t.device == residual.device
            and t.dtype == torch.bfloat16
            and t.is_contiguous()
            for t in (block, residual, normed, inject_weight, norm_weight)
        )
    )


def hc_combine_norm_small(block, residual, normed, inject_weight, norm_weight, eps):
    updated = torch.empty_like(residual)
    out = torch.empty_like(residual)
    rows = residual.shape[0]
    _hc_combine_norm_small_kernel[(rows, _HC)](
        normed,
        inject_weight,
        block,
        residual,
        norm_weight,
        updated,
        out,
        eps,
        num_warps=8 if rows <= 4 or rows >= 24 else 16,
        enable_fp_fusion=True,
    )
    return updated, out
