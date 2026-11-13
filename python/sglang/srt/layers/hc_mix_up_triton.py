"""HC mix up projection with the sigmoid-gate/mean-over-hc epilogue fused in.

The prefill mix in ``GatedResidual`` ends with

    u   = act @ W_up^T                     # [M, hc*hs], act = silu(down / hc)
    out = mean_g(sigmoid(u[:, g*hs + j]) * x[:, g*hs + j])   # [M, hs]

Unfused, the up GEMM writes the whole ``[M, hc*hs]`` tensor and a separate
elementwise kernel reads it back together with ``x`` (the normed stream).
Here each program owns one hidden block ``j`` for *all* hc branches, i.e. the
up GEMM's N dimension is regrouped as (hidden block, hc) so the mean over hc
is a tile-local reduction; only ``[M, hs]`` is written.

The regrouping is done in index space: the tile keeps one accumulator per
branch fed by rows ``g*hs + j`` of the original ``W_up``, so no permuted copy
of the weight is materialized.  On PPU this beats a single wide
``[BM, hc*BJ]`` dot over an hc-innermost permuted weight, whose epilogue
reshape/split costs more than the extra tile-local dots.  Sigmoid uses the
fast exp/divide intrinsics: with the fp32 accumulator the result is rounded
to bf16 anyway, and the precise-division sigmoid is what bounds the unfused
elementwise kernel on this part.

Unlike ``hc_mix_triton.fused_hc_mix`` (persistent, grid barrier, decode) this
is an ordinary tiled GEMM and targets prefill-size row counts; it also beats
the torch.compile chain at decode sizes that miss the persistent kernel.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_HC = 4
_MIN_BLOCK_K = 32


@triton.jit
def _fast_sigmoid(x):
    return libdevice.fast_dividef(1.0, 1.0 + libdevice.fast_expf(-x))


@triton.jit
def _hc_mix_up_kernel(
    act_ptr,
    w_up_ptr,
    x_ptr,
    out_ptr,
    M,
    R,
    HS,
    inv_hc,
    HC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    tl.static_assert(HC == 4)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_j = HS // BLOCK_J
    width = GROUP_M * num_pid_j
    first_m = (pid // width) * GROUP_M
    group_size = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % width) % group_size
    pid_j = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Tail rows re-read row M-1 so the mainloop loads stay unmasked; only the
    # store is masked.
    rows = tl.minimum(offs_m, M - 1).to(tl.int64)
    offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_k = tl.arange(0, BLOCK_K)

    acc0 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_J), dtype=tl.float32)
    a_ptrs = act_ptr + rows[:, None] * R + offs_k[None, :]
    w_ptrs = w_up_ptr + offs_j[:, None] * R + offs_k[None, :]
    w_branch = HS * R
    for _ in range(0, R, BLOCK_K):
        a = tl.load(a_ptrs)
        acc0 = tl.dot(a, tl.trans(tl.load(w_ptrs)), acc0)
        acc1 = tl.dot(a, tl.trans(tl.load(w_ptrs + w_branch)), acc1)
        acc2 = tl.dot(a, tl.trans(tl.load(w_ptrs + 2 * w_branch)), acc2)
        acc3 = tl.dot(a, tl.trans(tl.load(w_ptrs + 3 * w_branch)), acc3)
        a_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    x_ptrs = x_ptr + rows[:, None] * (HC * HS) + offs_j[None, :]
    out = _fast_sigmoid(acc0) * tl.load(x_ptrs).to(tl.float32)
    out += _fast_sigmoid(acc1) * tl.load(x_ptrs + HS).to(tl.float32)
    out += _fast_sigmoid(acc2) * tl.load(x_ptrs + 2 * HS).to(tl.float32)
    out += _fast_sigmoid(acc3) * tl.load(x_ptrs + 3 * HS).to(tl.float32)
    tl.store(
        out_ptr + offs_m[:, None].to(tl.int64) * HS + offs_j[None, :],
        (out * inv_hc).to(out_ptr.dtype.element_ty),
        mask=(offs_m < M)[:, None],
    )


def _pick_config(rows: int, lowrank: int):
    # (BLOCK_M, BLOCK_J, BLOCK_K, num_warps, num_stages), tuned on ZW-M890P
    # for hc=4 / hs=2560 / lowrank=320. Decode-size rows have sharp occupancy
    # transitions as the M grid gains another wave, so keep the measured bands.
    if lowrank % _MIN_BLOCK_K != 0:
        raise ValueError(f"lowrank must be divisible by {_MIN_BLOCK_K}, got {lowrank}")
    if lowrank != 320:
        if rows <= 128:
            config = (32, 64, 32, 8, 4)
        elif rows <= 512:
            config = (64, 32, 32, 8, 3)
        else:
            config = (64, 64, 32, 8, 4)
    elif rows <= 64:
        config = (32, 64, 32, 8, 4)
    elif rows <= 96:
        config = (32, 32, 64, 4, 2)
    elif rows <= 128:
        config = (64, 32, 64, 4, 3)
    elif rows <= 192:
        config = (64, 64, 32, 8, 3)
    elif rows <= 256:
        config = (64, 32, 32, 8, 3)
    elif rows <= 320:
        config = (64, 64, 64, 8, 3)
    elif rows <= 512:
        config = (64, 32, 32, 8, 3)
    else:
        config = (64, 64, 32, 8, 4)

    block_k = config[2]
    if lowrank % block_k != 0:
        config = (*config[:2], _MIN_BLOCK_K, *config[3:])
    return config


def hc_mix_up_shape_ok(hc_count: int, hidden_size: int, lowrank: int) -> bool:
    return hc_count == _HC and hidden_size % 64 == 0 and lowrank % _MIN_BLOCK_K == 0


def hc_mix_up_supported(hyper_input_normed: torch.Tensor, w_up: torch.Tensor) -> bool:
    return (
        hyper_input_normed.is_cuda
        and hyper_input_normed.dim() == 2
        and hyper_input_normed.dtype in (torch.bfloat16, torch.float16)
        and w_up.dtype == hyper_input_normed.dtype
        and hyper_input_normed.is_contiguous()
        and w_up.is_contiguous()
    )


def hc_mix_up(
    act: torch.Tensor,
    w_up: torch.Tensor,
    hyper_input_normed: torch.Tensor,
    hc: int,
    hs: int,
) -> torch.Tensor:
    """out[m, j] = mean_g(sigmoid(act[m] @ w_up[g*hs + j]) * x[m, g*hs + j]).

    act                : [M, lowrank], silu(down / hc)
    w_up               : [hc*hs, lowrank], ``input_mix_weight_up.weight``
    hyper_input_normed : [M, hc*hs]
    """
    rows, lowrank = act.shape
    act = act.contiguous()
    out = torch.empty(
        (rows, hs), dtype=hyper_input_normed.dtype, device=hyper_input_normed.device
    )
    if rows == 0:
        return out
    block_m, block_j, block_k, num_warps, num_stages = _pick_config(rows, lowrank)
    grid = (triton.cdiv(rows, block_m) * (hs // block_j),)
    _hc_mix_up_kernel[grid](
        act,
        w_up,
        hyper_input_normed,
        out,
        rows,
        lowrank,
        hs,
        1.0 / hc,
        HC=hc,
        BLOCK_M=block_m,
        BLOCK_J=block_j,
        BLOCK_K=block_k,
        GROUP_M=8,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
