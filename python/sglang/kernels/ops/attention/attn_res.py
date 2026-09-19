"""CUDA JIT wrapper for the Kimi-K3 SM100 attention-residual kernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
    override_jit_cuda_arch,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_DIM: int = 7168  # K3 hidden size, template parameter of the TMA kernel
_MAX_BANK_ROWS: int = 8  # K3 has <= 8 snapshots, upper bound of the nvb dispatch tables


def _make_name(*args):
    return "kimi_k3_attn_res_" + "_".join(str(a) for a in args)


@cache_once
def _jit_fused_tma_module(
    chunk_rows: int, occupancy: int, consumer_regs: int
) -> Module:
    """Compile and cache the warp-specialized TMA aggregation kernel (per-row
    bulk copies into chunk slots; chunk_rows / occupancy / consumer_regs are
    tuning knobs). The smem ring is frozen at 2 chunk slots and PDL is always
    on: the kernel targets SM100+ except SM12x."""
    major, minor = torch.cuda.get_device_capability()
    if major < 10 or major == 12:
        raise RuntimeError(
            "attn_res_fused_tma requires SM100+ excluding SM12x; "
            f"SM{major}{minor} is unsupported"
        )
    args = make_cpp_args(
        _DIM,
        _MAX_BANK_ROWS,
        chunk_rows,
        occupancy,
        consumer_regs,
    )
    with override_jit_cuda_arch(major, minor, suffix="a"):
        return load_jit(
            _make_name("fused_tma"),
            *args,
            cuda_files=["kimi_k3/attn_res/fused_tma.cuh"],
            cuda_wrappers=[
                ("run", f"AttnResFusedTmaKernel<{args}>::run"),
                ("run_pull_rs", f"AttnResFusedTmaKernel<{args}>::run_pull_rs"),
                ("run_direct_ag", f"AttnResFusedTmaKernel<{args}>::run_direct_ag"),
            ],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )


# Benchmarked-best (chunk_rows, occupancy, consumer_regs) per nvb
# (GB200/GB300-class, H=7168). nvb=1 is latency-bound per token, so 2 CTAs/SM
# (occupancy=2, which excludes setmaxnreg: 4*Nc + 2*Np > 512) wins large T by
# ~17%; nvb=4/8 fill 5-row chunks exactly ((nvb+1) % 5 == 0 or covers it in
# 1-2 chunks); everything else is fastest on the balanced 4-row chunk. All
# entries use the setmaxnreg producer/consumer split. consumer_regs sits on
# the 200..232 performance plateau (below 200 the consumer loop starves); we
# take 200, not the 232 budget cap, so each SMSP keeps 2K registers free for
# PDL-overlapped neighbor kernels instead of allocating them idle.
_TMA_BEST_CONFIG: dict[int, tuple[int, int, int]] = {
    1: (2, 2, 0),
    2: (4, 1, 200),
    3: (4, 1, 200),
    4: (5, 1, 200),
    5: (3, 1, 200),
    6: (4, 1, 200),
    7: (4, 1, 200),
    8: (5, 1, 200),
}


def _tuning(nvb: int, num_tokens: int) -> tuple[int, int, int]:
    """(chunk_rows, occupancy, consumer_regs) for this aggregation point.

    Shared by attn_res_fused_tma and its collective variants in
    communication.attn_res_collective: they run the same kernel template and
    differ only in the collective fused onto it."""
    if not 1 <= nvb <= _MAX_BANK_ROWS:
        raise ValueError(f"attn_res: nvb must be in [1, {_MAX_BANK_ROWS}], got {nvb}")
    best = _TMA_BEST_CONFIG[nvb]
    if best[1] > 1 and num_tokens < 128:
        # occupancy=2 only pays off once there are enough tokens to fill both
        # CTAs per SM; below that its tighter register budget just costs ~10%.
        best = (4, 1, 200)
    return best


def attn_res_fused_tma(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    cw: torch.Tensor,
    ow: torch.Tensor,
    out: torch.Tensor,
    nvb: int,
    eps: float,
    *,
    write_prefix: bool = False,
) -> None:
    """Warp-specialized TMA aggregation (score -> online softmax -> weighted
    combine -> fused output RMSNorm), one persistent CTA per SM: a producer
    warp (group) fetches rows with one bulk copy each into chunk slots of
    `chunk_rows` rows (one barrier pair per chunk, double-buffered ring of 2
    slots); the 8 consumer warps score and fold one chunk per rendezvous,
    with cw / ow staged in TMEM.

    Restrictions: H == 7168, nvb in [1, 8], SM100a+. The launch config comes
    from _TMA_BEST_CONFIG via _tuning() (each combination compiles its own
    module).

    Parameters
    ----------
    prefix_sum : [T, H] bf16
    bank       : [T, NB, H] bf16 (rows 0..nvb-1 are aggregated)
    cw         : [H] bf16 — precomputed score_norm_weight * proj_weight
    ow         : [H] bf16 — output RMSNorm weight
    out        : [T, H] bf16 output buffer
    nvb        : number of valid bank rows (1..8)
    eps        : RMSNorm epsilon (shared by score and output norms)
    write_prefix : also snapshot the prefix row into bank[:, nvb, :]
                 (bit-exact copy, fused into the score pass which already
                 has the row in registers); requires NB > nvb
    """
    _jit_fused_tma_module(*_tuning(nvb, prefix_sum.shape[0])).run(
        prefix_sum, bank, cw, ow, out, nvb, eps, write_prefix
    )


# ---- SM80/SM89 fused aggregation --------------------------------------------


@triton.jit
def _aggregate_sm8x_kernel(
    prefix_ptr,  # [T, H]
    delta_ptr,  # [T, H] when HAS_DELTA
    prefix_out_ptr,  # [T, H], receives prefix + delta when HAS_DELTA
    bank_ptr,  # [T, NB_total, H]
    cw_ptr,  # [H], fp32
    out_weight_ptr,  # [H]
    out_ptr,  # [T, H]
    stride_pm: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_om: tl.constexpr,
    NVB: tl.constexpr,
    EPS: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    WRITE_BANK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One CTA per token for the SM80/SM89 K3 attention-residual path.

    The kernel deliberately rounds the pending prefix add back to the source
    dtype. This is the value consumed by the old eager add plus fallback and
    must also be the value retained by AttnResidual as its running prefix.
    """
    pid_t = tl.program_id(0).to(tl.int64)
    offs_h = tl.max_contiguous(tl.arange(0, BLOCK_H), BLOCK_H)
    mask_h = offs_h < H

    updated_prefix = tl.load(
        prefix_ptr + pid_t * stride_pm + offs_h, mask=mask_h, other=0.0
    ).to(tl.float32)
    if HAS_DELTA:
        delta = tl.load(
            delta_ptr + pid_t * stride_dm + offs_h, mask=mask_h, other=0.0
        ).to(tl.float32)
        updated_prefix = updated_prefix + delta
        updated_prefix = updated_prefix.to(prefix_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            prefix_out_ptr + pid_t * stride_pm + offs_h, updated_prefix, mask=mask_h
        )

    if WRITE_BANK:
        tl.store(
            bank_ptr + pid_t * stride_bm + NVB * stride_bb + offs_h,
            updated_prefix,
            mask=mask_h,
        )

    # A block write with NVB == 0 is the first K3 attention-residual point:
    # the sole source has softmax weight one, so it needs no score pass.
    if NVB == 0:
        mixed = updated_prefix
    else:
        # Do not retain the 8192-wide prefix vector over the source loop.
        # Selecting source pointers lets Triton reload it in the prefix tile,
        # avoiding the severe register pressure observed on SM89/PPU.
        if HAS_DELTA:
            tl.debug_barrier()
            prefix_source_ptr = prefix_out_ptr
        else:
            prefix_source_ptr = prefix_ptr
        cw = tl.load(cw_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
        max_score = tl.full((), -float("inf"), tl.float32)
        denom = tl.zeros((), tl.float32)
        mixed = tl.zeros([BLOCK_H], tl.float32)

        # Process bank rows plus current prefix in small tiles. Online softmax
        # avoids materializing scores and preserves a single kernel launch.
        for row0 in tl.static_range(0, NVB + 1, BLOCK_ROWS):
            offs_r = row0 + tl.arange(0, BLOCK_ROWS)
            row_mask = offs_r <= NVB
            is_prefix = offs_r == NVB
            bank_ptrs = (
                bank_ptr
                + pid_t * stride_bm
                + offs_r[:, None] * stride_bb
                + offs_h[None, :]
            )
            prefix_ptrs = (
                prefix_source_ptr
                + pid_t * stride_pm
                + offs_r[:, None] * 0
                + offs_h[None, :]
            )
            values = tl.load(
                tl.where(is_prefix[:, None], prefix_ptrs, bank_ptrs),
                mask=row_mask[:, None] & mask_h[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            inv_rms = tl.rsqrt(tl.sum(values * values, axis=1) * (1.0 / H) + EPS)
            scores = tl.sum(values * cw[None, :], axis=1) * inv_rms
            scores = tl.where(row_mask, scores, -float("inf"))

            new_max = tl.maximum(max_score, tl.max(scores, axis=0))
            old_scale = tl.exp(max_score - new_max)
            scales = tl.exp(scores - new_max)
            denom = denom * old_scale + tl.sum(scales, axis=0)
            mixed = mixed * old_scale + tl.sum(scales[:, None] * values, axis=0)
            max_score = new_max

        mixed = mixed / denom

    out_inv_rms = tl.rsqrt(tl.sum(mixed * mixed, axis=0) * (1.0 / H) + EPS)
    out_weight = tl.load(out_weight_ptr + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + pid_t * stride_om + offs_h,
        mixed * out_inv_rms * out_weight,
        mask=mask_h,
    )


def attn_res_fused_sm8x(
    prefix_sum: torch.Tensor,
    bank: torch.Tensor,
    nvb: int,
    cw: torch.Tensor,
    out_weight: torch.Tensor,
    eps: float,
    delta: torch.Tensor | None = None,
    write_bank_row: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate, normalize and optionally snapshot a 7168-wide SM8x prefix."""
    assert prefix_sum.shape[1] == 7168
    assert prefix_sum.is_contiguous() and bank.is_contiguous()
    assert delta is None or delta.is_contiguous()
    if prefix_sum.shape[0] == 0:
        return torch.empty_like(prefix_sum), prefix_sum
    T, H = prefix_sum.shape
    out = torch.empty_like(prefix_sum)
    prefix = torch.empty_like(prefix_sum) if delta is not None else prefix_sum
    block_rows = 1 if T >= 256 or nvb <= 1 else 4
    _aggregate_sm8x_kernel[(T,)](
        prefix_sum,
        prefix_sum if delta is None else delta,
        prefix,
        bank,
        cw,
        out_weight,
        out,
        prefix_sum.stride(0),
        0 if delta is None else delta.stride(0),
        bank.stride(0),
        bank.stride(1),
        out.stride(0),
        NVB=nvb,
        EPS=eps,
        HAS_DELTA=delta is not None,
        WRITE_BANK=write_bank_row,
        BLOCK_ROWS=block_rows,
        H=H,
        BLOCK_H=triton.next_power_of_2(H),
        num_warps=4 if block_rows == 1 else 8,
        num_stages=2,
    )
    return out, prefix
