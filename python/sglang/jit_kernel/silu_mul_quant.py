"""Fused SiLU+Mul + MXFP4/FP8 quantization kernels (PPU-only, JIT via tvm_ffi)."""
from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit
from sglang.srt.utils.common import is_ppu

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_PPU_ONLY_MSG = "{name} is PPU-only (__ppu_sgmdf); use the generic fallback on non-PPU."

_TP_BLOCK_N = 512
_EP_BLOCK_N = 256
_FP8_BLOCK_THREADS = 256


@cache_once
def _jit_tp_module(apply_clamp: bool) -> Module:
    if not is_ppu():
        raise RuntimeError(_PPU_ONLY_MSG.format(name="silu_and_mul_post_quant_mxfp4"))
    clamp_str = "true" if apply_clamp else "false"
    return load_jit(
        "silu_and_mul_post_quant_mxfp4", clamp_str,
        cuda_files=["elementwise/silu_and_mul_post_quant_mxfp4.cuh"],
        cuda_wrappers=[
            ("run", f"SiluMulMxfp4TP<{_TP_BLOCK_N},{clamp_str}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def silu_and_mul_post_quant_mxfp4(
    gateup: torch.Tensor,
    swiglu_limit: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert gateup.ndim == 2, "input must be 2D (N, 2H)"
    assert gateup.dtype == torch.bfloat16, "input must be bfloat16"
    gateup = gateup.contiguous()

    N, two_H = gateup.shape
    H = two_H // 2
    assert H % _TP_BLOCK_N == 0, f"H must be multiple of {_TP_BLOCK_N}"
    assert (H // 32) % 2 == 0, "H/32 must be even"

    S_pairs = (H // 32) // 2
    out_quant = torch.empty((N, H // 2), dtype=torch.uint8, device=gateup.device)
    out_scale = torch.empty((S_pairs, N), dtype=torch.uint16, device=gateup.device)

    if N == 0:
        return out_quant, out_scale.t()

    apply_clamp = swiglu_limit is not None
    limit_val = float(swiglu_limit) if apply_clamp else 0.0

    module = _jit_tp_module(apply_clamp)
    module.run(gateup, out_quant, out_scale, limit_val)

    return out_quant, out_scale.t()


@cache_once
def _jit_fp8_module(apply_clamp: bool) -> Module:
    if not is_ppu():
        raise RuntimeError(_PPU_ONLY_MSG.format(name="silu_and_mul_post_per_token_quant_fp8"))
    clamp_str = "true" if apply_clamp else "false"
    return load_jit(
        "silu_and_mul_post_per_token_quant_fp8", clamp_str,
        cuda_files=["elementwise/silu_and_mul_post_per_token_quant_fp8.cuh"],
        cuda_wrappers=[
            ("run", f"SiluMulFp8TP<{_FP8_BLOCK_THREADS},{clamp_str}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def silu_and_mul_post_per_token_quant_fp8(
    gateup: torch.Tensor,
    swiglu_limit: Optional[float] = None,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert gateup.ndim == 2, "input must be 2D (M, 2N)"
    assert gateup.dtype == torch.bfloat16, "input must be bfloat16"
    gateup = gateup.contiguous()

    M, two_N = gateup.shape
    N = two_N // 2
    assert two_N % 2 == 0, "input last dim must be even"
    assert N % 8 == 0, f"N must be multiple of 8, got N={N}"

    output = torch.empty(
        (M, N), dtype=torch.float8_e4m3fn, device=gateup.device
    )
    output_scale = torch.empty(
        (M, 1), dtype=torch.float32, device=gateup.device
    )

    if M == 0:
        return output, output_scale

    apply_clamp = swiglu_limit is not None
    limit_val = float(swiglu_limit) if apply_clamp else 0.0

    module = _jit_fp8_module(apply_clamp)
    module.run(gateup, output, output_scale, limit_val, float(eps))

    return output, output_scale


@cache_once
def _jit_ep_module(apply_clamp: bool) -> Module:
    if not is_ppu():
        raise RuntimeError(_PPU_ONLY_MSG.format(name="silu_and_mul_masked_post_quant_mxfp4"))
    clamp_str = "true" if apply_clamp else "false"
    return load_jit(
        "silu_and_mul_masked_post_quant_mxfp4", clamp_str,
        cuda_files=["elementwise/silu_and_mul_masked_post_quant_mxfp4.cuh"],
        cuda_wrappers=[
            ("run", f"SiluMulMxfp4EP<{_EP_BLOCK_N},{clamp_str}>::run"),
        ],
        extra_cuda_cflags=["-use_fast_math"],
    )


def silu_and_mul_masked_post_quant_mxfp4(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    masked_m: torch.Tensor,
    swiglu_limit: Optional[float] = None,
) -> None:
    assert input.dtype == torch.bfloat16, "input must be bfloat16"
    assert output.dtype == torch.uint8, "output must be uint8"
    assert output_scale.dtype == torch.uint16, "output_scale must be uint16"
    assert masked_m.dtype == torch.int32, "masked_m must be int32"

    apply_clamp = swiglu_limit is not None
    limit_val = float(swiglu_limit) if apply_clamp else 0.0

    module = _jit_ep_module(apply_clamp)
    module.run(input, output, output_scale, masked_m, limit_val)
