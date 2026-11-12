from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    make_cpp_args,
)
from sglang.kernels.jit.utils.common import is_sail_runtime
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_per_token_quant_fp8_module(dtype: torch.dtype) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"Unsupported dtype {dtype}. Supported: float16, bfloat16, float32"
        )
    arch = get_jit_cuda_arch()
    use_fast_math = (arch.major, arch.minor) == (9, 0)
    math_mode = "fast_math" if use_fast_math else "precise_math"
    cuda_cflags = ["--use_fast_math"] if use_fast_math else []
    if is_sail_runtime():
        cuda_cflags.append("-DSGL_PER_TOKEN_QUANT_SAIL")
    args = make_cpp_args(dtype)
    return load_jit(
        "per_token_quant_fp8",
        math_mode,
        *args,
        cuda_files=["gemm/per_token_quant_fp8.cuh"],
        cuda_wrappers=[
            ("per_token_quant_fp8", f"per_token_quant_fp8<{args}>"),
            (
                "rmsnorm_per_token_quant_fp8",
                f"rmsnorm_per_token_quant_fp8<{args}>",
            ),
            (
                "rmsnorm_per_token_quant_int8",
                f"rmsnorm_per_token_quant_int8<{args}>",
            ),
        ],
        extra_cuda_cflags=cuda_cflags,
    )


@register_custom_op(
    op_name="per_token_quant_fp8",
    mutates_args=["output_q", "output_s"],
)
def per_token_quant_fp8(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    """Dynamically quantize each row to FP8 E4M3."""
    module = _jit_per_token_quant_fp8_module(input.dtype)
    module.per_token_quant_fp8(input, output_q, output_s.view(output_s.shape[0], 1))


@register_custom_op(
    op_name="rmsnorm_per_token_quant_fp8",
    mutates_args=["output_y", "output_q", "output_s"],
)
def rmsnorm_per_token_quant_fp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    output_y: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    eps: float,
) -> None:
    """RMSNorm each row and quantize the normalized values to FP8 E4M3."""
    module = _jit_per_token_quant_fp8_module(input.dtype)
    module.rmsnorm_per_token_quant_fp8(
        input,
        weight,
        output_y,
        output_q,
        output_s.view(output_s.shape[0], 1),
        eps,
    )


@register_custom_op(
    op_name="rmsnorm_per_token_quant_int8",
    mutates_args=["output_y", "output_q", "output_s"],
)
def rmsnorm_per_token_quant_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    output_y: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    eps: float,
) -> None:
    """RMSNorm each row and quantize it to int8 with max(max|y|, 1e-10)/127."""
    module = _jit_per_token_quant_fp8_module(input.dtype)
    module.rmsnorm_per_token_quant_int8(
        input,
        weight,
        output_y,
        output_q,
        output_s.view(output_s.shape[0], 1),
        eps,
    )


def fused_rmsnorm_per_token_quant_fp8(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm + per-token FP8 quantize in one launch.

    Returns the normalized row (input dtype, contiguous), the FP8 payload and
    its ``[M, 1]`` fp32 per-token scale.
    """
    output_y = torch.empty(
        input.shape,
        dtype=input.dtype,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    output_q = torch.empty(
        input.shape,
        dtype=torch.float8_e4m3fn,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    output_s = torch.empty(
        (input.shape[0], 1), dtype=torch.float32, device=input.device
    )
    rmsnorm_per_token_quant_fp8(input, weight, output_y, output_q, output_s, eps)
    return output_y, output_q, output_s


def fused_rmsnorm_per_token_quant_int8(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm + per-token int8 quantize in one launch.

    Returns the normalized row (input dtype, contiguous), the int8 payload and
    its ``[M, 1]`` fp32 per-token scale, matching per_token_quant_int8's
    ``max(max|y|, 1e-10) / 127`` contract.
    """
    output_y = torch.empty(
        input.shape,
        dtype=input.dtype,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    output_q = torch.empty(
        input.shape,
        dtype=torch.int8,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    output_s = torch.empty(
        (input.shape[0], 1), dtype=torch.float32, device=input.device
    )
    rmsnorm_per_token_quant_int8(input, weight, output_y, output_q, output_s, eps)
    return output_y, output_q, output_s
