# SPDX-License-Identifier: Apache-2.0
"""Qwen38 Hopper W4A8 kernels and load-time checkpoint transformations.

The checkpoint uses signed two's-complement INT4 with adjacent K nibbles.
The kernel uses sign-magnitude INT4, packed K order [0,4,1,5,2,6,3,7],
CUTLASS lane interleaving, BF16 scales multiplied by 512, and interleaved
gate/up rows for fusion.
No transformation in this module belongs in the timed forward path.
"""

from pathlib import Path
from typing import NamedTuple

import torch

_loaded = False


def load_w4a8_sm_ops() -> None:
    global _loaded
    if _loaded:
        return
    library = Path(__file__).with_name("w4a8_sm_ops.so")
    if not library.is_file():
        raise RuntimeError(
            "Qwen38 W4A8 extension is missing. Build/install sglang-kernel "
            "from this checkout with SGL_KERNEL_BUILD_W4A8_SM=ON."
        )
    torch.ops.load_library(str(library))
    _loaded = True


class W4A8SMWeights(NamedTuple):
    weight: torch.Tensor
    scales: torch.Tensor
    scales_k512: torch.Tensor | None


def pack_w4a8_scales(scales: torch.Tensor, alignment: int) -> torch.Tensor:
    """Pack logical [E,N,K/128] scales for the selected mainloop tile K."""
    if scales.ndim != 3 or alignment not in (1, 2, 4):
        raise ValueError("scales must be [E,N,K/128], alignment must be 1, 2 or 4")
    e, n, groups = scales.shape
    if groups % alignment:
        raise ValueError("scale groups must be divisible by the packing alignment")
    return (
        scales.reshape(e, n, groups // alignment, alignment)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(e, groups // alignment, n * alignment)
    )


@torch.no_grad()
def prepare_w4a8_sm_weight(
    weight: torch.Tensor, scales: torch.Tensor, *, fused_gate_up: bool = False
) -> W4A8SMWeights:
    """Repack one logical checkpoint projection once, without requantizing it.

    ``weight`` is int8 [E,N,K/2], low nibble first, two's complement.
    ``scales`` is unpermuted floating point [E,N,K/128]. Work is bounded to
    one expert at a time. INT4 -8 is rejected: the sign-magnitude fast path
    represents [-7,7]. Callers may retain the original upstream path instead.
    """
    if weight.dtype != torch.int8 or weight.ndim != 3 or not weight.is_contiguous():
        raise ValueError("weight must be contiguous packed int8 [E,N,K/2]")
    e, n, half_k = weight.shape
    k = half_k * 2
    expected = (1280, 2560) if fused_gate_up else (2560, 640)
    if (n, k) != expected:
        raise ValueError(f"Qwen38 projection must have (N,K)={expected}, got {(n, k)}")
    if scales.shape != (e, n, k // 128) or scales.device != weight.device:
        raise ValueError("scales must be [E,N,K/128] on the weight device")
    if not scales.is_floating_point():
        raise ValueError("weight scales must be floating point")
    if not bool(torch.isfinite(scales).all()) or bool((scales < 0).any()):
        raise ValueError("weight scales must be finite and nonnegative")
    # This validation is deliberately before converting any checkpoint tensor.
    for expert in weight:
        u = expert.to(torch.uint8)
        if bool(((u & 15) == 8).any()) or bool(((u >> 4) == 8).any()):
            raise ValueError(
                "INT4 checkpoint contains -8; Ours sign-magnitude supports [-7,7]"
            )
    perm = torch.arange(n, device=weight.device)
    if fused_gate_up:
        perm = (perm // 16) * 8 + perm % 8 + (perm % 16 >= 8) * (n // 2)
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=weight.device)
    packed = torch.empty_like(weight)
    for i in range(e):
        u = weight[i].to(torch.uint8)
        nibble = torch.stack((u & 15, u >> 4), dim=-1).reshape(n, k)
        signed = nibble.to(torch.int16)
        signed = torch.where(signed >= 8, signed - 16, signed)
        sm = torch.where(signed < 0, 8 - signed, signed).to(torch.uint8)
        # The MMA fragment interleave crosses pairs of eight N rows and K/4.
        # Applying the register permutation directly to adjacent K would corrupt
        # the A/B pairing. Preserve the complete frozen benchmark layout.
        sm = sm[perm].reshape(n // 16, 2, 8, k // 4, 4).permute(0, 2, 3, 1, 4)
        sm = sm.contiguous().reshape(n // 16, 8, k // 4, 8)[..., order]
        sm = sm.reshape(n // 16, 8, k // 4, 2, 4).permute(0, 3, 1, 2, 4)
        sm = sm.contiguous().reshape(n, k)
        packed[i].copy_((sm[:, 0::2] | (sm[:, 1::2] << 4)).view(torch.int8))
    scaled = (scales[:, perm].float() * 512).to(torch.bfloat16)
    if not bool(torch.isfinite(scaled).all()):
        raise ValueError("scaled weight scales overflow BF16")
    return W4A8SMWeights(
        packed,
        pack_w4a8_scales(scaled, 2 if fused_gate_up else 1),
        pack_w4a8_scales(scaled, 4) if fused_gate_up else None,
    )


def select_w4a8_sm_config(
    projection: str, local_rows: int, num_experts: int
) -> tuple[int, int]:
    """Return (configuration id, scale alignment) using host-known row hints.

    The table comes from frozen H800 EP4 prefill/decode configurations. The
    hint affects scheduling only; device problem_sizes remain authoritative.
    No device tensor is copied to the host, including during CUDA graph capture.
    """
    if local_rows < 0 or num_experts <= 0:
        raise ValueError("invalid routing shape hint")
    if projection not in ("gate_up", "down"):
        raise ValueError("projection must be gate_up or down")
    mean_rows = local_rows / num_experts
    if projection == "gate_up":
        if mean_rows <= 32:
            if num_experts == 128 and local_rows in (8, 20, 30, 60):
                return 6, 4
            return 0, 2
        if mean_rows <= 80:
            return 1, 2
        if mean_rows <= 100:
            return 2, 2
        if mean_rows <= 120:
            return 3, 2
        if mean_rows <= 140:
            return 4, 2
        return 5, 2
    if mean_rows <= 32:
        return (4 if local_rows <= 5 else 0), 1
    if mean_rows <= 80:
        return 1, 1
    if mean_rows <= 120:
        return 2, 1
    return 3, 1


def cutlass_w4a8_sm_moe_mm(
    d,
    a,
    b,
    a_scales,
    b_scales,
    expert_offsets,
    problem_sizes,
    a_strides,
    b_strides,
    d_strides,
    s_strides,
    *,
    config_id=0,
):
    """Down GEMM, using prepared packed-K weights; produces BF16."""
    load_w4a8_sm_ops()
    torch.ops.sgl_kernel.qwen38_w4a8_sm_mm(
        d,
        a,
        b,
        a_scales,
        b_scales,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        d_strides,
        s_strides,
        config_id,
    )


def cutlass_w4a8_sm_moe_mm_swiglu(
    d,
    a,
    b,
    a_scales,
    b_scales,
    out_scales,
    expert_offsets,
    problem_sizes,
    a_strides,
    b_strides,
    d_strides,
    s_strides,
    *,
    config_id=0,
):
    """Gate-up GEMM + SwiGLU + static E4M3 quantization, with prepared weights."""
    load_w4a8_sm_ops()
    torch.ops.sgl_kernel.qwen38_w4a8_sm_swiglu(
        d,
        a,
        b,
        a_scales,
        b_scales,
        out_scales,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        d_strides,
        s_strides,
        config_id,
    )
