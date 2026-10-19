"""Triton gather of DeepSeek-V4.1 engram rows: fp8 e4m3 payload, e8m0 block scales.

The table pointers arrive as raw addresses so one kernel serves a device table, a
pinned host table, or (on Grace-Blackwell, through ATS) a plain host mapping. The
output is bf16 computed as fp32(row) * 2**(exp - 127) then rounded once, which is
the arithmetic of the torch lookup it replaces.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_device_sm, is_ppu

# e8m0 has no zero: the exponent byte 0 encodes 2**-127.
_E8M0_ZERO = 2.0**-127


@triton.jit
def _e4m3fn_to_float(payload):
    # Decode raw E4M3FN bytes without introducing an FP8 type into Triton IR.
    # This also covers SM80 PPU, where float8e4nv is rejected during lowering.
    bits = payload.to(tl.uint32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    normal = ((exponent + 120) << 23) | (mantissa << 20)
    subnormal = (mantissa.to(tl.float32) * (1.0 / 512.0)).to(tl.uint32, bitcast=True)
    magnitude = tl.where(exponent == 0, subnormal, normal)
    # E4M3FN has finite exponent-15 values through 448, and two signed NaNs.
    magnitude = tl.where((bits & 127) == 127, 0x7FC00000, magnitude)
    return (magnitude | ((bits & 128) << 24)).to(tl.float32, bitcast=True)


@triton.jit
def _engram_gather_kernel(
    w_ptr,
    s_ptr,
    ids_ptr,
    out_ptr,
    row_lo,
    row_hi,
    DIM: tl.constexpr,
    BLK: tl.constexpr,
    E8M0_ZERO: tl.constexpr,
    DECODE_E4M3_BYTES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    idx = tl.load(ids_ptr + row).to(tl.int64)
    # The table holds rows [row_lo, row_hi); an id outside it is not read and
    # comes out as zeros, which is what the sharded all-reduce sums.
    owned = (idx >= row_lo) & (idx < row_hi)
    local = tl.where(owned, idx - row_lo, 0)
    s = s_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    offs = tl.arange(0, DIM)
    if DECODE_E4M3_BYTES:
        w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
        payload = tl.load(w + local * DIM + offs, mask=owned, other=0)
        vals = _e4m3fn_to_float(payload)
    else:
        w = w_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
        vals = tl.load(w + local * DIM + offs, mask=owned, other=0.0).to(tl.float32)
    exps = tl.load(s + local * (DIM // BLK) + offs // BLK, mask=owned, other=0).to(
        tl.int32
    )
    # 2**(e - 127) from the exponent bits: exact, no exp2 rounding or denormal flush.
    scale = (exps << 23).to(tl.float32, bitcast=True)
    scale = tl.where(exps == 0, E8M0_ZERO, scale)
    scale = tl.where(exps == 255, float("nan"), scale)
    out = tl.where(owned, vals * scale, 0.0)
    tl.store(out_ptr + row * DIM + offs, out.to(tl.bfloat16))


def engram_gather(
    weight_ptr: int,
    scale_ptr: int,
    ids: torch.Tensor,
    out: torch.Tensor,
    dim: int,
    block_size: int,
    row_lo: int = 0,
    row_hi: int = 2**62,
) -> torch.Tensor:
    """Gather rows ``ids`` ([N] int) into ``out`` ([N, dim] bf16, contiguous).

    ``weight_ptr`` addresses [rows, dim] fp8 e4m3 bytes and ``scale_ptr``
    [rows, dim // block_size] e8m0 bytes for global rows [row_lo, row_hi); both
    may live in device or host memory. Ids outside the range produce zero rows.
    """
    assert dim > 0 and dim & (dim - 1) == 0 and block_size > 0, (dim, block_size)
    assert dim % block_size == 0, (dim, block_size)
    assert (
        ids.is_cuda and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)
    )
    assert out.is_contiguous() and out.dtype == torch.bfloat16
    assert out.device == ids.device and out.shape == (ids.numel(), dim), out.shape
    n = ids.numel()
    if n:
        _engram_gather_kernel[(n,)](
            weight_ptr,
            scale_ptr,
            ids,
            out,
            row_lo,
            row_hi,
            DIM=dim,
            BLK=block_size,
            E8M0_ZERO=_E8M0_ZERO,
            DECODE_E4M3_BYTES=is_ppu() and get_device_sm() < 89,
        )
    return out
