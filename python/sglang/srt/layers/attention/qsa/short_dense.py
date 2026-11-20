"""Pack physical KV once for bounded dense attention, preserving cache bits."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["WIDTH", "st_r"])
def _pack_short_dense_kv(
    K,
    V,
    Slots,
    Lengths,
    OutK,
    OutV,
    WIDTH,
    sk_t: tl.constexpr,
    sk_h: tl.constexpr,
    sv_t: tl.constexpr,
    sv_h: tl.constexpr,
    st_r,
    st_t: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    position = offsets // (HEADS * DIM)
    head = offsets // DIM % HEADS
    column = offsets % DIM
    length = tl.load(Lengths + row)
    valid = position < length
    slot = tl.load(Slots + row * st_r + position * st_t, valid, other=0).to(tl.int64)
    k = tl.load(K + slot * sk_t + head * sk_h + column, valid, other=0)
    v = tl.load(V + slot * sv_t + head * sv_h + column, valid, other=0)
    destination = row * WIDTH * HEADS * DIM + offsets
    tl.store(OutK + destination, k, position < WIDTH)
    tl.store(OutV + destination, v, position < WIDTH)


def pack_short_dense_kv(k, v, slots, lengths, out_k, out_v):
    """Copy NHD cache into BSHD scratch; unused padding is zero, never read.

    No widening, requantization, or modification of the persistent KV pool.
    Slots already follow the backend's per-request physical mapping.
    """
    if k.shape != v.shape or k.dtype != v.dtype:
        raise ValueError("Dense KV packing requires matching K/V cache layouts")
    if out_k.shape != out_v.shape or out_k.dtype != k.dtype or out_v.dtype != k.dtype:
        raise ValueError("Dense KV packing must preserve the cache dtype")
    if not out_k.is_contiguous() or not out_v.is_contiguous():
        raise ValueError("Dense KV scratch must be contiguous")
    if out_k.shape[0] != lengths.numel() or slots.shape[0] != lengths.numel():
        raise ValueError("Dense KV descriptors must have one row per request")
    if out_k.shape[2:] != k.shape[1:] or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Unsupported dense KV head layout")
    if lengths.numel() == 0:
        return
    _, width, heads, dim = out_k.shape
    if k.dtype == torch.float8_e4m3fn:
        # Copy storage bytes, including NaN payloads, with no FP8 conversion.
        k, v = k.view(torch.uint8), v.view(torch.uint8)
        out_k, out_v = out_k.view(torch.uint8), out_v.view(torch.uint8)
    _pack_short_dense_kv[(triton.cdiv(width * heads * dim, 1024), lengths.numel())](
        k,
        v,
        slots,
        lengths,
        out_k,
        out_v,
        width,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        slots.stride(0),
        slots.stride(1),
        HEADS=heads,
        DIM=dim,
        BLOCK=1024,
        num_warps=4,
    )
