"""Bounded, stream-owned scratch for widening FP8 KV once per prefill layer."""

import torch
import triton
import triton.language as tl


class QsaPrefillWorkspace:
    """One packed K/V allocation reused by sequential layers on one stream.

    The byte limit covers live tensor storage, not CUDA allocator reservations.
    A different stream/shape/dtype or a request over budget uses the caller's
    original tile-local FP8 path. This scratch is not used by CUDA graphs.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self._buffer = None
        self._owner = None

    def acquire(self, tokens, heads, dim, dtype, device, *, stream=None):
        if tokens <= 0 or self.max_bytes == 0:
            return None
        if dtype not in (torch.bfloat16, torch.float16):
            return None
        row_bytes = 2 * heads * dim * 2  # K + V, BF16/FP16.
        capacity_limit = self.max_bytes // row_bytes
        if tokens > capacity_limit:
            return None
        owner = (heads, dim, dtype, torch.device(device), stream)
        if self._owner is not None and self._owner != owner:
            return None
        capacity = 0 if self._buffer is None else self._buffer.shape[1]
        if tokens > capacity:
            capacity = min(capacity_limit, max(tokens, 2 * capacity))
            # Same-stream ordering permits allocator reuse after the previous
            # layer. Drop our old allocation before attempting a larger one.
            self._buffer = None
            try:
                self._buffer = torch.empty(
                    (2, capacity, heads, dim), dtype=dtype, device=device
                )
            except torch.OutOfMemoryError:
                return None
            self._owner = owner
        return self._buffer[0, :tokens], self._buffer[1, :tokens]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 512}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=4),
    ],
    key=["HEADS", "DIM"],
)
@triton.jit
def _gather_dequantize_prefill_kv(
    K,
    V,
    Table,
    Requests,
    Cu,
    OutK,
    OutV,
    k_scale,
    v_scale,
    sk_t: tl.constexpr,
    sk_h: tl.constexpr,
    sv_t: tl.constexpr,
    sv_h: tl.constexpr,
    st_r: tl.constexpr,
    st_t: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(1)
    begin = tl.load(Cu + request).to(tl.int64)
    end = tl.load(Cu + request + 1).to(tl.int64)
    request_id = tl.load(Requests + request).to(tl.int64)
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    position = offset // (HEADS * DIM)
    head = offset // DIM % HEADS
    column = offset % DIM
    mask = position < end - begin
    slot = tl.load(Table + request_id * st_r + position * st_t, mask, other=0).to(
        tl.int64
    )
    keys = tl.load(K + slot * sk_t + head * sk_h + column, mask, other=0.0).to(
        tl.float32
    )
    values = tl.load(V + slot * sv_t + head * sv_h + column, mask, other=0.0).to(
        tl.float32
    )
    destination = begin * HEADS * DIM + offset
    tl.store(OutK + destination, (keys * k_scale).to(OutK.dtype.element_ty), mask)
    tl.store(OutV + destination, (values * v_scale).to(OutV.dtype.element_ty), mask)


def gather_dequantize_prefill_kv(
    k,
    v,
    table,
    request_indices,
    cu_seqlens,
    max_seq_len,
    out_k,
    out_v,
    k_scale=1.0,
    v_scale=1.0,
):
    """Gather NHD cache rows into packed query-dtype K/V, applying scales once.

    Request rows and physical slots follow the backend's validated KV mapping.
    CPU length metadata supplies the launch bound; no device readback is needed.
    """
    assert k.shape == v.shape and k.dtype == v.dtype
    assert k.stride(-1) == v.stride(-1) == 1
    assert out_k.shape == out_v.shape and out_k.dtype == out_v.dtype
    assert out_k.shape[1:] == k.shape[1:]
    assert out_k.is_contiguous() and out_v.is_contiguous()
    assert request_indices.is_contiguous() and cu_seqlens.is_contiguous()
    if max_seq_len == 0 or request_indices.numel() == 0:
        return
    heads, dim = k.shape[1:]
    _gather_dequantize_prefill_kv[
        lambda meta: (
            triton.cdiv(max_seq_len * heads * dim, meta["BLOCK"]),
            request_indices.numel(),
        )
    ](
        k,
        v,
        table,
        request_indices,
        cu_seqlens,
        out_k,
        out_v,
        k_scale,
        v_scale,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        table.stride(0),
        table.stride(1),
        HEADS=heads,
        DIM=dim,
    )
