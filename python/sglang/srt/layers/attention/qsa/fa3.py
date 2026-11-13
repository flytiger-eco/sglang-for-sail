"""FA3 QSA over the physical KV pool, without gathering K/V per query."""

from functools import lru_cache
from inspect import signature

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def resolve_fa3_qsa():
    try:
        from flash_attn_interface import flash_attn_with_kvcache
    except ImportError as exc:
        raise ImportError(
            "SGLANG_QSA_BACKEND=fa3 requires a QSA-enabled FA3 wheel"
        ) from exc
    if "qsa_allow_aiu" not in signature(flash_attn_with_kvcache).parameters:
        raise RuntimeError("SGLANG_QSA_BACKEND=fa3 requires a QSA-enabled FA3 wheel")
    return flash_attn_with_kvcache


@triton.jit
def _physical_sparse_rows(
    indices,
    row_to_sequence,
    slots,
    positions,
    physical,
    counts,
    index_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    slot_width: tl.constexpr,
    pool_size: tl.constexpr,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    seq = tl.load(row_to_sequence + row).to(tl.int64)
    pos = tl.load(positions + row)
    logical = tl.load(indices + row * index_stride + cols, cols < topk, other=-1)
    valid = (cols < topk) & (logical >= 0) & (logical <= pos) & (logical < slot_width)
    token = tl.load(slots + seq * slot_stride + logical, valid, other=0)
    valid = valid & (token >= 0) & (token < pool_size)
    # Compact invalid columns before FA3's column-based causal mask. The indexer
    # normally puts -1 padding last, but speculative selections can have holes.
    dest = tl.cumsum(valid.to(tl.int32)) - 1
    tl.store(physical + row * topk + dest, token, valid)
    tl.store(counts + row, tl.sum(valid.to(tl.int32), 0))


@triton.jit
def _quantize_q_fp8(
    q,
    output,
    descale,
    q_row_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    head = kv_head * GROUP + offsets // DIM
    values = tl.load(
        q + row * q_row_stride + head * q_head_stride + offsets % DIM,
        offsets < GROUP * DIM,
        other=0.0,
    ).to(tl.float32)
    # FA3 descales are per (query sequence, KV head), shared by its GQA group.
    scale = tl.maximum(tl.max(tl.abs(values), 0) / 448.0, 1e-12)
    quantized = tl.minimum(tl.maximum(values / scale, -448.0), 448.0)
    tl.store(
        output + row * NUM_KV_HEADS * GROUP * DIM + kv_head * GROUP * DIM + offsets,
        quantized,
        offsets < GROUP * DIM,
    )
    tl.store(descale + row * NUM_KV_HEADS + kv_head, scale)


def quantize_q_fp8(q, num_kv_heads):
    """Quantize BF16 Q per token/GQA group and return FP32 FA3 descales."""
    rows, heads, dim = q.shape
    group = heads // num_kv_heads
    output = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    descale = torch.empty((rows, num_kv_heads), dtype=torch.float32, device=q.device)
    _quantize_q_fp8[(rows, num_kv_heads)](
        q,
        output,
        descale,
        q.stride(0),
        q.stride(1),
        num_kv_heads,
        group,
        dim,
        BLOCK=triton.next_power_of_2(group * dim),
    )
    return output, descale


def fa3_sparse_attention(
    q, k, v, indices, row_to_sequence, slots, positions, scale,
    *, k_scale=1.0, v_scale=1.0,
):
    """Return [rows, heads, dim], honoring per-row causal physical selections.

    FA3 masks sparse *columns*, not logical token positions. Treat each query as
    a length-one sequence and pass its compacted valid count as cache_seqlens.
    This handles unequal prefixes and query lengths without a CPU readback.
    """
    if q.shape[0] == 0:
        return torch.empty_like(q)
    if q.shape[-1] != 256 or k.shape[-1] != 256 or v.shape[-1] != 256:
        raise ValueError("FA3 QSA requires head dimension 256")
    if (
        q.dtype != torch.bfloat16
        or k.dtype != v.dtype
        or k.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
    ):
        raise ValueError("FA3 QSA requires BF16 Q and matching BF16 or E4M3 K/V")
    if indices.ndim != 2 or indices.shape[0] != q.shape[0] or indices.shape[1] == 0:
        raise ValueError(
            "FA3 QSA requires a nonempty 2D sparse table with one row per query"
        )
    if row_to_sequence.numel() != q.shape[0] or positions.numel() != q.shape[0]:
        raise ValueError("FA3 QSA query metadata must have one entry per query")
    if (
        q.shape[1] <= 0
        or k.shape[1] <= 0
        or q.shape[1] % k.shape[1]
        or q.shape[1] // k.shape[1] > 32
    ):
        raise ValueError(
            "FA3 QSA requires positive Q/KV head counts, Q heads divisible by "
            "KV heads, and at most 32 query heads per KV head"
        )
    rows, topk = indices.shape
    physical = torch.empty((rows, topk), dtype=torch.int32, device=q.device)
    counts = torch.empty(rows, dtype=torch.int32, device=q.device)
    _physical_sparse_rows[(rows,)](
        indices,
        row_to_sequence,
        slots,
        positions,
        physical,
        counts,
        indices.stride(0),
        slots.stride(0),
        slots.shape[1],
        k.shape[0],
        topk,
        BLOCK=triton.next_power_of_2(topk),
    )
    cu_q = torch.arange(rows + 1, dtype=torch.int32, device=q.device)
    descales = {}
    page_size = 1
    if k.dtype == torch.float8_e4m3fn:
        q, q_descale = quantize_q_fp8(q, k.shape[1])
        # Match the per-layer scales used by QwenSparseAttnBackend._store_kv,
        # including scales baked into the checkpoint.
        descales = dict(
            q_descale=q_descale,
            k_descale=torch.full_like(q_descale, k_scale),
            v_descale=torch.full_like(q_descale, v_scale),
        )
        # The PPU six-head/two-KV-head FP8 specializations use 64-token pages.
        if k.shape[0] % 64 == 0:
            page_size = 64
    # Splitting the slot dimension into pages is a view of the physical pool.
    return resolve_fa3_qsa()(
        q.contiguous(),
        k.view(-1, page_size, k.shape[1], k.shape[2]),
        v.view(-1, page_size, v.shape[1], v.shape[2]),
        page_table=physical.unsqueeze(1),
        cache_seqlens=counts,
        cu_seqlens_q=cu_q,
        max_seqlen_q=1,
        softmax_scale=scale,
        causal=True,
        num_splits=0,
        pack_gqa=True,
        # QSA indexer blocks contain four tokens; they do not certify the
        # sixteen physically contiguous tokens required by the AIU shortcut.
        qsa_allow_aiu=False,
        **descales,
    )
