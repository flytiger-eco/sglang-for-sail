"""CUDA kernels and tensor transforms for simple QSA."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class QSAPrefillPhysicalSelection:
    """已映射的物理 KV slots，防止后端将其误当作逻辑 token 索引。"""

    slots: torch.Tensor

    @property
    def shape(self):
        return self.slots.shape

    def __getitem__(self, rows):
        return QSAPrefillPhysicalSelection(self.slots[rows])

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        self.slots.record_stream(stream)


def average_pool_qsa_keys(key_groups: torch.Tensor) -> torch.Tensor:
    """FP32-average complete key groups shaped ``[groups, ratio, kv_heads, dim]``."""

    if key_groups.ndim != 4:
        raise ValueError(
            "QSA key groups must be [groups, ratio, kv_heads, head_dim], "
            f"got {key_groups.shape}"
        )
    return key_groups.float().mean(dim=1).to(key_groups.dtype)


@lru_cache(maxsize=32)
def _qsa_prefill_topk_metadata(device: torch.device, rows: int):
    """缓存 identity page mapping，避免逐层分配常量张量。"""
    return (
        torch.zeros((1, 1), dtype=torch.int32, device=device),
        torch.zeros((rows,), dtype=torch.int32, device=device),
    )


def _qsa_prefill_bf16_topk(logits, row_starts, row_ends, topk):
    """直接选择 BF16 分数，输出各 packed request 的相对 block 索引。"""
    from sglang.kernels.ops.attention.dsv4 import top_k_per_row_prefill_bf16

    rows, columns = logits.shape
    output = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    if rows == 0:
        return output
    page_table, row_to_page = _qsa_prefill_topk_metadata(logits.device, rows)
    page_size = 1 << max(columns - 1, 0).bit_length()
    top_k_per_row_prefill_bf16(
        logits,
        row_starts.contiguous(),
        row_ends.to(device=logits.device, dtype=torch.int32).contiguous(),
        page_table,
        output,
        page_size,
        row_to_page=row_to_page,
    )
    return output


def qsa_fast_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select compressed blocks with fixed-width, row-relative output."""

    lengths = (row_ends - row_starts).to(device=logits.device, dtype=torch.int32)
    starts = row_starts.to(device=logits.device, dtype=torch.int32)
    if logits.is_cuda:
        from sglang.srt.utils import is_ppu

        if logits.dtype == torch.bfloat16 and is_ppu() and topk in (512, 2048):
            return _qsa_prefill_bf16_topk(logits, starts, row_ends, topk)
        if is_ppu():
            from sglang.kernels.ops.attention.qsa_fast_topk import fast_topk
        else:
            from sglang.kernels.ops.attention.fast_topk import fast_topk

        return fast_topk(logits, lengths, topk=topk, row_starts=starts)

    output = torch.full(
        (logits.shape[0], topk),
        -1,
        dtype=torch.int32,
        device=logits.device,
    )
    for row in range(logits.shape[0]):
        start = int(starts[row])
        length = int(lengths[row])
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def torch_expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand compressed block indices into fixed-width logical token indices."""

    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [M, {block_topk}], got "
            f"{tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("query positions and sequence lengths must match top-k rows")

    device = block_indices.device
    blocks = block_indices.long()
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]

    query_positions = query_positions.to(device=device, dtype=torch.long)
    sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    visible_tokens = query_positions + 1
    tail_start = (
        torch.div(visible_tokens, compress_ratio, rounding_mode="floor")
        * compress_ratio
    )
    tail_count = visible_tokens - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat([expanded, tail], dim=1)
    # Keep all valid entries contiguous. This is required by the FA2 packing path.
    order = torch.arange(final_topk, device=device).unsqueeze(0).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + final_topk)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


@triton.jit
def _expand_qsa_block_indices_kernel(
    block_indices,
    query_positions,
    sequence_lengths,
    output,
    token_to_batch_idx,
    token_slot_table,
    block_stride: tl.constexpr,
    output_stride: tl.constexpr,
    token_stride: tl.constexpr,
    token_width: tl.constexpr,
    PHYSICAL_SLOTS: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    FINAL_TOPK: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, OUTPUT_BLOCK_SIZE)
    sequence_length = tl.load(sequence_lengths + row)

    source_block_cols = cols // COMPRESS_RATIO
    offsets = cols % COMPRESS_RATIO
    blocks = tl.load(
        block_indices + row * block_stride + source_block_cols,
        mask=(cols < TOKEN_TOPK) & (source_block_cols < BLOCK_TOPK),
        other=-1,
    )
    expanded = blocks * COMPRESS_RATIO + offsets
    expanded_valid = (
        (cols < TOKEN_TOPK)
        & (blocks >= 0)
        & (expanded >= 0)
        & (expanded < sequence_length)
    )

    valid_block_count = tl.sum(
        (
            (cols < BLOCK_TOPK)
            & (
                tl.load(
                    block_indices + row * block_stride + cols,
                    mask=cols < BLOCK_TOPK,
                    other=-1,
                )
                >= 0
            )
        ).to(tl.int32),
        axis=0,
    )
    valid_token_count = tl.minimum(valid_block_count * COMPRESS_RATIO, TOKEN_TOPK)

    query_position = tl.load(query_positions + row)
    visible_tokens = query_position + 1
    tail_start = (visible_tokens // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_offset = cols - valid_token_count
    tail_count = visible_tokens - tail_start
    tail = tail_start + tail_offset
    tail_valid = (
        (tail_offset >= 0)
        & (tail_offset < COMPRESS_RATIO - 1)
        & (tail_offset < tail_count)
        & (tail < sequence_length)
    )

    result = tl.where(
        expanded_valid & (cols < valid_token_count),
        expanded,
        tl.where(tail_valid, tail, -1),
    )
    if PHYSICAL_SLOTS:
        sequence_id = tl.load(token_to_batch_idx + row).to(tl.int64)
        valid = (cols < FINAL_TOPK) & (result >= 0) & (result < token_width)
        result = tl.load(
            token_slot_table
            + sequence_id * token_stride
            + tl.where(valid, result, 0).to(tl.int64),
            mask=valid,
            other=-1,
        )
    tl.store(
        output + row * output_stride + cols,
        result,
        mask=cols < FINAL_TOPK,
    )


def triton_expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """CUDA fast path for fast_topk_v2 output (valid blocks precede -1 padding)."""
    rows, block_topk = block_indices.shape
    final_topk = token_topk + compress_ratio - 1
    output = torch.empty(
        (rows, final_topk), dtype=torch.int32, device=block_indices.device
    )
    if rows == 0:
        return output
    _expand_qsa_block_indices_kernel[(rows,)](
        block_indices,
        query_positions,
        sequence_lengths,
        output,
        block_indices,
        block_indices,
        block_indices.stride(0),
        output.stride(0),
        token_stride=0,
        token_width=0,
        PHYSICAL_SLOTS=False,
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        FINAL_TOPK=final_topk,
        OUTPUT_BLOCK_SIZE=triton.next_power_of_2(final_topk),
        num_warps=8,
    )
    return output


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand compressed blocks with Triton on CUDA and Torch elsewhere."""

    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [M, {block_topk}], got "
            f"{tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("query positions and sequence lengths must match top-k rows")
    if block_indices.is_cuda:
        # The Triton kernel loads positions/lengths as scalars, so any integer
        # dtype works; skip the int64 conversion copies.
        return triton_expand_qsa_block_indices(
            block_indices.contiguous(),
            query_positions.to(device=block_indices.device).contiguous(),
            sequence_lengths.to(device=block_indices.device).contiguous(),
            compress_ratio,
            token_topk,
        )
    return torch_expand_qsa_block_indices(
        block_indices,
        query_positions,
        sequence_lengths,
        compress_ratio,
        token_topk,
    )


def expand_map_qsa_physical_slots(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_batch_idx: torch.Tensor,
    token_slot_table: torch.Tensor,
    *,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """逐 query 融合 block 展开与 KV 映射，不共享其他 query 的 selection。"""
    rows = block_indices.shape[0]
    if (
        block_indices.ndim != 2
        or block_indices.shape[1] != token_topk // compress_ratio
    ):
        raise ValueError("QSA block indices do not match the configured budget")
    if any(
        t.numel() != rows
        for t in (query_positions, sequence_lengths, token_to_batch_idx)
    ):
        raise ValueError("QSA mapping metadata must have one entry per query")
    if token_slot_table.ndim != 2 or token_slot_table.stride(-1) != 1:
        raise ValueError("QSA token slot table must have a contiguous last dimension")
    if not block_indices.is_cuda:
        logical = expand_qsa_block_indices(
            block_indices, query_positions, sequence_lengths, compress_ratio, token_topk
        )
        valid = (logical >= 0) & (logical < token_slot_table.shape[1])
        slots = token_slot_table[
            token_to_batch_idx.long()[:, None],
            logical.clamp(0, token_slot_table.shape[1] - 1).long(),
        ]
        return torch.where(valid, slots, -1).to(torch.int32)
    final_topk = token_topk + compress_ratio - 1
    slots = torch.empty(
        (rows, final_topk), dtype=torch.int32, device=block_indices.device
    )
    if rows:
        _expand_qsa_block_indices_kernel[(rows,)](
            block_indices,
            query_positions,
            sequence_lengths,
            slots,
            token_to_batch_idx,
            token_slot_table,
            block_indices.stride(0),
            slots.stride(0),
            token_stride=token_slot_table.stride(0),
            token_width=token_slot_table.shape[1],
            PHYSICAL_SLOTS=True,
            BLOCK_TOPK=block_indices.shape[1],
            COMPRESS_RATIO=compress_ratio,
            TOKEN_TOPK=token_topk,
            FINAL_TOPK=final_topk,
            OUTPUT_BLOCK_SIZE=triton.next_power_of_2(final_topk),
            num_warps=8,
        )
    return slots


def qsa_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for sparse GQA over physical token slots."""

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("q, k_cache and v_cache must be rank-3 tensors")
    if token_slots.ndim != 2 or token_slots.shape[0] != q.shape[0]:
        raise ValueError(
            "token slots must be [query_tokens, selected_tokens], got "
            f"{token_slots.shape}"
        )
    if q.shape[-1] != k_cache.shape[-1] or q.shape[-1] != v_cache.shape[-1]:
        raise ValueError("Q/K/V head dimensions must match")
    if q.shape[1] % k_cache.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    return qsa_sparse_attention_reference(
        q, k_cache, v_cache, token_slots, softmax_scale, k_scale, v_scale
    )


def qsa_sparse_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
) -> torch.Tensor:
    """Device-agnostic sparse GQA reference."""

    from sglang.srt.layers.attention.qsa.sparse_attn import _unit_scale

    scale = softmax_scale or q.shape[-1] ** -0.5
    k_scale, v_scale = _unit_scale(k_scale), _unit_scale(v_scale)
    outputs = []
    repeats = q.shape[1] // k_cache.shape[1]
    for row in range(q.shape[0]):
        valid = token_slots[row] >= 0
        slots = token_slots[row, valid].long()
        if slots.numel() == 0:
            outputs.append(torch.zeros_like(q[row]))
            continue
        keys = (
            k_cache.index_select(0, slots).float().repeat_interleave(repeats, dim=1)
            * k_scale
        )
        values = (
            v_cache.index_select(0, slots).float().repeat_interleave(repeats, dim=1)
            * v_scale
        )
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys) * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,khd->hd", probabilities, values).to(q.dtype))
    return torch.stack(outputs)


__all__ = [
    "average_pool_qsa_keys",
    "expand_qsa_block_indices",
    "torch_expand_qsa_block_indices",
    "triton_expand_qsa_block_indices",
    "qsa_fast_topk",
    "qsa_sparse_attention",
    "qsa_sparse_attention_reference",
]
