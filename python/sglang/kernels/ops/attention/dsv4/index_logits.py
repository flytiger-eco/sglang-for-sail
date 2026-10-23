"""The index logits of the DeepSeek V4.1 ratio-1/2 index layers, by the layout of
the index K they read: the flattened K of a prefill chunk (tile by tile under a
memory budget), the paged pool, or the published blocks of a sparse table."""

from __future__ import annotations

from typing import Iterator, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.mqa_logits_utils import (
    mqa_logits_row_bytes,
    mqa_logits_rows_per_chunk,
)
from sglang.srt.utils.common import ceil_align, is_ppu

from .candidate_table import CANDIDATE_BLOCK_SIZE


def flat_index_logits_rows_per_tile(
    rows: int, width: int, *, heads: int, budget_bytes: int
) -> int:
    """Rows per fp32 logits tile within ``budget_bytes``, at the kernel's row
    alignment."""
    row_alignment = 128 // heads
    rows_per_chunk = mqa_logits_rows_per_chunk(
        num_rows=ceil_align(rows, row_alignment),
        row_bytes=mqa_logits_row_bytes(width),
        budget_bytes=budget_bytes,
    )
    if rows_per_chunk is None:
        return rows
    return max(row_alignment, rows_per_chunk // row_alignment * row_alignment)


def flat_index_logits_tiles(
    *,
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    context_lengths: list[int],
    budget_bytes: int,
    width_align: int = 4,
) -> Iterator[tuple[slice, torch.Tensor]]:
    """``(rows, logits)`` per row tile, each fp32 tile within ``budget_bytes``:
    ``logits[i, j]`` scores query row ``rows.start + i`` against ``kv[starts + j]``,
    garbage past the row's ``lengths``; the width is ``max(context_lengths)``
    aligned to ``width_align``."""
    from deep_gemm import fp8_fp4_mqa_logits

    rows = q[0].shape[0]
    width = ceil_align(max(context_lengths, default=0), width_align)
    if rows == 0 or width == 0:
        return
    if is_ppu():
        max_logits_mb = envs.SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB.get()
        if max_logits_mb <= 0:
            raise ValueError("SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB must be positive")
        # PPU obeys the same MB knob as the INT8 indexer in place of the
        # caller's generic budget; the sliced FP32 tile is its only live
        # logits tensor, so the planner's row accounting applies as-is.
        budget_bytes = max_logits_mb * 1024 * 1024
    rows_per_chunk = flat_index_logits_rows_per_tile(
        rows, width, heads=q[0].shape[1], budget_bytes=budget_bytes
    )
    for offset in range(0, rows, rows_per_chunk):
        tile = slice(offset, min(offset + rows_per_chunk, rows))
        yield (
            tile,
            fp8_fp4_mqa_logits(
                (q[0][tile], q[1][tile]),
                kv,
                weights[tile],
                starts[tile],
                starts[tile] + lengths[tile],
                False,
                width,
            ),
        )


def deep_gemm_fp4_paged_mqa_logits(
    q_fp4: Tuple[torch.Tensor, torch.Tensor],
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata,
    max_seq_len: int,
) -> torch.Tensor:
    """DeepGEMM paged fp4 logits; no hadamard, the reference does not apply one."""
    from deep_gemm import fp8_fp4_paged_mqa_logits

    sl = seq_lens.to(torch.int32)
    if sl.dim() == 1:
        sl = sl.unsqueeze(-1)
    return fp8_fp4_paged_mqa_logits(
        q_fp4,
        k_cache,
        weights,
        sl,
        page_table,
        deep_gemm_metadata,
        max_seq_len,
        False,
    )


def sparse_logits(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    schedule: torch.Tensor,
    topk_blocks: int,
) -> torch.Tensor:
    """bf16 logits ``[rows, topk_blocks * 8]`` of the published blocks: ``q_fp4``
    ``[rows, 1, heads, 64]`` int8 with ``q_sf`` ``[rows, 1, heads]`` int32 (packed
    ue8m0), ``k_cache`` ``[pages, page_size, 1, 68]`` uint8 whose page stride is
    a multiple of 512 bytes, ``weights`` ``[rows, heads]`` bf16, ``schedule`` the
    ``build_sparse_indexer_schedule`` of the ``topk_blocks`` published blocks."""
    import deep_gemm

    return deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
        (q_fp4, q_sf),
        k_cache,
        weights,
        schedule,
        topk_blocks,
        CANDIDATE_BLOCK_SIZE,
    )


def sparse_varlen_schedule(
    ks: torch.Tensor,
    ke: torch.Tensor,
    num_kv_tokens: int,
    blocks: torch.Tensor,
    q_dtype: torch.dtype,
) -> torch.Tensor:
    """DeepGEMM's schedule for the published blocks of a *row set*: ``ks``/``ke``
    ``[rows]`` int32 windows into the flattened index K of ``num_kv_tokens``,
    ``blocks`` ``[rows, topk_blocks]`` int32 block ids in that flattened K. The
    schedule is per row set, so a different row set (a late-layer tail) needs a
    new one; it is not sliceable. Unaligned ``ks`` anchor every block at
    ``floor(ks / 8) * 8 + 8 * j + ks % 8``, i.e. the row's own causal grid."""
    import deep_gemm

    return deep_gemm.get_sparse_mqa_logits_metadata(
        ks.contiguous(),
        ke.contiguous(),
        num_kv_tokens,
        blocks,
        q_dtype,
        CANDIDATE_BLOCK_SIZE,
        True,
    )


def sparse_varlen_logits(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    kv: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    schedule: torch.Tensor,
    topk_blocks: int,
) -> torch.Tensor:
    """bf16 logits ``[rows, topk_blocks * 8]`` of ``kv``, the flattened index K
    ``(int8 [n, 64], int32 [n])`` of a prefill request set; ``schedule`` comes
    from :func:`sparse_varlen_schedule` over the same row set."""
    import deep_gemm

    return deep_gemm.fp8_fp4_sparse_mqa_logits(
        (q_fp4, q_sf),
        kv,
        weights,
        schedule,
        topk_blocks,
        CANDIDATE_BLOCK_SIZE,
        True,
    )
