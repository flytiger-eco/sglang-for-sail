"""Metadata owned by the simple QSA implementation.

QSA intentionally does not inherit the NSA metadata abstraction. This module
contains only fields and transforms consumed by the indexer.
"""

from __future__ import annotations

from typing import Optional, Tuple

import msgspec
import torch

from sglang.srt.layers.attention.qsa.kernel import qsa_fast_topk


def build_qsa_row_ranges(
    sequence_lengths: torch.Tensor,
    query_positions: torch.Tensor,
    query_sequence_ids: torch.Tensor,
    compress_ratio: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build packed compressed-key ranges for prefill scoring."""

    sequence_lengths = sequence_lengths.to(dtype=torch.int32)
    compressed_lengths = torch.div(
        sequence_lengths, compress_ratio, rounding_mode="floor"
    )
    compressed_cu_seqlens = torch.nn.functional.pad(
        compressed_lengths.cumsum(0), (1, 0)
    ).to(torch.int32)
    query_sequence_ids = query_sequence_ids.to(
        device=sequence_lengths.device, dtype=torch.long
    )
    row_starts = compressed_cu_seqlens.index_select(0, query_sequence_ids)
    visible_blocks = torch.div(
        query_positions.to(device=sequence_lengths.device, dtype=torch.int32) + 1,
        compress_ratio,
        rounding_mode="floor",
    )
    max_blocks = compressed_lengths.index_select(0, query_sequence_ids)
    row_ends = row_starts + torch.minimum(visible_blocks, max_blocks)
    return row_starts, row_ends, compressed_cu_seqlens


class QSAPrefillMQAPlan(msgspec.Struct, frozen=True):
    """Layer-independent prefill scoring inputs, built once per forward.

    Every QSA layer of a forward reads the same compressed slots, row ranges
    and per-row lengths; only the per-layer compressed buffer differs.
    """

    compressed_locs: torch.Tensor
    row_starts: torch.Tensor
    row_ends: torch.Tensor
    row_sequence_lengths: torch.Tensor


def build_qsa_prefill_mqa_plan(
    *,
    sequence_lengths: torch.Tensor,
    sequence_lengths_cpu: list[int],
    token_slot_table: torch.Tensor,
    query_positions: torch.Tensor,
    query_sequence_ids: torch.Tensor,
    compress_ratio: int,
) -> QSAPrefillMQAPlan:
    """Compressed slots and ranges for prefill MQA, from host-side lengths.

    Compressed slots come straight from the token-slot rows: each sequence's
    complete blocks live at ``page * page_size + offset`` of its assigned
    pages, in block order. DSV4-style addressing: a group's compressed slot
    is its first raw slot // ratio (the page-aligned allocator keeps the group
    contiguous in one page).
    """

    parts = []
    for sequence_id, length in enumerate(sequence_lengths_cpu):
        complete_blocks = int(length) // compress_ratio
        if complete_blocks == 0:
            continue
        parts.append(
            token_slot_table[
                sequence_id, : complete_blocks * compress_ratio : compress_ratio
            ].long()
            // compress_ratio
        )
    compressed_locs = (
        torch.cat(parts)
        if parts
        else torch.empty(0, dtype=torch.long, device=token_slot_table.device)
    )
    sequence_lengths = sequence_lengths.to(torch.int32)
    query_sequence_ids = query_sequence_ids.to(sequence_lengths.device)
    row_starts, row_ends, _ = build_qsa_row_ranges(
        sequence_lengths,
        query_positions.to(sequence_lengths.device),
        query_sequence_ids,
        compress_ratio,
    )
    return QSAPrefillMQAPlan(
        compressed_locs=compressed_locs,
        row_starts=row_starts,
        row_ends=row_ends,
        row_sequence_lengths=sequence_lengths.index_select(
            0, query_sequence_ids.long()
        ),
    )


class QSAIndexerMetadata(msgspec.Struct, frozen=True):
    """All per-forward metadata consumed specifically by ``QSAIndexer``.

    Row layout contract:

    - ``sequence_lengths``/``token_slot_table`` carry one row per *sequence*
      for extend modes and one row per *query token* for the paged modes
      (decode, target_verify, draft_extend).
    - ``token_to_batch_idx`` maps every query/token row handled by the indexer
      onto a row of ``sequence_lengths``/``token_slot_table``; DP attention
      token padding adds physical rows beyond this mapping, never inside it.
    - For the paged modes the mapping is the identity
      (``arange(num_query_rows)``), so page-table/MQA inputs built per
      ``sequence_lengths`` row line up with per-query sparse-attention rows.
    """

    sequence_lengths: torch.Tensor
    token_to_batch_idx: torch.Tensor
    token_slot_table: torch.Tensor
    out_cache_loc: torch.Tensor
    token_to_kv_pool: object
    compress_ratio: int
    block_topk: int
    req_pool_indices: Optional[torch.Tensor] = None
    # Parallel per-group arrays for the groups compressed this forward:
    # slot, sequence-local group-end position, and owning metadata row.
    # The first member's token row in this forward's packed tensors is extend only,
    # where group-aligned chunks keep every member in-chunk; None on paged forwards.
    write_locs: Optional[torch.Tensor] = None
    compress_group_positions: Optional[torch.Tensor] = None
    compress_sequence_ids: Optional[torch.Tensor] = None
    compress_member_rows: Optional[torch.Tensor] = None
    is_cuda_graph: bool = False
    graph_write_locs: Optional[torch.Tensor] = None
    graph_compressed_page_table: Optional[torch.Tensor] = None
    graph_compressed_lengths: Optional[torch.Tensor] = None
    graph_prefix_lengths: Optional[torch.Tensor] = None
    decode_page_table: Optional[torch.Tensor] = None
    decode_lengths: Optional[torch.Tensor] = None
    decode_logical_positions: Optional[torch.Tensor] = None
    pending_ring_slots: Optional[torch.Tensor] = None
    compress_group_ring_locs: Optional[torch.Tensor] = None
    extend_rope_matrix: Optional[torch.Tensor] = None
    graph_ring_group_locs: Optional[torch.Tensor] = None
    # Built once per eager extend forward from host-side lengths; when absent
    # (component callers, tests), get_prefill_mqa_inputs derives it per layer.
    prefill_mqa_plan: Optional[QSAPrefillMQAPlan] = None
    # Upper bound of this forward's RoPE coordinates for the cos/sin cache
    # length check; None makes the indexer read it off the device per layer.
    rope_max_position: Optional[int] = None

    def get_seqlens_int32(self) -> torch.Tensor:
        return self.sequence_lengths.to(torch.int32)

    def get_token_slot_table(self) -> torch.Tensor:
        return self.token_slot_table

    def get_seqlens_expanded(self) -> torch.Tensor:
        return self.get_seqlens_int32().index_select(
            0, self.get_token_to_batch_idx().long()
        )

    def get_token_to_batch_idx(self) -> torch.Tensor:
        return self.token_to_batch_idx

    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
        row_starts: Optional[torch.Tensor] = None,
        row_ends: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if topk != self.block_topk:
            raise ValueError(
                f"QSA compressed top-k must be {self.block_topk}, got {topk}"
            )
        if row_starts is None or row_ends is None:
            raise ValueError("QSA top-k transform requires row_starts and row_ends")
        return qsa_fast_topk(logits, row_starts, row_ends, topk=self.block_topk)

    def get_prefill_mqa_inputs(
        self,
        layer_id: int,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather packed compressed K and ragged ranges for prefill MQA.

        Returns ``(compressed_keys, row_starts, row_ends, row_sequence_lengths)``
        where the last is already expanded to one entry per query row.
        """

        pool = self.token_to_kv_pool
        compressed_buffer = pool.get_qsa_compressed_k_buffer(layer_id)
        plan = self.prefill_mqa_plan
        if plan is None:
            num_valid_tokens = self.token_to_batch_idx.numel()
            if positions.numel() < num_valid_tokens:
                raise ValueError(
                    "QSA prefill positions are shorter than the request mapping: "
                    f"positions={positions.numel()}, mapping={num_valid_tokens}"
                )
            plan = build_qsa_prefill_mqa_plan(
                sequence_lengths=self.sequence_lengths,
                sequence_lengths_cpu=self.sequence_lengths.tolist(),
                token_slot_table=self.token_slot_table,
                query_positions=positions[:num_valid_tokens],
                query_sequence_ids=self.token_to_batch_idx,
                compress_ratio=self.compress_ratio,
            )
        if plan.compressed_locs.numel():
            compressed_keys = compressed_buffer.index_select(0, plan.compressed_locs)
        else:
            compressed_keys = compressed_buffer.new_empty(
                (0, pool.qsa_index_kv_heads, pool.qsa_index_head_dim)
            )
        return (
            compressed_keys,
            plan.row_starts,
            plan.row_ends,
            plan.row_sequence_lengths,
        )

    def get_decode_mqa_inputs(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Paged compressed-K cache inputs for decode MQA, one row per query row."""

        pool = self.token_to_kv_pool
        num_rows = self.sequence_lengths.numel()
        if self.token_slot_table.shape[0] != num_rows:
            raise ValueError(
                "QSA decode page-table rows must match the per-query sequence "
                f"lengths: table_rows={self.token_slot_table.shape[0]}, "
                f"rows={num_rows}"
            )
        compressed_cache = pool.get_qsa_compressed_k_buffer(layer_id).reshape(
            -1,
            pool.qsa_compressed_page_size,
            pool.qsa_index_kv_heads,
            pool.qsa_index_head_dim,
        )
        if self.is_cuda_graph:
            if (
                self.graph_compressed_page_table is None
                or self.graph_compressed_lengths is None
            ):
                raise RuntimeError("QSA CUDA graph decode metadata is incomplete")
            return (
                compressed_cache,
                self.graph_compressed_page_table,
                self.graph_compressed_lengths,
                self.graph_compressed_page_table.shape[1]
                * pool.qsa_compressed_page_size,
            )
        if self.decode_page_table is not None and self.decode_lengths is not None:
            return (
                compressed_cache,
                self.decode_page_table,
                self.decode_lengths,
                self.decode_page_table.shape[1] * pool.qsa_compressed_page_size,
            )
        compressed_page_table, compressed_lengths = compressed_decode_view(
            compressed_page_size=pool.qsa_compressed_page_size,
            compress_ratio=self.compress_ratio,
            sequence_lengths=self.sequence_lengths,
            token_slot_table=self.token_slot_table,
        )
        return (
            compressed_cache,
            compressed_page_table,
            compressed_lengths,
            compressed_page_table.shape[1] * pool.qsa_compressed_page_size,
        )


def build_pending_ring_slots(
    *,
    token_to_batch_idx: torch.Tensor,
    req_pool_indices: torch.Tensor,
    sequence_lengths: torch.Tensor,
    logical_positions: torch.Tensor,
    compress_ratio: int,
    is_extend: bool,
) -> torch.Tensor:
    """Pending-ring slot ``req_pool_idx * ratio + position % ratio`` per token.
    On extend, tokens before the pending tail dump into rows [0, ratio),
    which no request owns (request slot 0 is never allocated); CUDA-graph safe."""
    rows = token_to_batch_idx.long()[: logical_positions.numel()]
    requests = req_pool_indices.long()[rows]
    positions = logical_positions.long()
    slots = requests * compress_ratio + positions % compress_ratio
    if is_extend:
        lengths = sequence_lengths.long()[rows]
        pending = positions >= (lengths // compress_ratio) * compress_ratio
        slots = torch.where(pending, slots, positions % compress_ratio)
    return slots


def build_group_ring_slots(
    *,
    req_pool_indices: torch.Tensor,
    group_end_positions: torch.Tensor,
    sequence_ids: torch.Tensor,
    compress_ratio: int,
) -> torch.Tensor:
    """Ring slots of a planned group's members, oldest first."""
    requests = req_pool_indices.long()[sequence_ids]
    offsets = torch.arange(
        compress_ratio - 1,
        -1,
        -1,
        device=group_end_positions.device,
        dtype=torch.long,
    )
    positions = (group_end_positions[:, None] - offsets[None, :]).clamp_min(0)
    return requests[:, None] * compress_ratio + positions % compress_ratio


def build_rope_position_matrix(
    rope_positions: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """This forward's RoPE coordinates as the [tokens, 3] layout the fused
    compress kernel reads."""
    if rope_positions.ndim == 1:
        return (
            rope_positions[:num_tokens].long().unsqueeze(1).expand(-1, 3)
        ).contiguous()
    return rope_positions[:, :num_tokens].long().transpose(0, 1).contiguous()


def compressed_decode_view(
    *,
    compressed_page_size: int,
    compress_ratio: int,
    sequence_lengths: torch.Tensor,
    token_slot_table: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compressed page table and lengths for decode MQA.

    Page-table entries are full-KV page ids read off the page-aligned
    token-slot rows; the scoring kernel converts them to compressed
    slots as page_id * compressed_page_size + block_in_page. Entries
    past a row's compressed length are stale-but-unread (bounded by
    compressed_lengths); clamp keeps them non-negative.
    """
    full_page = compressed_page_size * compress_ratio
    compressed_lengths = torch.div(
        sequence_lengths.to(torch.int32),
        compress_ratio,
        rounding_mode="floor",
    )
    compressed_page_table = (
        (token_slot_table[:, ::full_page].long() // full_page)
        .clamp_min(0)
        .to(torch.int32)
    )
    return compressed_page_table, compressed_lengths


__all__ = [
    "QSAIndexerMetadata",
    "QSAPrefillMQAPlan",
    "build_qsa_prefill_mqa_plan",
    "build_qsa_row_ranges",
    "build_pending_ring_slots",
    "build_group_ring_slots",
    "build_rope_position_matrix",
    "compressed_decode_view",
]
