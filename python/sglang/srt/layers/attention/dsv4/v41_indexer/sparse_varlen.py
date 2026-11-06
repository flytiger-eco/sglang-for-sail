"""The candidate scheme of a prefill under context parallelism: a source publishes
its best blocks, a consumer scores only those blocks with DeepGEMM's varlen
``fp8_fp4_sparse_mqa_logits`` over the flattened index K.

A CP rank's rows are an interleaved subset of the batch, so the paged sparse
table (whose page table rows follow the batch) does not apply. The varlen
schedule is built from the row set itself and can therefore be rebuilt for any
subset, including the late-layer tail, instead of being sliced. Consumers score
budget-sized row tiles and each tile rebuilds its own schedule, so the bf16
logits stay within the same memory budget as the dense prefill path."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import candidate_row_lens
from sglang.kernels.ops.attention.dsv4.candidate_table import CANDIDATE_BLOCK_SIZE
from sglang.kernels.ops.attention.dsv4.index_logits import (
    sparse_varlen_logits,
    sparse_varlen_schedule,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_sparse
from sglang.srt.environ import envs
from sglang.srt.utils import is_ppu

from .scoring import DeepGEMMPrefillData
from .types import (
    CandidateMetadata,
    PrefillInputs,
    Selection,
    get_tail_row_indices,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

_PAD_BLOCK = torch.iinfo(torch.int32).max
# The dense prefill tiles use the same 2 GiB budget on non-PPU devices.
_SPARSE_LOGITS_BUDGET_BYTES = 2 << 30
# The sparse kernel downclocks on large chunks; cap a tile even when memory allows.
_MAX_ROWS_PER_CHUNK = 8192


def _sparse_logits_rows_per_chunk(topk_blocks: int) -> int:
    """Rows per bf16 logits tile within the dense path's score budget."""
    row_bytes = topk_blocks * CANDIDATE_BLOCK_SIZE * 2
    if is_ppu():
        max_mb = envs.SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB.get()
        if max_mb <= 0:
            raise ValueError("SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB must be positive")
        budget = max_mb * 1024 * 1024
    else:
        budget = _SPARSE_LOGITS_BUDGET_BYTES
    rows = budget // row_bytes
    # Keep the DeepGEMM row pairing (two rows per KV pass) inside one tile.
    return min(max(8, rows // 8 * 8), _MAX_ROWS_PER_CHUNK)


class _VarlenSparseTable(CandidateMetadata, msgspec.Struct):
    # [rows, topk_blocks] int32: ascending block ids on each row's causal grid,
    # INT32_MAX past the row's kept count (the kernel reads only that count)
    blocks: torch.Tensor
    # [rows] int32: sparse logits length once the kept blocks are laid out back
    # to back, the newest block possibly partial
    valid_lens: torch.Tensor
    # one DeepGEMM schedule (uint8) per row tile; metadata is per row set and
    # cannot be sliced, so a tile rebuilds it instead
    schedules: List[torch.Tensor]
    chunk_rows: int
    # the remaining inputs stay attached so a tail can rebuild the schedule
    compress_lens: torch.Tensor  # [rows] int32
    request_starts: torch.Tensor  # [rows] int32, first flattened-K column
    num_kv_tokens: int
    q_dtype: torch.dtype
    rows_per_request: List[int]

    def tail(self, rows_per_request: List[int]) -> _VarlenSparseTable:
        rows = get_tail_row_indices(
            full_rows_per_request=self.rows_per_request,
            tail_rows_per_request=rows_per_request,
            device=self.blocks.device,
        )
        return _build_varlen_table(
            blocks=self.blocks[rows].contiguous(),
            valid_lens=self.valid_lens[rows].contiguous(),
            compress_lens=self.compress_lens[rows].contiguous(),
            request_starts=self.request_starts[rows].contiguous(),
            num_kv_tokens=self.num_kv_tokens,
            q_dtype=self.q_dtype,
            rows_per_request=rows_per_request,
        )


class VarlenSparsePrefillBackend:
    def __init__(
        self,
        *,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        req_to_token: torch.Tensor,
        candidate_topk_blocks: int,
    ) -> None:
        self.token_to_kv_pool = token_to_kv_pool
        self.req_to_token = req_to_token
        self.topk_blocks = candidate_topk_blocks

    def publish_prefill(self, inputs: PrefillInputs, out: Selection):
        from .scoring import get_deep_gemm_prefill_data

        out.reset()
        data = get_deep_gemm_prefill_data(inputs, self.req_to_token)
        if data is None:
            return None
        return publish_varlen_prefill(
            data=data,
            kv=self.token_to_kv_pool.get_low_ratio_index_k_fp4(
                inputs.layer_id, data.k_slots
            ),
            topk=inputs.indexer.index_topk,
            topk_blocks=self.topk_blocks,
            out=out,
        )

    def consume_prefill(
        self,
        inputs: PrefillInputs,
        published: Optional[_VarlenSparseTable],
        out: Selection,
    ) -> None:
        from .scoring import get_deep_gemm_prefill_data

        out.reset()
        data = get_deep_gemm_prefill_data(inputs, self.req_to_token)
        if data is None:
            return
        assert published is not None
        consume_varlen_prefill(
            data=data,
            kv=self.token_to_kv_pool.get_low_ratio_index_k_fp4(
                inputs.layer_id, data.k_slots
            ),
            published=published,
            topk=inputs.indexer.index_topk,
            out=out,
        )


def publish_varlen_prefill(
    *,
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    topk: int,
    topk_blocks: int,
    out: Selection,
) -> _VarlenSparseTable:
    """The source's own top-k and the table of its blocks, from one dense pass:
    only the source layer scores the whole context, consumers read the blocks."""
    from .dense_blocks import _publish_prefill_blocks

    selected, blocks = _publish_prefill_blocks(
        data=data,
        kv=kv,
        topk=topk,
        topk_blocks=topk_blocks,
        block_size=CANDIDATE_BLOCK_SIZE,
    )
    data.write_selection(selected=selected, out=out)
    # DeepGEMM requires a sparse-block stride that is a multiple of 4.
    pad = -blocks.shape[1] % 4
    if pad:
        blocks = torch.cat([blocks, blocks.new_full((blocks.shape[0], pad), -1)], dim=1)
    # The kept count follows the logical budget, not the padded stride.
    _, valid_lens = candidate_row_lens(
        data.compress_lens, topk_blocks, block_size=CANDIDATE_BLOCK_SIZE
    )
    # Ascending ids put the row's newest block last, which is the layout
    # ``valid_lens`` describes; the source's picks are unordered and -1 padded.
    blocks = blocks.masked_fill(blocks < 0, _PAD_BLOCK).sort(dim=-1).values
    return _build_varlen_table(
        blocks=blocks,
        valid_lens=valid_lens,
        compress_lens=data.compress_lens,
        request_starts=data.request_starts,
        num_kv_tokens=data.k_slots.numel(),
        q_dtype=data.q_fp4.dtype,
        rows_per_request=data.rows_per_request,
    )


def consume_varlen_prefill(
    *,
    data: DeepGEMMPrefillData,
    kv: Tuple[torch.Tensor, torch.Tensor],
    published: _VarlenSparseTable,
    topk: int,
    out: Selection,
) -> None:
    """One consumer's top-k among the published blocks of its rows."""
    rows = data.num_rows
    assert published.blocks.shape[0] == rows
    topk_blocks = published.blocks.shape[1]
    weights = data.weights.to(torch.bfloat16)
    # Column j of a sparse row is request-relative position
    # blocks[b, j // 8] * 8 + j % 8, which is the source's published grid.
    selected = data.empty_selection(topk)
    chunk = published.chunk_rows
    for index, start in enumerate(range(0, rows, chunk)):
        tile = slice(start, min(start + chunk, rows))
        logits = sparse_varlen_logits(
            data.q_fp4[tile],
            data.q_sf[tile],
            kv,
            weights[tile],
            published.schedules[index],
            topk_blocks,
        )
        topk_transform_sparse(
            logits, published.valid_lens[tile], published.blocks[tile], selected[tile]
        )
        del logits
    selected.add_(torch.where(selected >= 0, data.request_starts[:, None], 0))
    data.write_selection(selected=selected, out=out)


def _build_varlen_table(
    *,
    blocks: torch.Tensor,
    valid_lens: torch.Tensor,
    compress_lens: torch.Tensor,
    request_starts: torch.Tensor,
    num_kv_tokens: int,
    q_dtype: torch.dtype,
    rows_per_request: List[int],
) -> _VarlenSparseTable:
    # Absolute ids for the schedule: the unaligned-ks kernel rebuilds each block
    # as ``id * 8 + start % 8``, i.e. the row's causal grid. The padding past a
    # row's kept count is never read; map it to a valid slot anyway. The table
    # keeps the relative ids that ``topk_transform_sparse`` maps columns through.
    absolute = (
        blocks.masked_fill(blocks == _PAD_BLOCK, 0)
        + (request_starts // CANDIDATE_BLOCK_SIZE)[:, None]
    )
    rows = blocks.shape[0]
    chunk = _sparse_logits_rows_per_chunk(blocks.shape[1])
    schedules = [
        sparse_varlen_schedule(
            request_starts[tile],
            request_starts[tile] + compress_lens[tile],
            num_kv_tokens,
            absolute[tile].contiguous(),
            q_dtype,
        )
        for tile in (slice(s, min(s + chunk, rows)) for s in range(0, rows, chunk))
    ]
    return _VarlenSparseTable(
        blocks=blocks,
        valid_lens=valid_lens,
        schedules=schedules,
        chunk_rows=chunk,
        compress_lens=compress_lens,
        request_starts=request_starts,
        num_kv_tokens=num_kv_tokens,
        q_dtype=q_dtype,
        rows_per_request=rows_per_request,
    )
