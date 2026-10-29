from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_exec, get_spec
from sglang.srt.utils import is_xpu

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_is_xpu = is_xpu()

_FLASHINFER_TIE_BREAK_VALUES = {
    "small": 1,
    "large": 2,
}


class TopkTransformMethod(IntEnum):
    # Transform topk indices to indices to the page table (page_size = 1)
    PAGED = auto()
    # Transform topk indices to indices to ragged kv (non-paged)
    RAGGED = auto()


class DSATopKBackend(Enum):
    SGL_KERNEL = "sgl-kernel"
    TORCH = "torch"
    FLASHINFER = "flashinfer"

    @classmethod
    def resolve(cls, model_runner: ModelRunner) -> DSATopKBackend:
        """Resolve the DSA top-k backend for one model runner.

        ``--dsa-topk-backend`` selects the target backend, while
        ``--speculative-dsa-topk-backend`` independently selects the draft.
        """
        if model_runner.is_draft_worker:
            return cls(get_spec().speculative_dsa_topk_backend)
        return cls(get_exec().kernel.dsa_topk_backend)

    def is_sgl_kernel(self) -> bool:
        return self == DSATopKBackend.SGL_KERNEL

    def is_torch(self) -> bool:
        return self == DSATopKBackend.TORCH

    def is_flashinfer(self) -> bool:
        return self == DSATopKBackend.FLASHINFER

    def should_use_topk_v2(self) -> bool:
        return self.is_sgl_kernel() and envs.SGLANG_OPT_USE_TOPK_V2.get()

    def topk_func(
        self,
        score: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        row_starts: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_sgl_kernel():
            from sglang.kernels.ops.attention.fast_topk import fast_topk_v2

            return fast_topk_v2(score, lengths, topk, row_starts=row_starts)
        if self.is_torch():
            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=torch.topk,
                topk_op_kwargs={"dim": -1},
            )
        if self.is_flashinfer():
            import flashinfer

            return _topk_unfused(
                score,
                lengths,
                topk,
                row_starts=row_starts,
                topk_op=flashinfer.top_k,
                topk_op_kwargs={
                    "sorted": False,
                    "deterministic": envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    "tie_break": _flashinfer_tie_break_value(),
                    "dsa_graph_safe": True,
                },
            )
        raise RuntimeError(f"Unsupported {self = }.")

    def _topk_transform_dsv4_bf16(
        self,
        logits: torch.Tensor,
        seq_lens_topk: torch.Tensor,
        page_table_size_1: Optional[torch.Tensor],
        cu_seqlens_q_topk: Optional[torch.Tensor],
        topk: int,
        ks: Optional[torch.Tensor],
        row_to_page: Optional[torch.Tensor] = None,
        topk_indices_offset: Optional[torch.Tensor] = None,
        page_size: int = 1,
    ) -> torch.Tensor:
        """Select directly from BF16 scores and return physical KV slots for
        PAGED mode or local indices plus output offsets for RAGGED mode.

        row_to_page reuses request-level page tables without per-query expansion.
        Slots beyond the valid top-k count are padded with -1. The caller routes
        decode-shaped PAGED inputs to v2 first when v2 is enabled.
        """
        from sglang.kernels.ops.attention.dsv4 import top_k_per_row_prefill_bf16

        assert logits.dim() == 2
        assert logits.dtype == torch.bfloat16
        device = logits.device
        num_q = logits.shape[0]

        # row_starts / row_ends (per query). NSA passes ks as row_starts; the dsv4
        # kernel needs an explicit row_ends = row_starts + length.
        row_starts = (
            torch.zeros(num_q, dtype=torch.int32, device=device)
            if ks is None
            else ks.to(torch.int32)
        )
        row_ends = row_starts + seq_lens_topk.to(torch.int32)

        req_idx = None
        if page_table_size_1 is not None:
            if row_to_page is not None:
                req_idx = row_to_page
            elif cu_seqlens_q_topk is not None:
                q_ids = torch.arange(num_q, device=device)
                req_idx = torch.searchsorted(
                    cu_seqlens_q_topk[1:],
                    q_ids,
                    right=True,
                    out_int32=True,
                )
            else:
                assert page_table_size_1.shape[0] == num_q
        else:
            assert topk_indices_offset is not None

        # Use new_full(-1) so unfilled slots are safe when logits has fewer positions than topk.
        page_indices = logits.new_full((num_q, topk), -1, dtype=torch.int32)
        top_k_per_row_prefill_bf16(
            logits,
            row_starts,
            row_ends,
            page_table_size_1,
            page_indices,
            page_size,
            row_to_page=req_idx,
            out_offsets=topk_indices_offset,
        )
        return page_indices

    def topk_transform(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        topk: int,
        topk_transform_method: TopkTransformMethod,
        attn_metadata,
        cu_seqlens_q_topk: Optional[torch.Tensor] = None,
        topk_indices_offset: Optional[torch.Tensor] = None,
        row_starts: Optional[torch.Tensor] = None,
        batch_idx_list: Optional[Union[List[int], torch.Tensor]] = None,
        force_unfused_topk: bool = False,
    ) -> torch.Tensor:
        if not envs.SGLANG_DSA_FUSE_TOPK.get() or force_unfused_topk:
            return self.topk_func(logits, lengths, topk, row_starts=row_starts)

        # Decode-shaped PAGED top-k for the SGL backend (plain decode AND spec
        # verify / draft-extend, whose expanded rows match the same shape) routes
        # to the DeepSeek-V4 top-k v2 JIT kernel. It fuses top-k selection and the
        # page-table transform in one launch and consumes the indexer's own
        # page_size>=1 table directly, so no page_size=1 table is materialized.
        # Shared by DeepSeek-V3.2 and GLM DSA.
        # This is a deterministic dispatch on the work shape, not a best-effort
        # attempt: the fused-decode CUDA graph drops the page_size=1 table for
        # exactly this case (see dsa_drop_wide_page_table), so once the shape
        # matches we commit to v2 and never silently fall back to the legacy
        # page_size=1 path from here.
        if (
            self.should_use_topk_v2()
            and topk_transform_method == TopkTransformMethod.PAGED
            and row_starts is None
            and batch_idx_list is None
            and 0 < topk <= 2048
            and lengths.shape[0]
            == logits.shape[0]
            == attn_metadata.real_page_table.shape[0]
        ):
            return _topk_transform_v2_paged(logits, lengths, topk, attn_metadata)

        # Extend-shaped RAGGED top-k for the SGL backend routes to the same v2
        # kernel through its ragged entry point: no page table (the columns are
        # already flattened-KV positions), no plan (prefill has enough rows that
        # the cluster path never applies), just a per-row window and an additive
        # output transform. `batch_idx_list` is not None only on the prefill-CP
        # path, whose `topk_indices_offset` is built from cu_seqlens_q rather
        # than the KV bases -- leave that one on the legacy kernel.
        if (
            self.should_use_topk_v2()
            and topk_transform_method == TopkTransformMethod.RAGGED
            and topk_indices_offset is not None
            and batch_idx_list is None
            and 0 < topk <= 2048
            and lengths.shape[0] == logits.shape[0] == topk_indices_offset.shape[0]
        ):
            return _topk_transform_v2_ragged(
                logits, lengths, topk, topk_indices_offset, row_starts
            )

        # CUDA/PPU/ROCm PAGED prefill: select within packed windows, then map
        # each query through its request's compact page table. CP request lists
        # expand only the row map, not the table; explicit interleaved/chunk maps
        # take precedence.
        if (
            not _is_xpu
            and self.should_use_topk_v2()
            and topk_transform_method == TopkTransformMethod.PAGED
            and 0 < topk <= 2048
            and lengths.shape[0] == logits.shape[0]
            and logits.dtype in (torch.float32, torch.bfloat16)
            and logits.stride(1) == 1
            and logits.stride(0) % 4 == 0
            and logits.stride(0) >= logits.shape[1]
            and row_starts is not None
            and row_starts.shape[0] == logits.shape[0]
        ):
            row_to_batch = _build_packed_row_to_batch(
                attn_metadata,
                batch_idx_list,
                cu_seqlens_q_topk,
                logits.shape[0],
                logits.device,
            )
            if row_to_batch is not None:
                return _topk_transform_v2_packed(
                    logits,
                    lengths,
                    topk,
                    attn_metadata,
                    row_starts=row_starts,
                    row_to_batch=row_to_batch,
                )

        use_bf16_topk = (
            self.is_sgl_kernel()
            and envs.SGLANG_DSA_USE_DSV4_BF16_TOPK.get()
            and logits.dtype == torch.bfloat16
        )
        if use_bf16_topk:
            row_to_page = None
            page_size = 1
            if topk_transform_method == TopkTransformMethod.PAGED:
                topk_indices_offset = None
                if (
                    row_starts is None
                    and batch_idx_list is None
                    and lengths.shape[0]
                    == logits.shape[0]
                    == attn_metadata.real_page_table.shape[0]
                ):
                    page_table_size_1 = attn_metadata.real_page_table
                    page_size = attn_metadata.page_size
                    cu_seqlens_q_topk = None
                else:
                    assert attn_metadata.page_table_1 is not None
                    if (
                        isinstance(batch_idx_list, torch.Tensor)
                        and batch_idx_list.ndim == 1
                        and batch_idx_list.shape[0] == logits.shape[0]
                    ):
                        row_to_page = batch_idx_list
                    page_table_size_1 = (
                        attn_metadata.page_table_1[batch_idx_list]
                        if batch_idx_list is not None and row_to_page is None
                        else attn_metadata.page_table_1
                    )
            elif topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                page_table_size_1 = None
                cu_seqlens_q_topk = None
            else:
                raise RuntimeError(f"Unsupported {topk_transform_method = }.")
            return self._topk_transform_dsv4_bf16(
                logits=logits,
                seq_lens_topk=lengths,
                page_table_size_1=page_table_size_1,
                cu_seqlens_q_topk=cu_seqlens_q_topk,
                topk=topk,
                ks=row_starts,
                row_to_page=row_to_page,
                topk_indices_offset=topk_indices_offset,
                page_size=page_size,
            )

        # The legacy transforms below read attn_metadata.page_table_1 (page_size=1),
        # which is always present here: the fold only drops it for the decode case
        # dispatched to v2 above.
        if topk_transform_method == TopkTransformMethod.PAGED:
            assert attn_metadata.page_table_1 is not None

        if self.is_sgl_kernel():
            from sglang.kernels.ops.attention.fast_topk import (
                fast_topk_transform_fused,
                fast_topk_transform_ragged_fused,
            )

            if topk_transform_method == TopkTransformMethod.PAGED:
                # A full-forward call already maps rows through cu_seqlens_q_topk;
                # reuse its request-level page table without gathering a copy.
                # Chunk slices are distinct objects and still require remapping.
                if batch_idx_list is attn_metadata.token_to_batch_idx:
                    batch_idx_list = None
                per_query_page_map = (
                    isinstance(batch_idx_list, torch.Tensor)
                    and batch_idx_list.ndim == 1
                    and batch_idx_list.shape[0] == logits.shape[0]
                )
                if per_query_page_map:
                    batch_idx_list = batch_idx_list[cu_seqlens_q_topk[:-1]]
                page_table_size_1 = (
                    attn_metadata.page_table_1[batch_idx_list]
                    if batch_idx_list is not None
                    else attn_metadata.page_table_1
                )
                return fast_topk_transform_fused(
                    score=logits,
                    lengths=lengths,
                    page_table_size_1=page_table_size_1,
                    cu_seqlens_q=cu_seqlens_q_topk,
                    topk=topk,
                    row_starts=row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return fast_topk_transform_ragged_fused(
                    score=logits,
                    lengths=lengths,
                    topk_indices_offset=topk_indices_offset,
                    topk=topk,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        if self.is_flashinfer():
            import flashinfer

            if topk_transform_method == TopkTransformMethod.PAGED:
                row_to_batch, page_table_row_starts = _build_flashinfer_paged_args(
                    attn_metadata=attn_metadata,
                    row_starts=row_starts,
                    cu_seqlens_q_topk=cu_seqlens_q_topk,
                    batch_idx_list=batch_idx_list,
                    device=logits.device,
                    num_rows=logits.shape[0],
                )
                return flashinfer.top_k_page_table_transform(
                    logits.contiguous(),
                    attn_metadata.page_table_1.contiguous(),
                    lengths.contiguous(),
                    topk,
                    row_to_batch=row_to_batch,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                    page_table_row_starts=page_table_row_starts,
                )
            if topk_transform_method == TopkTransformMethod.RAGGED:
                if topk_indices_offset is None:
                    raise RuntimeError(
                        "RAGGED topk_transform requires topk_indices_offset; "
                        "expected extend-without-speculative metadata."
                    )
                return flashinfer.top_k_ragged_transform(
                    logits.contiguous(),
                    topk_indices_offset.contiguous(),
                    lengths.contiguous(),
                    topk,
                    deterministic=envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
                    tie_break=_flashinfer_tie_break_value(),
                    dsa_graph_safe=True,
                    row_starts=row_starts,
                )
            raise RuntimeError(f"Unsupported {topk_transform_method = }.")

        raise RuntimeError(f"Unsupported {self = } for SGLANG_DSA_FUSE_TOPK.")


def _topk_unfused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
    topk_op: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = torch.topk,
    topk_op_kwargs: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    batch_size, max_score_len = score.shape
    topk_indices = score.new_full((batch_size, topk), -1, dtype=torch.int32)
    if batch_size == 0 or topk == 0 or max_score_len == 0:
        return topk_indices

    if row_starts is None:
        row_starts = torch.zeros_like(lengths, dtype=torch.int32, device=score.device)
    else:
        row_starts = row_starts.to(dtype=torch.int32, device=score.device)
    lengths = lengths.to(dtype=torch.int32, device=score.device)

    col_indices = torch.arange(max_score_len, dtype=torch.int32, device=score.device)
    col_indices = col_indices.unsqueeze(0)
    row_starts_unsqueezed = row_starts.unsqueeze(1)
    row_ends_unsqueezed = (row_starts + lengths).unsqueeze(1)
    valid_mask = (col_indices >= row_starts_unsqueezed) & (
        col_indices < row_ends_unsqueezed
    )

    masked_logits = score.masked_fill(~valid_mask, float("-inf"))
    valid_topk = min(topk, max_score_len)
    topk_kwargs = topk_op_kwargs or {}
    topk_scores, topk_col_indices = topk_op(masked_logits, valid_topk, **topk_kwargs)
    topk_local_indices = topk_col_indices.to(torch.int32) - row_starts_unsqueezed
    topk_local_indices = topk_local_indices.masked_fill(
        topk_scores == float("-inf"), -1
    )
    topk_indices[:, :valid_topk] = topk_local_indices

    return topk_indices


def _topk_transform_v2_paged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    attn_metadata,
) -> torch.Tensor:
    """Fused top-k + page-table transform via the DeepSeek-V4 v2 JIT kernel.

    Returns the transformed page indices ``(num_rows, topk)`` int32 (physical
    page_size=1 KV slots, ``-1`` padded) -- identical in meaning to
    ``fast_topk_transform_fused`` / ``flashinfer.top_k_page_table_transform``.
    The kernel selects, per row, the top-k of ``logits[row, :lengths[row]]`` and
    maps each selected position ``p`` through the page table as
    ``real_page_table[row, p // page_size] * page_size + (p % page_size)``. Feeding
    it the indexer's compact ``real_page_table`` (page_size = pool page size,
    typically 64) yields the same physical slots as gathering the page_size=1
    table, without materializing that wide table.

    For DSA extend's packed batch-global scores see
    :func:`_topk_transform_v2_packed`.

    This is a committed contract, not a best-effort path: ``topk_transform`` routes
    here only for the decode-shaped PAGED case, and the fused-decode CUDA graph
    drops the page_size=1 table for exactly this case (see
    ``dsa_drop_wide_page_table``). The preconditions below are therefore
    invariants the caller must uphold -- they assert (raise) on violation rather
    than fall back to the slow legacy path (which may not even have a page_size=1
    table to fall back to) or silently paper over bad input (padding, recomputing
    the plan) at the cost of the performance this path exists to deliver.

    ``lengths`` entries must be NON-NEGATIVE: the kernel reads them as
    ``uint32_t``, so a negative row length (DP-padded / idle-companion rows)
    reinterprets as ~4e9 tokens and illegal-addresses. Metadata producers clamp
    padded rows to 0 (see ``fused_dsa_draft_extend_metadata`` /
    ``seqlens_expand_kernel``); 0 takes the trivial all-(-1) output path.
    """
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2

    num_rows = logits.shape[0]

    # Row strides count input elements: four FP32 values occupy 16B, while
    # four BF16 values occupy 8B. The column stride must remain one.
    assert (
        logits.dtype in (torch.float32, torch.bfloat16)
        and logits.stride(1) == 1
        and logits.stride(0) % 4 == 0
    ), (
        f"v2 top-k expects fp32/bf16 scores with unit column stride and a row stride divisible by 4, got {logits.dtype=} {logits.stride()=}"
    )
    assert 0 < topk <= 2048, f"v2 top-k supports 0 < topk <= 2048, got {topk=}"

    page_table = attn_metadata.real_page_table

    # The plan is preprocessed once per forward (DSAMetadata.topk_v2_plan,
    # refreshed in-place under CUDA graph) and reused across layers. A missing or
    # mismatched plan means the caller skipped that preprocessing -- fail loudly
    # rather than silently recompute it per layer.
    plan = attn_metadata.topk_v2_plan
    assert plan is not None and plan.shape[0] == num_rows + 1, (
        "topk_v2_plan must be preprocessed per forward (see DSAMetadata.topk_v2_plan)"
    )

    page_size = attn_metadata.page_size
    out = logits.new_empty((num_rows, topk), dtype=torch.int32)
    topk_transform_paged_v2(logits, lengths, page_table, out, page_size, plan)
    return out


def _build_packed_row_to_batch(
    attn_metadata,
    batch_idx_list: Optional[Union[List[int], torch.Tensor]],
    cu_seqlens_q: Optional[torch.Tensor],
    num_rows: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Map CP request lists or contiguous query segments to page-table rows."""
    if isinstance(batch_idx_list, torch.Tensor):
        # A tensor maps individual queries; do not rebuild interleaved ordering
        # from cu_seqlens.
        if batch_idx_list.ndim != 1 or batch_idx_list.shape[0] != num_rows:
            raise RuntimeError("packed top-k requires one row_to_batch entry per query")
        return batch_idx_list

    if batch_idx_list is None:
        row_map = attn_metadata.token_to_batch_idx
        if row_map is not None and row_map.ndim == 1 and row_map.shape[0] == num_rows:
            return row_map
    elif cu_seqlens_q is None or cu_seqlens_q.numel() != len(batch_idx_list) + 1:
        raise RuntimeError("CP request list requires matching cu_seqlens_q metadata")

    if cu_seqlens_q is None:
        if attn_metadata.real_page_table.shape[0] == num_rows:
            return torch.arange(num_rows, dtype=torch.int32, device=device)
        return None

    # right=True skips empty query segments, including trailing empty requests.
    segment_ids = torch.searchsorted(
        cu_seqlens_q[1:],
        torch.arange(num_rows, device=device),
        right=True,
        out_int32=True,
    )
    if batch_idx_list is None:
        return segment_ids
    requests = torch.tensor(batch_idx_list, dtype=torch.int32, device=device)
    return requests[segment_ids]


def _topk_transform_v2_packed(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    attn_metadata,
    row_starts: torch.Tensor,
    row_to_batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused packed-row top-k + page-table transform (DSA extend prefill).

    Same output contract as :func:`_topk_transform_v2_paged` -- ``(num_rows,
    topk)`` int32 physical KV slots, ``-1`` padded -- but the scores are packed:
    row ``i`` owns the window at ``row_starts[i]`` of one batch-global buffer and
    maps through page-table row ``row_to_batch[i]`` (prefill expands one request
    into many query-token rows). Selected indices stay row-local.

    Being a prefill-only path it dispatches per row inside the kernel, so unlike
    the paged entry point it needs no ``topk_v2_plan``.

    NOTE: ``logits`` is MODIFIED IN PLACE (the <= 3 columns ahead of each window
    are masked); the caller must not reuse it. ``lengths`` must be NON-NEGATIVE,
    for the same reason as in :func:`_topk_transform_v2_paged`.
    """
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_packed_v2

    num_rows = logits.shape[0]
    assert (
        logits.dtype in (torch.float32, torch.bfloat16)
        and logits.stride(1) == 1
        and logits.stride(0) % 4 == 0
        and logits.stride(0) >= logits.shape[1]
    ), (
        f"packed v2 top-k expects non-overlapping fp32/bf16 rows with unit column stride and row stride divisible by 4, got {logits.dtype=} {logits.stride()=}"
    )
    assert 0 < topk <= 2048, f"v2 top-k supports 0 < topk <= 2048, got {topk=}"

    out = logits.new_empty((num_rows, topk), dtype=torch.int32)
    topk_transform_packed_v2(
        logits,
        lengths,
        attn_metadata.real_page_table,
        out,
        attn_metadata.page_size,
        row_starts=row_starts.to(torch.int32),
        row_to_batch=(None if row_to_batch is None else row_to_batch.to(torch.int32)),
    )
    return out


def _topk_transform_v2_ragged(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    topk_indices_offset: torch.Tensor,
    row_starts: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused ragged top-k via the DeepSeek-V4 v2 JIT kernel.

    ``logits`` is written in place: the kernel reads from a 16-byte-aligned base
    and masks the <= 3 columns that pulls in ahead of the window. Those columns
    belong to a preceding request of the same row, and the score buffer is dead
    after the top-k (see ``DSAIndexer._get_topk_ragged``).
    """
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

    out = logits.new_empty((logits.shape[0], topk), dtype=torch.int32)
    topk_transform_ragged_v2(
        logits,
        lengths,
        out_offsets=topk_indices_offset,
        out_indices=out,
        row_starts=row_starts,
    )
    return out


def _build_flashinfer_paged_args(
    attn_metadata,
    row_starts: Optional[torch.Tensor],
    cu_seqlens_q_topk: Optional[torch.Tensor],
    batch_idx_list: Optional[List[int]],
    device: torch.device,
    num_rows: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    row_to_batch = (
        torch.as_tensor(batch_idx_list, dtype=torch.int32, device=device)
        if batch_idx_list is not None
        else None
    )

    # Both dynamic mappings contain one entry per logit row. Supplying the known
    # size avoids synchronizing CUDA to infer the sum of the repeat counts.
    if (
        row_to_batch is not None
        and cu_seqlens_q_topk is not None
        and row_to_batch.shape[0] != num_rows
    ):
        q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
            dtype=torch.int32, device=device
        )
        row_to_batch = torch.repeat_interleave(
            row_to_batch, q_lens, output_size=num_rows
        )

    if row_to_batch is None and cu_seqlens_q_topk is not None:
        # Decode-like case (one query row per batch) does not need an explicit mapping.
        # Avoid dynamic tensor construction in this branch to keep CUDA graph capture safe.
        num_batches = cu_seqlens_q_topk.shape[0] - 1
        if not (row_starts is None and num_rows == num_batches):
            q_lens = (cu_seqlens_q_topk[1:] - cu_seqlens_q_topk[:-1]).to(
                dtype=torch.int32, device=device
            )
            row_to_batch = torch.repeat_interleave(
                torch.arange(q_lens.shape[0], dtype=torch.int32, device=device),
                q_lens,
                output_size=num_rows,
            )

    if row_starts is not None and row_to_batch is None:
        raise RuntimeError(
            "PAGED topk_transform with row_starts requires cu_seqlens_q metadata."
        )

    page_table_row_starts = row_starts
    if page_table_row_starts is not None and row_to_batch is not None:
        page_table_row_starts = (
            page_table_row_starts - attn_metadata.cu_seqlens_k[:-1][row_to_batch]
        )

    return row_to_batch, page_table_row_starts


def _flashinfer_tie_break_value() -> int:
    mode = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    if mode is None:
        return 0
    mode = mode.lower()
    if mode not in _FLASHINFER_TIE_BREAK_VALUES:
        raise RuntimeError(
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK must be one of "
            f"{tuple(_FLASHINFER_TIE_BREAK_VALUES)} or unset, got {mode!r}."
        )
    return _FLASHINFER_TIE_BREAK_VALUES[mode]
