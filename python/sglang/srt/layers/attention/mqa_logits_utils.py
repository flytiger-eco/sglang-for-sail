"""Memory budget for the fp32 MQA-logits scratch of the DeepSeek sparse
attention indexers (DSA, DSV4, DSV4.1).

The indexer scores every query row against every key column into one fp32
logits matrix that no pool sized by mem_fraction_static accounts for. These
helpers bound that matrix and slice it by query rows.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import get_device_module, is_hip, is_xpu
from sglang.srt.utils.common import ceil_div

MQA_LOGITS_BYTES_PER_ELEM = 4
MQA_LOGITS_STATIC_SKIP_ELEMS = 8_000_000
MQA_LOGITS_TOTAL_MEM_FRACTION = 0.3
# aiter's fp8_mqa_logits only compiles below 2 GiB of logits (buffer_store).
MQA_LOGITS_MAX_BYTES_ROCM = 2**31 - 1
# DeepGEMM pads the logits row stride to 1024 bytes, i.e. 256 fp32 columns.
MQA_LOGITS_ROW_ALIGN_ELEMS = 256


def sail_mqa_logits(q, kv, weights, starts, ends, *, max_seqlen_k=0):
    """PPU weighted logits: request-relative MXFP4 columns, weights-selected INT8 dtype."""
    import deep_gemm

    if isinstance(q, tuple):
        if max_seqlen_k <= 0:
            raise ValueError("MXFP4 indexer requires a positive max_seqlen_k")
        q = (q[0].contiguous().view(torch.int8), q[1].contiguous())
        kv = (kv[0].contiguous().view(torch.int8), kv[1].contiguous())
        return deep_gemm.fp8_fp4_mqa_logits(
            q,
            kv,
            weights.to(torch.bfloat16).contiguous(),
            starts,
            ends,
            max_seqlen_k,
        )
    fn = (
        deep_gemm.int8_mqa_logits if q.dtype == torch.int8 else deep_gemm.fp8_mqa_logits
    )
    dtype = weights.dtype if q.dtype == torch.int8 else torch.float32
    return fn(
        q,
        kv,
        weights.to(dtype).contiguous(),
        starts,
        ends,
        clean_logits=False,
        max_seqlen_k=max_seqlen_k,
    )


def sail_paged_mqa_metadata(
    context_lens, page_size, heads, head_dim, qk_dtype, indices=None
):
    """Use logical D and torch.dtype in the hint; weighted next_n is fixed at 1."""
    import deep_gemm

    context_lens = context_lens.to(torch.int32).reshape(-1, 1).contiguous()
    if indices is not None:
        indices = indices.to(device=context_lens.device, dtype=torch.int32).contiguous()
        if indices.ndim != 1 or indices.numel() != context_lens.shape[0]:
            raise ValueError("request indices must contain one owner per query row")
    return deep_gemm.get_paged_mqa_logits_metadata(
        context_lens,
        page_size,
        deep_gemm.get_num_sms(),
        indices=indices,
        metadata_extra=(1, heads, head_dim, qk_dtype),
    )


def sail_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    *,
    indices=None,
):
    """Generate the schedule with the compute SM configuration for graph replay."""
    import deep_gemm

    packed = isinstance(q, tuple)
    values = q[0] if packed else q
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("PPU weighted paged indexer requires [Q, 1, H, Dstorage]")
    context_lens = context_lens.to(torch.int32).reshape(-1, 1).contiguous()
    block_table = block_table.to(torch.int32).contiguous()
    if (
        context_lens.shape[0] != values.shape[0]
        or block_table.ndim != 2
        or block_table.shape[0] != values.shape[0]
    ):
        raise ValueError("context lengths and block table must have one row per query")
    if packed:
        if indices is None:
            raise ValueError("MXFP4 paged indexer requires real request indices")
        indices = indices.to(device=values.device, dtype=torch.int32).contiguous()
        if indices.ndim != 1 or indices.numel() != values.shape[0]:
            raise ValueError("request indices must contain one owner per query row")
        # Give zero-context graph padding rows distinct dummy groups.
        dummy_ids = -torch.arange(
            1, values.shape[0] + 1, device=values.device, dtype=torch.int32
        )
        indices = torch.where(context_lens[:, 0] == 0, dummy_ids, indices)
        q = (values.contiguous().view(torch.int8), q[1].contiguous())
        dtype, head_dim = torch.int8, values.shape[-1] * 2
        weights = weights.to(torch.bfloat16).contiguous()
        fn = deep_gemm.fp8_fp4_paged_mqa_logits
    else:
        if indices is not None:
            raise ValueError("FP8/INT8 paged indexer does not accept request indices")
        dtype, head_dim = values.dtype, values.shape[-1]
        weights = weights.to(
            weights.dtype if dtype == torch.int8 else torch.float32
        ).contiguous()
        fn = (
            deep_gemm.int8_paged_mqa_logits
            if dtype == torch.int8
            else deep_gemm.fp8_paged_mqa_logits
        )
    # Schedules are backend-opaque; do not reuse SGLang JIT or other-device metadata.
    schedule_meta = sail_paged_mqa_metadata(
        context_lens, kv_cache.shape[1], values.shape[2], head_dim, dtype, indices
    )
    return fn(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        indices=indices,
    )


def mqa_logits_free_mem_fraction() -> float:
    return envs.SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION.get()


def mqa_logits_needs_budget_check(*, num_rows: int, num_cols: int) -> bool:
    return num_rows * num_cols >= MQA_LOGITS_STATIC_SKIP_ELEMS


def mqa_logits_row_bytes(num_cols: int) -> int:
    aligned_cols = (
        ceil_div(num_cols, MQA_LOGITS_ROW_ALIGN_ELEMS) * MQA_LOGITS_ROW_ALIGN_ELEMS
    )
    return aligned_cols * MQA_LOGITS_BYTES_PER_ELEM


def mqa_logits_static_budget_bytes(*, device_index: int) -> int:
    """Budget from configuration alone (no device query); safe during graph capture."""
    total_mem = get_device_module().get_device_properties(device_index).total_memory
    total_mem_budget = int(total_mem * MQA_LOGITS_TOTAL_MEM_FRACTION)
    mem_fraction_static = get_schedule().mem_fraction_static
    if mem_fraction_static is None:
        budget = total_mem_budget
    else:
        static_free_mem = int(total_mem * max(0.0, 1.0 - mem_fraction_static))
        budget = min(
            int(static_free_mem * mqa_logits_free_mem_fraction()), total_mem_budget
        )
    return max(1, budget)


def mqa_logits_budget_bytes(*, device_index: int, allow_sync: bool) -> int:
    """Static budget capped by current free memory; mem_get_info syncs, so
    callers pass allow_sync=False under CUDA graph capture."""
    budget = mqa_logits_static_budget_bytes(device_index=device_index)
    if allow_sync and not is_xpu():
        free_mem, _ = torch.cuda.mem_get_info(device_index)
        budget = min(int(free_mem * mqa_logits_free_mem_fraction()), budget)
    if is_hip():
        budget = min(budget, MQA_LOGITS_MAX_BYTES_ROCM)
    return max(1, budget)


def mqa_logits_should_chunk(
    *,
    num_rows: int,
    num_cols: int,
    get_budget_bytes: Callable[[], int],
    rocm: bool,
) -> Tuple[bool, int]:
    """Whether a [num_rows, num_cols] fp32 logits matrix must be row-chunked.

    Returns (need_chunk, effective_budget_bytes). Matrices below
    MQA_LOGITS_STATIC_SKIP_ELEMS never chunk and the budget is not queried
    (it may synchronize the host), hence the callable. On ROCm the budget is
    also capped by aiter's logits ceiling.
    """
    if not mqa_logits_needs_budget_check(num_rows=num_rows, num_cols=num_cols):
        return False, 0
    budget_bytes = get_budget_bytes()
    if rocm:
        budget_bytes = min(budget_bytes, MQA_LOGITS_MAX_BYTES_ROCM)
    logits_bytes = num_rows * num_cols * MQA_LOGITS_BYTES_PER_ELEM
    return logits_bytes > budget_bytes, budget_bytes


def mqa_logits_rows_per_chunk(
    *, num_rows: int, row_bytes: int, budget_bytes: int
) -> Optional[int]:
    """Query rows per chunk so one logits chunk fits the budget; None if all rows fit."""
    if num_rows * row_bytes <= budget_bytes:
        return None
    rows = max(budget_bytes // max(row_bytes, 1), 1)
    return int(rows) if rows < num_rows else None
