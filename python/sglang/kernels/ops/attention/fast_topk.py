from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_FAST_TOPK_SUPPORTED_K = (512, 2048)
_FAST_TOPK_SUPPORTED_DTYPES = (torch.float32, torch.bfloat16)


@cache_once
def _jit_fast_topk_module(topk: int, dtype: torch.dtype) -> Module:
    if topk not in _FAST_TOPK_SUPPORTED_K:
        raise RuntimeError(
            f"Unsupported topk {topk}. Supported: {_FAST_TOPK_SUPPORTED_K}"
        )
    if dtype not in _FAST_TOPK_SUPPORTED_DTYPES:
        raise RuntimeError(
            f"Unsupported score dtype {dtype}. Supported: {_FAST_TOPK_SUPPORTED_DTYPES}"
        )
    args = make_cpp_args(topk, is_arch_support_pdl(), dtype)
    return load_jit(
        "fast_topk",
        *args,
        cuda_files=["elementwise/fast_topk.cuh"],
        cuda_wrappers=[
            ("fast_topk", f"FastTopKKernel<{args}>::run"),
            ("fast_topk_transform_fused", f"FastTopKKernel<{args}>::transform"),
            (
                "fast_topk_transform_ragged_fused",
                f"FastTopKKernel<{args}>::transform_ragged",
            ),
        ],
    )


def _validate_topk_inputs(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor],
) -> None:
    """
    Per-row top-k selection over a fp32/bf16 score matrix.

    Row b selects the `topk` largest values in
    ``score[b, row_starts[b] : row_starts[b] + lengths[b]]`` and returns their
    indices relative to ``row_starts[b]``. Slots beyond ``lengths[b]`` are -1.
    Output order within a row is unspecified (atomic collection order).

    Parameters
    ----------
    score      : CUDA fp32 or bf16 tensor [B, L]
    lengths    : CUDA int32 tensor [B]
    topk       : number of indices per row; 512 or 2048
    row_starts : optional CUDA int32 tensor [B]; defaults to zeros
    """
    if topk not in _FAST_TOPK_SUPPORTED_K:
        raise RuntimeError(
            f"Unsupported topk {topk}. Supported: {_FAST_TOPK_SUPPORTED_K}"
        )
    if score.dtype not in _FAST_TOPK_SUPPORTED_DTYPES:
        raise RuntimeError(
            f"Unsupported score dtype {score.dtype}. Supported: {_FAST_TOPK_SUPPORTED_DTYPES}"
        )
    if not score.is_cuda:
        raise RuntimeError("score must be a CUDA tensor")
    if score.dim() != 2 or score.stride(1) != 1:
        raise RuntimeError("score must be a 2D tensor contiguous in its last dimension")
    if (
        lengths.dim() != 1
        or lengths.shape[0] != score.shape[0]
        or lengths.dtype != torch.int32
        or not lengths.is_contiguous()
        or lengths.device != score.device
    ):
        raise RuntimeError("lengths must be contiguous CUDA int32 with shape [batch]")
    if row_starts is not None and (
        row_starts.dim() != 1
        or row_starts.shape[0] != score.shape[0]
        or row_starts.dtype != torch.int32
        or not row_starts.is_contiguous()
        or row_starts.device != score.device
    ):
        raise RuntimeError(
            "row_starts must be contiguous CUDA int32 with shape [batch]"
        )


def fast_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return row-relative top-k indices for FP32 or BF16 CUDA scores."""
    _validate_topk_inputs(score, lengths, topk, row_starts)
    indices = torch.empty(
        (score.shape[0], topk), dtype=torch.int32, device=score.device
    )
    if score.shape[0] == 0:
        return indices
    module = _jit_fast_topk_module(topk, score.dtype)
    module.fast_topk(score, indices, lengths, row_starts)
    return indices


def fast_topk_v2(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compatibility entry point matching ``sgl_kernel.fast_topk_v2``."""
    return fast_topk(score, lengths, topk, row_starts=row_starts)


def fast_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select top-k positions and gather their page-table entries."""
    _validate_topk_inputs(score, lengths, topk, row_starts)
    batch = score.shape[0]
    if (
        page_table_size_1.dim() != 2
        or page_table_size_1.dtype != torch.int32
        or page_table_size_1.stride(1) != 1
        or page_table_size_1.device != score.device
    ):
        raise RuntimeError(
            "page_table_size_1 must be CUDA int32 [query_sequences, width] "
            "and contiguous in its last dimension"
        )
    if (
        cu_seqlens_q.dim() != 1
        or cu_seqlens_q.dtype != torch.int32
        or not cu_seqlens_q.is_contiguous()
        or cu_seqlens_q.device != score.device
        or cu_seqlens_q.shape[0] != page_table_size_1.shape[0] + 1
    ):
        raise RuntimeError(
            "cu_seqlens_q must be contiguous CUDA int32 with one more entry "
            "than page-table rows"
        )
    # Empty query segments can make request rows outnumber score rows.
    # cu_seqlens_q describes their mapping; its length must still match the table.
    output = torch.empty((batch, topk), dtype=torch.int32, device=score.device)
    if batch == 0:
        return output
    if page_table_size_1.shape[0] == 0:
        raise RuntimeError("non-empty scores require at least one page-table row")
    module = _jit_fast_topk_module(topk, score.dtype)
    module.fast_topk_transform_fused(
        score,
        lengths,
        output,
        page_table_size_1,
        cu_seqlens_q,
        row_starts,
    )
    return output


def fast_topk_transform_ragged_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select row-relative top-k positions and add ragged-row offsets."""
    _validate_topk_inputs(score, lengths, topk, row_starts)
    if (
        topk_indices_offset.dim() != 1
        or topk_indices_offset.shape[0] != score.shape[0]
        or topk_indices_offset.dtype != torch.int32
        or not topk_indices_offset.is_contiguous()
        or topk_indices_offset.device != score.device
    ):
        raise RuntimeError(
            "topk_indices_offset must be contiguous CUDA int32 with shape [batch]"
        )

    output = torch.empty((score.shape[0], topk), dtype=torch.int32, device=score.device)
    if score.shape[0] == 0:
        return output
    module = _jit_fast_topk_module(topk, score.dtype)
    module.fast_topk_transform_ragged_fused(
        score, lengths, output, topk_indices_offset, row_starts
    )
    return output
