"""The top-k selection of the DeepSeek V4.1 ratio-1/2 index layers: the full top-k
(``full_topk``), and the candidate schemes split by what a source publishes, a
DeepGEMM sparse table (``sparse_table``) or dense block ids (``dense_blocks``)."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.layers.attention.dsv4.indexer_quant import use_dsv41_int8_indexer
from sglang.srt.utils import get_device_sm, is_ppu

from .full_topk import FullTopKIndexer
from .types import (
    CandidateMetadata,
    CapturedPrefillInputs,
    DecodeCandidates,
    DecodeInputs,
    PrefillCandidates,
    PrefillInputs,
    Selection,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool

logger = logging.getLogger(__name__)

__all__ = [
    "CandidateMetadata",
    "CapturedPrefillInputs",
    "DecodeCandidates",
    "DecodeInputs",
    "FullTopKIndexer",
    "PrefillCandidates",
    "PrefillInputs",
    "Selection",
    "has_dense_fp4_indexer",
    "is_sm100_or_newer",
    "make_candidate_indexer",
    "make_full_topk_indexer",
]


@functools.cache
def is_sm100_or_newer() -> bool:
    # DeepGEMM's fp8_fp4 mqa-logits kernels need SM100+; Hopper takes the torch indexer.
    return torch.cuda.get_device_capability()[0] >= 10


def use_deep_gemm_fp4_indexer() -> bool:
    return is_ppu() or is_sm100_or_newer()


@functools.cache
def has_dense_fp4_indexer() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return hasattr(deep_gemm, "fp8_fp4_mqa_logits")


@functools.cache
def _use_deep_gemm_prefill() -> bool:
    # A CP rank scores rank-local rows, which only DeepGEMM serves, so CP ignores the env.
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsv4.indexer_quant import use_dsv41_int8_indexer
    from sglang.srt.runtime_context import get_parallel

    return (
        not use_dsv41_int8_indexer()
        and use_deep_gemm_fp4_indexer()
        and has_dense_fp4_indexer()
        and (
            get_parallel().attn_cp_size > 1
            or not envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get()
        )
    )


def make_full_topk_indexer(
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
) -> FullTopKIndexer:
    from sglang.srt.layers.attention.dsv4.indexer_quant import use_dsv41_int8_indexer

    return FullTopKIndexer(
        token_to_kv_pool=token_to_kv_pool,
        req_to_token=req_to_token,
        use_deep_gemm_prefill=_use_deep_gemm_prefill(),
        use_deep_gemm_decode=use_deep_gemm_fp4_indexer()
        and not use_dsv41_int8_indexer(),
    )


def make_candidate_indexer(
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    page_size: int,
    candidate_topk_blocks: int,
    candidate_block_size: int,
) -> Tuple[PrefillCandidates, DecodeCandidates]:
    from .dense_blocks import DenseBlocksBackend

    use_deep_gemm_prefill = _use_deep_gemm_prefill()
    dense_blocks = DenseBlocksBackend(
        token_to_kv_pool=token_to_kv_pool,
        req_to_token=req_to_token,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        use_deep_gemm_prefill=use_deep_gemm_prefill,
    )
    # The sparse table path packs FP4 index-K; an INT8 pool stores 132-byte
    # slots, and the dense backend dispatches on indexer.use_int8 itself.
    if use_dsv41_int8_indexer():
        return dense_blocks, dense_blocks
    # Without candidate blocks no layer is a candidate source or consumer.
    supports_sparse = get_device_sm() >= 89 if is_ppu() else is_sm100_or_newer()
    if not supports_sparse or candidate_topk_blocks <= 0:
        return dense_blocks, dense_blocks

    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
        DEEPGEMM_SPARSE_MQA_LOGITS,
    )

    # Decode reads the paged index-K pool; without that API it stays dense.
    decode_backend = dense_blocks
    if DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
        from .sparse_table import SparseTableBackend

        decode_backend = SparseTableBackend(
            token_to_kv_pool=token_to_kv_pool,
            req_to_token=req_to_token,
            page_size=page_size,
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=candidate_block_size,
        )
    else:
        logger.warning(
            "Candidate indexer requested but DeepGEMM has no paged sparse MQA "
            "logits API (sgl-deep-gemm >= 0.2.0 with "
            "SGLANG_ENABLE_JIT_DEEPGEMM on); decode uses dense blocks."
        )

    # The torch prefill env keeps prefill on dense blocks.
    if not use_deep_gemm_prefill:
        return dense_blocks, decode_backend
    # The varlen path scores the published blocks over the request-flattened
    # index K, so it serves both CP and non-CP prefill.
    if DEEPGEMM_SPARSE_MQA_LOGITS:
        from .sparse_varlen import VarlenSparsePrefillBackend

        logger.info("Prefill candidate indexer: DeepGEMM varlen sparse MQA logits")
        return (
            VarlenSparsePrefillBackend(
                token_to_kv_pool=token_to_kv_pool,
                req_to_token=req_to_token,
                candidate_topk_blocks=candidate_topk_blocks,
            ),
            decode_backend,
        )
    logger.warning(
        "Prefill candidate indexer: DeepGEMM has no varlen sparse MQA logits "
        "API; using the paged sparse table."
    )
    return decode_backend, decode_backend
