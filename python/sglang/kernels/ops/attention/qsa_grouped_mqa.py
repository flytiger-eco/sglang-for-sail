"""PPU grouped MQA scoring for paged speculative verification."""

import math

import tilelang
import torch
from tilelang import language as T

from sglang.srt.utils import is_ppu

_is_ppu = is_ppu()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def _qsa_mqa_kernel(
    heads: int,
    physical_heads: int,
    head_dim: int,
    queries: int,
    block_keys: int = 64,
    page_size: int = 16,
    key_splits: int = 64,
    threads: int = 128,
    dtype: str = "float8_e4m3fn",
):
    qpad = 1 << (queries - 1).bit_length()
    head_multiple = max(1, 16 // qpad)
    hpad = (heads + head_multiple - 1) // head_multiple * head_multiple
    groups = T.dynamic("groups")
    keys = T.dynamic("keys")
    max_pages = T.dynamic("max_pages")
    output_keys = T.dynamic("output_keys")
    rows = groups * queries
    q_shape = [rows, physical_heads, head_dim]
    k_shape = [keys, page_size, 1, head_dim]

    @T.macro
    def score_and_store(
        qs: T.Buffer([qpad * hpad, head_dim], dtype),
        ks: T.Buffer([block_keys, head_dim], dtype),
        scores: T.Buffer([block_keys, qpad * hpad], T.float32),
        reduced: T.Buffer([block_keys, qpad], T.float32),
        ends: T.Buffer([rows], T.int32),
        out: T.Buffer([rows, output_keys], T.float32),
        row_base: T.int32,
        column_base: T.int32,
        scale: T.float32,
    ):
        T.gemm(
            ks,
            qs,
            scores,
            transpose_B=True,
            clear_accum=True,
            policy=T.GemmWarpPolicy.FullCol,
        )
        for ki, qhi in T.Parallel(block_keys, qpad * hpad):
            scores[ki, qhi] = T.max(scores[ki, qhi], 0.0)
        view = T.reshape(scores, [block_keys, qpad, hpad])
        T.reduce_sum(view, reduced, dim=2, clear=True)
        for qi, ki in T.Parallel(qpad, block_keys):
            if qi < queries:
                if (
                    column_base + ki < ends[row_base + qi]
                    and column_base + ki < output_keys
                ):
                    out[row_base + qi, column_base + ki] = reduced[ki, qi] / scale

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        Table: T.Tensor([rows, max_pages], T.int32),
        Ends: T.Tensor([rows], T.int32),
        Out: T.Tensor([rows, output_keys], T.float32),
        Scale: T.float32,
    ):
        with T.Kernel(groups, key_splits, threads=threads) as (bx, sx):
            qs = T.alloc_shared([qpad * hpad, head_dim], dtype)
            ks = T.alloc_shared([block_keys, head_dim], dtype)
            scores = T.alloc_fragment([block_keys, qpad * hpad], T.float32)
            reduced = T.alloc_fragment([block_keys, qpad], T.float32)
            base = bx * queries
            T.clear(qs)
            for qi in range(queries):
                T.copy(Q[base + qi, :heads, :], qs[qi * hpad : qi * hpad + heads, :])
            last = base + queries - 1
            end = T.min(Ends[last], output_keys)
            count = T.ceildiv(end, block_keys)
            for step in T.serial(T.ceildiv(T.max(count - sx, 0), key_splits)):
                block = sx + step * key_splits
                if block * block_keys < end:
                    for sp in range(block_keys // page_size):
                        if block * block_keys + sp * page_size < end:
                            T.copy(
                                K[
                                    Table[
                                        last, block * (block_keys // page_size) + sp
                                    ],
                                    :,
                                    0,
                                    :,
                                ],
                                ks[sp * page_size : (sp + 1) * page_size, :],
                            )
                    score_and_store(
                        qs,
                        ks,
                        scores,
                        reduced,
                        Ends,
                        Out,
                        base,
                        block * block_keys,
                        Scale,
                    )

    return kernel.with_attr("global_symbol", "qsa_mqa")


def qsa_grouped_mqa_supported(
    q, k_cache, page_table, context_lens, *, num_query_heads, num_queries_per_request
):
    return (
        _is_ppu
        and q.is_cuda
        and q.ndim == 3
        and q.shape[0] > 0
        and q.shape[2] == 128
        and num_queries_per_request in (3, 4, 5, 6)
        and q.shape[0] % num_queries_per_request == 0
        and num_query_heads in (1, 2, 4, 8)
        and num_query_heads <= q.shape[1]
        and k_cache.ndim == 4
        and k_cache.shape[1] in (8, 16, 32, 64)
        and k_cache.shape[2:] == (1, 128)
        and q.dtype == k_cache.dtype
        and q.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and page_table.ndim == 2
        and page_table.shape[0] == q.shape[0]
        and context_lens.shape == (q.shape[0],)
        and page_table.dtype == context_lens.dtype == torch.int32
        and q.device == k_cache.device == page_table.device == context_lens.device
        and all(t.is_contiguous() for t in (q, k_cache, page_table, context_lens))
    )


def qsa_grouped_mqa_decode(
    *,
    q,
    k_cache,
    page_table,
    context_lens,
    max_model_len,
    num_query_heads,
    num_queries_per_request,
):
    """Score ordered queries from one request with one shared K scan.

    Each query retains its own causal length, head reduction, and output row.
    The producer must guarantee same-request contiguous groups and a final-row
    page table that covers every earlier query's visible keys.
    """
    if not qsa_grouped_mqa_supported(
        q,
        k_cache,
        page_table,
        context_lens,
        num_query_heads=num_query_heads,
        num_queries_per_request=num_queries_per_request,
    ):
        raise ValueError("unsupported grouped QSA scoring inputs")
    out = torch.full(
        (q.shape[0], max_model_len), -float("inf"), device=q.device, dtype=torch.float32
    )
    if not max_model_len:
        return out
    dtype = "float8_e4m3fn" if q.dtype == torch.float8_e4m3fn else "bfloat16"
    _qsa_mqa_kernel(
        heads=num_query_heads,
        physical_heads=q.shape[1],
        head_dim=q.shape[2],
        queries=num_queries_per_request,
        page_size=k_cache.shape[1],
        dtype=dtype,
    )(q, k_cache, page_table, context_lens, out, math.sqrt(q.shape[-1]))
    return out
