# ruff: noqa
"""Standalone TileLang lowering for native-indexed PPU/AIU QSA."""

from __future__ import annotations

import tilelang
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def sparse_qsa_fwd(
    Q,
    K,
    V,
    Indices,
    PartialMax,
    PartialSum,
    PartialAcc,
    Counters,
    num_kv_heads: int,
    topk: int,
    num_splits: int,
    sm_scale: float,
    q_per_kv: int = 12,
    kv_head_base: int = -1,
):
    """BF16 QSA with compile-time split-KV and last-arriver FP32 merge."""

    assert num_kv_heads in (1, 2)
    assert num_splits in (1, 2, 4, 8)
    assert 1 <= q_per_kv <= 16
    assert -1 <= kv_head_base < num_kv_heads

    num_queries = T.dynamic("num_queries")
    num_slots = T.dynamic("num_slots")
    scratch_queries = T.dynamic("scratch_queries")

    head_dim = 256
    block_i = 64
    threads = 128
    cp_elems = 8
    copy_rounds = (block_i * head_dim) // (cp_elems * threads)
    # Keep the original multi-KV-group ABI when no mapped head is supplied.
    # TP-sharded queries instead consume just their single mapped KV head,
    # directly from the compact cache without a per-step K/V copy.
    kv_groups = num_kv_heads if kv_head_base < 0 else 1
    kv_head_offset = max(kv_head_base, 0)
    padded_q_per_kv = max(16, 1 << (q_per_kv - 1).bit_length())
    num_q_heads = kv_groups * q_per_kv
    dtype = T.bfloat16
    accum_dtype = T.float32
    num_tiles = tilelang.cdiv(topk, block_i)
    tiles_per_split = tilelang.cdiv(num_tiles, num_splits)
    sm_scale_log2 = sm_scale * 1.4426950408889634

    Q: T.Tensor([num_queries, num_q_heads, head_dim], dtype)  # type: ignore
    K: T.Tensor([num_slots, num_kv_heads, head_dim], dtype)  # type: ignore
    V: T.Tensor([num_slots, num_kv_heads, head_dim], dtype)  # type: ignore
    Indices: T.Tensor([num_queries, topk], T.int32)  # type: ignore
    PartialMax: T.Tensor(  # type: ignore
        [scratch_queries, kv_groups, num_splits, padded_q_per_kv],
        accum_dtype,
    )
    PartialSum: T.Tensor(  # type: ignore
        [scratch_queries, kv_groups, num_splits, padded_q_per_kv],
        accum_dtype,
    )
    PartialAcc: T.Tensor(  # type: ignore
        [
            scratch_queries,
            kv_groups,
            num_splits,
            padded_q_per_kv,
            head_dim,
        ],
        accum_dtype,
    )
    Counters: T.Tensor([scratch_queries, kv_groups], T.int32)  # type: ignore
    Output = T.empty([num_queries, num_q_heads, head_dim], dtype)

    with T.Kernel(num_queries, kv_groups, num_splits, threads=threads) as (
        q_idx,
        kv_head,
        split_idx,
    ):
        tx = T.get_thread_binding()
        q_shared = T.alloc_shared([padded_q_per_kv, head_dim], dtype)
        kv_shared = T.alloc_shared([block_i, head_dim], dtype)
        probability_shared = T.alloc_shared([padded_q_per_kv, block_i], dtype)
        last_arrival = T.alloc_shared([1], "int32")
        valid = T.alloc_fragment([block_i], "bool")

        scores = T.alloc_fragment([padded_q_per_kv, block_i], accum_dtype)
        output_acc = T.alloc_fragment([padded_q_per_kv, head_dim], accum_dtype)
        row_max = T.alloc_fragment([padded_q_per_kv], accum_dtype)
        previous_max = T.alloc_fragment([padded_q_per_kv], accum_dtype)
        tile_sum = T.alloc_fragment([padded_q_per_kv], accum_dtype)
        row_sum = T.alloc_fragment([padded_q_per_kv], accum_dtype)
        rescale = T.alloc_fragment([padded_q_per_kv], accum_dtype)

        T.fill(output_acc, 0)
        T.fill(row_sum, 0)
        # Finite sentinel avoids -inf - -inf for an entirely invalid split.
        T.fill(row_max, -(2**30))

        for h, d in T.Parallel(padded_q_per_kv, head_dim):
            q_shared[h, d] = T.if_then_else(
                h < q_per_kv,
                Q[q_idx, kv_head * q_per_kv + T.min(h, q_per_kv - 1), d],
                0,
            )
        T.sync_threads()

        tile_begin = split_idx * tiles_per_split
        for tile_offset in T.serial(tiles_per_split):
            global_tile = tile_begin + tile_offset
            tile_active = global_tile < num_tiles
            for col in T.Parallel(block_i):
                token_offset = global_tile * block_i + col
                safe_offset = T.min(token_offset, topk - 1)
                index = Indices[q_idx, safe_offset]
                valid[col] = (
                    tile_active
                    & (token_offset < topk)
                    & (index >= 0)
                    & (index < num_slots)
                )
            for h, col in T.Parallel(padded_q_per_kv, block_i):
                scores[h, col] = T.if_then_else(
                    valid[col], 0, -T.infinity(scores.dtype)
                )

            for copy_round in T.unroll(copy_rounds):
                chunk = tx + copy_round * threads
                row = chunk // (head_dim // cp_elems)
                col = (chunk % (head_dim // cp_elems)) * cp_elems
                token_offset = global_tile * block_i + row
                safe_offset = T.min(token_offset, topk - 1)
                index = Indices[q_idx, safe_offset]
                is_valid = (
                    tile_active
                    & (token_offset < topk)
                    & (index >= 0)
                    & (index < num_slots)
                )
                T.ptx_cp_async(
                    T.access_ptr(kv_shared[row, col], "w", cp_elems),
                    T.access_ptr(
                        K[T.max(index, 0), kv_head + kv_head_offset, col], "r", cp_elems
                    ),
                    cp_elems,
                    predicate=is_valid,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.sync_threads()

            T.gemm(
                q_shared,
                kv_shared,
                scores,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )

            T.copy(row_max, previous_max)
            T.reduce_max(scores, row_max, dim=1, clear=False)
            for h in T.Parallel(padded_q_per_kv):
                row_max[h] = T.max(row_max[h], previous_max[h])
                rescale[h] = T.exp2((previous_max[h] - row_max[h]) * sm_scale_log2)
            for h, col in T.Parallel(padded_q_per_kv, block_i):
                scores[h, col] = T.exp2(
                    scores[h, col] * sm_scale_log2 - row_max[h] * sm_scale_log2
                )
            T.reduce_sum(scores, tile_sum, dim=1)
            for h in T.Parallel(padded_q_per_kv):
                row_sum[h] = row_sum[h] * rescale[h] + tile_sum[h]
            for h, d in T.Parallel(padded_q_per_kv, head_dim):
                output_acc[h, d] *= rescale[h]

            T.copy(scores, probability_shared)
            T.sync_threads()

            for copy_round in T.unroll(copy_rounds):
                chunk = tx + copy_round * threads
                row = chunk // (head_dim // cp_elems)
                col = (chunk % (head_dim // cp_elems)) * cp_elems
                token_offset = global_tile * block_i + row
                safe_offset = T.min(token_offset, topk - 1)
                index = Indices[q_idx, safe_offset]
                is_valid = (
                    tile_active
                    & (token_offset < topk)
                    & (index >= 0)
                    & (index < num_slots)
                )
                T.ptx_cp_async(
                    T.access_ptr(kv_shared[row, col], "w", cp_elems),
                    T.access_ptr(
                        V[T.max(index, 0), kv_head + kv_head_offset, col], "r", cp_elems
                    ),
                    cp_elems,
                    predicate=is_valid,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.sync_threads()

            T.gemm(
                probability_shared,
                kv_shared,
                output_acc,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.sync_threads()

        if num_splits == 1:
            for h, d in T.Parallel(q_per_kv, head_dim):
                Output[q_idx, kv_head * q_per_kv + h, d] = T.if_then_else(
                    row_sum[h] > 0,
                    output_acc[h, d] / row_sum[h],
                    0,
                )
        else:
            for h in T.Parallel(padded_q_per_kv):
                PartialMax[q_idx, kv_head, split_idx, h] = row_max[h]
                PartialSum[q_idx, kv_head, split_idx, h] = row_sum[h]
            for h, d in T.Parallel(padded_q_per_kv, head_dim):
                PartialAcc[q_idx, kv_head, split_idx, h, d] = output_acc[h, d]
            T.sync_threads()

            if tx == 0:
                arrival = T.atomic_add(
                    Counters[q_idx, kv_head],
                    1,
                    memory_order="acq_rel",
                    return_prev=True,
                )
                last_arrival[0] = T.if_then_else(arrival == num_splits - 1, 1, 0)
            T.sync_threads()

            if last_arrival[0] == 1:
                merged_max = T.alloc_fragment([padded_q_per_kv], accum_dtype)
                merged_sum = T.alloc_fragment([padded_q_per_kv], accum_dtype)
                merged_acc = T.alloc_fragment([padded_q_per_kv, head_dim], accum_dtype)
                merge_weight = T.alloc_fragment([padded_q_per_kv], accum_dtype)
                T.fill(merged_max, -(2**30))
                T.fill(merged_sum, 0)
                T.fill(merged_acc, 0)

                for part in T.serial(num_splits):
                    for h in T.Parallel(padded_q_per_kv):
                        merged_max[h] = T.max(
                            merged_max[h],
                            PartialMax[q_idx, kv_head, part, h],
                        )
                for part in T.serial(num_splits):
                    for h in T.Parallel(padded_q_per_kv):
                        merge_weight[h] = T.exp2(
                            (PartialMax[q_idx, kv_head, part, h] - merged_max[h])
                            * sm_scale_log2
                        )
                        merged_sum[h] += (
                            merge_weight[h] * PartialSum[q_idx, kv_head, part, h]
                        )
                    for h, d in T.Parallel(padded_q_per_kv, head_dim):
                        merged_acc[h, d] += (
                            merge_weight[h] * PartialAcc[q_idx, kv_head, part, h, d]
                        )

                for h, d in T.Parallel(q_per_kv, head_dim):
                    Output[q_idx, kv_head * q_per_kv + h, d] = T.if_then_else(
                        merged_sum[h] > 0,
                        merged_acc[h, d] / merged_sum[h],
                        0,
                    )
                T.sync_threads()
                if tx == 0:
                    T.atomic_store(
                        Counters[q_idx, kv_head],
                        0,
                        memory_order="release",
                    )

    return Output


__all__ = ["sparse_qsa_fwd"]
