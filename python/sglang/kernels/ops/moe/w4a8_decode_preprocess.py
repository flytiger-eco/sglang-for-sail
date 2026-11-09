"""Bounded decode layout for Qwen38 W4A8 Standard MoE.

One CTA builds the same stable expert layout as torch.sort(stable=True),
normalizes the EP -1 sentinel, and writes CUTLASS problem sizes/offsets.
The existing input quant/reorder, grouped GEMMs, and output sum are unchanged.
"""

import torch
import triton
import triton.language as tl

MAX_DECODE_ROUTED_SLOTS = 1024


@triton.autotune(
    configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)],
    key=["N", "E"],
)
@triton.jit
def _w4a8_decode_preprocess_kernel(
    Ids,
    NormalizedIds,
    SortedIds,
    Src2Dst,
    ExpertOffsets,
    Problems1,
    Problems2,
    N: tl.constexpr,
    E: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    row = tl.arange(0, BLOCK_N)
    mask = row < N
    expert = tl.load(Ids + row, mask, other=E).to(tl.int32)
    expert = tl.where(expert == -1, E, expert)
    tl.store(NormalizedIds + row, expert, mask)
    # Unique position bits reproduce stable sorting, including the EP sentinel.
    # N <= 1024; 12 position bits leave room for all padded lanes.
    packed = tl.sort((expert << 12) | row, descending=False)
    sorted_expert, source = packed >> 12, packed & 4095
    tl.store(SortedIds + row, sorted_expert, mask)
    tl.store(Src2Dst + source, row, mask)
    tl.debug_barrier()

    # lower_bound(e) gives exact unpadded expert offsets. All valid experts
    # precede sentinel E, so offset[E] excludes EP-filtered slots.
    e = tl.arange(0, BLOCK_E)
    emask = e <= E
    lo = tl.zeros([BLOCK_E], tl.int32)
    hi = tl.full([BLOCK_E], N, tl.int32)
    for _ in tl.static_range(12):
        mid = (lo + hi) // 2
        value = tl.load(SortedIds + mid, emask & (mid < N), other=E + 1)
        right = value < e
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    tl.store(ExpertOffsets + e, lo, emask)
    tl.debug_barrier()
    counts = tl.load(ExpertOffsets + e + 1, e < E, other=0) - lo
    tl.store(Problems1 + e * 3, 2 * I, e < E)
    tl.store(Problems1 + e * 3 + 1, counts, e < E)
    tl.store(Problems1 + e * 3 + 2, H, e < E)
    tl.store(Problems2 + e * 3, H, e < E)
    tl.store(Problems2 + e * 3 + 1, counts, e < E)
    tl.store(Problems2 + e * 3 + 2, I, e < E)


def supports_w4a8_decode_preprocess(topk_ids: torch.Tensor, experts: int) -> bool:
    return (
        experts == 128
        and topk_ids.is_cuda
        and topk_ids.dtype in (torch.int32, torch.int64)
        and topk_ids.is_contiguous()
        and 0 < topk_ids.numel() <= MAX_DECODE_ROUTED_SLOTS
    )


def w4a8_decode_preprocess(
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes1: torch.Tensor,
    problem_sizes2: torch.Tensor,
    hidden_size: int,
    intermediate_size: int,
):
    """Return normalized ids and stable src2dst; update real CUTLASS buffers."""
    experts = problem_sizes1.shape[0]
    if not supports_w4a8_decode_preprocess(topk_ids, experts):
        raise ValueError(
            "requires contiguous CUDA int32/int64 ids, E128, 1..1024 slots"
        )
    n = topk_ids.numel()
    normalized = torch.empty_like(topk_ids, dtype=torch.int32)
    sorted_ids = torch.empty(n, device=topk_ids.device, dtype=torch.int32)
    src2dst = torch.empty_like(sorted_ids)
    _w4a8_decode_preprocess_kernel[(1,)](
        topk_ids,
        normalized,
        sorted_ids,
        src2dst,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        n,
        experts,
        hidden_size,
        intermediate_size,
        BLOCK_N=max(triton.next_power_of_2(n), 16),
        BLOCK_E=triton.next_power_of_2(experts + 1),
    )
    return normalized, src2dst
