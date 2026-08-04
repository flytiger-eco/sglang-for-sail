from __future__ import annotations

from typing import Optional

from sglang.srt.runtime_context import get_server_args
from sglang.srt.server_args import ServerArgs


def get_alloc_len_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    if server_args is None:
        server_args = get_server_args()

    if server_args.speculative_algorithm is None:
        return 1

    # Spec decoding allocates max(topk * num_steps, num_draft_tokens) per decode step.
    spec_steps = server_args.speculative_num_steps or 1
    spec_topk = server_args.speculative_eagle_topk or 1
    spec_tokens = server_args.max_speculative_num_draft_tokens
    page_size = server_args.page_size

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    spec_algo = SpeculativeAlgorithm.from_string(server_args.speculative_algorithm)
    if page_size == 1 or spec_topk == 1 or not spec_algo.has_draft_kv():
        return max(spec_steps * spec_topk, spec_tokens)
    else:
        # spec v2 tree (page>1, topk>1): worst-case page-aligned footprint per
        # topk branch is ceil((page_size-1 + num_steps) / page) pages, each branch
        # duplicated -- reserve for all topk branches.
        num_new_pages_per_topk = (
            (page_size - 1) + spec_steps + page_size - 1
        ) // page_size
        return max(num_new_pages_per_topk * page_size * spec_topk, spec_tokens)


def get_alloc_reserve_per_decode(server_args: Optional[ServerArgs] = None) -> int:
    """KV length reserved per request at each decode step.

    The 2x is a double-buffer that absorbs the kv_committed_len lag in overlap
    mode; see eagle_utils.eagle_prepare_for_decode.
    """
    return 2 * get_alloc_len_per_decode(server_args)


def get_req_to_token_extra_context_len(server_args: ServerArgs) -> int:
    """req_to_token row headroom beyond the model context length.

    Sized to hold the decode over-allocation; the spec v2 page>1 topk>1 holey
    draft footprint can outgrow the default num_draft_tokens headroom.
    """
    # FIXME(lsyin): temporary fix for the context length issue under spec decoding
    extra = 4 + (server_args.max_speculative_num_draft_tokens or 0)
    if server_args.speculative_algorithm is not None and server_args.page_size > 1:
        # kv_allocated_len is page-aligned (eagle_prepare_for_decode), so near
        # the context limit the aligned reserve can overshoot by page_size - 1;
        # without the headroom the row write silently lands in the neighbor row.
        extra = max(
            extra,
            get_alloc_reserve_per_decode(server_args) + server_args.page_size - 1,
        )
    return extra


def estimate_max_running_requests(
    token_capacity: int,
    context_len: int,
    server_args: ServerArgs,
    attn_dp_size: int,
    mamba_req_cap: Optional[int] = None,
) -> int:
    """Single source of truth for the max_running_requests derivation.

    Called twice per boot with the same formula so the two call sites can
    never drift apart:
    - KVCacheConfigurator._estimate_req_to_token_pool_bytes sizes the
      req_to_token pool deduction during memory profiling (one-iteration
      estimate from a provisional token capacity);
    - KVCacheConfigurator.resolve_max_num_reqs resolves the final
      max_running_requests once token capacity is fixed.

    ``mamba_req_cap`` is the hybrid-mamba state-cache clamp
    (max_mamba_cache_size // mamba_ratio); pass None for non-mamba models.
    """
    # Estimate pool size (used as upper bound when user specifies max_running_requests)
    estimated = int(token_capacity / context_len * 512)
    estimated = max(min(estimated, 4096), 2048)

    max_num_reqs = server_args.max_running_requests
    if max_num_reqs is not None:
        requested_per_worker = max_num_reqs // attn_dp_size
        max_num_reqs = min(requested_per_worker, token_capacity // 2)
    else:
        max_num_reqs = min(estimated, token_capacity // 2)

    if mamba_req_cap is not None:
        max_num_reqs = min(max_num_reqs, mamba_req_cap)
    return max_num_reqs


def get_req_to_token_pool_num_slots(max_num_reqs: int, server_args: ServerArgs) -> int:
    """Rows in the req_to_token map for a given max_running_requests.

    The +1 is the padding row at index 0 (ReqToTokenPool._alloc_size); decode
    mode additionally subscribes rows for pre-allocated in-transfer requests
    (DecodeReqToTokenPool._alloc_size).
    """
    num_slots = max_num_reqs + 1
    if server_args.disaggregation_mode == "decode":
        num_slots += server_args.disaggregation_decode_extra_slots or 0
    return num_slots


def estimate_req_to_token_pool_bytes(
    token_capacity: int,
    context_len: int,
    server_args: ServerArgs,
    attn_dp_size: int,
    mamba_req_cap: Optional[int] = None,
) -> int:
    """GPU bytes the req_to_token map will occupy for this token capacity
    (num_slots x max_context_len x int32)."""
    max_num_reqs = estimate_max_running_requests(
        token_capacity=token_capacity,
        context_len=context_len,
        server_args=server_args,
        attn_dp_size=attn_dp_size,
        mamba_req_cap=mamba_req_cap,
    )
    max_context_len = context_len + get_req_to_token_extra_context_len(server_args)
    return (
        get_req_to_token_pool_num_slots(max_num_reqs, server_args)
        * max_context_len
        * 4  # int32
    )
