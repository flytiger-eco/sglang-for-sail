from __future__ import annotations

from typing import Optional

import torch


def bf16_mega_gate(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
    *,
    routed_scaling_factor: float,
    ep_rank: int,
    scoring_func: str = "sqrtsoftplus",
    bias: Optional[torch.Tensor] = None,
    image_bias: Optional[torch.Tensor] = None,
    image_token_mask: Optional[torch.Tensor] = None,
    valid_token_mask: Optional[torch.Tensor] = None,
    hash_topk_ids: Optional[torch.Tensor] = None,
    to_physical_map: Optional[torch.Tensor] = None,
    logical_count: Optional[torch.Tensor] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (weights, ids). Supply out in the same order to reuse both tensors.

    `hash_topk_ids` is a writable per-call buffer. Masked rows are overwritten
    with -1, so refill them with valid expert IDs before reusing them as active rows.
    """
    import deep_gemm

    if out is None:
        ids = torch.empty(
            (hidden_states.shape[0], top_k),
            dtype=torch.int64,
            device=hidden_states.device,
        )
        weights = torch.empty_like(ids, dtype=torch.float32)
    else:
        weights, ids = out
    deep_gemm.bf16_mega_gate(
        hidden_states,
        weight,
        top_k,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        ep_rank=ep_rank,
        scoring_func=scoring_func,
        mask=valid_token_mask,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        fix_routing_mask=(
            torch.ones_like(ids[:, 0], dtype=torch.bool)
            if hash_topk_ids is not None
            else None
        ),
        unmapped_topk_idx=hash_topk_ids,
        to_physical_map=to_physical_map,
        logical_count=logical_count,
        out=(ids, weights),
    )
    return weights, ids
