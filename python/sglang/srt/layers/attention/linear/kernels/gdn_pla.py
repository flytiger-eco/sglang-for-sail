"""Adapt the PLA FlashQLA prefill API to SGLang's indexed state pool."""

import logging
from functools import lru_cache
from itertools import accumulate

import torch

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def log_pla_prefill_track_fallback():
    logger.warning(
        "PLA GDN prefill does not expose intermediate chunk states; "
        "using Triton for batches requiring intermediate prefix-cache states."
    )


@lru_cache(maxsize=1)
def _log_pla_prefill():
    logger.info("USE PPU SAIL CUDA PLA GDN prefill kernel")


def pla_gdn_prefill(
    q,
    k,
    v,
    g,
    beta,
    *,
    ssm_states,
    cache_indices,
    query_start_loc,
    seq_lens_cpu=None,
):
    # Import only on the enabled prefill path: PLA's prefill extension is
    # packaged separately from decode and is not needed for Triton serving.
    from pla.prefill.flashqla import chunk_gated_delta_rule_fwd

    raw_cu_seqlens = (
        (0, *accumulate(seq_lens_cpu)) if seq_lens_cpu is not None else None
    )
    # Both indexed APIs store [slot, H, V, K]. In particular, do not transpose
    # to PLA's *non-indexed* legacy [batch, H, K, V] initial-state layout.
    needs_gather = not ssm_states.is_contiguous()
    indices = cache_indices.to(dtype=torch.int32).contiguous()
    if needs_gather:
        state = ssm_states[cache_indices].contiguous()
        indices = torch.arange(
            indices.numel(), dtype=torch.int32, device=indices.device
        )
    else:
        state = ssm_states

    _, _, output, _, _ = chunk_gated_delta_rule_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        initial_state=state,
        initial_state_indices=indices,
        cu_seqlens=query_start_loc.to(dtype=torch.int32).contiguous(),
        # Indexed PLA updates the pool even without materializing ht. The
        # caller reads final states from the pool, just as on the Triton path.
        output_final_state=False,
        normalize_qk=True,
        raw_cu_seqlens=raw_cu_seqlens,
        final_state_dtype=ssm_states.dtype,
    )
    if needs_gather:
        ssm_states[cache_indices] = state
    _log_pla_prefill()
    # PLA's ht is a final state, NOT Triton's per-chunk h. Batches needing h
    # must be dispatched to Triton before entering this adapter.
    return output, None, None
