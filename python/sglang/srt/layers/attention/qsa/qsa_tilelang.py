"""TileLang QSA runtime: sparse attention straight off physical KV slots.

This module owns the split-KV scratch lifetime, the ``num_splits`` choice and
the shape guards around ``kernels/ops/attention/qsa_ppu.sparse_qsa_fwd``. It
exposes the same contract as ``qsa.kernel.qsa_sparse_attention``:
``token_slots`` rows index the KV cache's first dimension, and negatives are
padding.

Because the kernel consumes ``token_slots`` directly it needs neither the packed
K/V scratch nor the ``cu_seqlens`` metadata that the FA2/FA3 fallback builds, so
routing here removes the gather from the decode critical path.

The kernel keeps the legacy 12:1 multi-KV-group layout and also accepts a
TP-sharded query slice mapping to one cached KV head. The caller resolves that
head from the attention-TP rank; unsupported or straddling slices fall back.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import is_ppu

logger = logging.getLogger(__name__)

# Mirrors the kernel's compile-time constants; a mismatch means the launch would
# read out of bounds rather than merely run slowly, so these are hard gates.
_KERNEL_HEAD_DIM = 256
_KERNEL_Q_PER_KV = 12
_KERNEL_PADDED_Q_PER_KV = 16
_SUPPORTED_KV_HEADS = (1, 2)
_SUPPORTED_Q_PER_KV = (1, 2, 3, 4, 6, 8, 12, 16)
_SUPPORTED_SPLITS = (1, 2, 4, 8)
# Measured on ZW-M890P for the 24-q-head (TP1) shape; see _choose_num_splits.
_TP1_TUNED_SPLITS = {1: 8, 32: 2, 64: 1, 128: 1, 256: 2}


@lru_cache(maxsize=1)
def is_available() -> bool:
    """Whether the TileLang QSA kernel can be used in this process."""

    if not is_ppu():
        return False
    if not envs.SGLANG_QSA_USE_TILELANG.get():
        return False
    try:
        import tilelang  # noqa: F401

        from sglang.kernels.ops.attention.qsa_ppu import sparse_qsa_fwd  # noqa: F401
    except ImportError as exc:
        logger.warning("TileLang QSA unavailable (%s); using the packed path.", exc)
        return False
    return True


def is_supported(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    """Whether this (q, kv) shape/dtype pair matches the kernel's constants."""

    if q.dtype is not torch.bfloat16 or k_cache.dtype is not torch.bfloat16:
        return False
    if q.shape[-1] != _KERNEL_HEAD_DIM or k_cache.shape[-1] != _KERNEL_HEAD_DIM:
        return False
    num_kv_heads = k_cache.shape[1]
    if num_kv_heads not in _SUPPORTED_KV_HEADS:
        return False
    return (
        q.shape[1] == num_kv_heads * _KERNEL_Q_PER_KV
        or q.shape[1] in _SUPPORTED_Q_PER_KV
    )


def _choose_num_splits(rows: int, num_kv_heads: int, num_q_heads: int) -> int:
    """Pick the kernel's compile-time split-KV count.

    Transcribed from ppu-dev ``d24b34d328``'s
    ``benchmark/kernels/attention/bench_qsa_ppu_vs_fa3.py``, measured on
    ZW-M890P. The split count tracks how much row-level parallelism the launch
    already has -- ``rows * num_kv_heads`` -- and is independent of ``topk``:
    with enough rows the device is busy already and the extra FP32 merge pass
    only costs time. The 24-head (TP1) entries are a measured table that does
    not follow the generic rule.

    ``num_splits`` is a kernel constexpr, so each distinct value pays one
    TileLang compile (~15s) on first use; serving warmup touches at most a few.
    """

    override = envs.SGLANG_QSA_TILELANG_NUM_SPLITS.get()
    if override:
        if override not in _SUPPORTED_SPLITS:
            raise ValueError(
                f"SGLANG_QSA_TILELANG_NUM_SPLITS must be one of "
                f"{_SUPPORTED_SPLITS}, got {override}"
            )
        return override
    if num_q_heads == 24 and rows in _TP1_TUNED_SPLITS:
        return _TP1_TUNED_SPLITS[rows]
    if rows > 256:
        return 1
    if rows * num_kv_heads <= 32:
        return 4
    return 2


# The automatic policy shares one baseline capacity per split specialization.
# A manual override beyond that capacity gets a separate, row-sized entry. No
# entry is ever resized: a CUDA graph captured against an old allocation would
# otherwise keep replaying into freed memory. Oversizing is correct because
# `scratch_queries` is dynamic and the merge phase touches rows < num_queries.
_MAX_ROWS_FOR_SPLITS = {
    1: 1,  # single-pass path never touches the partials
    2: 256,  # generic else-branch covers rows <= 256
    4: 32,  # generic branch requires rows * kv_heads <= 32
    8: 1,  # only reachable from the TP1 table at rows == 1
}

_SCRATCH: Dict[
    Tuple[int, int, int, int, torch.device],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _scratch_capacity_rows(rows: int, num_splits: int) -> int:
    """Stable capacity for one split specialization.

    The automatic policy fits the measured caps above. A manual environment
    override may deliberately use more rows; give that shape its own cached
    allocation rather than rejecting it or resizing storage captured by an
    earlier CUDA graph.
    """

    if num_splits == 1:
        return 1
    return max(rows, _MAX_ROWS_FOR_SPLITS[num_splits])


def _scratch(
    rows: int,
    kv_groups: int,
    num_splits: int,
    padded_q_per_kv: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scratch_rows = _scratch_capacity_rows(rows, num_splits)
    key = (scratch_rows, kv_groups, num_splits, padded_q_per_kv, device)
    buffers = _SCRATCH.get(key)
    if buffers is None:
        partial_shape = (
            scratch_rows,
            kv_groups,
            num_splits,
            padded_q_per_kv,
        )
        partial_max = torch.empty(partial_shape, device=device, dtype=torch.float32)
        partial_sum = torch.empty_like(partial_max)
        partial_acc = torch.empty(
            (*partial_shape, _KERNEL_HEAD_DIM), device=device, dtype=torch.float32
        )
        # The kernel releases each counter back to 0 on its last arriver, so a
        # cached buffer stays reusable across launches; only the first launch
        # depends on this zero-init.
        counters = torch.zeros(
            (scratch_rows, kv_groups), device=device, dtype=torch.int32
        )
        buffers = (partial_max, partial_sum, partial_acc, counters)
        _SCRATCH[key] = buffers
    return buffers


def sparse_qsa_tilelang(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: Optional[float] = None,
    kv_head_base: int = -1,
) -> torch.Tensor:
    """Sparse GQA over physical KV slots, shaped ``[query_rows, q_heads, 256]``.

    Same contract as ``qsa.kernel.qsa_sparse_attention``; prefill rows work too
    because the kernel reads one selection row per query row (the flattened
    ``B*S`` case its test covers).
    """

    from sglang.kernels.ops.attention.qsa_ppu import sparse_qsa_fwd

    # ``num_queries`` is one dynamic dim shared by Q and Indices, so a short
    # selection would make the kernel read past the end of ``token_slots``
    # instead of failing.  Callers trim padded query rows before this point
    # (see ``forward_extend``), so a mismatch is a caller bug.
    if token_slots.shape[0] != q.shape[0]:
        raise ValueError(
            "TileLang QSA needs one selection row per query row: "
            f"slots={token_slots.shape[0]}, query={q.shape[0]}"
        )

    num_kv_heads = k_cache.shape[1]
    if not is_supported(q, k_cache):
        raise ValueError("Unsupported TileLang QSA query/cache shape or dtype")
    if kv_head_base == -1 and q.shape[1] == num_kv_heads * _KERNEL_Q_PER_KV:
        # Preserve the original TP1/TP2 wrapper and raw-kernel ABI.
        kv_groups, q_per_kv = num_kv_heads, _KERNEL_Q_PER_KV
    else:
        if kv_head_base == -1 and num_kv_heads == 1:
            kv_head_base = 0
        if not 0 <= kv_head_base < num_kv_heads:
            raise ValueError("TP-sharded QSA queries require a mapped cached KV head")
        if q.shape[1] not in _SUPPORTED_Q_PER_KV:
            raise ValueError("Unsupported TileLang QSA per-rank query head count")
        kv_groups, q_per_kv = 1, q.shape[1]
    padded_q_per_kv = max(_KERNEL_PADDED_Q_PER_KV, 1 << (q_per_kv - 1).bit_length())
    topk = token_slots.shape[1]
    num_splits = _choose_num_splits(q.shape[0], kv_groups, q.shape[1])
    partial_max, partial_sum, partial_acc, counters = _scratch(
        q.shape[0], kv_groups, num_splits, padded_q_per_kv, q.device
    )
    return sparse_qsa_fwd(
        q.contiguous(),
        k_cache,
        v_cache,
        token_slots.to(torch.int32).contiguous(),
        partial_max,
        partial_sum,
        partial_acc,
        counters,
        num_kv_heads,
        topk,
        num_splits,
        softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5,
        q_per_kv,
        kv_head_base,
    )


__all__ = ["is_available", "is_supported", "sparse_qsa_tilelang"]
