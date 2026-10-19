"""DSv4.1 shifted MegaMHC boundary, adapted from SGLang PR #42808.

Credit: DeepSeek's DeepGEMM project provides the fused MegaMHC kernel:
https://github.com/deepseek-ai/DeepGEMM

PPU SM89+ requires a DeepGEMM build exporting a compatible ``mega_mhc`` API.
This adapter does not make the upstream SM100 kernel executable on PPU.
"""

from __future__ import annotations

import functools
import logging
from typing import NamedTuple, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM
from sglang.srt.utils import get_device_sm, is_cuda, is_ppu

logger = logging.getLogger(__name__)
HcTriplet = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class MegaMHCInput(NamedTuple):
    """Normalized input and mixing stats for exactly the next sublayer's rows."""

    normalized: torch.Tensor
    stats: HcTriplet


@functools.cache
def _mega_mhc_kernel():
    try:
        import deep_gemm
    except ImportError:
        kernel = None
    else:
        kernel = getattr(deep_gemm, "mega_mhc", None)
    if not callable(kernel):
        logger.warning(
            "MegaMHC prefill requested but DeepGEMM has no mega_mhc API; "
            "using the existing mHC path."
        )
        return None
    return kernel


def can_use_mega_mhc_prefill(forward_batch) -> bool:
    if (
        not envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.get()
        or not ENABLE_JIT_DEEPGEMM
        or not forward_batch.forward_mode.is_extend_without_speculative()
    ):
        return False
    sm = get_device_sm()
    if not (sm >= 89 if is_ppu() else is_cuda() and 100 <= sm < 110):
        return False
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    if is_batch_invariant_mode_enabled():
        return False
    return _mega_mhc_kernel() is not None


def try_mhc_mega_boundary(
    x,
    residual,
    pre,
    post,
    comb,
    fn,
    scale,
    base,
    weight,
    rms_eps,
    hc_eps,
    norm_eps,
    sinkhorn_iters,
) -> Optional[Tuple[torch.Tensor, MegaMHCInput]]:
    """Update residual, collapse with carried pre, and compute next stats.

    The collapse uses the preceding sublayer's ``pre``, not the newly computed
    pre. BF16 output keeps the existing attention/MoE quantization contracts.
    Warm on each execution stream before CUDA Graph capture to initialize the
    backend's per-stream barriers. A failure after launch propagates; it must
    not trigger a retry against potentially modified buffers.
    """
    if x.ndim != 2 or residual.ndim != 3:
        return None
    rows, hidden = x.shape
    if not 0 < rows <= 1 << 20 or hidden <= 0 or hidden % 1024 or sinkhorn_iters < 1:
        return None
    tensors = (
        (x, (rows, hidden), torch.bfloat16),
        (residual, (rows, 4, hidden), torch.bfloat16),
        (pre, (rows, 4), torch.float32),
        (post, (rows, 4), torch.float32),
        (comb, (rows, 4, 4), torch.float32),
        (fn, (24, 4 * hidden), torch.float32),
        (scale, (3,), torch.float32),
        (base, (24,), torch.float32),
        (weight, (hidden,), torch.bfloat16),
    )
    if not x.is_cuda or any(
        tensor.shape != shape
        or tensor.dtype != dtype
        or tensor.device != x.device
        or not tensor.is_contiguous()
        for tensor, shape, dtype in tensors
    ):
        return None
    kernel = _mega_mhc_kernel()
    if kernel is None:
        return None
    updated = torch.empty_like(residual)
    normalized = torch.empty_like(x)
    next_pre = torch.empty((rows, 4, 1), device=x.device, dtype=torch.float32)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty_like(comb)
    kernel(
        x=x,
        residual=residual,
        shifted_prev_mix=pre.unsqueeze(-1),
        post_mix=post.unsqueeze(-1),
        comb_res_mix=comb,
        fn=fn,
        mix_scales=scale,
        mix_bases=base,
        hc_mult=4,
        hc_norm_eps=rms_eps,
        hc_pre_eps=hc_eps,
        hc_post_scale=2.0,
        sinkhorn_eps=hc_eps,
        num_sinkhorn_iters=sinkhorn_iters,
        rmsnorm_weight=weight,
        rmsnorm_eps=norm_eps,
        rmsnorm_scale=1.0,
        new_residual=updated,
        new_prev_mix=next_pre,
        new_post_mix=next_post,
        new_comb_res_mix=next_comb,
        y_bf16=normalized,
    )
    return updated, MegaMHCInput(
        normalized, (next_pre.squeeze(-1), next_post.squeeze(-1), next_comb)
    )
