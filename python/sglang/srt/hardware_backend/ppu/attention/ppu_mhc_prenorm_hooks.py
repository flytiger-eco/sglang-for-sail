"""PPU DSv4.1 mHC coefficient hooks.

Uses ``@plugin_hook`` to run the independent coefficient path (the boundaries
MegaMHC cannot fuse: the first layer and the layers after an Engram) through
the DeepGEMM tf32 prenorm projection, without modifying the community source.

Registered hooks:

1. ``DeepseekV4DecoderLayer.refresh_mhc_norm_weight_cache`` (AROUND) — the
   community gate only builds the tf32/bf16 weight split on SM90/SM100, so PPU
   keeps the plain Triton projection. The hook builds the tf32 split there and
   rebinds the hyper-connections; the bf16x3 compensated Triton projection is
   skipped because it is fixed to 16 K slices and PPU DeepGEMM has no split-k.
2. ``sglang.srt.models.deepseek_v4_mhc._mix_stats_impl`` (AROUND) — for
   prefill-sized batches, compute the triplet with
   ``hc_mix_stats_sinkhorn_deepgemm`` (one K slice) instead of the separate
   ``hc_mix_stats`` + ``hc_split_sinkhorn`` kernels. Small batches keep the
   Triton path, where the DeepGEMM dispatch cost dominates.

The MegaMHC boundary and the shifted pre/post/comb hand-off are untouched.
"""

import torch

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
from sglang.srt.runtime_context import get_platform

_HC_WEIGHT_SHAPE = (24, 4 * 5120)
# PPU DeepGEMM's fixed dispatch cost dominates below 1k rows, the same
# crossover the legacy hc_pre gate uses.
_PPU_DEEPGEMM_MIN_ROWS = 1024


def _build_tf32_parts(layer) -> None:
    from sglang.srt.layers.deep_gemm_wrapper.configurer import ENABLE_JIT_DEEPGEMM

    if not ENABLE_JIT_DEEPGEMM:
        return
    import deep_gemm

    if not callable(getattr(deep_gemm, "tf32_hc_prenorm_gemm", None)):
        return
    from sglang.kernels.ops.layernorm.mhc import split_tf32_hc_weight

    layer._hc_attn_tf32_parts = split_tf32_hc_weight(layer.hc_attn_fn.data)
    layer._hc_ffn_tf32_parts = split_tf32_hc_weight(layer.hc_ffn_fn.data)


@plugin_hook(
    "sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer."
    "refresh_mhc_norm_weight_cache",
    type=HookType.AROUND,
)
def _ppu_build_tf32_parts(original_fn, layer):
    original_fn(layer)
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    if (
        not get_platform().is_ppu
        or not layer.hc_pre_from_prev_sublayer
        or layer.hc_attn_fn.shape != _HC_WEIGHT_SHAPE
        or not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        or getattr(layer.config, "model_type", None) != "deepseek_v41"
        or is_batch_invariant_mode_enabled()
    ):
        return
    _build_tf32_parts(layer)
    if layer._hc_attn_tf32_parts is not None:
        layer._init_hyper_connections()


@plugin_hook(
    "sglang.srt.models.deepseek_v4_mhc._mix_stats_impl",
    type=HookType.AROUND,
)
def _ppu_mix_stats_deepgemm(original_fn, hc, x):
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    if (
        get_platform().is_ppu
        and hc.tf32_parts is not None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and x.shape[0] >= _PPU_DEEPGEMM_MIN_ROWS
        and envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
        and not is_batch_invariant_mode_enabled()
    ):
        from sglang.kernels.ops.layernorm.mhc import (
            hc_mix_stats_sinkhorn_deepgemm,
        )

        cfg = hc.cfg
        return hc_mix_stats_sinkhorn_deepgemm(
            x.flatten(1),
            hc.tf32_parts,
            hc.scale,
            hc.base,
            cfg.sinkhorn_iters,
            cfg.rms_eps,
            cfg.eps,
        )
    return original_fn(hc, x)
