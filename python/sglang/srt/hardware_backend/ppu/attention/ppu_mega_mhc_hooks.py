"""PPU DSv4.1 prefill-CP hooks.

Uses ``@plugin_hook`` to keep community-gated DSv4.1 prefill fusions available
on PPU without modifying the community source.

Registered hooks:

1. ``sglang.srt.models.deepseek_v4_mhc._can_use_mega_mhc`` (AROUND) — the
   community gate returns False whenever ``cfg.cp_prefill`` is set, which
   drops the mHC boundary back onto the separate ``hc_mix_stats`` /
   ``mhc_post`` kernels. Measured on 890P tp8/cp8 prefill (con1, in100k) the
   fallback costs ~5x the GPU time of the fused DeepGEMM ``mega_mhc`` kernel,
   +10.5pp of an iteration. The hook re-runs the gate with the CP clause
   removed and keeps every other condition intact.

``SGLANG_OPT_DSV41_MEGA_MHC_PREFILL`` keeps its platform default (true on SAIL)
and still opts out explicitly.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.models.deepseek_v4_mhc._can_use_mega_mhc",
    type=HookType.AROUND,
)
def _ppu_mega_mhc_with_cp_prefill(original_fn, cfg, forward_batch):
    """Allow the fused MegaMHC boundary under CP prefill."""
    if original_fn(cfg, forward_batch):
        return True
    if not cfg.cp_prefill:
        return False

    from sglang.kernels.ops.layernorm.mhc_mega import can_use_mega_mhc_prefill
    from sglang.srt.runtime_context import get_forward

    return (
        cfg.pre_from_prev
        and cfg.mult == 4
        and not get_forward().sp_active
        and forward_batch is not None
        and can_use_mega_mhc_prefill(forward_batch)
    )
