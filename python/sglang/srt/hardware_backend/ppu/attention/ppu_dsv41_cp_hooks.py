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

``SGLANG_OPT_DSV41_MEGA_MHC_PREFILL`` still opts out explicitly; an unset
value is treated as enabled so the fused boundary is the default.
"""

import logging
import os

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

_MEGA_MHC_ENV = "SGLANG_OPT_DSV41_MEGA_MHC_PREFILL"


def _mega_mhc_enabled_by_default() -> bool:
    """Opt-out env: only an explicit value disables the fused boundary."""
    if os.environ.get(_MEGA_MHC_ENV) is None:
        return True
    from sglang.srt.environ import envs

    return envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.get()


@plugin_hook(
    "sglang.srt.models.deepseek_v4_mhc._can_use_mega_mhc",
    type=HookType.AROUND,
)
def _ppu_mega_mhc_with_cp_prefill(original_fn, cfg):
    """Allow the fused MegaMHC boundary under CP prefill."""
    if original_fn(cfg):
        return True
    if not cfg.cp_prefill or not _mega_mhc_enabled_by_default():
        return False

    from sglang.kernels.ops.layernorm.mhc_mega import mega_mhc_supported
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    if is_batch_invariant_mode_enabled():
        return False
    return mega_mhc_supported()
