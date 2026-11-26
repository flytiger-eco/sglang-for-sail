"""PPU MegaMoE hooks.

Uses ``@plugin_hook`` to inject PPU-specific behavior into the MegaMoE path
without modifying the community source code.

Registered hooks
~~~~~~~~~~~~~~~~

1. ``_configure_mega_moe_deep_gemm_num_sms`` (AROUND) — The shared path rounds
   the DeepGEMM SM budget down to an even CTA count because the SM100
   whole-grid clustered launch requires it. The PPU SM89 mega launch does not
   use that launch, so keep the odd budget there and spend the full SM count
   instead of dropping SMs.

   PPU also defaults ``SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS`` to 0: the
   residency margin exists for the SM100 clustered launch, so reserving SMs
   here would only shrink the mega kernel. An explicit env value still wins.
"""

import os
from contextlib import contextmanager

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

os.environ.setdefault("SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS", "0")


@plugin_hook(
    "sglang.srt.layers.moe.mega_moe._configure_mega_moe_deep_gemm_num_sms",
    type=HookType.AROUND,
)
@contextmanager
def _ppu_configure_mega_moe_deep_gemm_num_sms(original_fn, deep_gemm):
    from sglang.srt.layers.moe import mega_moe

    max_num_sms = mega_moe._mega_moe_max_num_sms()
    if max_num_sms is None or mega_moe._device_sm != 89:
        with original_fn(deep_gemm):
            yield
        return

    current_num_sms = deep_gemm.get_num_sms()
    # Same nesting rule as the shared path: never claim SMs back from an outer
    # context, but keep the odd count this time.
    target_num_sms = min(max_num_sms, current_num_sms)
    if target_num_sms == current_num_sms:
        yield
        return

    deep_gemm.set_num_sms(target_num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(current_num_sms)
