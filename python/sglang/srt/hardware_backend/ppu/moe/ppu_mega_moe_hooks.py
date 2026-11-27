"""SM89 MegaMoE budget for Qwen's disaggregated MXFP4 decode."""

from contextlib import contextmanager

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.moe.mega_moe._configure_mega_moe_deep_gemm_num_sms",
    type=HookType.AROUND,
)
@contextmanager
def _ppu_configure_mega_moe_deep_gemm_num_sms(
    original_fn, deep_gemm, *, use_full_ppu_sms=False
):
    from sglang.srt.layers.moe import mega_moe

    if not use_full_ppu_sms or mega_moe._device_sm != 89:
        with original_fn(deep_gemm):
            yield
        return

    reserved_sms = envs.SGLANG_OPT_DEEPGEMM_MEGA_MOE_RESERVED_SMS
    current_num_sms = deep_gemm.get_num_sms()
    if reserved_sms.is_set():
        max_num_sms = mega_moe._mega_moe_max_num_sms()
        target_num_sms = min(current_num_sms, max_num_sms)
    else:
        # The outer DeepGEMM budget already reflects the physical SM limit.
        target_num_sms = current_num_sms

    if target_num_sms == current_num_sms:
        yield
        return

    deep_gemm.set_num_sms(target_num_sms)
    try:
        yield
    finally:
        deep_gemm.set_num_sms(current_num_sms)
