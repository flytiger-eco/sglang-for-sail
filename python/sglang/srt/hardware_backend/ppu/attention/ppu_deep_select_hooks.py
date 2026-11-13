"""Report PPU DeepSelect wheel support through the shared capability gate.

The PPU backend requires SM89 or newer, BF16 index logits and the precompiled
wheel. The BF16 dtype resolver honors SGLANG_OPT_DSV41_BF16_INDEX_LOGITS,
including its platform default when unset.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.kernels.ops.attention.deep_select.is_deep_select_supported",
    type=HookType.REPLACE,
)
def _ppu_is_deep_select_supported() -> bool:
    """Use the wheel's requirements instead of the CUDA JIT architecture list."""
    import torch

    from sglang.kernels.ops.attention.deep_select import _wheel_available
    from sglang.kernels.ops.attention.dsv4.index_logits import index_logits_dtype
    from sglang.srt.utils.common import get_device_sm

    return (
        get_device_sm() >= 89
        and index_logits_dtype() == torch.bfloat16
        and _wheel_available()
    )
