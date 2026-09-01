"""PPU FA3 ops hooks.

Uses ``@plugin_hook`` with ``HookType.REPLACE`` to replace the module-level
FA3 functions (``flash_attn_varlen_func``, ``flash_attn_with_kvcache``) in
the ``flashattention_backend`` module, and ``HookType.AROUND`` on
``FlashAttentionBackend.__init__`` to replace the instance-level
``_get_scheduler_metadata`` attribute.

This avoids any modification to ``flashattention_backend.py`` itself — the
PPU ops are injected purely through the plugin hook mechanism.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.attention.flashattention_backend.flash_attn_varlen_func",
    type=HookType.REPLACE,
)
def _ppu_flash_attn_varlen_func(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_attention import (
        flash_attn_varlen_func,
    )

    return flash_attn_varlen_func(*args, **kwargs)


@plugin_hook(
    "sglang.srt.layers.attention.flashattention_backend.flash_attn_with_kvcache",
    type=HookType.REPLACE,
)
def _ppu_flash_attn_with_kvcache(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_attention import (
        flash_attn_with_kvcache,
    )

    # [PPU-fix] Pop only_qv: PPU flash_attn_interface does not support this parameter
    # (added by community for GLM-5.3-Flash KDA layers where q_rope=None)
    kwargs.pop("only_qv", None)

    # [PPU-fix] Handle k_cache=None: when only_qv=True (KDA layers), there is
    # no k_rope data and k_cache may be None. PPU FA3 requires a valid tensor.
    if "k_cache" in kwargs and kwargs["k_cache"] is None:
        import torch

        v_cache = kwargs.get("v_cache")
        if v_cache is not None:
            kwargs["k_cache"] = torch.zeros_like(v_cache)

    # [PPU-fix] INT8 KV cache dtype → BF16: PPU FA3 only supports BF16 KV.
    # When the model uses INT8 quantization, k_cache/v_cache may arrive as int8.
    import torch

    for key in ("k_cache", "v_cache"):
        tensor = kwargs.get(key)
        if tensor is not None and tensor.dtype == torch.int8:
            kwargs[key] = tensor.to(torch.bfloat16)

    return flash_attn_with_kvcache(*args, **kwargs)


@plugin_hook(
    "sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.__init__",
    type=HookType.AROUND,
)
def _ppu_fa3_init(original_fn, self, *args, **kwargs):
    original_fn(self, *args, **kwargs)
    if self.fa_impl_ver == 3:
        from sglang.srt.hardware_backend.ppu.attention.flash_attention import (
            get_scheduler_metadata,
        )

        self._get_scheduler_metadata = get_scheduler_metadata
