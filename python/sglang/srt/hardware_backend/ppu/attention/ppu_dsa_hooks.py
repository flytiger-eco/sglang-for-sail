"""PPU DSA backend hooks.

Uses ``@plugin_hook`` with ``HookType.REPLACE`` on DSA backend methods to
inject PPU-specific behavior without modifying the community source code.

None of these hooks call the original function, so REPLACE is used instead
of AROUND to avoid the unnecessary ``original_fn`` parameter.

Registered hooks:

1. ``_compute_flashmla_metadata`` (REPLACE) — Replaces the method body on
   PPU so that the PPU ``get_mla_metadata`` is used instead of the community
   ``sgl_kernel.flash_mla.get_mla_metadata``.  The DSA backend imports
   ``get_mla_metadata`` locally inside this method, so the flashmla_backend
   REPLACE hook does not cover it.  On PPU, ``get_mla_metadata`` returns
   ``(FlashMLASchedMeta_like, _)`` where the first element carries
   ``.tile_scheduler_metadata`` / ``.num_splits`` attributes.  The hook
   extracts these to build ``DSAFlashMLAMetadata``.

2. ``DSAFlashMLAMetadata.copy_`` (REPLACE) — Skips the copy on PPU because
   PPU metadata tensors are cached/immutable and must not be overwritten
   during CUDA graph replay.

3. ``DSAFlashMLAMetadata.slice`` (REPLACE) — Returns ``num_splits`` without
   slicing on PPU because the tensor shape semantics differ from CUDA.

4. ``get_mla_metadata`` (REPLACE) — Replaces the community
   ``sgl_kernel.flash_mla.get_mla_metadata`` with the PPU implementation.

5. ``flash_mla_with_kvcache`` (REPLACE) — Same, for the decode kernel.

6  ``flash_mla_with_kvcache`` (REPLACE) — Same, for the decode kernel.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._compute_flashmla_metadata",
    type=HookType.REPLACE,
)
def _ppu_dsa_compute_flashmla_metadata(self, cache_seqlens, seq_len_q):
    """Use PPU ``get_mla_metadata`` for DSA backend metadata computation.

    On PPU, ``get_mla_metadata`` returns ``(FlashMLASchedMeta_like, _)``
    where the first element has ``.tile_scheduler_metadata`` and
    ``.num_splits`` attributes.  This hook discards the second return value
    and extracts the two tensor fields from the sched-meta object.
    """
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
        get_mla_metadata,
    )
    from sglang.srt.layers.attention.dsa_backend import DSAFlashMLAMetadata

    num_heads_q = self.flashmla_kv_num_q_heads
    flashmla_metadata, _ = get_mla_metadata(
        cache_seqlens=cache_seqlens,
        num_q_tokens_per_head_k=seq_len_q * num_heads_q // 1,
        num_heads_k=1,
        num_heads_q=num_heads_q,
        is_fp8_kvcache=True,
        topk=self.dsa_index_topk,
    )
    return DSAFlashMLAMetadata(
        flashmla_metadata=flashmla_metadata.tile_scheduler_metadata,
        num_splits=flashmla_metadata.num_splits,
    )


@plugin_hook(
    "sglang.srt.layers.attention.dsa_backend.DSAFlashMLAMetadata.copy_",
    type=HookType.REPLACE,
)
def _ppu_dsa_flashmla_copy(self, other):
    """Skip copy on PPU — metadata is cached/immutable."""
    return


@plugin_hook(
    "sglang.srt.layers.attention.dsa_backend.DSAFlashMLAMetadata.slice",
    type=HookType.REPLACE,
)
def _ppu_dsa_flashmla_slice(self, sli):
    """Skip num_splits slicing on PPU — shape semantics differ from CUDA."""
    from sglang.srt.layers.attention.dsa_backend import DSAFlashMLAMetadata

    return DSAFlashMLAMetadata(
        flashmla_metadata=self.flashmla_metadata,
        num_splits=self.num_splits,
    )
