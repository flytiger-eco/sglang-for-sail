"""PPU DSA backend hooks.

Uses ``@plugin_hook`` to inject PPU-specific behavior into DSA backend
methods without modifying the community source code.

Registered hooks:

1. ``DSAFlashMLAMetadata.copy_`` (REPLACE) — Skips the copy on PPU because
   PPU metadata tensors are cached/immutable and must not be overwritten
   during CUDA graph replay.

2. ``DSAFlashMLAMetadata.slice`` (REPLACE) — Returns ``num_splits`` without
   slicing on PPU because the tensor shape semantics differ from CUDA.

3. ``DeepseekSparseAttnBackend.init_forward_metadata_in_graph`` (REPLACE) —
   Resets ``flashmla_metadata.have_initialized`` before CUDA graph capture
   so that PPU metadata generation occurs within the capture scope.

4. ``DeepseekSparseAttnBackend.init_forward_metadata_replay_cuda_graph_from_precomputed``
   (AROUND) — Skips flashmla_metadata copy on PPU by temporarily clearing
   and restoring the precomputed metadata.

5. ``DeepseekSparseAttnBackend._copy_replay_metadata_from_sibling`` (AROUND) —
   Skips opaque FlashMLA metadata only for PPU flashmla_kv decode sibling replay.

6. ``DeepseekV4AttnBackend._use_sparse_prefill`` (AROUND) — the community gate
   excludes any batch with active CP, so CP prefill attention runs the paged
   (decode-style) FlashMLA kernel instead of the sparse prefill one: 2.58ms vs
   1.77ms per call at seqlen_q=2496, +4.8pp of an iteration. The sparse chunk
   cache already builds its per-rank query layout through
   ``interleave_rows_per_request`` when CP is active.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


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


@plugin_hook(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.init_forward_metadata_in_graph",
    type=HookType.REPLACE,
)
def _ppu_dsa_init_forward_metadata_in_graph(self, forward_batch):
    """Replace init_forward_metadata_in_graph on PPU.

    For PPU FlashMLA, `get_mla_metadata` runs only on the first
    `flash_mla_with_kvcache` call. To capture this init logic in
    the CUDA Graph, reset flashmla_metadata before starting capture.
    This forces metadata generation to occur within the capture scope.
    """
    if self.dsa_decode_impl == "flashmla_kv":
        assert self.forward_metadata.flashmla_metadata is not None
        assert self.forward_metadata.flashmla_metadata.flashmla_metadata is not None
        self.forward_metadata.flashmla_metadata.flashmla_metadata.have_initialized = (
            False
        )
        # Reset the HS64 CUDA Graph tile-scheduler metadata so replay rebuilds it.
        metadata = self.forward_metadata.flashmla_metadata.flashmla_metadata
        metadata.tile_scheduler_metadata = None
        metadata.num_splits = None


@plugin_hook(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.init_forward_metadata_replay_cuda_graph_from_precomputed",
    type=HookType.AROUND,
)
def _ppu_dsa_init_forward_metadata_replay_cuda_graph_from_precomputed(
    original_fn, self, bs, precomputed, forward_mode
):
    """Skip flashmla_metadata copy on PPU."""
    flashmla_metadata = precomputed.flashmla_metadata
    precomputed.flashmla_metadata = None
    original_fn(self, bs, precomputed, forward_mode)
    precomputed.flashmla_metadata = flashmla_metadata


@plugin_hook(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend._copy_replay_metadata_from_sibling",
    type=HookType.AROUND,
)
def _ppu_dsa_copy_replay_metadata_from_sibling(
    original_fn, self, src_backend, bs, precomputed, forward_mode
):
    if (
        self.dsa_decode_impl != "flashmla_kv"
        or not forward_mode.is_decode_or_idle()
        or precomputed.flashmla_metadata is None
    ):
        return original_fn(self, src_backend, bs, precomputed, forward_mode)

    flashmla_metadata = precomputed.flashmla_metadata
    precomputed.flashmla_metadata = None
    original_fn(self, src_backend, bs, precomputed, forward_mode)
    precomputed.flashmla_metadata = flashmla_metadata


@plugin_hook(
    "sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend"
    "._use_sparse_prefill",
    type=HookType.AROUND,
)
def _ppu_sparse_prefill_with_cp(original_fn, self, forward_batch, *, num_qo_tokens):
    """Allow FlashMLA sparse prefill under CP prefill."""
    if original_fn(self, forward_batch, num_qo_tokens=num_qo_tokens):
        return True

    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        _LARGE_INDEXER_QUERY_THRESHOLD,
    )
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.runtime_context import get_platform

    if not is_cp_active(forward_batch):
        return False
    if not forward_batch.forward_mode.is_extend_without_speculative():
        return False
    if get_platform().is_sm120:
        return False
    return (
        num_qo_tokens > _LARGE_INDEXER_QUERY_THRESHOLD
        or envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
    )
