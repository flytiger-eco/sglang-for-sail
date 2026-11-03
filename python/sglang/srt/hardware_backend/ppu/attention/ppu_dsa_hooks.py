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

6. ``validate_deepseek_v4_cp`` (AROUND) — Restores
   ``SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`` after the community CP validation
   disables it for non-V4.1 models. PPU builds the CP-aware per-rank query
   layout in ``_build_sparse_prefill_chunk_cache`` through
   ``interleave_rows_per_request``, so the sparse prefill kernels stay usable.
"""

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


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
    "sglang.srt.arg_groups.deepseek_v4_hook.validate_deepseek_v4_cp",
    type=HookType.AROUND,
)
def _ppu_keep_sparse_prefill_under_cp(original_fn, server_args):
    """Keep FlashMLA sparse prefill enabled under PPU prefill CP.

    The community validation still runs in full (and still raises on the CP
    topologies it rejects); this hook only revisits its last rule, which
    disables the sparse prefill switch for non-V4.1 models. It restores the
    switch only when the same CP configuration the rule accepted can actually
    run the sparse kernels on PPU.
    """
    from sglang.srt.arg_groups.overrides import resolving_view
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
        use_dsv4_q8kv8_sparse_prefill,
    )

    # original_fn overwrites the env var with whatever the CP rule decided, so
    # snapshot the pre-validation value first: the operator's setting, or the
    # EnvBool default (True) when unset. Without it the two disable reasons —
    # operator opt-out vs. the CP rule — are indistinguishable afterwards.
    want_sparse = envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
    original_fn(server_args)

    if not want_sparse:
        # Operator disabled sparse prefill explicitly; keep it disabled.
        return
    if envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get():
        # The CP rule did not disable it; nothing to revisit.
        return

    try:
        cfg = resolving_view(server_args)
        resumable = (
            cfg.enable_prefill_cp
            and cfg.cp_strategy == "interleave"
            and cfg.dp_size == 1
            and cfg.tp_size > 1
            and cfg.moe_a2a_backend in ("none", "deepep", "megamoe", "mori")
            and not use_dsv4_q8kv8_sparse_prefill(cfg.dsv4_prefill_backend)
        )
    except Exception:
        logger.exception(
            "PPU: cannot evaluate the sparse prefill CP configuration; "
            "keeping SGLANG_OPT_FLASHMLA_SPARSE_PREFILL disabled."
        )
        return
    if not resumable:
        return

    envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(want_sparse)
    logger.warning(
        "PPU: keeping SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=%s under "
        "DeepSeekV4 context parallelism (tp=%d, cp_strategy=%s, "
        "prefill_backend=%s).",
        want_sparse,
        cfg.tp_size,
        cfg.cp_strategy,
        cfg.dsv4_prefill_backend,
    )
