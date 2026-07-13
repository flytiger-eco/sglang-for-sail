"""PPU FlashMLA ops hooks.

Uses ``@plugin_hook`` to inject PPU-specific behavior into
``sglang.srt.layers.attention.flashmla_backend`` without modifying the
community source code.

Registered hooks
~~~~~~~~~~~~~~~~

1. ``get_mla_metadata`` (REPLACE) — Replaces the community
   ``sgl_kernel.flash_mla.get_mla_metadata`` with the PPU implementation.

2. ``flash_mla_with_kvcache`` (REPLACE) — Same, for the decode kernel.

3. ``FlashMLABackend.init_cuda_graph_state`` (AROUND) — Caps the
   ``cuda_graph_mla_metadata`` buffer to PPU's hard-limit of 320 sm_parts.

4. ``FlashMLABackend.init_forward_metadata`` (REPLACE) — Replaces the
   method body on PPU.  Uses ``mla_metadata, _ = get_mla_metadata(...)``
   and stores the sched-meta object as ``flashmla_metadata`` (no separate
   ``num_splits``).  Also adds the ``is_draft_extend_v2()`` branch.

5. ``FlashMLABackend.init_forward_metadata_out_graph`` (AROUND) — Routes
   DRAFT_EXTEND_V2 through the decode/target-verify metadata path.

6. ``FlashMLABackend._apply_decode_target_verify_metadata`` (REPLACE) —
   Replaces the method body on PPU with the PPU-specific logic:
   ``scheduler_metadata, _ = get_mla_metadata(...)``, the
   ``if is_ppu(): raise`` guard, and the ``.tile_scheduler_metadata`` /
   ``.num_splits`` attribute pattern.

7. ``FlashMLABackend.forward_decode`` (AROUND) — Adapts the PPU metadata
   format (``.flashmla_metadata.tile_scheduler_metadata`` /
   ``.flashmla_metadata.num_splits``) to the community format
   (``.flashmla_metadata`` / ``.num_splits``) before delegating to the
   original function.

8. ``FlashMLABackend.forward_extend`` (AROUND) — Same adaptation as
   ``forward_decode``.
"""

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_PPU_MAX_SM_PARTS = 320  # hard limit on num_sm_parts returned by PPU flash_mla


# ---------------------------------------------------------------------------
# 1 & 2. Replace the module-level ops with PPU implementations
# ---------------------------------------------------------------------------


@plugin_hook(
    "sgl_kernel.flash_mla.flash_mla_with_kvcache",
    type=HookType.REPLACE,
)
def _ppu_flash_mla_with_kvcache(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
        flash_mla_with_kvcache,
    )

    return flash_mla_with_kvcache(*args, **kwargs)


@plugin_hook(
    "sgl_kernel.flash_mla.flash_mla_sparse_fwd",
    type=HookType.REPLACE,
)
def _ppu_flash_mla_sparse_fwd(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
        flash_mla_sparse_fwd,
    )

    return flash_mla_sparse_fwd(*args, **kwargs)


@plugin_hook(
    "sgl_kernel.flash_mla.get_mla_metadata",
    type=HookType.REPLACE,
)
def _ppu_get_mla_metadata(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import get_mla_metadata

    return get_mla_metadata(*args, **kwargs)


# ---------------------------------------------------------------------------
# 3. init_cuda_graph_state — cap cuda_graph_mla_metadata buffer
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.flashmla_backend.FlashMLABackend.init_cuda_graph_state",
    type=HookType.AROUND,
)
def _ppu_flashmla_cuda_graph_state(original_fn, self, *args, **kwargs):
    original_fn(self, *args, **kwargs)
    buf = self.cuda_graph_mla_metadata
    if buf is not None and buf.shape[0] > _PPU_MAX_SM_PARTS:
        self.cuda_graph_mla_metadata = torch.empty(
            (_PPU_MAX_SM_PARTS, 8),
            dtype=torch.int32,
            device=buf.device,
        )


# ---------------------------------------------------------------------------
# 4. init_forward_metadata — PPU metadata format + DRAFT_EXTEND_V2 branch
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.flashmla_backend.FlashMLABackend.init_forward_metadata",
    type=HookType.REPLACE,
)
def _ppu_flashmla_init_forward_metadata(self, forward_batch):
    """Replace init_forward_metadata on PPU.

    PPU ``get_mla_metadata`` returns ``(FlashMLASchedMeta_like, _)``; the
    community code unpacks ``mla_metadata, num_splits = ...`` which is
    incompatible.  This hook uses ``mla_metadata, _ = ...`` and stores
    the sched-meta object as ``flashmla_metadata`` (no separate num_splits).

    Also adds the ``is_draft_extend_v2()`` branch (same as target_verify).
    """
    import triton

    from sglang.srt.hardware_backend.ppu.attention.flash_mla import get_mla_metadata
    from sglang.srt.layers.attention.flashmla_backend import (
        PAGE_SIZE,
        FlashMLADecodeMetadata,
    )
    from sglang.srt.layers.attention.utils import (
        create_flashmla_kv_indices_triton,
        get_num_kv_index_blocks_flashmla,
    )

    bs = forward_batch.batch_size
    if forward_batch.forward_mode.is_decode_or_idle():
        max_seqlen_pad = triton.cdiv(forward_batch.seq_lens_cpu.max().item(), PAGE_SIZE)
        block_kv_indices = torch.full(
            (bs, max_seqlen_pad),
            -1,
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )
        create_flashmla_kv_indices_triton[
            (bs, get_num_kv_index_blocks_flashmla(max_seqlen_pad, PAGE_SIZE))
        ](
            self.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_seqlen_pad,
        )
        mla_metadata, _ = get_mla_metadata(
            forward_batch.seq_lens.to(torch.int32),
            self.num_q_heads,
            1,
            is_fp8_kvcache=self.is_fp8_kvcache,
        )
        self.forward_metadata = FlashMLADecodeMetadata(
            mla_metadata,
            None,
            block_kv_indices,
        )
    elif forward_batch.forward_mode.is_target_verify():
        seq_lens_cpu = forward_batch.seq_lens_cpu + self.num_draft_tokens
        seq_lens = forward_batch.seq_lens + self.num_draft_tokens

        max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
        block_kv_indices = torch.full(
            (bs, max_seqlen_pad),
            -1,
            dtype=torch.int32,
            device=seq_lens.device,
        )
        create_flashmla_kv_indices_triton[
            (bs, get_num_kv_index_blocks_flashmla(max_seqlen_pad, PAGE_SIZE))
        ](
            self.req_to_token,
            forward_batch.req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_seqlen_pad,
        )
        mla_metadata, _ = get_mla_metadata(
            seq_lens.to(torch.int32),
            self.num_draft_tokens * self.num_q_heads,
            1,
            is_fp8_kvcache=self.is_fp8_kvcache,
        )
        self.forward_metadata = FlashMLADecodeMetadata(
            mla_metadata,
            None,
            block_kv_indices,
        )
    elif forward_batch.forward_mode.is_draft_extend_v2():
        # [Fix] DRAFT_EXTEND_V2 reuses target_verify logic
        seq_lens_cpu = forward_batch.seq_lens_cpu + self.num_draft_tokens
        seq_lens = forward_batch.seq_lens + self.num_draft_tokens

        max_seqlen_pad = triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
        block_kv_indices = torch.full(
            (bs, max_seqlen_pad),
            -1,
            dtype=torch.int32,
            device=seq_lens.device,
        )
        create_flashmla_kv_indices_triton[
            (bs, get_num_kv_index_blocks_flashmla(max_seqlen_pad, PAGE_SIZE))
        ](
            self.req_to_token,
            forward_batch.req_pool_indices,
            seq_lens,
            None,
            block_kv_indices,
            self.req_to_token.stride(0),
            max_seqlen_pad,
        )
        mla_metadata, _ = get_mla_metadata(
            seq_lens.to(torch.int32),
            self.num_draft_tokens * self.num_q_heads,
            1,
            is_fp8_kvcache=self.is_fp8_kvcache,
        )
        self.forward_metadata = FlashMLADecodeMetadata(
            mla_metadata,
            None,
            block_kv_indices,
        )
    else:
        super(type(self), self).init_forward_metadata(forward_batch)


# ---------------------------------------------------------------------------
# 5. init_forward_metadata_out_graph — route DRAFT_EXTEND_V2
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.flashmla_backend.FlashMLABackend.init_forward_metadata_out_graph",
    type=HookType.AROUND,
)
def _ppu_flashmla_init_forward_metadata_out_graph(
    original_fn, self, forward_batch, in_capture=False
):
    """Route DRAFT_EXTEND_V2 through the decode/target-verify metadata path.

    ``init_forward_metadata_out_graph`` normally routes decode and
    target_verify modes to ``_apply_decode_target_verify_metadata``.
    On PPU, DRAFT_EXTEND_V2 needs the same routing so that FlashMLA
    metadata is built instead of falling through to the parent class
    (which uses FA3 -- Hopper-only on PPU).
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    forward_mode = forward_batch.forward_mode
    if forward_mode.is_decode_or_idle() or forward_mode.is_target_verify():
        original_fn(self, forward_batch, in_capture=in_capture)
    elif forward_mode.is_draft_extend_v2():
        # Pass TARGET_VERIFY so that q_head_mult is computed correctly
        # inside _apply_decode_target_verify_metadata.
        self._apply_decode_target_verify_metadata(
            bs=forward_batch.batch_size,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            forward_mode=ForwardMode.TARGET_VERIFY,
        )
    else:
        original_fn(self, forward_batch, in_capture=in_capture)


# ---------------------------------------------------------------------------
# 6. _apply_decode_target_verify_metadata — PPU metadata handling
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.flashmla_backend.FlashMLABackend._apply_decode_target_verify_metadata",
    type=HookType.REPLACE,
)
def _ppu_flashmla_apply_decode_target_verify_metadata(
    self, bs, req_pool_indices, seq_lens, seq_lens_cpu, forward_mode
):
    """Replace _apply_decode_target_verify_metadata on PPU.

    Key differences from the community version:
    * Uses ``scheduler_metadata, _ = get_mla_metadata(...)`` (discards
      second return value) instead of ``mla_metadata, num_splits = ...``.
    * Contains ``if is_ppu(): raise`` guard inside the fp8 kvcache path.
    * Sets ``.tile_scheduler_metadata`` and ``.num_splits`` attributes on
      the scheduler_metadata object for CUDA graph buffer views.
    * Creates ``FlashMLADecodeMetadata`` with sched-meta as flashmla_metadata
      and None as num_splits (no separate num_splits).
    """
    import logging

    import triton

    from sglang.srt.hardware_backend.ppu.attention.flash_mla import get_mla_metadata
    from sglang.srt.layers.attention.flashmla_backend import (
        PAGE_SIZE,
        FlashMLADecodeMetadata,
    )
    from sglang.srt.layers.attention.utils import (
        create_flashmla_kv_indices_triton,
        get_num_kv_index_blocks_flashmla,
    )
    from sglang.srt.utils import is_ppu

    _logger = logging.getLogger(__name__)

    if True:
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs] if seq_lens_cpu is not None else None

        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.num_draft_tokens
            if seq_lens_cpu is not None:
                seq_lens_cpu = seq_lens_cpu + self.num_draft_tokens

        seq_max = (
            seq_lens_cpu.max().item()
            if seq_lens_cpu is not None
            else seq_lens.max().item()
        )
        max_seqlen_pad = triton.cdiv(seq_max, PAGE_SIZE)

        create_flashmla_kv_indices_triton[
            (
                bs,
                get_num_kv_index_blocks_flashmla(
                    self.cuda_graph_kv_indices.stride(0), PAGE_SIZE
                ),
            )
        ](
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens,
            None,
            self.cuda_graph_kv_indices,
            self.req_to_token.stride(0),
            self.cuda_graph_kv_indices.stride(0),
        )

        q_head_mult = self.num_draft_tokens if forward_mode.is_target_verify() else 1
        scheduler_metadata, _ = get_mla_metadata(
            seq_lens.to(torch.int32),
            q_head_mult * self.num_q_heads,
            1,
            is_fp8_kvcache=self.is_fp8_kvcache,
        )
        if self.is_fp8_kvcache:
            if is_ppu():
                raise
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                q_head_mult * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            assert actual_num_sm_parts <= self.cuda_graph_mla_metadata.shape[0], (
                f"num_sm_parts {actual_num_sm_parts} exceeds preallocated max "
                f"{self.cuda_graph_mla_metadata.shape[0]}"
            )

            if (
                self.cuda_graph_mla_metadata_view is None
                or actual_num_sm_parts != self.cuda_graph_mla_metadata_view.shape[0]
            ):
                if self.cuda_graph_mla_metadata_view is not None:
                    _logger.warning(
                        f"num_sm_parts mismatch in CUDA Graph replay: "
                        f"capture={self.cuda_graph_mla_metadata_view.shape[0]}, "
                        f"replay={actual_num_sm_parts}. "
                        f"This may indicate batch size changed between capture and replay."
                    )
                self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                    :actual_num_sm_parts
                ]
            # num_splits has shape (bs+1,) -- always update for the current bs.
            self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            scheduler_metadata.tile_scheduler_metadata = (
                self.cuda_graph_mla_metadata_view
            )
            scheduler_metadata.num_splits = self.cuda_graph_num_splits_view

        self.forward_metadata = FlashMLADecodeMetadata(
            scheduler_metadata,
            None,
            self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
        )


# ---------------------------------------------------------------------------
# 7. forward_decode — adapt PPU metadata format for community code
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.flashmla_backend.FlashMLABackend.forward_decode",
    type=HookType.AROUND,
)
def _ppu_flashmla_forward_decode(
    original_fn, self, q, k, v, layer, forward_batch, save_kv_cache=True
):
    """Adapt PPU metadata format before calling the community forward_decode.

    PPU stores a FlashMLASchedMeta-like object as ``flashmla_metadata``
    (with ``.tile_scheduler_metadata`` / ``.num_splits`` attributes).
    The community code expects ``flashmla_metadata`` and ``num_splits`` to
    be plain tensors.  This hook temporarily adapts the metadata so the
    original function works unchanged.
    """
    from sglang.srt.layers.attention.flashmla_backend import FlashMLADecodeMetadata

    meta = self.forward_metadata
    # Only adapt when metadata is in PPU format (FlashMLASchedMeta-like
    # flashmla_metadata with .tile_scheduler_metadata / .num_splits).
    sched = meta.flashmla_metadata
    if sched is not None and hasattr(sched, "tile_scheduler_metadata"):
        adapted = FlashMLADecodeMetadata(
            flashmla_metadata=sched.tile_scheduler_metadata,
            num_splits=sched.num_splits,
            block_kv_indices=meta.block_kv_indices,
        )
        self.forward_metadata = adapted
        try:
            return original_fn(self, q, k, v, layer, forward_batch, save_kv_cache)
        finally:
            self.forward_metadata = meta
    else:
        return original_fn(self, q, k, v, layer, forward_batch, save_kv_cache)


# ---------------------------------------------------------------------------
# 8. forward_extend — adapt PPU metadata format for community code
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.flashmla_backend.FlashMLABackend.forward_extend",
    type=HookType.AROUND,
)
def _ppu_flashmla_forward_extend(
    original_fn, self, q, k, v, layer, forward_batch, save_kv_cache=True
):
    """Adapt PPU metadata format before calling the community forward_extend.

    Same adaptation as ``_ppu_flashmla_forward_decode``, but with a guard
    because ``forward_extend`` is also called for EXTEND / DRAFT_EXTEND /
    DRAFT_EXTEND_V2 modes whose metadata is NOT in PPU format.  In those
    cases the original function is called without adaptation.
    """
    from sglang.srt.layers.attention.flashmla_backend import FlashMLADecodeMetadata

    meta = self.forward_metadata
    # Only adapt when metadata is in PPU format (FlashMLASchedMeta-like
    # flashmla_metadata with .tile_scheduler_metadata / .num_splits).
    # For EXTEND / DRAFT_EXTEND modes the metadata is NOT in PPU format,
    # so we skip the adaptation and call the original function directly.
    sched = getattr(meta, "flashmla_metadata", None)
    if sched is not None and hasattr(sched, "tile_scheduler_metadata"):
        adapted = FlashMLADecodeMetadata(
            flashmla_metadata=sched.tile_scheduler_metadata,
            num_splits=sched.num_splits,
            block_kv_indices=meta.block_kv_indices,
        )
        self.forward_metadata = adapted
        try:
            return original_fn(self, q, k, v, layer, forward_batch, save_kv_cache)
        finally:
            self.forward_metadata = meta
    else:
        return original_fn(self, q, k, v, layer, forward_batch, save_kv_cache)
