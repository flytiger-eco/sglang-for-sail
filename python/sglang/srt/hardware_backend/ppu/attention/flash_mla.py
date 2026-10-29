"""PPU-specific FlashMLA ops.

Wraps the ``flash_mla`` library (T-HEAD PPU) with the same signatures
expected by :mod:`sglang.srt.layers.attention.flashmla_backend`.

Exported symbols
----------------
get_mla_metadata
flash_mla_with_kvcache
flash_mla_sparse_fwd
"""

from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs

try:
    import flash_mla as _flashmla
    from flash_mla import FlashMLASchedMeta

    _import_error = None
except Exception as _e:
    _flashmla = None
    _import_error = _e

_IMPORT_ERROR = ImportError(
    "Failed to import flash_mla for PPU. "
    "Ensure the flash_mla package is installed in your PPU environment."
)


# Add for nvtx profiling
SGLANG_PROFILE_NVTX = envs.SGLANG_PROFILE_NVTX.get()
SGLANG_PROFILE_NVTX_PRINT_SEQLEN = envs.SGLANG_PROFILE_NVTX_PRINT_SEQLEN.get()
if SGLANG_PROFILE_NVTX:
    try:
        from torch.cuda.nvtx import range_pop as th_nvtx_range_pop
        from torch.cuda.nvtx import range_push as th_nvtx_range_push
    except ImportError as e:
        SGLANG_PROFILE_NVTX = False
        SGLANG_PROFILE_NVTX_PRINT_SEQLEN = False


def validate_dense_fp8_config(
    dtype, head_dim, head_dim_v, page_size, device, dcp_size=1
):
    """Check the actual PPU library's dense Q8KV8 contract, including auto dtype."""
    if dtype != torch.float8_e4m3fn:
        raise ValueError("PPU FlashMLA dense FP8 requires E4M3 Q and KV cache.")
    major, minor = torch.cuda.get_device_capability(device)
    # The library checks major == 8 even though its FP8 helper is named
    # is_sm89_or_newer. Do not assume V900/SM90 support.
    if major != 8 or minor < 9:
        raise ValueError("PPU FlashMLA dense FP8 requires an SM89 device.")
    if head_dim != 576 or head_dim_v != 512:
        raise ValueError("PPU FlashMLA dense FP8 requires Q/K dim 576 and V dim 512.")
    if page_size % 64:
        raise ValueError("PPU FlashMLA dense FP8 requires page size divisible by 64.")
    if dcp_size != 1:
        raise ValueError("PPU FlashMLA FP8 KV cache does not support DCP.")


def get_mla_metadata(
    cache_seqlens: Optional[torch.Tensor] = None,
    num_q_tokens_per_head_k: Optional[int] = None,
    num_heads_k: Optional[int] = None,
    num_heads_q: Optional[int] = None,
    is_fp8_kvcache: bool = False,
    topk: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return MLA tile scheduler metadata for PPU.

    Delegates to ``flash_mla.get_mla_metadata``.

    Arguments mirror :func:`sgl_kernel.flash_mla.get_mla_metadata`.
    """
    if _import_error is not None:
        raise _IMPORT_ERROR from _import_error

    if cache_seqlens is None:
        return FlashMLASchedMeta(), None

    assert num_q_tokens_per_head_k is not None
    assert num_heads_k is not None

    return _flashmla.get_mla_metadata(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
        num_heads_q=num_heads_q,
        is_fp8_kvcache=is_fp8_kvcache,
        topk=topk,
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor | FlashMLASchedMeta,
    num_splits: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run FlashMLA decode attention on PPU.

    Dense FP8 uses unscaled E4M3 Q and KV, with BF16 output. Sparse FP8
    uses BF16 Q and the library's packed KV layout. Metadata is generated
    lazily by FlashMLA 2.1+.
    """
    if _import_error is not None:
        raise _IMPORT_ERROR from _import_error

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if indices is None and q.element_size() == 1:
        if k_cache.dtype != q.dtype or k_cache.shape[-1] != q.shape[-1]:
            raise ValueError(
                "PPU FlashMLA dense FP8 requires matching Q/KV dtype and dim."
            )
        validate_dense_fp8_config(
            q.dtype, q.shape[-1], head_dim_v, k_cache.shape[1], q.device
        )
        if descale_q is not None or descale_k is not None:
            raise ValueError(
                "PPU FlashMLA dense FP8 requires unscaled Q/KV; "
                "descale_q/descale_k are not supported."
            )

    nvtx_pushed = SGLANG_PROFILE_NVTX and num_splits is not None
    if nvtx_pushed:
        batch_size = len(num_splits) - 1
        if q.dim() == 4:
            max_seqlen_q = q.shape[-3]
        else:
            max_seqlen_q = 1

        if torch.cuda.is_current_stream_capturing():
            nvtx_message = f"[FW_FMHA] --format=MLA,Forward,type:D,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{head_dim_v},num_heads_kv:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:{causal}"
        else:
            if SGLANG_PROFILE_NVTX_PRINT_SEQLEN:
                cu_seqlens_k_list = (
                    cache_seqlens.flatten().cpu().tolist()
                    if cache_seqlens is not None
                    else "[]"
                )
                nvtx_message = f"[FW_FMHA] --format=MLA,Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{head_dim_v},num_heads_kv:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:{causal},num_blocks:{k_cache.shape[-4]},page_block_size:{k_cache.shape[-3]},cu_seqlens_k:{cu_seqlens_k_list}"
            else:
                nvtx_message = f"[FW_FMHA] --format=MLA,Forward,type:P,seqlen_q:{max_seqlen_q},head_dim:{q.shape[-1]},head_dim_v:{head_dim_v},num_heads_kv:{k_cache.shape[-2]},num_heads:{q.shape[-2]},batch_size:{batch_size},data_type:{q.dtype},causal:{causal}"
        th_nvtx_range_push(nvtx_message)

    try:
        out, softmax_lse = _flashmla.flash_mla_with_kvcache(
            q,
            k_cache,
            block_table,
            cache_seqlens,
            head_dim_v,
            tile_scheduler_metadata,
            num_splits,
            softmax_scale=softmax_scale,
            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            indices=indices,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=topk_length,
            extra_topk_length=extra_topk_length,
        )
    finally:
        if nvtx_pushed:
            th_nvtx_range_pop()
    return out, softmax_lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sparse prefill attention on PPU.

    Delegates to ``flash_mla.flash_mla_sparse_fwd``.
    """
    if _import_error is not None:
        raise _IMPORT_ERROR from _import_error

    return _flashmla.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
