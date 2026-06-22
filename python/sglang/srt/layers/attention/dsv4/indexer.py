from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

from sglang.jit_kernel.deepseek_v4 import (
    fused_q_indexer_rope_hadamard_quant,
    fused_q_indexer_rope_hadamard_quant_int8,
    fused_q_indexer_rope_hadamard_quant_mxfp4,
    fused_rope,
    top_k_per_row_prefill,
    top_k_per_row_prefill_bf16,
    topk_transform_512,
    topk_transform_512_v2,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_prefill_cp_round_robin_split,
    nsa_cp_round_robin_split_q_seqs_cpu,
)
from sglang.srt.layers.dp_attention import get_attention_cp_size
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.utils import add_prefix, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.attention.dsv4.compressor import (
        CompressorBackendMixin,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang.srt.layers.attention.nsa.triton_kernel import (
    MXFP_BLOCK_SIZE,
    _supports_fp8,
    is_fp4_indexer_cache_enabled,
)

if is_hip():
    FP8_DTYPE = torch.float8_e4m3fnuz
    FP8_MAX = torch.finfo(FP8_DTYPE).max
else:
    FP8_DTYPE = torch.float8_e4m3fn
    FP8_MAX = torch.finfo(FP8_DTYPE).max

# Determine which quant dtype to use based on hardware capability
_USE_INT8 = not _supports_fp8() or envs.SGLANG_SAIL_DSV4_USE_INT8.get()


def _log_sparse_indexer_budget_once() -> None:
    # The prefill chunking-budget banner only prints once (the first C4Indexer
    # the model loads triggers it -- effectively at server startup).
    mb = envs.SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB.get()
    logger.info_once(
        "[C4Indexer] SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB=%d MiB (prefill "
        "logits-tile budget per chunk). If you hit OOM during long-prefill "
        "indexer forward, try lowering this value (e.g. 256 or 128).",
        mb,
    )


def split_indexer_prefill_chunks(
    seq_lens_cpu: List[int],
    query_lens_cpu: List[int],
    workspace_size: int,
    max_logits_bytes: int,
    logits_dtype: torch.dtype,
) -> List[Tuple[slice, slice]]:
    chunks: List[Tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    bytes_per_elem = torch.empty((), dtype=logits_dtype).element_size()
    max_logits_elems = max(1, max_logits_bytes // bytes_per_elem)
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end], seq_lens_cpu[end]
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end], seq_lens_cpu[end]
            end += 1

        req_slice = slice(start, end)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else chunk_m
        for q_off in range(0, chunk_m, max_q):
            chunks.append((req_slice, slice(q_off, min(q_off + max_q, chunk_m))))

    return chunks


@triton.jit
def _build_prefill_c4_logits_metadata_kernel(
    seq_lens_ptr,
    extend_lens_ptr,
    per_row_c4_seq_lens_ptr,
    page_table_ptr,
    c4_seq_lens_ptr,
    page_indices_ptr,
    row_starts_ptr,
    row_ends_ptr,
    page_table_stride: tl.constexpr,
    page_index_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_REQS: tl.constexpr,
):
    req_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    seq_len = tl.load(seq_lens_ptr + req_id)
    c4_seq_len = seq_len // 4
    tl.store(c4_seq_lens_ptr + req_id, c4_seq_len, mask=block_id == 0)

    req_offsets = tl.arange(0, BLOCK_REQS)
    before_req = req_offsets < req_id
    kv_offset = tl.sum(
        tl.load(seq_lens_ptr + req_offsets, mask=before_req, other=0) // 4
    )
    query_offset = tl.sum(
        tl.load(extend_lens_ptr + req_offsets, mask=before_req, other=0)
    )

    extend_len = tl.load(extend_lens_ptr + req_id)
    mask_q = offs < extend_len
    out_q = query_offset + offs
    per_row_c4_seq_len = tl.load(per_row_c4_seq_lens_ptr + out_q, mask=mask_q, other=0)
    tl.store(row_starts_ptr + out_q, kv_offset, mask=mask_q)
    tl.store(row_ends_ptr + out_q, kv_offset + per_row_c4_seq_len, mask=mask_q)

    num_pages = tl.cdiv(c4_seq_len, 64)
    page_mask = offs < num_pages
    pages = tl.load(
        page_table_ptr + query_offset * page_table_stride + offs,
        mask=page_mask,
    )
    tl.store(
        page_indices_ptr + req_id * page_index_stride + offs, pages, mask=page_mask
    )


def _build_prefill_c4_logits_metadata(
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    per_row_c4_seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_c4_seq_len: int,
    num_q_tokens: int,
    total_kv_len: int,
    max_extend_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    num_reqs = seq_lens.shape[0]
    c4_seq_lens = torch.empty((num_reqs,), dtype=torch.int32, device=seq_lens.device)
    row_starts = torch.empty((num_q_tokens,), dtype=torch.int32, device=seq_lens.device)
    row_ends = torch.empty((num_q_tokens,), dtype=torch.int32, device=seq_lens.device)

    num_pages = triton.cdiv(max_c4_seq_len, 64)
    page_indices = torch.empty(
        (num_reqs, num_pages),
        dtype=torch.int32,
        device=seq_lens.device,
    )

    block_size = 1024
    max_blocks = max(
        triton.cdiv(max_extend_len, block_size),
        triton.cdiv(num_pages, block_size),
    )
    _build_prefill_c4_logits_metadata_kernel[(num_reqs, max_blocks)](
        seq_lens,
        extend_lens,
        per_row_c4_seq_lens,
        page_table,
        c4_seq_lens,
        page_indices,
        row_starts,
        row_ends,
        page_table.stride(0),
        page_indices.stride(0),
        BLOCK_SIZE=block_size,
        BLOCK_REQS=triton.next_power_of_2(num_reqs),
    )

    return (
        c4_seq_lens,
        page_indices,
        row_starts,
        row_ends,
        total_kv_len,
    )


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128, "torch reference impl hardcodes DSV4 indexer head_dim=128"
    assert block_size == 64, "torch reference impl hardcodes block_size=64 cache layout"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
    for i in range(batch_size):
        q = q_fp8[i, 0]
        q = q.to(torch.float32)
        q_scale = weight[i]
        seq_len = int(seq_lens[i].item())
        assert seq_len <= max_seq_len
        num_pages = (seq_len + block_size - 1) // block_size
        padded_seq_len = num_pages * block_size
        pages = page_table[i, :num_pages]
        kvcache_fp8 = kvcache_fp8.view(-1, block_size * (head_dim + 4))
        kvcache = kvcache_fp8[pages]
        SCALE_OFFSET = block_size * head_dim
        kvcache_value = kvcache[..., :SCALE_OFFSET].view(dtype=FP8_DTYPE)
        kvcache_scale = kvcache[..., SCALE_OFFSET:].view(dtype=torch.float32)
        kvcache_value = kvcache_value.to(torch.float32)
        kvcache_scale = kvcache_scale.contiguous()
        kvcache_value = kvcache_value.view(padded_seq_len, head_dim)
        kvcache_scale = kvcache_scale.view(padded_seq_len)
        score = F.linear(kvcache_value, q)
        score = F.relu(score)
        score *= q_scale[None, :]
        score = score.sum(dim=1)
        score *= kvcache_scale
        logits[i, :seq_len] = score[:seq_len]

    return logits


def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:

    TOPK = 512
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    positions = (
        torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    valid_mask = positions < seq_lens.unsqueeze(1)

    masked_scores = scores.clone()
    masked_scores[~valid_mask] = float("-inf")

    actual_k = min(TOPK, max_seq_len)
    _, raw_indices = torch.topk(
        masked_scores, k=actual_k, dim=1, largest=True, sorted=False
    )
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        padding = torch.zeros(
            (batch_size, TOPK - actual_k), dtype=torch.int32, device=device
        )
        raw_indices = torch.cat([raw_indices, padding], dim=1)

    batch_indices = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, TOPK)
    )
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = torch.arange(TOPK, device=device).unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    needs_sequential = seq_lens <= TOPK
    if needs_sequential.any():
        sequential_indices = (
            torch.arange(TOPK, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        sequential_valid = sequential_indices < seq_lens.unsqueeze(1)

        raw_indices = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK),
            torch.where(
                sequential_valid,
                sequential_indices,
                torch.tensor(-1, device=device, dtype=torch.int32),
            ),
            raw_indices,
        )
        valid_topk = torch.where(
            needs_sequential.unsqueeze(1).expand(-1, TOPK), sequential_valid, valid_topk
        )

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)

    page_indices = torch.where(
        valid_topk, page_indices, torch.tensor(-1, device=device, dtype=torch.int32)
    )

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = torch.where(
            valid_topk, raw_indices, torch.tensor(-1, device=device, dtype=torch.int32)
        )
        out_raw_indices.copy_(raw_indices)


@triton.jit
def _fused_scale_kernel(
    weight_ptr,
    q_scale_ptr,
    out_ptr,
    numel,
    out_scale,
    APPLY_Q_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    w = tl.load(weight_ptr + offs, mask=mask)
    acc = w.to(tl.float32) * out_scale
    if APPLY_Q_SCALE:
        qs = tl.load(q_scale_ptr + offs, mask=mask)
        acc = acc * qs.to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_scale(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``weight * out_scale * q_scale`` (or ``weight * out_scale`` when
    ``q_scale`` is None — used by the MXFP4 indexer path where per-block Q
    scales live with the Q values, so there is no per-token scalar to fold)."""
    assert weight.is_contiguous()
    apply_q_scale = q_scale is not None
    B, H = weight.shape
    numel = B * H
    if apply_q_scale:
        assert q_scale.is_contiguous()
        out_dtype = torch.promote_types(weight.dtype, q_scale.dtype)
    else:
        # MXFP4 path: DeepGEMM's fp8_fp4_mqa_logits requires weights to be
        # fp32 (DG_HOST_ASSERT in csrc/apis/attention.hpp). The FP8 path lands
        # on fp32 naturally via promote_types(bf16, fp32); without a q_scale
        # we have to pin it explicitly.
        out_dtype = torch.float32
    out = torch.empty((B, H, 1), device=weight.device, dtype=out_dtype)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_scale_kernel[grid](
        weight,
        q_scale if apply_q_scale else weight,  # placeholder ptr; gated by APPLY_Q_SCALE
        out,
        numel,
        out_scale,
        APPLY_Q_SCALE=apply_q_scale,
        BLOCK=BLOCK,
    )
    return out


class C4IndexerBackendMixin:
    def __init__(self):
        super().__init__()
        self.debug_use_external_c4_sparse_indices: bool = False

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

        assert alt_streams is not None
        assert len(alt_streams) >= 2
        current_stream = torch.cuda.current_stream()
        stream_q = alt_streams[0]
        stream_weights = alt_streams[1]

        stream_q.wait_stream(current_stream)
        stream_weights.wait_stream(current_stream)

        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )

        # The weight projection is small and fast; compute it on its own
        # stream, then have the Q stream wait on it before launching the big
        # fused Q kernel. Both FP8 and FP4 fused kernels take `weight` as
        # input and emit the final weights tensor on stream_q, so we only
        # need to join stream_q at the end (stream_weights is transitively
        # joined via stream_q's wait on `weights_ready`).
        with torch.cuda.stream(stream_weights):
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            weights_ready = stream_weights.record_event()

        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            stream_q.wait_event(weights_ready)
            if c4_indexer.use_fp4_cache:
                # FP4: fused kernel returns ((q_packed, q_sf), weights_out).
                # q_scale is NOT folded into weights (4 per-token sub-block
                # scales); it stays alongside q values as q_sf.
                q_fp8, weights = c4_indexer.compute_q_mxfp4(q_lora, positions, weights)
            else:
                # FP8: fused kernel folds rope + hadamard + fp8 quant +
                # weight*weight_scale*q_scale into one pass.
                q_fp8, weights = c4_indexer.compute_q(q_lora, positions, weights)

        current_stream.wait_stream(stream_q)
        return q_fp8, weights, c4_indexer_kv_cache

    def _forward_prepare_normal(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

        weights = c4_indexer.compute_weights(x, skip_scale=True)
        if c4_indexer.use_fp4_cache:
            q_fp8, weights = c4_indexer.compute_q_mxfp4(q_lora, positions, weights)
        else:
            q_fp8, weights = c4_indexer.compute_q(q_lora, positions, weights)
        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )
        return q_fp8, weights, c4_indexer_kv_cache

    def _use_prefill_logits(self, forward_batch: ForwardBatch) -> bool:
        forward_mode = forward_batch.forward_mode
        return (
            not is_hip()
            and forward_mode.is_extend()
            and not forward_mode.is_mixed()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend(include_v2=True)
            and not envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
            and not envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        )

    def _get_paged_c4_logits(
        self,
        q_quant,
        c4_indexer_kv_cache: torch.Tensor,
        weights: torch.Tensor,
        indexer_metadata: PagedIndexerMetadata,
        c4_indexer: C4Indexer,
    ) -> torch.Tensor:
        use_fp4 = c4_indexer.use_fp4_cache
        if use_fp4:
            # MXFP4 paged path: only DeepGEMM's fp8_fp4_paged_mqa_logits is
            # supported. The torch / tilelang reference paths are FP8-only.
            assert (
                not envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
            ), "tilelang indexer does not support FP4 cache"
            assert (
                not envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
            ), "torch reference paged-mqa-logits is FP8-only"

            from deep_gemm import fp8_fp4_paged_mqa_logits as fn

            q_packed, q_sf = q_quant
            # DeepGEMM expects q values cast to int8 (kPackedFP4) and 4-D
            # (batch, next_n=1, num_heads, head_dim/2). q_sf is already 3-D
            # (batch, next_n=1, num_heads) -- the fused kernel emits it in
            # that layout directly, so no unsqueeze needed.
            q_packed_4d = q_packed.unsqueeze(1).contiguous().view(torch.int8)
            q_sf_3d = q_sf.contiguous()
            block_kv = 64
            num_heads_kv = 1
            # Per-token bytes in cache: 64 packed FP4 + 4 ue8m0 = 68. The
            # page layout is segregated (all values, then all scales) — but
            # DeepGEMM uses internal from_blob views with the right per-segment
            # offsets; the 4-D view we pass here only carries the page-bytes
            # stride and the assertion-required `stride(1) == fp4_with_sf_bytes`.
            fp4_with_sf_bytes = 68
            kv_cache_view = c4_indexer_kv_cache.view(
                c4_indexer_kv_cache.shape[0],
                block_kv,
                num_heads_kv,
                fp4_with_sf_bytes,
            )
            _c4sl = indexer_metadata.c4_seq_lens
            if _c4sl.dim() == 1:
                _c4sl = _c4sl.unsqueeze(-1)
            return fn(
                (q_packed_4d, q_sf_3d),
                kv_cache_view,
                weights,
                _c4sl,
                indexer_metadata.page_table,
                indexer_metadata.deep_gemm_metadata,
                indexer_metadata.max_c4_seq_len,
                False,
            )

        if _USE_INT8:
            from deep_gemm import int8_paged_mqa_logits

            fn = int8_paged_mqa_logits
        elif envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
            from sglang.srt.layers.attention.dsv4.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits as fn,
            )
        elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
            fn = fp8_paged_mqa_logits_torch
        else:
            from deep_gemm import fp8_paged_mqa_logits as fn

        q_fp8 = q_quant.unsqueeze(1)
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        c4_indexer_kv_cache = c4_indexer_kv_cache.view(
            c4_indexer_kv_cache.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        _c4sl = indexer_metadata.c4_seq_lens
        if _c4sl.dim() == 1:
            _c4sl = _c4sl.unsqueeze(-1)
        logits = fn(
            q_fp8,
            c4_indexer_kv_cache,
            weights,
            _c4sl,
            indexer_metadata.page_table,
            indexer_metadata.deep_gemm_metadata,
            indexer_metadata.max_c4_seq_len,
            False,
        )
        return logits

    def _get_prefill_c4_logits(
        self,
        q_quant,
        weights: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        indexer_metadata: PagedIndexerMetadata,
        request_slice: Optional[slice] = None,
        query_slice: Optional[slice] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import deep_gemm

        assert forward_batch.seq_lens_cpu is not None
        assert forward_batch.extend_seq_lens_cpu is not None
        assert forward_batch.extend_seq_lens is not None

        use_fp4 = c4_indexer.use_fp4_cache
        if use_fp4:
            q_packed, q_sf = q_quant
            num_q_tokens = q_packed.shape[0]
            device = q_packed.device
        else:
            num_q_tokens = q_quant.shape[0]
            device = q_quant.device

        c4_page_size = indexer_metadata.c4_page_size
        assert c4_page_size == 64

        extend_lens_cpu = forward_batch.extend_seq_lens_cpu
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if isinstance(extend_lens_cpu, torch.Tensor):
            extend_lens_cpu = [int(x) for x in extend_lens_cpu.tolist()]
        if isinstance(seq_lens_cpu, torch.Tensor):
            seq_lens_cpu = [int(x) for x in seq_lens_cpu.tolist()]

        # CP: tokens are round-robin split across CP ranks, so q_quant only
        # holds CP-local tokens while extend_seq_lens_cpu has full per-request
        # lengths.  Compute CP-local extend lengths to match.
        if can_nsa_prefill_cp_round_robin_split(forward_batch):
            cp_local_extend, bs_idx = nsa_cp_round_robin_split_q_seqs_cpu(
                extend_lens_cpu
            )
            extend_lens_cpu = cp_local_extend
            seq_lens_cpu = [seq_lens_cpu[i] for i in bs_idx]
            extend_seq_lens = torch.tensor(
                extend_lens_cpu,
                dtype=torch.int32,
                device=forward_batch.extend_seq_lens.device,
            )
            # Also filter the GPU seq_lens tensor so that downstream
            # forward_batch.seq_lens[req_start:req_stop] inside
            # _get_prefill_c4_logits indexes the correct (CP-local) requests.
            seq_lens = forward_batch.seq_lens[bs_idx].contiguous()
        else:
            extend_seq_lens = forward_batch.extend_seq_lens
            seq_lens = forward_batch.seq_lens

        if request_slice is None:
            request_slice = slice(0, len(extend_lens_cpu))
        req_start = 0 if request_slice.start is None else request_slice.start
        req_stop = (
            len(extend_lens_cpu) if request_slice.stop is None else request_slice.stop
        )
        global_query_start = sum(extend_lens_cpu[:req_start])
        local_extend_lens_cpu = extend_lens_cpu[req_start:req_stop]
        local_seq_lens_cpu = seq_lens_cpu[req_start:req_stop]
        local_num_q_tokens = sum(local_extend_lens_cpu)
        if query_slice is None:
            query_slice = slice(0, local_num_q_tokens)
        query_start = 0 if query_slice.start is None else query_slice.start
        query_stop = (
            local_num_q_tokens if query_slice.stop is None else query_slice.stop
        )
        global_token_start = global_query_start + query_start
        global_token_end = global_query_start + query_stop

        final_c4_lens_cpu = [int(seq_len) // 4 for seq_len in local_seq_lens_cpu]
        total_kv_len = sum(final_c4_lens_cpu)
        max_c4_seq_len = max(final_c4_lens_cpu) if final_c4_lens_cpu else 0
        max_extend_len = max(local_extend_lens_cpu) if local_extend_lens_cpu else 0
        assert sum(extend_lens_cpu) <= num_q_tokens, (
            f"CP-local extend sum {sum(extend_lens_cpu)} > num_q_tokens {num_q_tokens}; "
            f"cp_size={get_attention_cp_size()}"
        )

        if total_kv_len == 0:
            logits = torch.empty(
                (
                    global_token_end - global_token_start,
                    indexer_metadata.max_c4_seq_len,
                ),
                device=device,
                dtype=torch.float32,
            )
            row_starts = torch.zeros(logits.shape[0], dtype=torch.int32, device=device)
            row_ends = torch.zeros(logits.shape[0], dtype=torch.int32, device=device)
            return logits, row_starts, row_ends

        c4_seq_lens, page_indices, ks, ke, total_kv_len = (
            _build_prefill_c4_logits_metadata(
                seq_lens[req_start:req_stop],
                extend_seq_lens[req_start:req_stop],
                indexer_metadata.c4_seq_lens[
                    global_query_start : global_query_start + local_num_q_tokens
                ],
                indexer_metadata.page_table[
                    global_query_start : global_query_start + local_num_q_tokens
                ],
                max_c4_seq_len,
                local_num_q_tokens,
                total_kv_len,
                max_extend_len,
            )
        )
        k_buf, k_scale_buf = token_to_kv_pool.get_index_k_scale_buffer(
            c4_indexer.layer_id,
            c4_seq_lens,
            page_indices,
            total_kv_len,
            max_c4_seq_len,
        )

        # Slice Q / weights / cu-seqlen for the chunked-prefill window.
        weights = weights[global_token_start:global_token_end]
        ks = ks[query_start:query_stop]
        ke = ke[query_start:query_stop]

        if use_fp4:
            # FP4 path: K values are 64 packed bytes/token (uint8); SF is 4
            # ue8m0 bytes/token reinterpreted as int32 to match DeepGEMM's
            # `fp8_fp4_mqa_logits` contract.
            kv_packed = k_buf.contiguous().view(torch.int8)
            kv_sf = k_scale_buf.contiguous().view(torch.int32).squeeze(-1)
            q_packed_chunk = q_packed[global_token_start:global_token_end]
            # q_sf is (T, next_n=1, H) (canonical paged layout). Reshape to
            # (T_chunk, H, 1) for the prefill `fp8_fp4_mqa_logits` path so
            # the byte layout matches what the previous Triton downcast
            # produced. Both squeeze(1) and unsqueeze(-1) are zero-copy
            # metadata ops; the int32 strides land at (H, 1, 1).
            q_sf_chunk = (
                q_sf[global_token_start:global_token_end].squeeze(1).unsqueeze(-1)
            )
            q_packed_int8 = q_packed_chunk.contiguous().view(torch.int8)
            logits = deep_gemm.fp8_fp4_mqa_logits(
                (q_packed_int8, q_sf_chunk),
                (kv_packed, kv_sf),
                weights,
                ks,
                ke,
                clean_logits=False,
                logits_dtype=torch.bfloat16,
            )
        else:
            q_chunk = q_quant[global_token_start:global_token_end]
            kv_scale = k_scale_buf.view(torch.float32).squeeze(-1)
            if _USE_INT8:
                kv_int8 = k_buf.view(torch.int8)
                logits = deep_gemm.int8_mqa_logits(
                    q_chunk,
                    (kv_int8, kv_scale),
                    weights,
                    ks,
                    ke,
                    clean_logits=False,
                )
            else:
                kv_fp8 = k_buf.view(FP8_DTYPE)
                logits = deep_gemm.fp8_mqa_logits(
                    q_chunk,
                    (kv_fp8, kv_scale),
                    weights,
                    ks,
                    ke,
                    clean_logits=False,
                )
        return logits, ks, ke

    def _forward_prefill_c4_topk_chunked(
        self,
        q_quant,
        weights: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        indexer_metadata: PagedIndexerMetadata,
        core_metadata: Any,
        raw_indices: Optional[torch.Tensor],
    ) -> None:
        assert forward_batch.seq_lens_cpu is not None
        assert forward_batch.extend_seq_lens_cpu is not None

        seq_lens_cpu = forward_batch.seq_lens_cpu
        extend_lens_cpu = forward_batch.extend_seq_lens_cpu
        if isinstance(seq_lens_cpu, torch.Tensor):
            seq_lens_cpu = [int(x) for x in seq_lens_cpu.tolist()]
        if isinstance(extend_lens_cpu, torch.Tensor):
            extend_lens_cpu = [int(x) for x in extend_lens_cpu.tolist()]

        # CP: tokens are round-robin split across ranks.  Use CP-local extend
        # lengths for chunk planning and token offsets so that indexing into
        # CP-local core_metadata (page_table, c4_sparse_page_indices, …) is
        # correct.
        if can_nsa_prefill_cp_round_robin_split(forward_batch):
            cp_local_extend, bs_idx = nsa_cp_round_robin_split_q_seqs_cpu(
                extend_lens_cpu
            )
            chunk_extend_lens_cpu = cp_local_extend
            chunk_seq_lens_cpu = [seq_lens_cpu[i] for i in bs_idx]
        else:
            chunk_extend_lens_cpu = extend_lens_cpu
            chunk_seq_lens_cpu = seq_lens_cpu

        c4_seq_lens_cpu = [int(seq_len) // 4 for seq_len in chunk_seq_lens_cpu]
        workspace_size = indexer_metadata.max_seq_len * 40
        max_logits_bytes = envs.SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB.get() * 1024 * 1024
        logits_dtype = torch.bfloat16 if c4_indexer.use_fp4_cache else torch.float32
        chunk_specs = split_indexer_prefill_chunks(
            c4_seq_lens_cpu,
            chunk_extend_lens_cpu,
            workspace_size,
            max_logits_bytes,
            logits_dtype,
        )

        for req_slice, query_slice in chunk_specs:
            req_start = 0 if req_slice.start is None else req_slice.start
            global_query_start = sum(chunk_extend_lens_cpu[:req_start])
            query_start = 0 if query_slice.start is None else query_slice.start
            query_stop = query_slice.stop
            assert query_stop is not None
            token_start = global_query_start + query_start
            token_end = global_query_start + query_stop

            logits, row_starts, row_ends = self._get_prefill_c4_logits(
                q_quant,
                weights,
                c4_indexer,
                forward_batch,
                token_to_kv_pool,
                indexer_metadata,
                request_slice=req_slice,
                query_slice=query_slice,
            )
            if logits.dtype == torch.bfloat16:
                top_k_per_row_prefill_bf16(
                    logits,
                    row_starts,
                    row_ends,
                    core_metadata.page_table[token_start:token_end],
                    core_metadata.c4_sparse_page_indices[token_start:token_end],
                    indexer_metadata.c4_page_size,
                    None if raw_indices is None else raw_indices[token_start:token_end],
                )
            else:
                top_k_per_row_prefill(
                    logits,
                    row_starts,
                    row_ends,
                    core_metadata.page_table[token_start:token_end],
                    core_metadata.c4_sparse_page_indices[token_start:token_end],
                    indexer_metadata.c4_page_size,
                    None if raw_indices is None else raw_indices[token_start:token_end],
                )

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG lazy upgrade: this runs from MQALayer._forward_prepare,
        # before attn_backend.forward() would trigger the upgrade.
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool

        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
            assert isinstance(self, CompressorBackendMixin)

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DSV4AttnMetadata,
        )

        assert isinstance(core_metadata, DSV4AttnMetadata)
        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        if enable_multi_stream:
            q_quant, weights, c4_indexer_kv_cache = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_quant, weights, c4_indexer_kv_cache = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )

        if c4_indexer.use_fp4_cache:
            q_packed, _q_sf = q_quant
            assert len(q_packed.shape) == 3
        else:
            assert len(q_quant.shape) == 3
        assert len(c4_indexer_kv_cache.shape) == 2
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        use_prefill_logits = self._use_prefill_logits(forward_batch)
        logits = None
        if not use_prefill_logits:
            logits = self._get_paged_c4_logits(
                q_quant,
                c4_indexer_kv_cache,
                weights,
                indexer_metadata,
                c4_indexer,
            )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer is not None

        hisparse_coordinator = forward_batch.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None and forward_batch.forward_mode.is_decode()
        )

        # Sparse-fwd prefill consumes LOCAL c4 positions: when active, allocate
        # raw_indices and publish on the shared core_metadata so the attention
        # backend (DeepseekV4AttnBackend.forward) can pick them up. Skip if
        # another path (capture/hisparse_decode) already owns raw_indices.
        fm = forward_batch.forward_mode
        use_sparse_fwd_prefill = (
            envs.SGLANG_SAIL_DSV4_USE_FLASH_MLA_SPARSE_FWD.get()
            and fm.is_extend()
            and not fm.is_target_verify()
            and not fm.is_draft_extend(include_v2=True)
        )
        # Reset stale pointer from a prior forward in the same metadata object.
        core_metadata.c4_local_topk_indices = None

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(core_metadata.c4_sparse_page_indices)
        elif hisparse_decode:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : core_metadata.c4_sparse_page_indices.size(0)
            ]
        elif use_sparse_fwd_prefill:
            raw_indices = torch.empty_like(core_metadata.c4_sparse_page_indices)

        if use_sparse_fwd_prefill and raw_indices is not None:
            core_metadata.c4_local_topk_indices = raw_indices

        if use_prefill_logits:
            self._forward_prefill_c4_topk_chunked(
                q_quant,
                weights,
                c4_indexer,
                forward_batch,
                token_to_kv_pool,
                indexer_metadata,
                core_metadata,
                raw_indices,
            )
        elif envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
            assert logits is not None
            topk_transform_512_pytorch_vectorized(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            assert logits is not None
            topk_transform_512_v2(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                indexer_metadata.topk_metadata,
            )
        else:
            assert logits is not None
            topk_transform_512(
                logits,
                indexer_metadata.c4_seq_lens,
                core_metadata.page_table,
                core_metadata.c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        if hisparse_coordinator is not None:
            if hisparse_decode:
                compress_layer_id = token_to_kv_pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                core_metadata.c4_sparse_page_indices = (
                    hisparse_coordinator.swap_in_selected_pages(
                        req_pool_indices=forward_batch.req_pool_indices,
                        compressed_seq_lens=indexer_metadata.c4_seq_lens,
                        top_k_result=raw_indices,
                        layer_id=compress_layer_id,
                    )
                )
            else:
                core_metadata.c4_sparse_page_indices = (
                    token_to_kv_pool.c4_kv_pool.translate_loc_to_hisparse_device(
                        core_metadata.c4_sparse_page_indices
                    )
                )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)


class C4Indexer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        freqs_cis: torch.Tensor,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.n_local_heads = self.n_heads
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = Compressor(
            config,
            self.layer_id,
            True,
            freqs_cis,
            compress_ratio=4,
            head_dim=self.head_dim,
            rotate=True,
            prefix=add_prefix("compressor", prefix),
        )
        self.freqs_cis = freqs_cis
        self.weight_scale: float = self.softmax_scale * self.n_heads**-0.5
        self.alt_streams = alt_streams
        self.use_fp4_cache = is_fp4_indexer_cache_enabled()
        if self.use_fp4_cache:
            assert self.head_dim % MXFP_BLOCK_SIZE.value == 0, (
                f"index_head_dim={self.head_dim} must be a multiple of MXFP4 block "
                f"size {MXFP_BLOCK_SIZE.value} for FP4 indexer cache"
            )
            assert (
                not is_hip()
            ), "FP4 indexer cache is not supported on HIP/AMD platforms yet"

        _log_sparse_indexer_budget_once()

    def compute_q(
        self,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if _USE_INT8:
            return fused_q_indexer_rope_hadamard_quant_int8(
                q, weight, self.weight_scale, self.freqs_cis, positions
            )
        return fused_q_indexer_rope_hadamard_quant(
            q, weight, self.weight_scale, self.freqs_cis, positions
        )

    # Fallback path: kept for parity-check / rollback. The fused
    # `compute_q_mxfp4` below is the production path.
    def compute_q_no_fused(
        self, q_lora: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        fused_rope(
            q[..., -self.rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
        )
        q = rotate_activation(q)
        return q

    def compute_q_mxfp4(
        self,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Fused MXFP4 Q path: wq_b GEMM + (rope + Hadamard + MXFP4 quant +
        weight*weight_scale) in one CUDA launch.

        Returns ((q_packed, q_sf), weights_out) -- same shape contract as the
        old three-step path (`compute_q_no_fused` + `downcast_to_mxfp4_indexer`
        + `fused_scale(weights, weight_scale)`).
        """
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        q_packed, q_sf, weights_out = fused_q_indexer_rope_hadamard_quant_mxfp4(
            q, weight, self.weight_scale, self.freqs_cis, positions
        )
        return (q_packed, q_sf), weights_out

    def compute_weights(self, x: torch.Tensor, skip_scale=False) -> torch.Tensor:
        out, _ = self.weights_proj(x)
        if not skip_scale:
            out = out * self.weight_scale
        return out

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.attn_backend, DeepseekV4AttnBackend)
        return forward_batch.attn_backend.forward_c4_indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            c4_indexer=self,
            alt_streams=self.alt_streams,
            enable_multi_stream=enable_multi_stream,
            q_lora_ready=q_lora_ready,
        )
