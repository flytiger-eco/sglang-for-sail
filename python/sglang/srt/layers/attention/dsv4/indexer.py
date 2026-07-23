from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, TypeAlias, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.jit_kernel.dsv4 import (
    fused_q_indexer_rope_hadamard_fp4_quant,
    fused_q_indexer_rope_hadamard_int8_quant,
    fused_q_indexer_rope_hadamard_quant,
    top_k_per_row_prefill,
    top_k_per_row_prefill_bf16,
    topk_transform_512,
    topk_transform_512_v2,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import (
    can_dsa_prefill_cp_round_robin_split,
    dsa_cp_round_robin_split_q_seqs_cpu,
)
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.metadata import (
    NonPagedIndexerPlan,
    PagedIndexerMetadata,
)
from sglang.srt.layers.dp_attention import get_attention_cp_size
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.utils import add_prefix, is_cuda, is_hip, is_ppu
from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.dsv4.compressor import (
        CompressorBackendMixin,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang.srt.layers.attention.dsa.triton_kernel import (
    _supports_fp8,
)

FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


IndexerQuery: TypeAlias = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


_arange_cache = {}
# Determine which quant dtype to use based on hardware capability
_USE_INT8 = not _supports_fp8() or envs.SGLANG_SAIL_DSV4_USE_INT8.get()


def split_indexer_prefill_chunks(
    seq_lens_cpu: List[int],
    query_lens_cpu: List[int],
    workspace_size: int,
    max_logits_bytes: int,
) -> List[Tuple[slice, slice]]:
    chunks: List[Tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max(1, max_logits_bytes // 4)
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


def split_indexer_prefill_chunks(
    seq_lens_cpu: List[int],
    query_lens_cpu: List[int],
    workspace_size: int,
    max_logits_bytes: int,
) -> List[Tuple[slice, slice]]:
    chunks: List[Tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max(1, max_logits_bytes // 4)
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
    """Vectorized implementation compatible with CUDA graph capture."""
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128
    assert block_size == 64
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    max_num_pages = page_table.shape[1]
    SCALE_OFFSET = block_size * head_dim
    total_dim = block_size * (head_dim + 4)

    kvcache_flat = kvcache_fp8.view(-1, total_dim)

    pages_clamped = page_table.clamp(min=0)
    kvcache_gathered = kvcache_flat[pages_clamped]

    kv_values_raw = kvcache_gathered[..., :SCALE_OFFSET].contiguous()
    kv_values_fp8 = kv_values_raw.view(dtype=FP8_DTYPE)
    kv_values = kv_values_fp8.to(torch.float32)
    kv_values = kv_values.reshape(batch_size, max_num_pages * block_size, head_dim)

    kv_scales_raw = kvcache_gathered[..., SCALE_OFFSET:].contiguous()
    kv_scales = kv_scales_raw.view(dtype=torch.float32)
    kv_scales = kv_scales.reshape(batch_size, max_num_pages * block_size)

    q_float = q_fp8[:, 0].to(torch.float32)
    scores = torch.bmm(kv_values, q_float.transpose(1, 2))
    scores = F.relu(scores)
    scores = scores * weight.unsqueeze(1)
    scores = scores.sum(dim=2)
    scores = scores * kv_scales

    padded_seq_len = max_num_pages * block_size
    cache = _arange_cache
    arange_key = f"arange_{padded_seq_len}_{scores.device}"
    if arange_key not in cache:
        cache[arange_key] = torch.arange(padded_seq_len, device=scores.device)
    positions = cache[arange_key].unsqueeze(0)
    valid_mask = positions < seq_lens.unsqueeze(1)
    scores = scores.masked_fill(~valid_mask, 0.0)

    if padded_seq_len < max_seq_len:
        scores = F.pad(scores, (0, max_seq_len - padded_seq_len), value=0.0)
    else:
        scores = scores[:, :max_seq_len]

    return scores


def _aiter_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Wrapper adapting aiter's deepgemm_fp8_paged_mqa_logits to SGLang's interface."""
    from aiter.ops.triton.attention.pa_mqa_logits import (
        deepgemm_fp8_paged_mqa_logits,
    )

    batch_size = q_fp8.shape[0]
    next_n = q_fp8.shape[1]
    total_tokens = batch_size * next_n
    _sl = seq_lens.squeeze(-1) if seq_lens.dim() == 2 else seq_lens
    kv_block_size = kvcache_fp8.shape[1]
    logits = torch.empty(
        total_tokens,
        max_seq_len,
        dtype=torch.float32,
        device=q_fp8.device,
    )
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kvcache_fp8,
        weight,
        logits,
        _sl.to(torch.int32),
        page_table.to(torch.int32),
        max_seq_len,
        KVBlockSize=kv_block_size,
        Preshuffle=True,
    )
    return logits


def fp8_paged_mqa_logits_torch_sm120(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """CUDA-graph-compatible FP8 paged MQA logits for SM120 (vectorized, no .item())."""
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    device = q_fp8.device

    assert head_dim == 128, "Vectorized torch impl hardcodes DSV4 indexer head_dim=128"
    assert (
        block_size == 64
    ), "Vectorized torch impl hardcodes block_size=64 cache layout"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    max_pages = (max_seq_len + block_size - 1) // block_size
    max_padded_seq = max_pages * block_size

    kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    SCALE_OFFSET = block_size * head_dim

    page_ids = page_table[:, :max_pages]
    kvcache_gathered = kvcache_flat[page_ids]

    kv_value_raw = kvcache_gathered[..., :SCALE_OFFSET]
    kv_scale_raw = kvcache_gathered[..., SCALE_OFFSET:]

    kv_value = kv_value_raw.contiguous().view(dtype=FP8_DTYPE).to(torch.float32)
    kv_value = kv_value.view(batch_size, max_padded_seq, head_dim)

    kv_scale = kv_scale_raw.contiguous().view(dtype=torch.float32)
    kv_scale = kv_scale.view(batch_size, max_padded_seq)

    q = q_fp8[:, 0].to(torch.float32)

    score = torch.bmm(kv_value, q.transpose(1, 2))

    score = F.relu(score)
    score = score * weight.unsqueeze(1)
    score = score.sum(dim=2)

    score = score * kv_scale

    out_width = min(max_padded_seq, max_seq_len)
    logits = score.new_full((batch_size, max_seq_len), float("-inf"))
    logits[:, :out_width] = score[:, :out_width]

    positions = torch.arange(max_seq_len, device=device)
    invalid_mask = positions.unsqueeze(0) >= seq_lens.unsqueeze(1)
    logits.masked_fill_(invalid_mask, float("-inf"))

    return logits


def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Vectorized PyTorch fallback for topk_transform_512.
    All helper tensors (arange, zeros) are cached to avoid device-tensor
    creation during HIP/CUDA graph capture."""

    TOPK = out_page_indices.shape[1]
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    cache = _arange_cache
    key_seq = f"arange_{max_seq_len}_{device}"
    key_topk = f"arange_{TOPK}_{device}"
    key_bs = f"arange_{batch_size}_{device}"
    if key_seq not in cache:
        cache[key_seq] = torch.arange(max_seq_len, device=device)
    if key_topk not in cache:
        cache[key_topk] = torch.arange(TOPK, device=device, dtype=torch.int32)
    if key_bs not in cache:
        cache[key_bs] = torch.arange(batch_size, device=device)

    positions = cache[key_seq].unsqueeze(0).expand(batch_size, -1)
    valid_mask = positions < seq_lens.unsqueeze(1)

    masked_scores = scores.clone()
    masked_scores.masked_fill_(~valid_mask, float("-inf"))

    actual_k = min(TOPK, max_seq_len)
    _, raw_indices = torch.topk(
        masked_scores, k=actual_k, dim=1, largest=True, sorted=False
    )
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        raw_indices = F.pad(raw_indices, (0, TOPK - actual_k), value=0)

    batch_indices = cache[key_bs].unsqueeze(1).expand(-1, TOPK)
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = cache[key_topk].unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    needs_sequential = seq_lens <= TOPK
    sequential_indices = cache[key_topk].unsqueeze(0).expand(batch_size, -1)
    sequential_valid = sequential_indices < seq_lens.unsqueeze(1)

    seq_indices_or_neg1 = sequential_indices.clone()
    seq_indices_or_neg1.masked_fill_(~sequential_valid, -1)

    needs_seq_mask = needs_sequential.unsqueeze(1).expand(-1, TOPK)
    raw_indices = torch.where(needs_seq_mask, seq_indices_or_neg1, raw_indices)
    valid_topk = torch.where(needs_seq_mask, sequential_valid, valid_topk)

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)
    page_indices.masked_fill_(~valid_topk, -1)

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = raw_indices.clone()
        raw_indices.masked_fill_(~valid_topk, -1)
        out_raw_indices.copy_(raw_indices)


@triton.jit
def _fused_scale_kernel(
    weight_ptr,
    q_scale_ptr,
    out_ptr,
    numel,
    out_scale,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    w = tl.load(weight_ptr + offs, mask=mask)
    qs = tl.load(q_scale_ptr + offs, mask=mask)

    acc = w.to(tl.float32) * out_scale * qs.to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_scale(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    assert weight.is_contiguous() and q_scale.is_contiguous()
    B, H = weight.shape
    numel = B * H
    out_dtype = torch.promote_types(weight.dtype, q_scale.dtype)
    out = torch.empty((B, H, 1), device=weight.device, dtype=out_dtype)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_scale_kernel[grid](
        weight,
        q_scale,
        out,
        numel,
        out_scale,
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
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> Tuple[IndexerQuery, torch.Tensor]:
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

        # The weight projection is small and fast; compute it on its own
        # stream, then have the Q stream wait on it before launching the big
        # fused Q kernel (which folds rope, hadamard, quantization, and
        # weight scaling into one pass).
        with torch.cuda.stream(stream_weights):
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            weights_ready = stream_weights.record_event()

        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            stream_q.wait_event(weights_ready)
            q, weights = c4_indexer.compute_q(q_lora, positions, weights)

        current_stream.wait_stream(stream_q)
        return q, weights

    def _forward_prepare_normal(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        skip_compressor: bool = False,
    ) -> Tuple[IndexerQuery, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

        weights = c4_indexer.compute_weights(x, skip_scale=True)
        q, weights = c4_indexer.compute_q(q_lora, positions, weights)
        if not skip_compressor:
            self.forward_indexer_compressor(
                x=x,
                forward_batch=forward_batch,
                layer_id=c4_indexer.layer_id,
                compressor=c4_indexer.compressor,
            )
        return q, weights

    def _can_use_nonpaged_indexer(
        self,
        *,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        indexer_metadata: PagedIndexerMetadata,
    ) -> bool:
        if not envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.get():
            return False
        # This path calls CUDA DeepGEMM and assumes the CUDA FP8+FP32 packed
        # indexer cache layout. Explicitly reject HIP, NPU, and other devices.
        if not is_cuda() or is_hip():
            return False
        # The gather plan is built from eager, child-local ForwardBatch metadata.
        # Rewritten, TBO-split, and graph-backed batches must use the paged path.
        if (
            forward_batch.forward_mode != ForwardMode.EXTEND
            or forward_batch._original_forward_mode is not None
            or forward_batch.tbo_parent_token_range is not None
            or forward_batch.batch_size != 1
            or indexer_metadata.use_prefill_cuda_graph
        ):
            return False
        if (
            c4_indexer.use_fp4_indexer
            or envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
            or envs.SGLANG_OPT_USE_AITER_INDEXER.get()
            or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        ):
            return False
        if (
            get_attention_cp_size() != 1
            or self.hisparse_coordinator is not None
            or is_in_tc_piecewise_cuda_graph()
            or is_in_breakable_cuda_graph()
        ):
            return False
        return not torch.cuda.is_current_stream_capturing()

    def _get_nonpaged_indexer_plan(
        self,
        *,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        indexer_metadata: PagedIndexerMetadata,
        page_table: torch.Tensor,
        c4_seq_lens: torch.Tensor,
        query_rows: int,
    ) -> Optional[NonPagedIndexerPlan]:
        # Bypass upstream non-paged indexer for PPU, use PPU-specific path
        if is_ppu():
            return None
        if query_rows < envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS.get():
            return None
        if not self._can_use_nonpaged_indexer(
            c4_indexer=c4_indexer,
            forward_batch=forward_batch,
            indexer_metadata=indexer_metadata,
        ):
            return None
        if indexer_metadata.nonpaged_plan is not None:
            return indexer_metadata.nonpaged_plan

        if (
            forward_batch.seq_lens is None
            or forward_batch.seq_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
            or forward_batch.extend_seq_lens is None
            or forward_batch.extend_start_loc is None
            or forward_batch.extend_num_tokens is None
        ):
            return None

        def to_cpu_int_list(values) -> Optional[List[int]]:
            if isinstance(values, torch.Tensor):
                if values.device.type != "cpu":
                    return None
                values = values.tolist()
            return [int(value) for value in values]

        extend_lens_cpu = to_cpu_int_list(forward_batch.extend_seq_lens_cpu)
        seq_lens_cpu = to_cpu_int_list(forward_batch.seq_lens_cpu)
        if (
            extend_lens_cpu is None
            or seq_lens_cpu is None
            or len(extend_lens_cpu) != 1
            or len(seq_lens_cpu) != 1
            or extend_lens_cpu[0] <= 0
        ):
            return None

        actual_queries = extend_lens_cpu[0]
        if (
            actual_queries != query_rows
            or int(forward_batch.extend_num_tokens) != query_rows
            or forward_batch.seq_lens.numel() != 1
            or forward_batch.extend_seq_lens.numel() != 1
            or forward_batch.extend_start_loc.numel() != 1
            or page_table.dim() != 2
            or page_table.shape[0] < query_rows
            or c4_seq_lens.numel() < query_rows
        ):
            return None

        final_c4_len = seq_lens_cpu[0] // 4
        if final_c4_len <= 0:
            return None

        request_page_table = page_table[:1].contiguous()
        ke = c4_seq_lens[:query_rows].reshape(-1).to(torch.int32).contiguous()
        gather_seq_lens = ke[-1:]
        ks = torch.zeros_like(ke)
        c4_page_size = indexer_metadata.c4_page_size
        max_seqlen_k = (final_c4_len + c4_page_size - 1) // c4_page_size * c4_page_size
        plan = NonPagedIndexerPlan(
            page_table=request_page_table,
            gather_seq_lens=gather_seq_lens,
            ks=ks,
            ke=ke,
            seq_len_sum=final_c4_len,
            max_seq_len=final_c4_len,
            max_seqlen_k=max_seqlen_k,
            query_rows=query_rows,
        )
        indexer_metadata.nonpaged_plan = plan
        return plan

    @staticmethod
    def _forward_nonpaged_indexer(
        *,
        q_indexer: torch.Tensor,
        weights: torch.Tensor,
        c4_indexer: C4Indexer,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        plan: NonPagedIndexerPlan,
    ) -> torch.Tensor:
        import deep_gemm

        k_u8, scale_u8 = token_to_kv_pool.get_index_k_scale_buffer(
            layer_id=c4_indexer.layer_id,
            seq_len_tensor=plan.gather_seq_lens,
            page_indices=plan.page_table,
            seq_len_sum=plan.seq_len_sum,
            max_seq_len=plan.max_seq_len,
        )
        k_fp8 = k_u8.view(FP8_DTYPE)
        k_scale = scale_u8.view(torch.float32).squeeze(-1)
        return deep_gemm.fp8_mqa_logits(
            q_indexer[: plan.query_rows],
            (k_fp8, k_scale),
            weights[: plan.query_rows],
            plan.ks,
            plan.ke,
            clean_logits=False,
            max_seqlen_k=plan.max_seqlen_k,
        )

    def _use_prefill_logits(self, forward_batch: ForwardBatch) -> bool:
        forward_mode = forward_batch.forward_mode
        return (
            not is_hip()
            and forward_mode.is_extend()
            and not forward_mode.is_mixed()
            and not forward_mode.is_target_verify()
            and not forward_mode.is_draft_extend_v2()
            and not envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
            and not envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
            and not _USE_INT8
        )

    def _get_prefill_c4_logits(
        self,
        q_fp8: torch.Tensor,
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

        use_fp4 = c4_indexer.use_fp4_indexer
        if use_fp4:
            q_packed, q_sf = q_fp8
            num_q_tokens = q_packed.shape[0]
            device = q_packed.device
        else:
            num_q_tokens = q_fp8.shape[0]
            device = q_fp8.device

        c4_page_size = indexer_metadata.c4_page_size
        assert c4_page_size == 64

        extend_lens_cpu = forward_batch.extend_seq_lens_cpu
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if isinstance(extend_lens_cpu, torch.Tensor):
            extend_lens_cpu = [int(x) for x in extend_lens_cpu.tolist()]
        if isinstance(seq_lens_cpu, torch.Tensor):
            seq_lens_cpu = [int(x) for x in seq_lens_cpu.tolist()]

        # q_fp8 and indexer metadata are already CP-local. ForwardBatch keeps
        # global per-request lengths, so reconstruct only the CP-local request
        # boundaries needed by the non-paged gather plan.
        if can_dsa_prefill_cp_round_robin_split(forward_batch):
            extend_lens_cpu, bs_idx = dsa_cp_round_robin_split_q_seqs_cpu(
                extend_lens_cpu
            )
            seq_lens_cpu = [seq_lens_cpu[i] for i in bs_idx]
            extend_seq_lens = forward_batch.extend_seq_lens.new_tensor(extend_lens_cpu)
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
            f"CP-local extend sum {sum(extend_lens_cpu)} > "
            f"num_q_tokens {num_q_tokens}; cp_size={get_attention_cp_size()}"
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
            q_sf_chunk = q_sf[global_token_start:global_token_end]
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
            q_fp8 = q_fp8[global_token_start:global_token_end]
            kv_fp8 = k_buf.view(FP8_DTYPE)
            kv_scale = k_scale_buf.view(torch.float32).squeeze(-1)
            kv = (kv_fp8, kv_scale)
            logits = deep_gemm.fp8_mqa_logits(
                q_fp8,
                kv,
                weights,
                ks,
                ke,
                clean_logits=False,
            )
        return logits, ks, ke

    def _forward_prefill_c4_topk_chunked(
        self,
        q_fp8: torch.Tensor,
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

        # Chunk offsets index tensors that were reindexed by the outer CP
        # layer, so plan chunks using the matching CP-local request lengths.
        if can_dsa_prefill_cp_round_robin_split(forward_batch):
            chunk_extend_lens_cpu, bs_idx = dsa_cp_round_robin_split_q_seqs_cpu(
                extend_lens_cpu
            )
            chunk_seq_lens_cpu = [seq_lens_cpu[i] for i in bs_idx]
        else:
            chunk_extend_lens_cpu = extend_lens_cpu
            chunk_seq_lens_cpu = seq_lens_cpu

        c4_seq_lens_cpu = [int(seq_len) // 4 for seq_len in chunk_seq_lens_cpu]
        workspace_size = indexer_metadata.max_seq_len * 40
        max_logits_bytes = envs.SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB.get() * 1024 * 1024
        chunk_specs = split_indexer_prefill_chunks(
            c4_seq_lens_cpu,
            chunk_extend_lens_cpu,
            workspace_size,
            max_logits_bytes,
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
                q_fp8,
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
        skip_compressor: bool = False,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        token_to_kv_pool = self.token_to_kv_pool

        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
            assert isinstance(self, CompressorBackendMixin)

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        positions = core_metadata.positions
        num_queries = min(x.shape[0], q_lora.shape[0], positions.shape[0])
        if x.shape[0] != num_queries:
            x = x[:num_queries]
        if q_lora.shape[0] != num_queries:
            q_lora = q_lora[:num_queries]
        if positions.shape[0] != num_queries:
            positions = positions[:num_queries]

        if enable_multi_stream:
            q_indexer, weights = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=positions,
                forward_batch=forward_batch,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_indexer, weights = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=positions,
                forward_batch=forward_batch,
                skip_compressor=skip_compressor,
            )

        use_fp4_indexer = c4_indexer.use_fp4_indexer

        use_prefill_logits = self._use_prefill_logits(forward_batch)
        if use_fp4_indexer:
            q_fp4, q_sf = q_indexer
            assert len(q_fp4.shape) == 3
            assert len(q_sf.shape) == 2
            if not use_prefill_logits:
                q = (q_fp4.unsqueeze(1), q_sf.unsqueeze(1))
            else:
                q = (q_fp4, q_sf)
        else:
            assert len(q_indexer.shape) == 3
            if not use_prefill_logits:
                q = q_indexer.unsqueeze(1)
            else:
                q = q_indexer

        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        # upstream use paged interface for both prefill and decode
        # ppu use non-paged interface for prefill and paged interface for decode
        logits = None
        if not use_prefill_logits:
            if _USE_INT8:
                from deep_gemm import int8_paged_mqa_logits as fn
            elif use_fp4_indexer:
                weights = weights.float()
                if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
                    raise RuntimeError(
                        "DeepSeek V4 FP4 indexer requires DeepGEMM indexer."
                    )
                from deep_gemm import fp8_fp4_paged_mqa_logits as fn
            elif envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
                from sglang.srt.layers.attention.dsa.tilelang_kernel import (
                    tilelang_fp8_paged_mqa_logits as fn,
                )
            elif envs.SGLANG_OPT_USE_AITER_INDEXER.get():
                fn = _aiter_fp8_paged_mqa_logits
            elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
                if is_sm120_supported():
                    fn = fp8_paged_mqa_logits_torch_sm120
                else:
                    fn = fp8_paged_mqa_logits_torch
            else:
                from deep_gemm import fp8_paged_mqa_logits as fn

        query_rows = q_indexer[0].shape[0] if use_fp4_indexer else q_indexer.shape[0]

        def match_num_queries(tensor: torch.Tensor, value: int) -> torch.Tensor:
            if tensor.shape[0] == query_rows:
                return tensor
            if tensor.shape[0] > query_rows:
                return tensor[:query_rows]
            pad = (0, 0) * (tensor.dim() - 1) + (0, query_rows - tensor.shape[0])
            return F.pad(tensor, pad, value=value)

        c4_seq_lens = match_num_queries(indexer_metadata.c4_seq_lens, value=1)
        _c4sl = c4_seq_lens
        page_table = match_num_queries(indexer_metadata.page_table, value=0)
        c4_sparse_page_indices = match_num_queries(
            core_metadata.c4_sparse_page_indices, value=-1
        )
        _use_tilelang = (
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.get() and not use_fp4_indexer
        )
        _use_aiter = envs.SGLANG_OPT_USE_AITER_INDEXER.get() and not use_fp4_indexer
        if _c4sl.dim() == 1 and not _use_tilelang and not _use_aiter:
            _c4sl = _c4sl.unsqueeze(-1)
        nonpaged_plan = self._get_nonpaged_indexer_plan(
            c4_indexer=c4_indexer,
            forward_batch=forward_batch,
            indexer_metadata=indexer_metadata,
            page_table=page_table,
            c4_seq_lens=c4_seq_lens,
            query_rows=query_rows,
        )
        if nonpaged_plan is not None:
            assert isinstance(q_indexer, torch.Tensor)
            logits = self._forward_nonpaged_indexer(
                q_indexer=q_indexer,
                weights=weights,
                c4_indexer=c4_indexer,
                token_to_kv_pool=token_to_kv_pool,
                plan=nonpaged_plan,
            )
        else:
            c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=c4_indexer.layer_id,
            )
            assert c4_indexer_kv_cache.dim() == 2
            head_dim_with_sf = 68 if use_fp4_indexer else 132
            c4_indexer_kv_cache = c4_indexer_kv_cache.view(
                c4_indexer_kv_cache.shape[0], 64, 1, head_dim_with_sf
            )
        if not use_prefill_logits:
            logits = fn(
                q,
                c4_indexer_kv_cache,
                weights,
                _c4sl,
                page_table,
                indexer_metadata.deep_gemm_metadata,
                indexer_metadata.max_c4_seq_len,
                False,
            )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer is not None

        hisparse_coordinator = self.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None and forward_batch.forward_mode.is_decode()
        )

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(c4_sparse_page_indices)
        elif hisparse_decode:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : c4_sparse_page_indices.size(0)
            ]
        elif core_metadata.c4_sparse_raw_indices is not None:
            raw_indices = core_metadata.c4_sparse_raw_indices

        if use_prefill_logits:
            self._forward_prefill_c4_topk_chunked(
                q,
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
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            assert logits is not None
            topk_transform_512_v2(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                indexer_metadata.topk_metadata,
            )
        else:
            assert logits is not None
            topk_transform_512(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
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
                # flash_mla C4 attention requires int32 page indices.
                core_metadata.c4_sparse_page_indices = (
                    token_to_kv_pool.c4_kv_pool.translate_loc_to_hisparse_device(
                        core_metadata.c4_sparse_page_indices
                    ).to(torch.int32)
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
        rotary_emb=None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
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
            rotary_emb=rotary_emb,
        )
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis
        self.weight_scale: float = self.softmax_scale * self.n_heads**-0.5
        from sglang.srt.server_args import get_global_server_args

        self.use_fp4_indexer = get_global_server_args().enable_deepseek_v4_fp4_indexer
        self.alt_streams = alt_streams

    def compute_q(
        self,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[IndexerQuery, torch.Tensor]:
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self.use_fp4_indexer:
            return fused_q_indexer_rope_hadamard_fp4_quant(
                q.contiguous(), weight, self.weight_scale, self.freqs_cis, positions
            )
        elif _USE_INT8:
            return fused_q_indexer_rope_hadamard_int8_quant(
                q, weight, self.weight_scale, self.freqs_cis, positions
            )
        return fused_q_indexer_rope_hadamard_quant(
            q, weight, self.weight_scale, self.freqs_cis, positions
        )

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
        attn_backend: AttentionBackend,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
        skip_compressor: bool = False,
    ) -> None:
        return attn_backend.forward_c4_indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            c4_indexer=self,
            alt_streams=self.alt_streams,
            enable_multi_stream=enable_multi_stream,
            q_lora_ready=q_lora_ready,
            skip_compressor=skip_compressor,
        )
