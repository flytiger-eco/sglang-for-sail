"""DeepSeek-V4 attention kernels (RFC #29630, Phase 2.5)."""

from sglang.kernels.registry import register_kernel
from sglang.kernels.spec import CapabilityRequirement, KernelBackend, KernelSpec

register_kernel(
    KernelSpec(
        op="attention.dequantize_k_cache_paged_v2_cuda",
        backend=KernelBackend.JIT,
        target="sglang.kernels.ops.attention.dsv4.attn:dequantize_k_cache_paged_v2_cuda",
        capabilities=frozenset({CapabilityRequirement.CUDA}),
    )
)

# --- merged from sglang.kernels.ops.attention.dsv4 (RFC #29630 Phase 4) ---
from .attn import (
    dequantize_k_cache_paged_cuda,
    dequantize_k_cache_paged_v2_cuda,
    fused_store_cache,
    get_paged_mqa_logits_metadata,
)
from .compress import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_norm_rope_store,
)
from .norm_rope import (
    fused_k_norm_rope_flashmla,
    fused_norm_rope_inplace,
    fused_q_indexer_rope_hadamard_fp4_quant,
    fused_q_indexer_rope_hadamard_int8_quant,
    fused_q_indexer_rope_hadamard_quant,
    fused_q_norm_rope,
    fused_rope_inplace,
)
from .topk import (
    plan_topk_v2,
    top_k_per_row_prefill,
    top_k_per_row_prefill_bf16,
    topk_transform_512_bf16,
    topk_transform_kpool_ragged_v2,
    topk_transform_kpool_v2,
    topk_transform_paged,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from .utils import make_name

__all__ = [
    "CompressorDecodePlan",
    "CompressorPrefillPlan",
    "compress_forward",
    "compress_norm_rope_store",
    "dequantize_k_cache_paged_cuda",
    "dequantize_k_cache_paged_v2_cuda",
    "fused_norm_rope_inplace",
    "fused_store_cache",
    "fused_rope_inplace",
    "fused_q_norm_rope",
    "fused_q_indexer_rope_hadamard_int8_quant",
    "fused_q_indexer_rope_hadamard_fp4_quant",
    "fused_q_indexer_rope_hadamard_quant",
    "fused_k_norm_rope_flashmla",
    "make_name",
    "get_paged_mqa_logits_metadata",
    "topk_transform_paged",
    "topk_transform_kpool_v2",
    "topk_transform_kpool_ragged_v2",
    "topk_transform_512_bf16",
    "topk_transform_paged_v2",
    "topk_transform_ragged_v2",
    "top_k_per_row_prefill",
    "top_k_per_row_prefill_bf16",
    "plan_topk_v2",
]
