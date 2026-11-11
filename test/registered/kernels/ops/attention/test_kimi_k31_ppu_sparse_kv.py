"""PPU K3.1 sparse prefill and paged decode over one FP8 DSA KV cache."""

import pytest
import torch

from sglang.srt.utils.common import is_sail
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not is_sail(), reason="PPU FlashMLA kernels are required")
def test_sparse_prefill_and_kv_decode_share_fp8_cache():
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
        dequantize_k_cache_paged,
    )
    from sglang.kernels.ops.quantization.dsa_quant_k_cache import quantize_k_cache
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
        FlashMLASchedMeta,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
    )

    torch.manual_seed(17)
    kv = (torch.randn(4, 64, 1, 576, device="cuda") * 0.25).to(torch.bfloat16)
    q = (torch.randn(16, 64, 576, device="cuda") * 0.25).to(torch.bfloat16)
    fp8_cache = quantize_k_cache(kv, pow2_scale=True)
    slots = torch.arange(1, 129, device="cuda", dtype=torch.int32)
    selected_kv = dequantize_k_cache_paged(fp8_cache, slots)
    topk = torch.arange(128, device="cuda", dtype=torch.int32)
    indices = topk.view(1, 1, 128).expand(16, 1, 128).contiguous()
    lengths = torch.full((16,), 128, device="cuda", dtype=torch.int32)
    scale = 576**-0.5

    prefill, _, _ = flash_mla_sparse_fwd(
        q, selected_kv, indices, scale, d_v=512, topk_length=lengths
    )
    keys = selected_kv[:, 0].float()
    reference = torch.softmax(q.float() @ keys.T * scale, dim=-1) @ keys[:, :512]
    torch.testing.assert_close(prefill.float(), reference, atol=0.02, rtol=0.02)

    decode, _ = flash_mla_with_kvcache(
        q[:1, None],
        fp8_cache,
        torch.empty((1, 0), device="cuda", dtype=torch.int32),
        torch.tensor([256], device="cuda", dtype=torch.int32),
        512,
        FlashMLASchedMeta(),
        softmax_scale=scale,
        indices=slots.view(1, 1, 128),
        topk_length=lengths[:1],
        is_fp8_kvcache=False,
        kv_format="V32",
    )
    torch.testing.assert_close(
        decode.reshape(64, 512).float(), reference[0], atol=0.02, rtol=0.02
    )
