"""397B attention importance extraction adapted to the installed FA3 API."""

from sglang.srt.utils import is_ppu
if is_ppu():
    from sglang.srt.hardware_backend.ppu.attention.flash_attention import flash_attn_varlen_func
else:
    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.srt.layers.attention.compute_flash_attn_score_triton import compute_flash_attn_score


def flash_attn_varlen_func_with_score(q, k, v, cu_seqlens, max_seqlen, softmax_scale):
    result = flash_attn_varlen_func(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        softmax_scale=softmax_scale, causal=False,
        return_softmax_lse=True, ver=3,
    )
    output, lse = result[0], result[1]
    scores = compute_flash_attn_score(
        q=q, k=k, lse=lse, cu_seqlens=cu_seqlens,
        sm_scale=softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5,
    )
    return output, scores
