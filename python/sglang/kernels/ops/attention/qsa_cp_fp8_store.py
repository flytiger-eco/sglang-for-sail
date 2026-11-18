"""Fuse CP token-order restoration and native FP8 KV cache stores."""
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['N', 'CAP', 'SLOTS'])
def _store_cp_kv(
    K,
    V,
    KC,
    VC,
    LOC,
    N,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    CP: tl.constexpr,
    CAP,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    KS,
    VS,
    SLOTS,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    token = i // (HEADS * D)
    head = i // D % HEADS
    dim = i % D
    valid = token < N
    row = token % CP * CAP + token // CP
    slot = tl.load(LOC + token, valid, 0).to(tl.int64)
    k = tl.load(K + row * SK0 + head * SK1 + dim * SK2, valid, 0).to(tl.float32)
    v = tl.load(V + row * SV0 + head * SV1 + dim * SV2, valid, 0).to(tl.float32)
    # Torch in-place division rounds to BF16 before clamp and FP8 conversion.
    k = tl.div_rn(k, KS).to(K.dtype.element_ty).to(tl.float32)
    v = tl.div_rn(v, VS).to(V.dtype.element_ty).to(tl.float32)
    # Comparisons retain NaNs, including their propagation through clamp.
    k = tl.where(k > 448.0, 448.0, tl.where(k < -448.0, -448.0, k))
    v = tl.where(v > 448.0, 448.0, tl.where(v < -448.0, -448.0, v))
    mask = valid & (slot >= 0) & (slot < SLOTS)
    dest = slot * (HEADS * D) + head * D + dim
    tl.store(KC + dest, k, mask)
    tl.store(VC + dest, v, mask)

def store_cp_kv(
    k,
    v,
    k_cache,
    v_cache,
    loc,
    *,
    total_rows,
    cp_size,
    local_capacity,
    k_scale=1.0,
    v_scale=1.0,
):
    assert k.shape == v.shape and k.ndim == 3
    assert k_cache.shape == v_cache.shape and k_cache.shape[1:] == k.shape[1:]
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
    assert k.dtype == v.dtype == torch.bfloat16 and k.stride(2) == v.stride(2) == 1
    assert loc.ndim == 1 and loc.numel() >= total_rows
    assert k.shape[0] == cp_size * local_capacity
    if not total_rows:
        return
    h, d = k.shape[1:]
    block = 2048
    _store_cp_kv[triton.cdiv(total_rows * h * d, block),](
        k,
        v,
        k_cache,
        v_cache,
        loc,
        total_rows,
        h,
        d,
        cp_size,
        local_capacity,
        *k.stride(),
        *v.stride(),
        float(k_scale),
        float(v_scale),
        k_cache.shape[0],
        block,
        num_warps=4,
        enable_fp_fusion=False,
    )
