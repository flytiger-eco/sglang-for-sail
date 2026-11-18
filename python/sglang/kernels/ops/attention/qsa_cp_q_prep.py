# PPU CP Q preparation. Raw K and RoPE state are not written here.
import torch
import triton
import triton.language as tl

@triton.jit(do_not_specialize=['N', 'PS'])
def _qprep(
    X,
    W,
    COS,
    AX,
    POS,
    O,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    EPS,
    PS,
    AXES: tl.constexpr,
    BT: tl.constexpr,
):
    row = tl.program_id(0) * BT + tl.arange(0, BT)
    token = row // H
    head = row % H
    g = tl.arange(0, D // 8)
    ss = tl.full((BT, D // 8), 0, tl.float32)
    # Match the CUDA indexer: eight sequential FMAs followed by warp reduction.
    for j in tl.static_range(8):
        x = tl.load(
            X + token[:, None] * ((H + 1) * D) + head[:, None] * D + g[None, :] * 8 + j,
            token[:, None] < N,
            0,
        ).to(tl.float32)
        ss = tl.fma(x, x, ss)
    var = tl.sum(ss, 1)
    inv = tl.rsqrt(tl.fma(var, 1.0 / D, EPS))
    d = tl.arange(0, D)
    q = tl.load(
        X + token[:, None] * ((H + 1) * D) + head[:, None] * D + d[None, :],
        token[:, None] < N,
        0,
    ).to(tl.float32)
    w = tl.load(W + d).to(tl.float32)
    # Preserve the BF16 norm and per-operation RoPE rounding boundaries.
    norm = (q * inv[:, None] * (1.0 + w[None, :])).to(X.dtype.element_ty).to(tl.float32)
    pair = tl.where(d < ROT // 2, d + ROT // 2, tl.where(d < ROT, d - ROT // 2, d))
    qpair = tl.load(
        X + token[:, None] * ((H + 1) * D) + head[:, None] * D + pair[None, :],
        token[:, None] < N,
        0,
    ).to(tl.float32)
    wp = tl.load(W + pair).to(tl.float32)
    other = (qpair * inv[:, None] * (1.0 + wp[None, :])).to(X.dtype.element_ty).to(tl.float32)
    half = d % (ROT // 2)
    axis = tl.load(AX + half).to(tl.int64)
    if AXES == 1:
        axis = tl.full((D,), 0, tl.int64)
    p = tl.load(
        POS + axis[None, :] * PS + token[:, None],
        token[:, None] < N,
        0,
    ).to(tl.int64)
    c = tl.load(COS + p * ROT + half[None, :]).to(X.dtype.element_ty).to(tl.float32)
    s = tl.load(
        COS + p * ROT + ROT // 2 + half[None, :],
    ).to(X.dtype.element_ty).to(tl.float32)
    left = (norm * c).to(X.dtype.element_ty).to(tl.float32)
    right = (other * s).to(X.dtype.element_ty).to(tl.float32)
    result = tl.where(
        d[None, :] < ROT // 2,
        left - right,
        left + right,
    ).to(X.dtype.element_ty)
    result = tl.where(d[None, :] < ROT, result, norm.to(X.dtype.element_ty))
    tl.store(
        O + token[:, None] * (H * D) + head[:, None] * D + d[None, :],
        result,
        token[:, None] < N,
    )

def qprep(
    x,
    pos,
    cache,
    axis,
    weight,
    heads=4,
    rotary_dim=64,
    eps=1e-06,
    out=None,
):
    # Eight query-head rows retain the validated PPU reduction layout.
    bt = 8
    if out is None:
        out = torch.empty(
            (x.shape[0], heads, weight.numel()),
            device=x.device,
            dtype=x.dtype,
        )
    if not x.shape[0]:
        return out
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    _qprep[triton.cdiv(x.shape[0] * heads, bt),](
        x,
        weight,
        cache,
        axis,
        pos,
        out,
        x.shape[0],
        heads,
        weight.numel(),
        rotary_dim,
        eps,
        pos.stride(0),
        pos.shape[0],
        bt,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out
