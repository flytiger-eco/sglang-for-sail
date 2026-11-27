"""Launch-config table shared with the fused gated-HC megakernel.

The standalone down/gate Triton kernels that used to live in this module were
superseded by ``hc_gated_mega.fused_gated_hc``; only the config picker remains.
"""


def _pick_config(rows):
    # BM, BN, BK, warps, split-K, stages. Larger row counts need a shorter
    # partial reduction to keep the BF16 down projection close to its reference.
    if rows <= 32:
        return (32, 32, 64, 4, 16, 2)
    if rows <= 64:
        return (64, 32, 64, 8, 8, 3)
    if rows <= 128:
        return (64, 64, 64, 8, 16, 3)
    return (128, 64, 32, 8, 32, 3)
