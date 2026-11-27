"""Split-K selection for the production fused gated-HC shapes."""


def _pick_split_k(rows: int) -> int:
    return 8 if 32 < rows <= 64 else 16
