"""PPU/AIU TileLang QSA kernel entry point."""

from ._qsa_ppu_tilelang import sparse_qsa_fwd

__all__ = ["sparse_qsa_fwd"]
