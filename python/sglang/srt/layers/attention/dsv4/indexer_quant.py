"""Shared V4.1 indexer format selection for allocation and execution."""

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, is_ppu


def dsv41_requires_int8_indexer() -> bool:
    return is_ppu() and get_device_sm() < 89


def use_dsv41_int8_indexer() -> bool:
    # PPU below SM89 cannot use the V4.1 FP4 indexer, including when the legacy
    # opt-in is explicitly set to zero. All low-ratio consumers share this
    # decision so Q/K computation, cache layout and memory sizing agree.
    return dsv41_requires_int8_indexer() or (
        is_ppu() and envs.SGLANG_SAIL_DSV4_USE_INT8.get()
    )
