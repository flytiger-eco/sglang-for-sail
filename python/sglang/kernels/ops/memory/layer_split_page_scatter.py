"""Bit-preserving scatter of unique batch pages into stable LayerSplit pools."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_pages_kernel(
    destination,
    source,
    pages,
    PAGE_UNITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    destination_page = tl.load(pages + page).to(tl.int64)
    offset = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(source + page * PAGE_UNITS + offset, offset < PAGE_UNITS, other=0)
    tl.store(
        destination + destination_page * PAGE_UNITS + offset,
        value,
        offset < PAGE_UNITS,
    )


def scatter_unique_pages(destination, pages, packed):
    """Copy pages whose IDs are unique, valid and disjoint from packed storage.

    BatchPageBroadcastPlan obtains these IDs from a union bitmap and allocates
    separate packed storage. Unsupported layouts use PyTorch index_copy_.
    """
    if (
        not destination.is_cuda
        or not destination.is_contiguous()
        or not packed.is_contiguous()
        or not pages.is_contiguous()
        or destination.dtype != packed.dtype
        or destination.device != packed.device
        or pages.device != packed.device
        or pages.ndim != 1
        or pages.dtype not in (torch.int64, torch.int32)
        or tuple(destination.shape[1:]) != tuple(packed.shape[1:])
        or packed.shape[0] != pages.numel()
    ):
        destination.index_copy_(0, pages, packed)
        return destination
    if not pages.numel():
        return destination
    page_bytes = math.prod(destination.shape[1:]) * destination.element_size()
    if not page_bytes:
        return destination
    source = packed.view(torch.uint8).reshape(-1)
    target = destination.view(torch.uint8).reshape(-1)
    # Copy raw representations, preserving FP8 values and NaN payloads.
    if (
        page_bytes % 4 == 0
        and source.data_ptr() % 4 == 0
        and target.data_ptr() % 4 == 0
    ):
        source = source.view(torch.int32)
        target = target.view(torch.int32)
        units = page_bytes // 4
    else:
        units = page_bytes
    _scatter_pages_kernel[(pages.numel(), triton.cdiv(units, 2048))](
        target, source, pages, units, BLOCK=2048, num_warps=4
    )
    return destination
