"""Batch-scoped page transport; persistent pools keep their original addresses.

This module deliberately has no SGLang imports so its transport contract can
also be exercised with CPU tensors and Gloo. Production uses the CP group.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def build_batch_page_ids(token_table, seq_lens, page_size, num_pages, group=None):
    """Union of valid request pages, including cached prefixes and null page.

    Called on the *global*, unsharded request table before CP selects local
    query rows. The bitmap collective additionally unions requirements across
    CP ranks. All ranks consequently broadcast identical shapes and page order.
    Allocator pages are aligned: sampling logical offsets 0, P, 2P, ... names
    every page containing a valid token, including a partially filled last page.
    Padding columns of the request table must never enter the plan.
    """
    if page_size <= 0 or num_pages <= 0:
        raise ValueError("Invalid pool geometry")
    if token_table.ndim != 2 or seq_lens.ndim != 1:
        raise ValueError("Expected a request table and a sequence-length vector")
    if token_table.shape[0] != seq_lens.numel():
        raise ValueError("One sequence length is required per request")
    starts = torch.arange(0, token_table.shape[1], page_size, device=token_table.device)
    valid = starts.unsqueeze(0) < seq_lens.unsqueeze(1)
    pages = torch.div(
        token_table[:, ::page_size][valid], page_size, rounding_mode="floor"
    ).to(torch.int64)
    bitmap = torch.zeros(num_pages, dtype=torch.int32, device=token_table.device)
    # Page zero is used by padding/dummy accesses and must be refreshed too.
    bitmap[0] = 1
    bitmap.scatter_(0, pages, 1)
    if group is not None:
        dist.all_reduce(bitmap, op=dist.ReduceOp.MAX, group=group)
    return torch.nonzero(bitmap, as_tuple=False).flatten()


class BatchPageBroadcastPlan:
    """Pack -> collective -> scatter, preserving existing consumer addresses.

    Workspaces are reused across layers within a batch. Callers keep the plan
    alive until outstanding broadcast-stream work has been joined. Each caller
    supplies the same stream ordering and H2D dependencies as full transport.
    """

    def __init__(self, page_ids):
        self.page_ids = page_ids
        self.workspaces = {}

    def broadcast(self, destination_pages, source_pages, collective):
        key = (destination_pages.dtype, tuple(destination_pages.shape[1:]))
        packed = self.workspaces.get(key)
        if packed is None:
            packed = torch.empty(
                (self.page_ids.numel(),) + tuple(destination_pages.shape[1:]),
                dtype=destination_pages.dtype,
                device=destination_pages.device,
            )
            self.workspaces[key] = packed
        if source_pages is not None:
            torch.index_select(source_pages, 0, self.page_ids, out=packed)
        collective(packed)
        destination_pages.index_copy_(0, self.page_ids, packed)
