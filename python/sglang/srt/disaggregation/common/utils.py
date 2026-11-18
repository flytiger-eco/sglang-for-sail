import ctypes
import dataclasses
import struct
import threading
from collections import deque
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from sglang.srt.observability.trace import (
    TraceNullContext,
    TraceReqContext,
)


@dataclasses.dataclass
class TransferKVChunk:
    """Work unit for KV cache transfer from prefill to decode."""

    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last_chunk: bool
    prefill_aux_index: Optional[int]
    state_indices: Optional[List]
    chunk_id: Optional[int] = None
    num_kv_tokens: Optional[int] = None
    trace_ctx: Union[TraceReqContext, TraceNullContext] = dataclasses.field(
        default_factory=TraceNullContext
    )
    # Set when the staging worker first counts this chunk toward the per-room
    # outstanding count; stays set across re-enqueue on a watermark defer.
    staging_counted: bool = False


def pack_list_of_buffers(buffers: List[bytes]) -> bytes:
    if not buffers:
        return b""
    n = len(buffers)
    header = struct.pack(f"<{n+1}I", n, *(len(b) for b in buffers))
    return header + b"".join(buffers)


def unpack_list_of_buffers(buf: bytes) -> List[bytes]:
    if buf == b"":
        return []
    (n,) = struct.unpack("<I", buf[:4])
    lens = struct.unpack(f"<{n}I", buf[4 : 4 + 4 * n])
    out = []
    offset = 4 + 4 * n
    for length in lens:
        out.append(buf[offset : offset + length])
        offset += length
    return out


def pack_int_lists(lists, fmt: str) -> bytes:
    return pack_list_of_buffers([struct.pack(f"<{len(a)}{fmt}", *a) for a in lists])


def unpack_int_lists(buf: bytes, fmt: str) -> List[List[int]]:
    width = struct.calcsize(fmt)
    return [
        list(struct.unpack(f"<{len(b)//width}{fmt}", b))
        for b in unpack_list_of_buffers(buf)
    ]


class FastQueue:
    def __init__(self):
        self._buf = deque()
        self._cond = threading.Condition()

    def put(self, item):
        with self._cond:
            self._buf.append(item)
            # wake up a thread of wait()
            self._cond.notify()

    def get(self):
        with self._cond:
            # if queue is empty  ,block until is notified()
            while not self._buf:
                self._cond.wait()
            return self._buf.popleft()


class AuxDataCodec:
    """Handles serialization and deserialization of auxiliary data buffers."""

    @staticmethod
    def serialize_data_from_buffer(src_addr, data_length):
        """Serialize data from memory buffer to bytes."""
        buffer = (ctypes.c_byte * data_length).from_address(src_addr)
        return bytes(buffer)

    @staticmethod
    def deserialize_data_to_buffer(kv_args, buffer_index, aux_index, data):
        """Deserialize bytes into target memory buffer."""
        dst_aux_ptr = kv_args.aux_data_ptrs[buffer_index]
        item_len = kv_args.aux_item_lens[buffer_index]
        dst_addr = dst_aux_ptr + item_len * aux_index
        buffer = (ctypes.c_byte * len(data)).from_address(dst_addr)
        buffer[:] = data
        return


def group_concurrent_contiguous(
    src_indices: npt.NDArray[np.int32], dst_indices: npt.NDArray[np.int32]
) -> Tuple[List[npt.NDArray[np.int32]], List[npt.NDArray[np.int32]]]:
    """Vectorised NumPy implementation."""
    # src/dst indices are transferred pairwise, so an empty side means there is
    # nothing to transfer. Guarding both sides (not just src) avoids a cryptic
    # NumPy broadcast error from np.diff() below when only one side is empty, e.g.
    # a non-empty prefill DSA/SWA state list paired with an empty decode registration.
    if src_indices.size == 0 or dst_indices.size == 0:
        return [], []

    if src_indices.size != dst_indices.size:
        raise ValueError(
            "group_concurrent_contiguous requires equal-length src/dst index arrays, "
            f"got {src_indices.size} and {dst_indices.size}"
        )

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


@dataclasses.dataclass(frozen=True)
class DCPTokenTransferPlan:
    src_token_indices: npt.NDArray[np.int64]
    dst_token_indices: npt.NDArray[np.int64]


def build_dcp_token_transfer_plan(
    src_page_indices: npt.NDArray[np.int32],
    dst_page_indices: npt.NDArray[np.int32],
    *,
    physical_page_size: int,
    dcp_size: int,
    dcp_rank: int,
    src_page_offset: int = 0,
    decode_prefix_len: int = 0,
    num_kv_tokens: Optional[int] = None,
    replicate_to_virtual: bool = False,
) -> DCPTokenTransferPlan:
    if physical_page_size <= 0 or dcp_size <= 0 or not 0 <= dcp_rank < dcp_size:
        raise ValueError("Invalid DCP page size or rank")
    if src_page_offset < 0 or decode_prefix_len < 0:
        raise ValueError("Negative PD token offset")
    virtual_page_size = physical_page_size * dcp_size
    if decode_prefix_len % virtual_page_size != 0:
        raise ValueError(
            "PD DCP transfer requires decode_prefix_len to align to the virtual "
            f"DCP page size ({virtual_page_size}), got {decode_prefix_len}"
        )

    src_pages = np.asarray(src_page_indices, dtype=np.int64)
    dst_pages = np.asarray(dst_page_indices, dtype=np.int64)
    if (
        src_pages.ndim != 1
        or dst_pages.ndim != 1
        or np.any(src_pages < 0)
        or np.any(dst_pages < 0)
    ):
        raise ValueError("PD pages must be one-dimensional nonnegative indices")
    source_capacity = src_pages.size * physical_page_size
    if num_kv_tokens is None:
        num_kv_tokens = source_capacity
    if not 0 <= num_kv_tokens <= source_capacity:
        raise ValueError(
            "num_kv_tokens must fit in the provided source pages, "
            f"got tokens={num_kv_tokens}, capacity={source_capacity}"
        )
    if src_pages.size == 0:
        empty = np.empty((0,), dtype=np.int64)
        return DCPTokenTransferPlan(empty, empty.copy())

    chunk_start = decode_prefix_len + src_page_offset * physical_page_size
    first_owned_offset = (
        0 if replicate_to_virtual else (dcp_rank - chunk_start) % dcp_size
    )
    owned_offsets = np.arange(
        first_owned_offset,
        num_kv_tokens,
        1 if replicate_to_virtual else dcp_size,
        dtype=np.int64,
    )
    src_token_indices = (
        src_pages[owned_offsets // physical_page_size] * physical_page_size
        + owned_offsets % physical_page_size
    )

    relative_positions = src_page_offset * physical_page_size + owned_offsets
    dst_local_offsets = (
        relative_positions if replicate_to_virtual else relative_positions // dcp_size
    )
    dst_page_size = virtual_page_size if replicate_to_virtual else physical_page_size
    dst_page_ordinals = dst_local_offsets // dst_page_size
    if dst_page_ordinals.size and (
        dst_pages.size == 0 or int(dst_page_ordinals.max()) >= dst_pages.size
    ):
        required_pages = int(dst_page_ordinals.max()) + 1
        raise ValueError(
            "Insufficient destination DCP pages: "
            f"required={required_pages}, provided={dst_pages.size}, "
            f"src_page_offset={src_page_offset}, dcp_rank={dcp_rank}"
        )

    dst_token_indices = (
        dst_pages[dst_page_ordinals] * dst_page_size + dst_local_offsets % dst_page_size
    )
    return DCPTokenTransferPlan(src_token_indices, dst_token_indices)


def _draft_head_transfer_geometry(
    src_token_bytes: int,
    dst_token_bytes: int,
    total_heads: int,
    src_tp_size: int,
    src_tp_rank: int,
    dst_tp_size: int,
    dst_tp_rank: int,
) -> Tuple[int, int, int]:
    """Immutable head geometry, shared across chunks; token indices are not cached.

    Uses the common PD rank mapping: each D rank contacts a contiguous P-rank
    group when P TP is larger, or one P rank otherwise. Choose one sender for
    duplicate KV heads *within that contact group*, not globally.
    """
    if (
        min(total_heads, src_tp_size, dst_tp_size, src_token_bytes, dst_token_bytes)
        <= 0
    ):
        raise ValueError("Draft head geometry must be positive")
    if not (0 <= src_tp_rank < src_tp_size and 0 <= dst_tp_rank < dst_tp_size):
        raise ValueError("Draft TP rank out of range")
    if max(src_tp_size, dst_tp_size) % min(src_tp_size, dst_tp_size):
        raise ValueError("Draft TP sizes must divide each other")
    for tp in (src_tp_size, dst_tp_size):
        if max(total_heads, tp) % min(total_heads, tp):
            raise ValueError("Draft head count and TP size must divide each other")
    src_heads = max(1, total_heads // src_tp_size)
    dst_heads = max(1, total_heads // dst_tp_size)
    if src_token_bytes % src_heads or dst_token_bytes % dst_heads:
        raise ValueError("Draft token bytes must divide into whole heads")
    head_bytes = src_token_bytes // src_heads
    if head_bytes != dst_token_bytes // dst_heads:
        raise ValueError("Draft source/destination bytes per head differ")
    src_replication = max(1, src_tp_size // total_heads)
    dst_replication = max(1, dst_tp_size // total_heads)
    if src_tp_size > dst_tp_size:
        group_size = src_tp_size // dst_tp_size
        group_start = dst_tp_rank * group_size
        if not group_start <= src_tp_rank < group_start + group_size:
            return 0, 0, 0
        owner = max(group_start, src_tp_rank // src_replication * src_replication)
        if src_tp_rank != owner:
            return 0, 0, 0
    elif src_tp_rank != dst_tp_rank // (dst_tp_size // src_tp_size):
        return 0, 0, 0
    src_begin = src_tp_rank // src_replication * src_heads
    dst_begin = dst_tp_rank // dst_replication * dst_heads
    begin, end = max(src_begin, dst_begin), min(
        src_begin + src_heads, dst_begin + dst_heads
    )
    if begin >= end:
        return 0, 0, 0
    src_offset = (begin - src_begin) * head_bytes
    dst_offset = (begin - dst_begin) * head_bytes
    copy_bytes = (end - begin) * head_bytes
    return src_offset, dst_offset, copy_bytes


def build_draft_head_transfer_blocks(
    plan,
    *,
    src_ptr,
    dst_ptr,
    src_token_bytes,
    dst_token_bytes,
    total_heads,
    src_tp_size,
    src_tp_rank,
    dst_tp_size,
    dst_tp_rank,
):
    """Map each draft token's intersecting heads, choosing one contacted owner."""
    src_offset, dst_offset, width = _draft_head_transfer_geometry(
        src_token_bytes,
        dst_token_bytes,
        total_heads,
        src_tp_size,
        src_tp_rank,
        dst_tp_size,
        dst_tp_rank,
    )
    if (
        plan.src_token_indices.ndim != 1
        or plan.dst_token_indices.ndim != 1
        or len(plan.src_token_indices) != len(plan.dst_token_indices)
        or np.any(plan.src_token_indices < 0)
        or np.any(plan.dst_token_indices < 0)
    ):
        raise ValueError("Invalid draft token indices")
    if not width:
        return []
    if width == src_token_bytes == dst_token_bytes:
        # Full rows may be merged only when both address streams are contiguous.
        # Head slices leave gaps in at least one peer's rows and stay separate.
        src_groups, dst_groups = group_concurrent_contiguous(
            plan.src_token_indices, plan.dst_token_indices
        )
        return [
            (
                src_ptr + int(src[0]) * src_token_bytes + src_offset,
                dst_ptr + int(dst[0]) * dst_token_bytes + dst_offset,
                len(src) * width,
            )
            for src, dst in zip(src_groups, dst_groups)
        ]
    return [
        (
            src_ptr + int(src) * src_token_bytes + src_offset,
            dst_ptr + int(dst) * dst_token_bytes + dst_offset,
            width,
        )
        for src, dst in zip(plan.src_token_indices, plan.dst_token_indices)
    ]
