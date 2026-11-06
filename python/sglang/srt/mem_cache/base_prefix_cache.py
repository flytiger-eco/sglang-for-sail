from __future__ import annotations

import dataclasses
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_RADIX_CACHE,
    RadixCacheMetricsCollector,
    radix_cache_metric_labels,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_observability, get_parallel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import HiCacheController
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )


@dataclasses.dataclass(frozen=True)
class CacheRequestHandle:
    rid: str
    attempt_id: int


class CacheRequestOutcome(Enum):
    SUCCESS = auto()
    ABORT = auto()


@runtime_checkable
class PrefixCacheTrait(Protocol):
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int
    disable: bool


@dataclasses.dataclass
class MatchPrefixParams:
    """Unified parameters for match_prefix across different cache types"""

    key: RadixKey

    # Mamba specific
    cow_mamba: bool = False
    req: Optional[Req] = None


@dataclasses.dataclass
class InsertParams:
    """Unified parameters for insert across different cache types"""

    key: Optional[RadixKey] = None
    value: Optional[torch.Tensor] = None

    # Mamba specific
    mamba_value: Optional[torch.Tensor] = None

    # DSV4 NPU C128 sidecar pages, one page id per physical C128 page group.
    c128_value: Optional[torch.Tensor] = None

    # SWA specific
    prev_prefix_len: int = 0
    swa_branching_seqlen: Optional[int] = None

    # General
    component_evicted_seqlens: dict[ComponentType, int] = dataclasses.field(
        default_factory=dict, kw_only=True
    )
    chunked: bool = False
    priority: int = 0
    session_id: Optional[str] = None
    track_adopted_ranges: bool = False

    # Logical-page KV sharding: rotation base of the chain the inserted
    # values belong to (stamped onto new tree nodes; None when sharding is
    # off). See UnifiedTreeNode.rotation_base.
    rotation_base: Optional[int] = None

    def get_evicted_seqlen(self, component_type: ComponentType) -> int:
        return self.component_evicted_seqlens.get(component_type, 0)

    def set_evicted_seqlen(self, component_type: ComponentType, length: int) -> None:
        self.component_evicted_seqlens[component_type] = length


@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation"""

    prefix_len: int
    total_len: int = 0
    last_device_node: Any = None
    mamba_exist: bool = False
    swa_branch_inserted: bool = False

    # Logical-page KV sharding: the un-matched tail was NOT inserted because
    # its rotation base disagrees with the matched chain's (a cross-chain
    # graft would break the cyclic-owner gather contract). The tail's pages
    # stay owned by the inserting request; callers must not dedup/rebind
    # past prefix_len.
    rotation_tail_declined: bool = False
    inserted_host_node: Any = None
    host_insert_dropped: bool = False
    adopted_ranges: Optional[dict[ComponentType, list[tuple[int, int]]]] = None
    # Controller-applied actions from the non-stepped channels (e.g. insert_host); the stepped insert emits via InsertStepResult.actions.
    cache_actions: list[CacheAction | ComponentAction] = dataclasses.field(
        default_factory=list
    )

    def record_adopted_range(
        self, component_type: ComponentType, start: int, end: int
    ) -> None:
        if self.adopted_ranges is None or start >= end:
            return
        ranges = self.adopted_ranges.setdefault(component_type, [])
        if ranges and start <= ranges[-1][1]:
            prev_start, prev_end = ranges[-1]
            ranges[-1] = (min(prev_start, start), max(prev_end, end))
        else:
            ranges.append((start, end))


@dataclasses.dataclass
class EvictParams:
    """Unified parameters for evict across different cache types"""

    num_tokens: int = 0
    swa_num_tokens: int = 0
    mamba_num: int = 0


@dataclasses.dataclass
class EvictResult:
    """Result of an evict operation"""

    num_tokens_evicted: int = 0
    swa_num_tokens_evicted: int = 0
    mamba_num_evicted: int = 0


@dataclasses.dataclass
class IncLockRefResult:
    """Receipt returned by ``inc_lock_ref``.

    ``node_id`` is the anchor the lock was taken on; a release replays the
    receipt on that node only. A recorded UUID marks a segment boundary;
    ``None`` means root, while an absent entry means no receipt.
    ``skipped_lock_components`` records the components the acquire left
    untaken, so the release leaves them untouched.
    """

    delta: Optional[int] = None
    node_id: Optional[int] = None
    skipped_lock_components: tuple[ComponentType, ...] = ()
    component_lock_uuids: dict[ComponentType, Optional[int]] = dataclasses.field(
        default_factory=dict
    )
    component_host_lock_uuids: dict[ComponentType, Optional[int]] = dataclasses.field(
        default_factory=dict
    )

    def set_lock_uuid(
        self,
        component_type: ComponentType,
        uuid: Optional[int],
        *,
        lock_host: bool = False,
    ) -> None:
        uuids = (
            self.component_host_lock_uuids if lock_host else self.component_lock_uuids
        )
        uuids[component_type] = uuid

    def to_dec_params(self) -> DecLockRefParams:
        """Convert to the corresponding DecLockRefParams for dec_lock_ref."""
        return DecLockRefParams(
            node_id=self.node_id,
            skipped_lock_components=tuple(self.skipped_lock_components),
            component_lock_uuids=dict(self.component_lock_uuids),
            component_host_lock_uuids=dict(self.component_host_lock_uuids),
        )


@dataclasses.dataclass
class DecLockRefParams:
    """Receipt required by unified-tree ``dec_lock_ref``.

    A segment release requires its component's boundary entry; a missing
    entry must not be treated as a lock reaching the root. ``node_id`` is
    ``None`` only for receipts that never came from a unified-tree acquire
    (legacy caches, session sentinels).
    """

    node_id: Optional[int] = None
    skipped_lock_components: tuple[ComponentType, ...] = ()
    component_lock_uuids: dict[ComponentType, Optional[int]] = dataclasses.field(
        default_factory=dict
    )
    component_host_lock_uuids: dict[ComponentType, Optional[int]] = dataclasses.field(
        default_factory=dict
    )

    def get_lock_uuid(
        self, component_type: ComponentType, *, lock_host: bool = False
    ) -> Optional[int]:
        uuids = (
            self.component_host_lock_uuids if lock_host else self.component_lock_uuids
        )
        return uuids[component_type]


@dataclasses.dataclass
class DecLockRefResult:
    """Result of an dec_lock_ref operation."""

    delta: Optional[int] = None


@dataclasses.dataclass
class InitLoadBackParams:
    """Unified parameters for init_load_back across different cache types."""

    best_match_node: Any
    host_hit_length: int
    mem_quota: Optional[int] = None
    req: Optional[Req] = None


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
                            Reserved for L3 storage prefetch anchoring; L2 load_back
                            uses `best_match_node` instead.
        best_match_node :   Deepest node accepted by all component validators
                            during match_prefix. Anchor for every L2 host->device
                            load_back walk (FULL / SWA / ...). For legacy caches
                            that don't run multi-component validation, set this
                            equal to `last_host_node`.
        host_hit_length :   Number of Full-KV tokens that hit on host (CPU) and need to be
                            loaded back to device. Pure-KV cache semantics;
        swa_host_hit_length  :   Number of SWA tokens that hit on host (within the sliding
                            window) and will be load-back into the SWA device pool.
        swa_branching_seqlen: The SWA radix cache branching point, which is the longest
                              page-aligned position that could've been cache hit if there
                              exists an SWA window.
        mamba_host_hit_length:   Number of Mamba slots that hit on host and will be load-back
                            into the Mamba device pool. Typically 0 or 1.
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
        full_kv_hit_length: Longest Full-KV prefix available on either device or
                            host, independent of other components.
    """

    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    best_match_node: Any
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    swa_branching_seqlen: Optional[int] = None
    mamba_host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None
    cache_protected_len: Optional[int] = None
    full_kv_hit_length: int = 0
    # Actions the Controller applies: CacheActions itself, ComponentActions routed to the owning component.
    cache_actions: Sequence[CacheAction | ComponentAction] = ()


def zero_match_result(
    tree_cache, match_result: MatchResult, extra_key: Optional[str] = None
) -> MatchResult:
    if tree_cache.is_chunk_cache():
        # Chunk caches' match_prefix already returns a miss; no root_node to walk back to.
        return match_result
    root = tree_cache.root_node_handle(extra_key=extra_key)
    return match_result._replace(
        # [:0] keeps dtype and device of the original tensor (e.g. CUDA int64)
        # without allocating a fresh empty tensor.
        device_indices=match_result.device_indices[:0],
        last_device_node=root,
        last_host_node=root,
        best_match_node=root,
        host_hit_length=0,
        swa_host_hit_length=0,
        swa_branching_seqlen=None,
        mamba_host_hit_length=0,
        # A dropped match must not keep a mamba branching hint either: it
        # describes the discarded prefix and would mis-drive checkpointing on
        # the recomputed one.
        mamba_branching_seqlen=None,
        full_kv_hit_length=0,
    )


def rollback_matched_mamba_state(
    tree_cache, req, *, mamba_held_before_match: Optional[bool] = None
) -> None:
    """Undo the mamba state a discarded ``match_prefix`` left on ``req``.

    ``match_prefix(cow_mamba=True)`` may allocate a mamba slot and stage
    copy-on-write/clear metadata on ``req.kv``. A caller that drops the match
    (a request the scheduler does not admit, or a prefix reuse rejected by the
    DSA prefill-CP alignment gate) must roll that state back, otherwise the
    request leaks a slot and later carries stale COW hints.

    ``mamba_held_before_match`` is ``req.kv.holds_mamba`` captured before
    ``match_prefix``. Pass it at call sites where the request may already hold
    a slot from an earlier round, so only a freshly allocated slot is freed.
    When omitted, the current value is used, which keeps the scheduler's
    post-hoc behaviour for requests that have never been admitted.

    Session-held slots have their own lifecycle; freeing one here causes a
    double free, so they are always skipped.
    """
    req.kv.mamba_cow_src_index = None
    req.kv.mamba_needs_clear = False
    mamba_allocator = getattr(
        getattr(tree_cache, "req_to_token_pool", None), "mamba_allocator", None
    )
    if (
        (mamba_held_before_match is None or not mamba_held_before_match)
        and req.kv.holds_mamba
        and mamba_allocator is not None
        and not getattr(req, "session", None)
    ):
        mamba_allocator.free(req.kv.mamba_pool_idx.unsqueeze(-1))
        req.kv.mamba_pool_idx = None


@lru_cache(maxsize=1)
def dsa_cp_prefix_cache_alignment() -> int:
    """Prefix-hit length alignment required by the DSA prefill-CP kpool path.

    Returns 1 (no constraint) unless the experimental DSA prefill-CP prefix
    cache is active. When active, a reused prefix must satisfy
    ``prefix_len % lcm(page_size, index_kpool) == 0``: the kpool write path
    derives pooled slots from ``global_position // index_kpool`` and rejects a
    non-aligned chunk start deep inside the indexer, so an unaligned hit must
    be dropped before allocation instead of raising there. The radix tree
    matches whole pages and ``index_kpool`` divides ``page_size`` today, so
    this is a defence-in-depth guard for future granularity changes.
    """
    from sglang.srt.environ import envs

    if not envs.SGLANG_DSA_CP_PREFIX_CACHE.get():
        return 1

    from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_interleave

    if not is_dsa_prefill_cp_interleave():
        return 1

    from math import lcm

    from sglang.srt.configs.model_config import get_dsa_index_kpool
    from sglang.srt.runtime_context import get_server_args, process_model_config

    page_size = int(getattr(get_server_args(), "page_size", 1) or 1)
    index_kpool = int(get_dsa_index_kpool(process_model_config().hf_config))
    return lcm(max(1, page_size), max(1, index_kpool))


_MISALIGNED_PREFIX_HIT_WARNED = False


def dsa_cp_prefix_reuse_length(match_result: MatchResult) -> int:
    """Length of the prefix a match would actually reuse.

    ``device_indices`` only covers the device-resident part of the hit; with
    HiCache a match can extend into host memory (``host_hit_length``) and
    ``full_kv_hit_length`` reports the component-independent total. The
    alignment gate must judge the whole reusable prefix, not just its device
    tail, otherwise an unaligned host extension would slip through and reach
    the kpool path.
    """
    device_len = len(match_result.device_indices)
    host_len = int(getattr(match_result, "host_hit_length", 0) or 0)
    full_len = int(getattr(match_result, "full_kv_hit_length", 0) or 0)
    return max(device_len + host_len, full_len)


def is_dsa_cp_prefix_hit_misaligned(match_result: MatchResult) -> bool:
    """Whether a prefix hit violates the DSA prefill-CP kpool alignment."""
    alignment = dsa_cp_prefix_cache_alignment()
    if alignment <= 1:
        return False
    hit_len = dsa_cp_prefix_reuse_length(match_result)
    if hit_len == 0 or hit_len % alignment == 0:
        return False

    global _MISALIGNED_PREFIX_HIT_WARNED
    if not _MISALIGNED_PREFIX_HIT_WARNED:
        _MISALIGNED_PREFIX_HIT_WARNED = True
        logger.warning(
            "DSA prefill CP prefix cache: a %d-token prefix hit is not aligned "
            "to %d; dropping the reuse for this request (recompute).",
            hit_len,
            alignment,
        )
    return True


def drop_misaligned_dsa_cp_prefix_reuse(
    tree_cache,
    req,
    match_result: MatchResult,
    *,
    mamba_held_before_match: Optional[bool] = None,
    extra_key: Optional[str] = None,
) -> MatchResult:
    """Drop a misaligned reuse together with the mamba state its match staged.

    Every ``cow_mamba=True`` caller must use this instead of zeroing on its
    own: ``zero_match_result`` alone leaves the allocated mamba slot and the
    staged COW hints on the request (see ``rollback_matched_mamba_state``).
    Returns the match unchanged when it is aligned or the gate is inactive.
    """
    if not is_dsa_cp_prefix_hit_misaligned(match_result):
        return match_result
    rollback_matched_mamba_state(
        tree_cache, req, mamba_held_before_match=mamba_held_before_match
    )
    return zero_match_result(tree_cache, match_result, extra_key=extra_key)


def enforce_dsa_cp_prefix_cache_alignment(
    tree_cache, match_result: MatchResult, extra_key: Optional[str] = None
) -> MatchResult:
    """Drop a single prefix reuse whose hit length violates the kpool alignment.

    Never disables the radix cache: only this request's reuse is abandoned and
    its prefix is recomputed, which is the documented degradation contract for
    the experimental DSA prefill-CP prefix cache.

    NOTE: callers that matched with ``cow_mamba=True`` must use
    ``drop_misaligned_dsa_cp_prefix_reuse`` instead: zeroing a match alone
    leaves the allocated mamba slot and the staged COW hints on the request
    (see ``rollback_matched_mamba_state``).
    """
    if not is_dsa_cp_prefix_hit_misaligned(match_result):
        return match_result
    return zero_match_result(tree_cache, match_result, extra_key=extra_key)


def _dfs_weight_order(
    root_node: Any,
    node_handles: Sequence[Any],
    resolve_node_handle: Callable[[Any], Any],
) -> list[int]:
    last_node_to_indices: dict[Any, list[int]] = {}
    for index, node_handle in enumerate(node_handles):
        node = resolve_node_handle(node_handle)
        last_node_to_indices.setdefault(node, []).append(index)

    node_to_weight: dict[Any, int] = {
        node: len(indices) for node, indices in last_node_to_indices.items()
    }

    stack: list[tuple[Any, bool]] = [(root_node, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            weight = node_to_weight.get(node, 0)
            for child in node.children.values():
                weight += node_to_weight.get(child, 0)
            node_to_weight[node] = weight
            continue
        stack.append((node, True))
        for child in reversed(list(node.children.values())):
            stack.append((child, False))

    order: list[int] = []

    stack = [(root_node, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            order.extend(last_node_to_indices.get(node, ()))
            continue
        children = list(node.children.values())
        children.sort(key=lambda child: -node_to_weight.get(child, 0))
        stack.append((node, True))
        for child in reversed(children):
            stack.append((child, False))
    return order


class BasePrefixCache(ABC, PrefixCacheTrait):
    """Cache can be indexed by either rid or key."""

    metrics_collector: Optional[RadixCacheMetricsCollector] = (
        None  # metrics collector for the cache
    )
    cache_controller: Optional[HiCacheController] = None
    buffer_pipeline: Optional[BufferModePipeline] = None
    storage_prefetch_retries: Optional[StoragePrefetchRetries] = None
    # Set by caches that publish KV placement events; None means they don't.
    kv_events: Optional[KVCacheEventRecorder] = None

    def init_metrics_collector(self):
        from sglang.srt.layers.dp_attention import is_dp_attention_enabled

        labels = radix_cache_metric_labels(
            self.__class__.__name__, get_parallel(), is_dp_attention_enabled()
        )
        if get_observability().extra_metric_labels:
            labels.update(get_observability().extra_metric_labels)
        radix_cache_cls = resolve_collector_class(
            STAT_LOGGER_ROLE_RADIX_CACHE,
            RadixCacheMetricsCollector,
        )
        self.metrics_collector = radix_cache_cls(labels=labels)

    def update_eviction_metrics(self, num_evicted: int, start_time: float):
        if self.metrics_collector is not None and num_evicted > 0:
            self.metrics_collector.observe_eviction_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_eviction_num_tokens(num_evicted)

    def release_host_resources(self) -> None:
        """Release pinned host buffers in userspace on graceful shutdown.

        Kernel-side unpinning during process reclaim can stall teardown for
        tens of seconds (see HostKVCache.destroy). Idempotent.
        """

    def release_aborted_request(self, handle: CacheRequestHandle) -> None:
        """Release attempt state; caches without prefetch state have nothing to drop."""

    def finish(self, handle: CacheRequestHandle, outcome: CacheRequestOutcome) -> None:
        """Finish an attempt without cancelling successful asynchronous cache work."""
        if outcome != CacheRequestOutcome.SUCCESS:
            self.release_aborted_request(handle)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        pass

    def supports_fast_match_prefix(self) -> bool:
        return False

    def dfs_weight_order(self, node_handles: Sequence[Any]) -> list[int]:
        """Return request indices in depth-first, subtree-weight order."""
        return _dfs_weight_order(self.root_node, node_handles, self.resolve_node_handle)

    def resolve_node_handle(self, node_handle: Any) -> Any:
        """Map a node handle to its node -- e.g. UnifiedRadixCache looks up the
        node object from its NodeId. Temporary API for the Unified Radix Cache
        split migration.

        TODO(Jialin): Remove after the Unified Radix Cache split.
        """
        return node_handle

    def root_node_handle(self, extra_key: Optional[str] = None) -> Any:
        """The root handle as match results carry it -- the raw node by default,
        the root's NodeId for UnifiedRadixCache. extra_key scopes the root for
        implementations that shard trees per cache namespace."""
        return self.root_node

    def rotation_base_of(self, node: Any) -> Optional[int]:
        """Logical-page KV sharding: the rotation base stamped on ``node``.

        ``node`` is whatever this cache stores in ``req.last_node`` (a NodeId
        for the unified tree, None for caches without tree nodes). None means
        "no base available here", which sends the alloc path to the base the
        request recorded at its previous alloc. Tree caches that keep the
        per-chain base override this. See UnifiedTreeNode.rotation_base.
        """
        return None

    @abstractmethod
    def insert_req(self, req: Req, *, up_to: int, **kwargs):
        """Hand a finished request's KV up to row position ``up_to`` to the
        tree: insert what can be keyed and advance ``cache_protected_len``
        past it. The caller then frees ``[cache_protected_len, up_to)`` and
        everything after, and unpins; nothing here releases a slot."""

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    def free_kv_row(self, kv: Any, ranges: list[tuple[int, int]]) -> None:
        """Give back ascending, disjoint, half-open row-position ranges
        of the ``kv`` record's row; one call keeps a shared page freed once.
        """
        from sglang.srt.mem_cache.common import coalesce_ranges, free_kv_row_segments

        allocator = self.token_to_kv_pool_allocator
        row = self.req_to_token_pool.req_to_token[kv.req_pool_idx]
        # Adjacent pieces whose seam falls inside one (DCP-widened) page would
        # free that page twice; the allocator rejects that, so merge them first.
        free_kv_row_segments(
            allocator,
            [(row[start:end], start) for start, end in coalesce_ranges(ranges)],
            swa_evicted_seqlen=kv.get_evicted_seqlen(ComponentType.SWA),
            swa_dead_lo=kv.swa_dead_lo(allocator.page_size),
        )

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        """Evict cache entries to cover allocator shortfalls.

        The default implementation preserves the component-count semantics of
        :meth:`evict`. Multi-component caches backed by shared memory can
        override this entry point to stop once collateral frees make the
        requested allocation feasible.
        """
        return self.evict(params)

    @abstractmethod
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        pass

    @abstractmethod
    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        pass

    def unpin(self, req: Req) -> None:
        """Drop the tree lock the request holds on ``req.last_node``; a cache
        whose acquire returned a receipt releases with it here."""
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def claim_kv_row(self, req: Req) -> bool:
        """A streaming session keeps the request's kv row for the next turn.
        Return True after taking the row; the caller then releases nothing."""
        return False

    def on_release(self, req: Req, *, inserted: bool) -> None:
        """The row is freed and the lock dropped; ``inserted`` says whether the
        KV went into the tree first. Drop per-request state kept outside the tree."""

    def evictable_size(self):
        return 0

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def swa_protected_size(self):
        return 0

    def swa_transient_size(self):
        """Allocated SWA tokens owned outside the request and tree views."""
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> Optional[Tuple[torch.Tensor, Any]]:
        """
        Prepare host-to-device loading. None means retry admission; an empty
        tensor can be a successful auxiliary-only load or a recompute fallback.
        """
        raise NotImplementedError()

    def finish_storage_prefetch_admission(
        self,
        handle: CacheRequestHandle,
        fulfilled_tokens: int,
        reason: Optional[str],
    ) -> None:
        """Resolve storage-hit accounting once a request is admitted.

        Non-storage caches have no lifecycle state to resolve.
        """

    def discard_storage_prefetch_accounting(self, handle: CacheRequestHandle) -> None:
        """Forget storage-hit lifecycle state without emitting a result."""

    def pop_prefetch_loaded_span(
        self, handle: CacheRequestHandle
    ) -> tuple[int, Optional[int]]:
        """Pop L3-loaded tokens and their absolute prefix start, if known."""
        return self.pop_prefetch_loaded_tokens(handle), None

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def check_hicache_events(self) -> Any:
        """
        Check HiCache related activities to update radix tree and synchronize across TP workers if needed
        """
        raise NotImplementedError()

    def flush_pending_backups(self) -> None:
        """
        Submit queued host backups.
        Caches without deferred backups have nothing to flush.
        """
        pass

    def take_events(self):
        return [] if self.kv_events is None else self.kv_events.take()

    def supports_swa(self) -> bool:
        return False

    def supports_auxiliary_swa(self) -> bool:
        return False

    def evict_sliding_windows(
        self, req: Req, pre_len: int, *, eviction_interval: int = 1
    ) -> None:
        """Slide request-owned windows at the scheduler's safe eviction frontier."""
        from sglang.srt.mem_cache.common import free_swa_out_of_window_slots

        free_swa_out_of_window_slots(
            req,
            pre_len,
            sliding_window_size=self.sliding_window_size,
            page_size=self.page_size,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            is_chunk_cache=self.is_chunk_cache(),
            retain_floor=self.swa_retain_floor(req),
            eviction_interval=eviction_interval,
        )

    def swa_retain_floor(self, req) -> int | None:
        # A match lands on a state checkpoint rather than on the tail, so a cache
        # that pairs SWA with mamba/conv checkpoints has to keep the window behind
        # the last checkpoint. Those caches override this. Everyone else has
        # nothing deeper than the tail to protect.
        return None

    def swa_reprefill_tail_tokens(self) -> int:
        # Only the unified_kv compress-only HiCache layout needs to hold back a
        # trailing sliding window for re-prefill; every other cache keeps SWA
        # content-stable and overrides this where relevant.
        return 0

    def supports_mamba(self) -> bool:
        return False

    def supports_streaming_session(self) -> bool:
        return False

    def release_session(self, session_id: str) -> None:
        pass

    def release_radix_session(self, session_id: str) -> None:
        pass

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
