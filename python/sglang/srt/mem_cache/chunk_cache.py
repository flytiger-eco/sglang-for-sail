from __future__ import annotations

"""Cache for chunked prefill, used when RadixCache is disabled."""

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.mem_cache.allocator.hisparse import (
    DeepSeekV4HiSparseTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


logger = logging.getLogger(__name__)


class ChunkCache(BasePrefixCache):
    """
    ChunkCache is used when radix cache is disabled.

    That includes standard chunked-prefill setups and the decode side of P/D
    disaggregation when decode radix cache is not enabled.
    """

    def __init__(self, params: CacheInitParams):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        self.protected_size_ = 0
        # HiCache poll surfaces: ChunkCache never enqueues GPU<->host
        # write-through / load-back operations and has no storage backend.
        # Scheduler.is_fully_idle() polls these containers based purely on
        # `enable_hierarchical_cache`, so a radix-disabled worker (e.g. PD
        # DSA prefill CP with the experimental prefix-cache switch off,
        # which force-disables the radix cache) reached the base class with
        # no such attributes and raised AttributeError on the very first
        # idle poll, killing the process tree during startup. Overriding
        # them here matches the existing ChunkCache convention for
        # operations it cannot perform (`evict`, `reset`,
        # `check_hicache_events`, `ready_to_load_host_cache` are vacuous).
        self.ongoing_write_through: dict = {}
        self.ongoing_load_back: dict = {}
        self.enable_storage = False

    def is_chunk_cache(self) -> bool:
        return True

    # NOTE (csy): this is to determine if a cache has prefix matching feature.
    # Chunk cache always return True to indicate no prefix matching.
    # TODO (csy): Using a prefix cache trait to replace this
    @property
    def disable(self):
        return True

    def reset(self):
        pass

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            best_match_node=None,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        # ChunkCache does not support prefix caching, so insert is a no-op
        return InsertResult(prefix_len=0)

    def insert_req(self, req: Req, *, up_to: int):
        pass

    def cache_unfinished_req(self, req: Req, chunked=False):
        kv_indices = self.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, : req.extend_range.end
        ]
        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)

    def evict(self, params: EvictParams) -> EvictResult:
        return EvictResult()

    def check_hicache_events(self) -> Any:
        """HiCache drain is vacuous when the radix tree is switched off.

        ``ChunkCache`` is what the cache registry hands back whenever radix
        cache is disabled, including the automatic disable for DSA prefill CP.
        ``Scheduler._process_hicache_events`` calls this purely based on
        ``enable_hierarchical_cache``, so a PD prefill worker started with
        ``--enable-hierarchical-cache`` plus prefill CP reached the
        ``BasePrefixCache`` stub and raised ``NotImplementedError`` on the very
        first scheduled batch. Overriding it here matches the existing
        ChunkCache convention for operations it cannot perform (``evict``,
        ``inc_lock_ref``/``dec_lock_ref``, ``reset`` are all no-ops).
        """
        return None

    def ready_to_load_host_cache(self) -> Any:
        # -1 is ScheduleBatch.hicache_consumer_index's "no host load pending"
        # sentinel, and tp_worker.set_hicache_consumer is a no-op without a
        # hicache_layer_transfer_counter, which ChunkCache never creates.
        return -1

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        return IncLockRefResult(delta=0)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        return DecLockRefResult(delta=0)

    def protected_size(self):
        # NOTE: no protected size in chunk cache. Chunk cache's eviction is the same with request's lifecycle.
        return 0

    def pretty_print(self):
        return ""


class SWAChunkCache(ChunkCache):
    """ChunkCache with support for sliding window attention."""

    def __init__(self, params: CacheInitParams):
        # DeepSeek V4 HiSparse wraps SWATokenToKVPoolAllocator and exposes the same API.
        assert isinstance(
            params.token_to_kv_pool_allocator,
            (
                SWATokenToKVPoolAllocator,
                DeepSeekV4HiSparseTokenToKVPoolAllocator,
            ),
        )
        super().__init__(params)

        self.sliding_window_size = params.sliding_window_size
        self.chunked_prefill_size = params.chunked_prefill_size

    def supports_swa(self) -> bool:
        assert self.sliding_window_size is not None, (
            "sliding_window_size must be set for SWAChunkCache"
        )
        return True

    def evict(self, params: EvictParams) -> EvictResult:
        return EvictResult()


class PureSWAChunkCache(SWAChunkCache):
    """ChunkCache for all-SWA models (no full attention layers): no
    full_to_swa_index_mapping, so free_kv_row must skip the window-evicted span."""
