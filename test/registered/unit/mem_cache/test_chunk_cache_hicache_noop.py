"""Unit tests for the HiCache pump on ChunkCache (radix cache disabled)."""

import unittest

from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestChunkCacheHiCacheNoOp(unittest.TestCase):
    """ChunkCache must not inherit the raising BasePrefixCache HiCache stubs.

    A PD prefill worker started with --enable-hierarchical-cache plus prefill CP
    auto-disables the radix cache, so the registry hands back ChunkCache while
    Scheduler._process_hicache_events still pumps HiCache events: the first
    scheduled batch used to die with NotImplementedError.
    """

    def _bare_chunk_cache(self) -> ChunkCache:
        # The two methods under test touch no state, so skip the heavy __init__.
        return ChunkCache.__new__(ChunkCache)

    def test_hicache_methods_are_overridden_on_chunk_cache(self):
        from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

        for name in ("check_hicache_events", "ready_to_load_host_cache"):
            self.assertIn(
                name,
                ChunkCache.__dict__,
                f"{name} must be overridden on ChunkCache, not inherited",
            )
            self.assertIsNot(
                getattr(ChunkCache, name),
                getattr(BasePrefixCache, name),
                f"{name} still resolves to the base stub",
            )

    def test_check_hicache_events_is_no_op(self):
        self.assertIsNone(self._bare_chunk_cache().check_hicache_events())

    def test_ready_to_load_host_cache_returns_no_pending_sentinel(self):
        # -1 == ScheduleBatch.hicache_consumer_index default ("no host load pending").
        self.assertEqual(self._bare_chunk_cache().ready_to_load_host_cache(), -1)


if __name__ == "__main__":
    unittest.main()
