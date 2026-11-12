"""Unit tests for the DSA prefill-CP prefix-cache alignment gate.

A hybrid/mamba request matches its prefix with ``cow_mamba=True``, which may
allocate a mamba slot and stage copy-on-write metadata. When the alignment
gate drops the reuse, all of that must be rolled back: lingering COW fields,
a leaked slot, or a stale ``mamba_branching_seqlen`` would silently corrupt
the recomputed prefix.

Covers:
- ``rollback_matched_mamba_state`` directly (fresh vs. pre-held vs. session
  slot);
- the gate inside ``Req.init_next_round_input`` (mamba path) and its
  no-op behaviour when the hit is aligned;
- ``drop_misaligned_dsa_cp_prefix_reuse`` (shared by every ``cow_mamba=True``
  caller, including the decode-side PD match);
- reuse-length accounting for HiCache host extensions;
- the PD bypass predicates (only interleave prefill-CP can un-bypass).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
import unittest.mock
from array import array
from types import SimpleNamespace

import torch
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.disaggregation.utils import (
    should_bypass_dsa_cp_prefix_cache,
    supports_dsa_cp_prefix_cache,
    validate_dsa_cp_prefix_cache_geometry,
)
from sglang.srt.managers.schedule_batch import (
    Req,
    ReqKvInfo,
    ScheduleBatch,
)
from sglang.srt.managers.schedule_policy import match_prefix_for_req
from sglang.srt.mem_cache.base_prefix_cache import (
    MatchResult,
    drop_misaligned_dsa_cp_prefix_reuse,
    dsa_cp_prefix_reuse_length,
    is_dsa_cp_prefix_hit_misaligned,
    rollback_matched_mamba_state,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase


class _TrackingMambaAllocator:
    def __init__(self, slot: int = 5):
        self.slot = slot
        self.alloc_calls = 0
        self.free_calls = 0
        self.live = set()

    def alloc(self) -> torch.Tensor:
        self.alloc_calls += 1
        # 0-dim, matching every production producer (``mid[0]``,
        # ``dst_index[0]``); ``_collect_deferred_mamba_cow_and_clear``
        # relies on that shape to build its 1-D gather index.
        idx = torch.tensor(self.slot, dtype=torch.int64)
        self.live.add(self.slot)
        return idx

    def free(self, indices: torch.Tensor) -> None:
        self.free_calls += 1
        self.live.discard(int(indices[0]))


def _tree_cache(allocator) -> SimpleNamespace:
    return SimpleNamespace(req_to_token_pool=SimpleNamespace(mamba_allocator=allocator))


class TestRollbackMatchedMambaState(CustomTestCase):
    def test_freshly_allocated_slot_is_freed(self):
        allocator = _TrackingMambaAllocator()
        kv = ReqKvInfo(
            mamba_pool_idx=allocator.alloc(),
            mamba_cow_src_index=torch.tensor(2, dtype=torch.int64),
            mamba_needs_clear=True,
        )
        req = SimpleNamespace(kv=kv, session=None)

        rollback_matched_mamba_state(
            _tree_cache(allocator), req, mamba_held_before_match=False
        )

        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(allocator.free_calls, 1)
        self.assertEqual(len(allocator.live), 0)

    def test_pre_held_slot_is_kept(self):
        allocator = _TrackingMambaAllocator()
        kv = ReqKvInfo(
            mamba_pool_idx=torch.tensor(3, dtype=torch.int64),
            mamba_cow_src_index=torch.tensor(2, dtype=torch.int64),
            mamba_needs_clear=True,
        )
        req = SimpleNamespace(kv=kv, session=None)

        rollback_matched_mamba_state(
            _tree_cache(allocator), req, mamba_held_before_match=True
        )

        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(allocator.free_calls, 0)

    def test_session_slot_is_never_freed(self):
        allocator = _TrackingMambaAllocator()
        kv = ReqKvInfo(mamba_pool_idx=torch.tensor(4, dtype=torch.int64))
        req = SimpleNamespace(kv=kv, session=SimpleNamespace())

        rollback_matched_mamba_state(_tree_cache(allocator), req)

        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertEqual(allocator.free_calls, 0)


class _GateTreeCache:
    """Radix-cache stub whose match stages mamba CoW state like the real one."""

    def __init__(self, allocator, hit_len: int, mamba_branching_seqlen: int):
        self.req_to_token_pool = SimpleNamespace(mamba_allocator=allocator)
        self.root = object()
        self.allocator = allocator
        self.hit_len = hit_len
        self.mamba_branching_seqlen = mamba_branching_seqlen

    def supports_mamba(self) -> bool:
        return True

    def swa_reprefill_tail_tokens(self) -> int:
        return 0

    def is_chunk_cache(self) -> bool:
        return False

    def root_node_handle(self, extra_key=None):
        return self.root

    def match_prefix(self, params):
        req = params.req
        if params.cow_mamba and not req.kv.holds_mamba:
            req.kv.mamba_pool_idx = self.allocator.alloc()
            req.kv.mamba_cow_src_index = torch.tensor(2, dtype=torch.int64)
            req.kv.mamba_needs_clear = True
        return MatchResult(
            device_indices=torch.arange(self.hit_len, dtype=torch.int64),
            last_device_node=self.root,
            last_host_node=self.root,
            best_match_node=self.root,
            host_hit_length=0,
            mamba_branching_seqlen=self.mamba_branching_seqlen,
        )


def _make_req() -> Req:
    return Req(
        rid="dsa-cp-gate",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3, 4, 5, 6, 7, 8]),
        sampling_params=SamplingParams(max_new_tokens=8),
    )


class TestGateInInitNextRoundInput(CustomTestCase):
    def test_misaligned_hit_rolls_back_mamba_state(self):
        allocator = _TrackingMambaAllocator()
        tree = _GateTreeCache(allocator, hit_len=3, mamba_branching_seqlen=7)
        req = _make_req()

        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=4,
        ):
            req.init_next_round_input(tree)

        # Gate dropped the reuse: no CoW residue, no leaked slot, no hint.
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.mamba_branching_seqlen)
        self.assertEqual(allocator.alloc_calls, 1)
        self.assertEqual(allocator.free_calls, 1)
        self.assertEqual(len(allocator.live), 0)
        # Match zeroed to the root, so the request recomputes.
        self.assertEqual(int(req.prefix_indices.numel()), 0)
        self.assertIs(req.last_node, tree.root)

    def test_pre_held_slot_survives_g1_gate(self):
        # Later prefill rounds already hold a slot: the gate must clear only
        # the staged COW hints, never free the slot another round allocated.
        allocator = _TrackingMambaAllocator()
        tree = _GateTreeCache(allocator, hit_len=3, mamba_branching_seqlen=7)
        req = _make_req()
        req.kv.mamba_pool_idx = torch.tensor(3, dtype=torch.int64)

        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=4,
        ):
            req.init_next_round_input(tree)

        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertTrue(req.kv.mamba_needs_clear)
        self.assertIsNone(req.mamba_branching_seqlen)
        self.assertEqual(allocator.alloc_calls, 0)
        self.assertEqual(allocator.free_calls, 0)

    def test_aligned_hit_keeps_match_and_mamba_state(self):
        allocator = _TrackingMambaAllocator()
        tree = _GateTreeCache(allocator, hit_len=4, mamba_branching_seqlen=7)
        req = _make_req()

        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=4,
        ):
            req.init_next_round_input(tree)

        self.assertEqual(int(req.prefix_indices.numel()), 4)
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertEqual(allocator.free_calls, 0)
        self.assertEqual(req.mamba_branching_seqlen, 7)


class TestCollectDeferredMambaState(CustomTestCase):
    """Batch-level G2 invariant: deferred COW/clear matches held 0-dim slots."""

    def test_staged_cow_uses_0dim_slot(self):
        batch = SimpleNamespace()
        kv = ReqKvInfo(
            mamba_pool_idx=torch.tensor(5, dtype=torch.int64),
            mamba_cow_src_index=torch.tensor(7, dtype=torch.int64),
        )
        req = SimpleNamespace(kv=kv)

        ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])

        self.assertEqual(batch.mamba_cow_src_indices.tolist(), [7])
        self.assertEqual(batch.mamba_cow_dst_indices.tolist(), [5])
        self.assertIsNone(batch.mamba_clear_indices)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)

    def test_staged_clear_uses_0dim_slot(self):
        batch = SimpleNamespace()
        kv = ReqKvInfo(
            mamba_pool_idx=torch.tensor(3, dtype=torch.int64),
            mamba_needs_clear=True,
        )
        req = SimpleNamespace(kv=kv)

        ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])

        self.assertIsNone(batch.mamba_cow_src_indices)
        self.assertEqual(batch.mamba_clear_indices.tolist(), [3])

    def test_staged_cow_without_slot_fails(self):
        batch = SimpleNamespace()
        kv = ReqKvInfo(
            mamba_pool_idx=None,
            mamba_cow_src_index=torch.tensor(7, dtype=torch.int64),
        )
        req = SimpleNamespace(kv=kv)

        with self.assertRaises(AssertionError):
            ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])


class _ZeroingTreeCache:
    """Minimal cache stub for ``zero_match_result`` / ``inc_lock_ref`` paths."""

    def __init__(self, allocator):
        self.req_to_token_pool = SimpleNamespace(mamba_allocator=allocator)
        self.root = object()

    def is_chunk_cache(self) -> bool:
        return False

    def root_node_handle(self, extra_key=None):
        return self.root


def _match(device_len: int, host_len: int = 0, full_len: int = 0) -> MatchResult:
    node = object()
    return MatchResult(
        device_indices=torch.arange(device_len, dtype=torch.int64),
        last_device_node=node,
        last_host_node=node,
        best_match_node=node,
        host_hit_length=host_len,
        mamba_branching_seqlen=9,
        full_kv_hit_length=full_len,
    )


class TestDsaCpReuseLength(CustomTestCase):
    """The gate judges the whole reusable prefix, not just its device part."""

    def test_host_extension_counts_towards_alignment(self):
        match = _match(device_len=8, host_len=2)
        self.assertEqual(dsa_cp_prefix_reuse_length(match), 10)
        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=4,
        ):
            self.assertTrue(is_dsa_cp_prefix_hit_misaligned(match))

    def test_full_kv_hit_length_without_device_hit(self):
        match = _match(device_len=0, host_len=0, full_len=6)
        self.assertEqual(dsa_cp_prefix_reuse_length(match), 6)
        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=4,
        ):
            self.assertTrue(is_dsa_cp_prefix_hit_misaligned(match))

    def test_aligned_total_passes(self):
        match = _match(device_len=8, host_len=4)
        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=4,
        ):
            self.assertFalse(is_dsa_cp_prefix_hit_misaligned(match))

    def test_gate_inactive_when_alignment_is_one(self):
        match = _match(device_len=3)
        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=1,
        ):
            self.assertFalse(is_dsa_cp_prefix_hit_misaligned(match))


class TestDropMisalignedReuse(CustomTestCase):
    """``cow_mamba=True`` callers must use the shared drop helper."""

    def test_misaligned_drop_rolls_back_staged_slot(self):
        allocator = _TrackingMambaAllocator()
        kv = ReqKvInfo(
            mamba_pool_idx=allocator.alloc(),
            mamba_cow_src_index=torch.tensor(2, dtype=torch.int64),
            mamba_needs_clear=True,
        )
        req = SimpleNamespace(kv=kv, session=None)

        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=4,
        ):
            dropped = drop_misaligned_dsa_cp_prefix_reuse(
                _ZeroingTreeCache(allocator),
                req,
                _match(device_len=6)._replace(cache_protected_len=6),
                mamba_held_before_match=False,
            )

        self.assertEqual(int(dropped.device_indices.numel()), 0)
        self.assertEqual(dropped.cache_protected_len, 0)
        self.assertIsNone(dropped.mamba_branching_seqlen)
        self.assertIsNotNone(dropped.last_device_node)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(allocator.free_calls, 1)

    def test_aligned_match_is_returned_untouched(self):
        allocator = _TrackingMambaAllocator()
        kv = ReqKvInfo(mamba_pool_idx=torch.tensor(3, dtype=torch.int64))
        req = SimpleNamespace(kv=kv, session=None)
        match = _match(device_len=8)

        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=4,
        ):
            kept = drop_misaligned_dsa_cp_prefix_reuse(
                _ZeroingTreeCache(allocator), req, match
            )

        self.assertIs(kept, match)
        self.assertEqual(allocator.free_calls, 0)
        self.assertIsNotNone(req.kv.mamba_pool_idx)

    def test_retained_session_state_is_cleared_before_recomputation(self):
        allocator = _TrackingMambaAllocator()
        req = SimpleNamespace(
            kv=ReqKvInfo(mamba_pool_idx=torch.tensor(4, dtype=torch.int64)),
            session=object(),
        )
        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=4,
        ):
            match = drop_misaligned_dsa_cp_prefix_reuse(
                _ZeroingTreeCache(allocator),
                req,
                _match(device_len=6),
                mamba_held_before_match=True,
            )

        batch = SimpleNamespace()
        ScheduleBatch._collect_deferred_mamba_cow_and_clear(batch, [req])
        self.assertEqual(match.device_indices.numel(), 0)
        self.assertEqual(allocator.free_calls, 0)
        self.assertEqual(req.kv.mamba_pool_idx.item(), 4)
        self.assertEqual(batch.mamba_clear_indices.tolist(), [4])


def _pd_args(**overrides):
    from sglang.srt.runtime_context import get_context

    args = {
        "disaggregation_mode": "prefill",
        "enable_prefill_cp": True,
        "cp_strategy": "interleave",
    }
    args.update(overrides)
    return get_context().override_server_args(**args)


class TestBypassPredicates(CustomTestCase):
    """PD gating: only interleave prefill-CP deployments can un-bypass."""

    def test_interleave_is_supported(self):
        with _pd_args():
            self.assertTrue(supports_dsa_cp_prefix_cache())

    def test_zigzag_is_not_supported(self):
        with _pd_args(cp_strategy="zigzag"):
            self.assertFalse(supports_dsa_cp_prefix_cache())

    def test_decode_is_not_supported(self):
        with _pd_args(disaggregation_mode="decode"):
            self.assertFalse(supports_dsa_cp_prefix_cache())

    def test_unified_deployment_is_not_supported(self):
        with _pd_args(disaggregation_mode="null"):
            self.assertFalse(supports_dsa_cp_prefix_cache())

    def test_switch_off_keeps_bypass(self):
        with _pd_args(), unittest.mock.patch(
            "sglang.srt.disaggregation.utils.dsa_cp_prefix_cache_enabled",
            return_value=False,
        ):
            self.assertTrue(should_bypass_dsa_cp_prefix_cache())

    def test_switch_on_unbypasses_interleave(self):
        with _pd_args(), unittest.mock.patch(
            "sglang.srt.disaggregation.utils.dsa_cp_prefix_cache_enabled",
            return_value=True,
        ):
            self.assertFalse(should_bypass_dsa_cp_prefix_cache())

    def test_switch_on_keeps_bypass_for_non_interleave(self):
        with _pd_args(cp_strategy="zigzag"), unittest.mock.patch(
            "sglang.srt.disaggregation.utils.dsa_cp_prefix_cache_enabled",
            return_value=True,
        ):
            self.assertTrue(should_bypass_dsa_cp_prefix_cache())

    def test_non_dsa_never_bypasses(self):
        with _pd_args():
            self.assertFalse(should_bypass_dsa_cp_prefix_cache(is_dsa=False))

    def test_pooled_indexer_is_rejected_before_cache_construction(self):
        with self.assertRaisesRegex(ValueError, "index_kpool == 1"):
            validate_dsa_cp_prefix_cache_geometry(
                page_size=64, index_kpool=4, cache_layer_split=False
            )

    def test_unpooled_full_replica_geometry_is_supported(self):
        validate_dsa_cp_prefix_cache_geometry(
            page_size=64, index_kpool=1, cache_layer_split=False
        )

    def test_layer_split_hicache_layout(self):
        validate_dsa_cp_prefix_cache_geometry(
            page_size=64, index_kpool=1, cache_layer_split=True,
            hicache_layout="layer_first", hicache_io_backend="kernel",
        )
        with self.assertRaisesRegex(ValueError, "layer_first"):
            validate_dsa_cp_prefix_cache_geometry(
                page_size=64, index_kpool=1, cache_layer_split=True,
                hicache_layout="page_first", hicache_io_backend="kernel",
            )


class TestWrapperCowMambaGate(CustomTestCase):
    """``match_prefix_for_req(cow_mamba=True)`` must roll back, not just zero.

    The wrapper applies the alignment gate internally, so the decode-side PD
    match (the other ``cow_mamba=True`` caller) only gets protection if the
    wrapper itself is cow-aware.
    """

    def _run(self, hit_len: int):
        allocator = _TrackingMambaAllocator()
        tree = _GateTreeCache(allocator, hit_len=hit_len, mamba_branching_seqlen=7)
        req = _make_req()
        with unittest.mock.patch(
            "sglang.srt.mem_cache.base_prefix_cache.dsa_cp_prefix_cache_alignment",
            return_value=4,
        ):
            match_prefix_for_req(
                tree,
                req,
                req.origin_input_ids,
                cow_mamba=True,
                include_req=True,
            )
        return allocator, req

    def test_misaligned_wrapper_match_frees_staged_slot(self):
        allocator, req = self._run(hit_len=3)

        self.assertEqual(int(req.prefix_indices.numel()), 0)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(allocator.alloc_calls, 1)
        self.assertEqual(allocator.free_calls, 1)
        self.assertEqual(len(allocator.live), 0)

    def test_aligned_wrapper_match_keeps_staged_slot(self):
        allocator, req = self._run(hit_len=4)

        self.assertEqual(int(req.prefix_indices.numel()), 4)
        self.assertIsNotNone(req.kv.mamba_pool_idx)
        self.assertEqual(allocator.free_calls, 0)


class TestResolvedDsaCpAlignment(CustomTestCase):
    """The alignment follows resolved config and scoped feature overrides."""

    def setUp(self):
        super().setUp()
        from sglang.srt.mem_cache.base_prefix_cache import dsa_cp_prefix_cache_alignment

        # PR224 memoizes this globally. Clear between tests, but deliberately
        # not between overrides inside a test: that would hide the regression.
        clear = getattr(dsa_cp_prefix_cache_alignment, "cache_clear", None)
        if clear:
            clear()
            self.addCleanup(clear)

    def _context(self):
        from sglang.srt.runtime_context import get_context

        model = ModelConfig.__new__(ModelConfig)
        model.hf_config = PretrainedConfig(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_kpool=1,
        )
        return get_context().override_server_args(
            page_size=None,
            attn_cp_size=8,
            cp_strategy="interleave",
            enable_prefill_cp=True,
            disaggregation_mode="prefill",
            _model_config=model,
        )

    def test_resolved_page_size_replaces_unspecified_input(self):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.base_prefix_cache import dsa_cp_prefix_cache_alignment
        from sglang.srt.runtime_context import get_schedule

        with self._context(), envs.SGLANG_DSA_CP_PREFIX_CACHE.override(True):
            with get_schedule().override(page_size=64):
                self.assertEqual(dsa_cp_prefix_cache_alignment(), 64)

    def test_alignment_follows_scoped_page_override(self):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.base_prefix_cache import dsa_cp_prefix_cache_alignment
        from sglang.srt.runtime_context import get_schedule

        with self._context(), envs.SGLANG_DSA_CP_PREFIX_CACHE.override(True):
            with get_schedule().override(page_size=64):
                self.assertEqual(dsa_cp_prefix_cache_alignment(), 64)
            with get_schedule().override(page_size=128):
                self.assertEqual(dsa_cp_prefix_cache_alignment(), 128)

    def test_disabled_call_does_not_poison_later_enabled_call(self):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.base_prefix_cache import dsa_cp_prefix_cache_alignment
        from sglang.srt.runtime_context import get_schedule

        with self._context(), get_schedule().override(page_size=64):
            with envs.SGLANG_DSA_CP_PREFIX_CACHE.override(False):
                self.assertEqual(dsa_cp_prefix_cache_alignment(), 1)
            with envs.SGLANG_DSA_CP_PREFIX_CACHE.override(True):
                self.assertEqual(dsa_cp_prefix_cache_alignment(), 64)


if __name__ == "__main__":
    unittest.main()
