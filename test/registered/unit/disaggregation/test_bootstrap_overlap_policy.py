"""Exercise actual bootstrap admission, retaining v0.5.21 pause/cache guards."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.expanded_input import optimistic_prefill_supported
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestBootstrapOverlapPolicy(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def run_admission(
        self,
        *,
        overlap,
        attempts=0,
        retracted=False,
        metadata_available=True,
        unsafe_write_through=False,
    ):
        req = NS(
            disagg_kv_sender=object(),
            is_retracted=retracted,
            pending_bootstrap=True,
            prefill_attempt_count=0,
            swa_branching_seqlen=4 if unsafe_write_through else None,
            kv=NS(cache_protected_len=2),
            time_stats=NS(set_wait_queue_entry_time=Mock()),
        )
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue.queue = [req]
        queue.pp_size = 1
        queue.bootstrap_overlap = overlap
        queue.scheduler_stage_metrics = None
        queue.scheduler = NS(
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            tree_cache=object(),
            processed_tokens_counter=17,
        )
        queue.ensure_metadata_buffer = Mock(return_value=metadata_available)
        with (
            get_context().override_server_args(optimistic_prefill_attempts=attempts),
            patch(
                "sglang.srt.disaggregation.prefill._uses_write_through_cache",
                return_value=unsafe_write_through,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Bootstrapping],
            ),
        ):
            admitted = queue.pop_bootstrapped()
        return (queue, req, admitted)

    def test_overlap_admits_without_spending_optimistic_retries(self):
        queue, req, admitted = self.run_admission(overlap=True)
        self.assertEqual(admitted, [req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(req.pending_bootstrap)
        self.assertEqual(req.prefill_attempt_count, 0)
        self.assertEqual(req.arrival_processed_tokens, 17)
        req.time_stats.set_wait_queue_entry_time.assert_called_once_with()

    def test_default_waits_and_legacy_optimistic_budget_still_works(self):
        queue, req, admitted = self.run_admission(overlap=False)
        self.assertEqual(admitted, [])
        self.assertEqual(queue.queue, [req])
        _, req, admitted = self.run_admission(overlap=False, attempts=1)
        self.assertEqual(admitted, [req])
        self.assertEqual(req.prefill_attempt_count, 1)

    def test_overlap_respects_pause(self):
        queue, req, admitted = self.run_admission(overlap=True, retracted=True)
        self.assertEqual(admitted, [])
        self.assertEqual(queue.queue, [req])

    def test_overlap_respects_metadata_capacity(self):
        queue, req, admitted = self.run_admission(
            overlap=True, metadata_available=False
        )
        self.assertEqual(admitted, [])
        self.assertEqual(queue.queue, [req])

    def test_overlap_preserves_write_through_swa_safety(self):
        queue, req, admitted = self.run_admission(
            overlap=True, unsafe_write_through=True
        )
        self.assertEqual(admitted, [])
        self.assertEqual(queue.queue, [req])

    def test_overlap_reuses_pp_and_hicache_restrictions(self):
        cfg = resolving_view(ServerArgs(model_path="dummy"))
        self.assertTrue(optimistic_prefill_supported(cfg))
        for changes in [
            {"pp_size": 2},
            {
                "enable_hierarchical_cache": True,
                "hicache_storage_backend": "file",
                "hicache_host_memory_mode": "memory",
            },
        ]:
            args = ServerArgs(model_path="dummy", **changes)
            self.assertFalse(optimistic_prefill_supported(resolving_view(args)))


if __name__ == "__main__":
    unittest.main()
