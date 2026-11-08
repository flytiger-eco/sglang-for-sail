"""Reject unsupported QSA CPU retraction without disabling normal serving."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.srt.managers.schedule_batch import ScheduleBatch, release_req
from sglang.srt.mem_cache.common import backup_kv_cache
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestQSARetractionPolicy(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                disaggregation_decode_retraction_backup="cpu_tensor",
            ),
            role="scheduler",
        )

    @staticmethod
    def allocator():
        pool = QSATokenToKVPool.__new__(QSATokenToKVPool)
        pool.device = "cpu"
        return NS(get_kvcache=lambda: pool)

    def test_qsa_backup_refuses_incomplete_target_and_draft_copy(self):
        req = NS(rid="qsa", offload_kv_cache=Mock())
        with patch(
            "sglang.srt.mem_cache.common.dsv41_dspark_needs_rebootstrap",
            return_value=False,
        ):
            self.assertFalse(
                backup_kv_cache(req, None, None, self.allocator(), "cpu_tensor")
            )
        req.offload_kv_cache.assert_not_called()

    def test_non_qsa_cpu_backup_keeps_existing_behavior(self):
        req = NS(rid="ordinary", offload_kv_cache=Mock())
        allocator = NS(get_kvcache=lambda: NS())
        with patch(
            "sglang.srt.mem_cache.common.dsv41_dspark_needs_rebootstrap",
            return_value=False,
        ):
            self.assertTrue(backup_kv_cache(req, None, None, allocator, "cpu_tensor"))
        req.offload_kv_cache.assert_called_once_with(None, allocator)

    def test_refused_qsa_backup_still_releases_resources(self):
        req = NS(rid="qsa", offload_kv_cache=Mock(), reset_for_retract=Mock())
        with (
            patch(
                "sglang.srt.mem_cache.common.dsv41_dspark_needs_rebootstrap",
                return_value=False,
            ),
            patch("sglang.srt.managers.schedule_batch.release_kv_cache") as release,
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            self.assertFalse(
                release_req(
                    req=req,
                    remaing_req_count=1,
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=self.allocator(),
                    tree_cache=None,
                    hisparse_coordinator=None,
                )
            )
        release.assert_called_once_with(req, None, is_insert=False)
        req.reset_for_retract.assert_called_once()
        req.offload_kv_cache.assert_not_called()

    def test_inflight_target_and_draft_readers_drain_before_slots_are_freed(self):
        events = []
        req = NS(
            rid="qsa",
            offload_kv_cache=Mock(),
            reset_for_retract=lambda: events.append("reset"),
        )
        module = NS(synchronize=lambda device: events.append("synchronize"))
        with (
            patch(
                "sglang.srt.mem_cache.common.dsv41_dspark_needs_rebootstrap",
                return_value=False,
            ),
            patch(
                "sglang.srt.mem_cache.common.torch.get_device_module",
                return_value=module,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.release_kv_cache",
                side_effect=lambda *args, **kwargs: events.append("release"),
            ),
            patch("sglang.srt.managers.schedule_batch.evict_from_tree_cache"),
        ):
            self.assertFalse(
                release_req(
                    req=req,
                    remaing_req_count=1,
                    req_to_token_pool=None,
                    token_to_kv_pool_allocator=self.allocator(),
                    tree_cache=None,
                    hisparse_coordinator=None,
                )
            )
        self.assertEqual(events, ["synchronize", "release", "reset"])
        req.offload_kv_cache.assert_not_called()

    def test_qsa_retraction_is_retryable_abort_and_never_requeued(self):
        healthy = NS(rid="healthy", beam_group=None)
        victim = NS(rid="qsa", beam_group=None)
        batch = ScheduleBatch.__new__(ScheduleBatch)
        batch.reqs = [healthy, victim]
        batch.token_to_kv_pool_allocator = self.allocator()
        batch._get_decode_retraction_order = lambda reqs: [0, 1]
        batch.check_decode_mem = Mock(return_value=True)
        batch.release_req = Mock(return_value=False)
        batch.filter_batch = Mock()
        with (
            patch(
                "sglang.srt.managers.schedule_batch.beam_retraction_order",
                side_effect=lambda order, reqs: order,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
                return_value=1,
            ),
        ):
            requeued, _, aborted = batch.retract_decode()
        self.assertEqual(requeued, [])
        self.assertEqual(aborted, [victim])
        self.assertEqual(victim.to_finish.status_code, HTTPStatus.SERVICE_UNAVAILABLE)
        self.assertIn("QSA state", victim.to_finish.message)
        batch.release_req.assert_called_once_with(1, 1)
        batch.filter_batch.assert_called_once_with(keep_indices=[0])
        self.assertFalse(hasattr(healthy, "to_finish"))


if __name__ == "__main__":
    unittest.main()
