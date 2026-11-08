"""Cancellation credit admission and malformed metadata isolation."""

import threading
import unittest
from collections import OrderedDict
from http import HTTPStatus
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import KVTransferError
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
)
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPreallocAbortAdmission(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    @staticmethod
    def manager():
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._init_prealloc_abort_state()
        manager.request_status = {}
        manager._staging_outstanding = {}
        manager._deferred_ack_targets = {}
        manager._send_prealloc_abort_ack = Mock()
        return manager

    @staticmethod
    def sender(manager, room, pair):
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr = manager
        sender.bootstrap_room = room
        sender.conclude_state = None
        sender.set_prealloc_abort_pair_key(pair)
        manager.request_status[room] = KVPoll.Bootstrapping
        return sender

    def retire_failed(self, manager, room):
        sender = self.sender(manager, room, str(room))
        sender.conclude_state = KVPoll.Failed
        sender.clear()

    def test_invalid_generation_does_not_escape_scheduler_or_damage_owner(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._pending_expanded_lens = OrderedDict()
        queue._seen_expanded_metadata = OrderedDict()
        queue._expanded_len_cache_max = 4096
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.disagg_decode_prealloc_queue = queue
        self.assertTrue(queue.set_expanded_input_len("83", 100, 0, "owner"))
        expected = queue._pending_expanded_lens.copy()
        for generation in (None, [], 0, "", "x" * 129):
            with self.subTest(generation=generation):
                scheduler.handle_expanded_input_len(
                    NS(
                        rid="83",
                        expanded_input_len=100,
                        mrope_position_delta=0,
                        request_generation=generation,
                        metadata_version=2,
                    )
                )
                self.assertFalse(queue.set_expanded_input_len("83", 100, 0, generation))
                self.assertEqual(queue._pending_expanded_lens, expected)
                self.assertFalse(queue._seen_expanded_metadata)
        self.assertTrue(queue.set_expanded_input_len("84", 101, 1, "next-owner"))

    def test_last_admitted_sender_can_cancel_when_all_other_credits_are_retired(self):
        from sglang.srt.disaggregation.expanded_input import EXPANDED_INPUT_MAX_INFLIGHT

        manager = self.manager()
        for room in range(EXPANDED_INPUT_MAX_INFLIGHT - 1):
            self.retire_failed(manager, room)
        sender = self.sender(manager, 50000, "last")
        self.assertEqual(
            len(manager._prealloc_abort_reservations), EXPANDED_INPUT_MAX_INFLIGHT
        )
        self.assertFalse(manager.reserve_prealloc_abort(50001, "too-many"))
        manager._handle_prealloc_abort(
            [b"ABORT", b"50000", b"127.0.0.1", b"8999", b"WAIT_FOR_ROOM", b"last"]
        )
        self.assertEqual(manager.check_status(50000), KVPoll.Failed)
        manager._send_prealloc_abort_ack.assert_called_once_with(
            "127.0.0.1", 8999, 50000, "last"
        )
        sender.clear()
        # A later Decode rank still receives its generation-scoped ACK.
        manager._handle_prealloc_abort(
            [b"ABORT", b"50000", b"127.0.0.1", b"9000", b"WAIT_FOR_ROOM", b"last"]
        )
        self.assertEqual(manager._send_prealloc_abort_ack.call_count, 2)

    def test_healthy_sender_clear_does_not_accumulate_credits(self):
        manager = self.manager()
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.EXPANDED_INPUT_MAX_INFLIGHT", 2
        ):
            for room in range(20):
                sender = self.sender(manager, room, str(room))
                sender.conclude_state = KVPoll.Success
                sender.clear()
                self.assertFalse(manager._prealloc_abort_reservations)
                self.assertFalse(manager._prealloc_abort_replays)
                self.assertFalse(manager._prealloc_abort_attempts)

    def test_blocked_abort_ack_does_not_block_unrelated_admission(self):
        for path in ("new", "replay", "drain", "clear"):
            with self.subTest(path=path):
                manager = self.manager()
                sender = self.sender(manager, 83, "pair")
                frames = [
                    b"ABORT",
                    b"83",
                    b"127.0.0.1",
                    b"8999",
                    b"WAIT_FOR_ROOM",
                    b"pair",
                ]
                if path == "replay":
                    manager._handle_prealloc_abort(frames)
                elif path in ("drain", "clear"):
                    manager._staging_outstanding[83] = 1
                    manager._handle_prealloc_abort(frames)
                    manager._staging_outstanding[83] = 0
                entered, release, admitted = (
                    threading.Event(),
                    threading.Event(),
                    threading.Event(),
                )
                errors = []

                def blocked_send(*args):
                    entered.set()
                    if not release.wait(5):
                        errors.append("ACK was not released")

                manager._send_prealloc_abort_ack = blocked_send

                def notify():
                    try:
                        if path == "drain":
                            manager._maybe_ack_drained_abort(83)
                        elif path == "clear":
                            sender.clear()
                        else:
                            manager._handle_prealloc_abort(frames)
                    except BaseException as error:
                        errors.append(error)

                def reserve():
                    try:
                        if not manager.reserve_prealloc_abort(84, "unrelated"):
                            errors.append("unrelated admission was rejected")
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        admitted.set()

                worker = threading.Thread(target=notify)
                admission = threading.Thread(target=reserve)
                worker.start()
                try:
                    self.assertTrue(entered.wait(3))
                    admission.start()
                    self.assertTrue(
                        admitted.wait(1), "network I/O held the replay lock"
                    )
                finally:
                    release.set()
                    worker.join(5)
                    if admission.ident is not None:
                        admission.join(5)
                self.assertFalse(worker.is_alive())
                self.assertFalse(admission.is_alive())
                self.assertFalse(errors)

    def test_retained_cancel_resolves_routes_after_initial_retry_exhaustion(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._ensure_retry_count = {"prefill:8998": 14}
        queue._ensure_last_attempt_time = {}
        queue._ensure_retry_interval = 0
        queue._max_ensure_retries = 15
        queue._prefill_dp_rank_queries = {}
        manager = NS(
            try_ensure_parallel_info=Mock(side_effect=[False, True]),
            record_failure=Mock(),
            update_status=Mock(),
            waiting_timeout=300,
            _prealloc_abort_lock=threading.RLock(),
            _prealloc_abort_ack_tracker={},
        )
        queue.kv_manager = manager
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_room = 83
        receiver.init_time = None
        receiver.conclude_state = None
        receiver.bootstrap_infos = None
        receiver.prealloc_abort_pair_key = "pair"
        receiver._send_prealloc_abort_notification = Mock()
        receiver.init = Mock(
            side_effect=lambda rank: setattr(receiver, "bootstrap_infos", [NS()])
        )
        req = NS(
            needs_expanded_input_len=True,
            bootstrap_room=83,
            bootstrap_host="prefill",
            bootstrap_port=8998,
        )
        request = NS(req=req, kv_receiver=receiver)
        ready, remaining = queue._ensure_prefill_info({"prefill:8998": [request]})
        self.assertFalse(ready)
        self.assertEqual(remaining, [request])
        self.assertTrue(receiver._prealloc_local_abort)
        queue.pending_reqs = remaining
        queue._resolve_prefill_dp_rank = Mock(return_value=0)
        queue._resolve_pending_reqs()
        receiver.init.assert_called_once_with(0)
        self.assertFalse(queue.pending_reqs)
        self.assertFalse(receiver.finish_prealloc_abort())
        receiver._send_prealloc_abort_notification.assert_called_once()
        manager._prealloc_abort_ack_tracker[(83, "pair")].add(0)
        self.assertTrue(receiver.finish_prealloc_abort())
        self.assertIsNone(receiver.init_time)

    def test_legacy_failed_route_is_not_retained_for_paired_cancellation(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._ensure_retry_count = {}
        queue._ensure_last_attempt_time = {}
        queue._ensure_retry_interval = 0
        queue._max_ensure_retries = 1
        queue.kv_manager = NS(try_ensure_parallel_info=Mock(return_value=False))
        receiver = Mock()
        request = NS(req=NS(needs_expanded_input_len=False), kv_receiver=receiver)
        ready, remaining = queue._ensure_prefill_info({"prefill:8998": [request]})
        self.assertFalse(ready or remaining)
        receiver.abort.assert_called_once()

    def test_expiry_releases_retired_credit_but_keeps_live_reservation(self):
        manager = self.manager()
        with (
            patch(
                "sglang.srt.disaggregation.mooncake.conn.EXPANDED_INPUT_MAX_INFLIGHT", 2
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.conn.envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get",
                return_value=300,
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.conn.time.monotonic",
                return_value=10,
            ),
        ):
            self.retire_failed(manager, 1)
            live = self.sender(manager, 2, "live")
            self.assertFalse(manager.reserve_prealloc_abort(3, "next"))
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=310
        ):
            self.assertTrue(manager.reserve_prealloc_abort(3, "next"))
        self.assertEqual(
            manager._prealloc_abort_reservations, {(2, "live"), (3, "next")}
        )
        live.conclude_state = KVPoll.Failed
        live.clear()
        self.assertIn((2, "live"), manager._prealloc_abort_replays)

    def test_direct_binding_cannot_bypass_capacity_or_overwrite_a_room(self):
        manager = self.manager()
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.EXPANDED_INPUT_MAX_INFLIGHT", 1
        ):
            self.sender(manager, 1, "original")
            sender = MooncakeKVSender.__new__(MooncakeKVSender)
            sender.kv_mgr, sender.bootstrap_room = manager, 2
            with self.assertRaises(KVTransferError):
                sender.set_prealloc_abort_pair_key("overflow")
            self.assertFalse(hasattr(sender, "prealloc_abort_pair_key"))
            self.assertFalse(manager.reserve_prealloc_abort(1, "replacement"))
            self.assertEqual(manager._prealloc_abort_attempts, {1: "original"})

    def test_bound_room_rejects_same_pair_but_unbound_reservation_can_publish(self):
        manager = self.manager()
        self.assertTrue(manager.reserve_prealloc_abort(83, "pair"))
        # Rechecking an admission credit before publication remains idempotent.
        self.assertTrue(manager.reserve_prealloc_abort(83, "pair"))
        first = self.sender(manager, 83, "pair")
        self.assertFalse(manager.reserve_prealloc_abort(83, "pair"))
        duplicate = MooncakeKVSender.__new__(MooncakeKVSender)
        duplicate.kv_mgr, duplicate.bootstrap_room = manager, 83
        with self.assertRaises(KVTransferError):
            duplicate.set_prealloc_abort_pair_key("pair")
        self.assertFalse(hasattr(duplicate, "prealloc_abort_pair_key"))
        self.assertEqual(manager.check_status(83), KVPoll.Bootstrapping)
        self.assertEqual(manager._prealloc_abort_attempts, {83: "pair"})
        first.conclude_state = KVPoll.Success
        first.clear()
        self.assertFalse(manager._prealloc_abort_reservations)

    def queue_and_request(self):
        queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
        queue._check_if_req_exceed_kv_capacity = Mock(return_value=False)
        queue.transfer_backend = TransferBackend.MOONCAKE
        queue.tp_rank = queue.pp_rank = 0
        queue.bootstrap_port = 8998
        queue.kv_manager = self.manager()
        queue.scheduler = NS(
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            retire_unadmitted_request=Mock(),
        )
        queue._process_req = Mock()
        req = NS(
            rid="paired",
            bootstrap_host="172.17.0.214",
            bootstrap_room=83,
            disagg_prefill_dp_rank=None,
            _expanded_input_pair_key="paired",
            return_logprob=False,
        )
        return queue, req

    def test_admission_rejects_all_ranks_before_sender_publication(self):
        # Exercise both local exhaustion and an exhausted peer with local credit.
        for local_capacity in (0, 1):
            with self.subTest(local_capacity=local_capacity):
                queue, req = self.queue_and_request()
                with (
                    envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True),
                    patch(
                        "sglang.srt.disaggregation.mooncake.conn.EXPANDED_INPUT_MAX_INFLIGHT",
                        local_capacity,
                    ),
                    patch(
                        "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                        return_value=[KVPoll.Failed],
                    ),
                    patch("sglang.srt.disaggregation.prefill.get_kv_class") as factory,
                ):
                    self.assertFalse(queue.create_sender(req, 1))
                factory.assert_not_called()
                self.assertEqual(
                    req.finished_reason.status_code, HTTPStatus.SERVICE_UNAVAILABLE
                )
                queue.scheduler.retire_unadmitted_request.assert_called_once_with(req)
                queue._process_req.assert_not_called()
                self.assertFalse(queue.kv_manager.request_status)
                self.assertFalse(queue.kv_manager._prealloc_abort_reservations)

    def test_unpaired_or_disabled_requests_do_not_use_admission_collective(self):
        for enabled, pair in ((False, "paired"), (True, None)):
            with self.subTest(enabled=enabled, pair=pair):
                queue, req = self.queue_and_request()
                req._expanded_input_pair_key = pair
                queue.kv_manager.reserve_prealloc_abort = Mock()
                with (
                    envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(enabled),
                    patch(
                        "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group"
                    ) as consensus,
                    patch(
                        "sglang.srt.disaggregation.prefill.get_kv_class",
                        return_value=Mock(),
                    ),
                ):
                    self.assertTrue(queue.create_sender(req, 1))
                consensus.assert_not_called()
                queue.kv_manager.reserve_prealloc_abort.assert_not_called()

    def test_constructor_failure_returns_unpublished_credit(self):
        queue, req = self.queue_and_request()
        with (
            envs.SGLANG_DISAGG_EXPANDED_INPUT_LEN.override(True),
            patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                return_value=[KVPoll.Success],
            ),
            patch(
                "sglang.srt.disaggregation.prefill.get_kv_class",
                return_value=Mock(side_effect=RuntimeError("constructor")),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "constructor"):
                queue.create_sender(req, 1)
        self.assertFalse(queue.kv_manager._prealloc_abort_reservations)


if __name__ == "__main__":
    unittest.main()
