"""CPU regressions for Mooncake rejection and terminal-status races."""

import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferInfo,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_manager():
    # Exercise the production lifecycle without initializing a transfer engine.
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.failure_lock = threading.Lock()
    manager._pending_status_notifications = {}
    manager.failure_records = {}
    manager.request_status = {1: KVPoll.Transferring}
    manager.transfer_infos = {}
    manager.req_to_decode_prefix_len = {}
    manager.decode_kv_args_table = {}
    manager._dcp_pack_max_tokens = None
    manager._socket_send_locks = defaultdict(threading.Lock)
    manager.attn_tp_rank = manager.pp_rank = manager.attn_cp_rank = 0
    manager.pp_size = manager.attn_cp_size = 1
    manager.kv_status_msg_tag = None
    manager.kv_status_msg_carries_reason = True
    manager.required_prefill_response_num_table = {1: 1}
    manager.prefill_response_tracker = defaultdict(set)
    manager.addr_to_rooms_tracker = defaultdict(set)
    manager.enable_staging = False
    manager.is_mla_backend = False
    manager.is_dummy_cp_rank = False
    manager.messages = []
    manager._connect = Mock(
        return_value=SimpleNamespace(send_multipart=manager.messages.append)
    )
    return manager


def make_sender(manager, room=1):
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    sender.kv_mgr = manager
    sender.bootstrap_room = room
    return sender


def rejection(room=1, *, session="peer", port=1234, dummy=False, required=1):
    return TransferInfo(
        room=room,
        endpoint="127.0.0.1",
        dst_port=port,
        mooncake_session_id=session,
        dst_kv_indices=None,
        dst_aux_index=None,
        dst_state_indices=[],
        required_dst_info_num=required,
        is_dummy=dummy,
    )


class TestMooncakeRequestLifecycle(CustomTestCase):
    def test_failure_cannot_be_overwritten_by_a_paused_success_update(self):
        manager = make_manager()
        reading, release = threading.Event(), threading.Event()
        failing, finished = threading.Event(), threading.Event()
        errors = []

        class PausedStatus(dict):
            def get(self, key, default=None):
                value = super().get(key, default)
                if threading.current_thread().name == "success-update":
                    reading.set()
                    if not release.wait(5):
                        raise TimeoutError("Success update was not released")
                return value

        manager.request_status = PausedStatus(manager.request_status)

        def update(status):
            try:
                if status == KVPoll.Failed:
                    failing.set()
                manager.update_status(1, status)
            except BaseException as exc:
                errors.append(exc)
            finally:
                if status == KVPoll.Failed:
                    finished.set()

        success = threading.Thread(
            target=update, args=(KVPoll.Success,), name="success-update", daemon=True
        )
        failure = threading.Thread(target=update, args=(KVPoll.Failed,), daemon=True)
        success.start()
        try:
            self.assertTrue(reading.wait(5), errors)
            failure.start()
            self.assertTrue(failing.wait(5), errors)
            # The read/check/write operation must hold the status lock.
            self.assertFalse(finished.wait(0.1))
        finally:
            release.set()
            success.join(5)
            if failure.ident is not None:
                failure.join(5)
        self.assertFalse(success.is_alive())
        self.assertFalse(failure.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(manager.check_status(1), KVPoll.Failed)

    def test_rejection_notification_prevents_late_success_notification(self):
        self._assert_rejection_before_notification(clear=False)

    def test_rejection_and_clear_preserve_pending_failure_notification(self):
        self._assert_rejection_before_notification(clear=True)

    def _assert_rejection_before_notification(self, *, clear):
        manager = make_manager()
        req = rejection()
        manager.transfer_infos[1] = {req.mooncake_session_id: req}
        connecting, release = threading.Event(), threading.Event()
        errors = []
        socket = manager._connect.return_value

        def connect(*args, **kwargs):
            if threading.current_thread().name == "success-notification":
                connecting.set()
                if not release.wait(5):
                    raise TimeoutError("Success notification was not released")
            return socket

        manager._connect.side_effect = connect

        def complete():
            try:
                manager.conclude_transfer(
                    bootstrap_room=1,
                    status=KVPoll.Success,
                    targets=[("127.0.0.1", 1234)],
                )
            except BaseException as exc:
                errors.append(exc)

        success = threading.Thread(
            target=complete, name="success-notification", daemon=True
        )
        success.start()
        try:
            self.assertTrue(connecting.wait(5), errors)
            manager._fail_bootstrap_request(req, "invalid registration")
            self.assertEqual([int(msg[1]) for msg in manager.messages], [KVPoll.Failed])
            if clear:
                make_sender(manager).clear()
                # failure_exception consumes the reason after clearing the room.
                with manager.failure_lock:
                    manager.failure_records.pop(1)
        finally:
            release.set()
            success.join(5)
        self.assertFalse(success.is_alive())
        self.assertEqual(errors, [])
        if clear:
            self.assertNotIn(1, manager.request_status)
        else:
            self.assertEqual(manager.check_status(1), KVPoll.Failed)
        self.assertEqual(
            [int(msg[1]) for msg in manager.messages], [KVPoll.Failed, KVPoll.Failed]
        )
        self.assertTrue(
            all(msg[3] == b"invalid registration" for msg in manager.messages)
        )
        self.assertEqual(manager._pending_status_notifications, {})

    def test_identical_pending_notifications_are_removed_by_identity(self):
        manager = make_manager()
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        errors = []
        socket = manager._connect.return_value

        def connect(*args, **kwargs):
            index = int(threading.current_thread().name[-1])
            entered[index].set()
            if not release[index].wait(5):
                raise TimeoutError("Notification was not released")
            return socket

        manager._connect.side_effect = connect

        def complete():
            try:
                manager.conclude_transfer(
                    bootstrap_room=1,
                    status=KVPoll.Success,
                    targets=[("127.0.0.1", 1234)],
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=complete, name=f"notification-{i}", daemon=True)
            for i in range(2)
        ]
        threads[0].start()
        try:
            self.assertTrue(entered[0].wait(5), errors)
            with manager.failure_lock:
                first = manager._pending_status_notifications[1][0]
            threads[1].start()
            self.assertTrue(entered[1].wait(5), errors)
            release[1].set()
            threads[1].join(5)
            self.assertFalse(threads[1].is_alive())
            with manager.failure_lock:
                self.assertEqual(len(manager._pending_status_notifications[1]), 1)
                self.assertIs(manager._pending_status_notifications[1][0], first)
            manager.record_failure(1, "failed while first notification waited")
            manager.update_status(1, KVPoll.Failed)
        finally:
            for event in release:
                event.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(5)
        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(
            [int(msg[1]) for msg in manager.messages], [KVPoll.Success, KVPoll.Failed]
        )
        self.assertEqual(manager._pending_status_notifications, {})

    def test_clear_preserves_already_committed_notification(self):
        for status in (KVPoll.Success, KVPoll.Failed):
            with self.subTest(status=status):
                manager = make_manager()
                sender = make_sender(manager)
                socket = manager._connect.return_value

                def clear_before_send(*args, **kwargs):
                    sender.clear()
                    return socket

                manager._connect.side_effect = clear_before_send
                result = manager.conclude_transfer(
                    bootstrap_room=1,
                    status=status,
                    targets=[("127.0.0.1", 1234)],
                )
                self.assertEqual(result, status)
                self.assertEqual([int(msg[1]) for msg in manager.messages], [status])
                self.assertNotIn(1, manager.request_status)
                self.assertEqual(manager._pending_status_notifications, {})

    def test_clear_drops_late_completion(self):
        manager = make_manager()
        sender = make_sender(manager)
        sender.clear()
        for status in (KVPoll.Success, KVPoll.Failed):
            self.assertIsNone(
                manager.conclude_transfer(
                    bootstrap_room=1,
                    status=status,
                    targets=[("127.0.0.1", 1234)],
                )
            )
        self.assertEqual(manager.request_status, {})
        self.assertEqual(manager.failure_records, {})
        self.assertEqual(manager.messages, [])

    def test_early_rejection_is_retained_until_sender_creation(self):
        manager = make_manager()
        manager.request_status.clear()
        manager.is_dummy_cp_rank = False
        manager._reject_bootstrap_request(rejection(), "invalid registration")
        self.assertEqual(manager.request_status, {})
        self.assertEqual(manager.failure_records, {1: "invalid registration"})
        self.assertEqual([int(msg[1]) for msg in manager.messages], [KVPoll.Failed])
        with get_context().override_server_args(dp_size=1):
            with patch.object(MooncakeKVSender, "_init_trace_ctx"):
                sender = MooncakeKVSender(manager, "127.0.0.1:8000", 1, [0], 0)
        self.assertEqual(manager.check_status(sender.bootstrap_room), KVPoll.Failed)

    def test_cleared_request_ignores_late_registration_failure(self):
        manager = make_manager()
        req = rejection()
        manager.transfer_infos[1] = {req.mooncake_session_id: req}
        manager._fail_bootstrap_request(req, "invalid registration")
        make_sender(manager).clear()
        with manager.failure_lock:
            manager.failure_records.pop(1)
        manager.messages.clear()

        # The worker retained req before failure_exception consumed this room.
        manager._fail_bootstrap_request(req, "late registration failure")
        self.assertEqual(manager.failure_records, {})
        self.assertEqual(manager.request_status, {})
        self.assertEqual(manager.messages, [])
        self.assertEqual(manager._pending_status_notifications, {})
        with get_context().override_server_args(dp_size=1):
            with patch.object(MooncakeKVSender, "_init_trace_ctx"):
                MooncakeKVSender(manager, "127.0.0.1:8000", 1, [0], 0)
        self.assertEqual(manager.check_status(1), KVPoll.Bootstrapping)

    def test_replaced_request_ignores_old_snapshot_with_same_identifiers(self):
        manager = make_manager()
        old_req = rejection()
        replacement = rejection()
        self.assertIsNot(old_req, replacement)
        manager.transfer_infos[1] = {replacement.mooncake_session_id: replacement}
        manager._fail_bootstrap_request(old_req, "obsolete request failed")
        self.assertEqual(manager.failure_records, {})
        self.assertEqual(manager.check_status(1), KVPoll.Transferring)
        self.assertEqual(manager.messages, [])
        self.assertIs(manager.transfer_infos[1]["peer"], replacement)

    def test_registered_failure_notifies_all_real_peers_after_clear(self):
        manager = make_manager()
        reqs = [
            rejection(session="rejected", port=1234),
            rejection(session="sibling", port=1235),
            rejection(session="same-endpoint", port=1235),
            rejection(session="dummy", port=1236, dummy=True),
        ]
        manager.transfer_infos[1] = {req.mooncake_session_id: req for req in reqs}
        delivered = []

        def connect(endpoint, *, is_ipv6):
            self.assertFalse(is_ipv6)
            # The scheduler may clear as soon as Failed becomes observable.
            make_sender(manager).clear()
            with manager.failure_lock:
                manager.failure_records.pop(1, None)
            return SimpleNamespace(
                send_multipart=lambda msg: delivered.append((endpoint, msg))
            )

        manager._connect.side_effect = connect
        manager._fail_bootstrap_request(reqs[0], "registration rejected")
        self.assertEqual(
            [endpoint for endpoint, _ in delivered],
            ["tcp://127.0.0.1:1234", "tcp://127.0.0.1:1235"],
        )
        self.assertTrue(all(int(msg[1]) == KVPoll.Failed for _, msg in delivered))
        self.assertTrue(all(msg[3] == b"registration rejected" for _, msg in delivered))
        self.assertEqual(manager.request_status, {})
        self.assertEqual(manager.failure_records, {})
        self.assertEqual(manager._pending_status_notifications, {})

    def test_dummy_failure_still_notifies_real_siblings(self):
        manager = make_manager()
        dummy = rejection(session="dummy", port=1236, dummy=True)
        sibling = rejection(session="real", port=1235)
        manager.transfer_infos[1] = {"dummy": dummy, "real": sibling}
        manager._fail_bootstrap_request(dummy, "dummy registration rejected")
        manager._connect.assert_called_once_with("tcp://127.0.0.1:1235", is_ipv6=False)
        self.assertEqual([int(msg[1]) for msg in manager.messages], [KVPoll.Failed])
        self.assertEqual(manager.check_status(1), KVPoll.Failed)
        self.assertEqual(manager._pending_status_notifications, {})

    def test_incoming_rejection_notifies_registered_and_unregistered_peers(self):
        manager = make_manager()
        sibling = rejection(session="sibling", port=1235)
        manager.transfer_infos[1] = {"sibling": sibling}
        manager._reject_bootstrap_request(rejection(), "incoming peer rejected")
        self.assertEqual(
            manager._connect.call_args_list,
            [
                call("tcp://127.0.0.1:1235", is_ipv6=False),
                call("tcp://127.0.0.1:1234", is_ipv6=False),
            ],
        )
        self.assertEqual(
            [int(msg[1]) for msg in manager.messages], [KVPoll.Failed, KVPoll.Failed]
        )
        self.assertEqual(manager._pending_status_notifications, {})

    def test_presender_rejection_also_rejects_later_peer(self):
        manager = make_manager()
        manager.request_status.clear()
        manager._reject_bootstrap_request(rejection(), "first peer rejected")
        manager._connect.reset_mock()
        manager.messages.clear()
        manager._register_bootstrap_request(rejection(session="late-peer", port=1235))
        manager._connect.assert_called_once_with("tcp://127.0.0.1:1235", is_ipv6=False)
        self.assertEqual([int(msg[1]) for msg in manager.messages], [KVPoll.Failed])
        self.assertEqual(manager.messages[0][3], b"first peer rejected")
        self.assertEqual(manager.request_status, {})
        self.assertEqual(manager.transfer_infos, {})
        self.assertEqual(manager.failure_records, {1: "first peer rejected"})
        self.assertEqual(manager._pending_status_notifications, {})

    def test_bootstrap_registration_preserves_completion_and_prefix(self):
        manager = make_manager()
        manager.request_status[1] = KVPoll.Bootstrapping
        first = rejection(session="first", required=2)
        second = rejection(session="second", port=1235, required=2)
        second.decode_prefix_len = 64
        manager._register_bootstrap_request(first)
        self.assertEqual(manager.check_status(1), KVPoll.Bootstrapping)
        manager._register_bootstrap_request(second)
        self.assertEqual(manager.check_status(1), KVPoll.WaitingForInput)
        self.assertEqual(manager.req_to_decode_prefix_len[1], 64)
        self.assertEqual(manager.failure_records, {})
        self.assertEqual(manager.messages, [])

    def test_receiver_failure_is_terminal_and_clear_drops_late_status(self):
        manager = make_manager()
        manager.apply_prefill_status(
            bootstrap_room=1,
            status=KVPoll.Failed,
            prefill_rank=0,
            failure_reason="rejected",
        )
        manager.apply_prefill_status(
            bootstrap_room=1, status=KVPoll.Success, prefill_rank=0
        )
        self.assertEqual(manager.check_status(1), KVPoll.Failed)
        self.assertNotIn(1, manager.prefill_response_tracker)
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.kv_mgr = manager
        receiver.bootstrap_room = 1
        receiver.bootstrap_addr = "127.0.0.1:8000"
        receiver.clear()
        manager.apply_prefill_status(
            bootstrap_room=1, status=KVPoll.Success, prefill_rank=0
        )
        self.assertNotIn(1, manager.request_status)
        self.assertNotIn(1, manager.prefill_response_tracker)
        self.assertEqual(manager.failure_records, {1: "rejected"})

    def test_missing_registration_preserves_pack_limit(self):
        manager = make_manager()
        manager._dcp_pack_max_tokens = 64
        manager.transfer_infos[1] = {"rejected-peer": SimpleNamespace(is_dummy=False)}
        sender = make_sender(manager)
        # Registration rejection removes the peer before its queued room drains.
        manager.update_status(1, KVPoll.Failed)
        self.assertEqual(sender.get_max_transfer_tokens(), 64)
        self.assertEqual(manager.check_status(1), KVPoll.Failed)

    def test_pack_limit_only_for_active_relayout_peers(self):
        manager = make_manager()
        sender = make_sender(manager)
        manager.transfer_infos[1] = {"peer": SimpleNamespace(is_dummy=False)}
        self.assertIsNone(sender.get_max_transfer_tokens())
        manager._dcp_pack_max_tokens = 64
        for relayout in (False, True):
            with self.subTest(relayout=relayout):
                manager.decode_kv_args_table["peer"] = SimpleNamespace(
                    requires_dcp_relayout=relayout
                )
                self.assertEqual(
                    sender.get_max_transfer_tokens(), 64 if relayout else None
                )
        manager.decode_kv_args_table.clear()
        manager.transfer_infos[1]["peer"].is_dummy = True
        self.assertIsNone(sender.get_max_transfer_tokens())
        manager.transfer_infos.clear()
        self.assertIsNone(sender.get_max_transfer_tokens())


if __name__ == "__main__":
    unittest.main()
