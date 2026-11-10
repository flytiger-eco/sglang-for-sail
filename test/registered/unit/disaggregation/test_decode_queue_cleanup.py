import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeTransferQueue,
    HiCacheRestoreResult,
)
from sglang.srt.disaggregation.fake.conn import FakeKVManager, FakeKVReceiver
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.req_double import ReqDouble
from sglang.test.separate_buffer_allocator_double import (
    bind_separate_buffer_capacity,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class FakeReceiver:
    def __init__(self):
        self.clear_called = False
        self.conclude_state = None
        self.events = []

    def ensure_abort_notified(self):
        if "notify" not in self.events:
            self.events.append("notify")

    def clear(self):
        self.events.append("clear")
        self.clear_called = True

    def failure_exception(self):
        return None


def _pending_abort_queue(acknowledged=False):
    """Use real room/ACK bookkeeping; isolate allocation and output boundaries."""
    import threading
    import time
    from collections import defaultdict

    from sglang.srt.disaggregation.mooncake.conn import (
        MooncakeKVManager,
        MooncakeKVReceiver,
    )

    room = 83
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager._init_prealloc_abort_state()
    manager.__dict__.update(
        _prealloc_abort_ack_tracker={(room, "pair"): {0} if acknowledged else set()},
        waiting_timeout=30,
        local_ip="127.0.0.1",
        rank_port=9000,
        request_status={room: KVPoll.WaitingForInput},
        required_prefill_response_num_table={},
        prefill_response_tracker={},
        addr_to_rooms_tracker=defaultdict(set),
    )
    receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
    receiver.prealloc_abort_pair_key = "pair"
    receiver.__dict__.update(
        bootstrap_room=room,
        bootstrap_addr="127.0.0.1:8998",
        kv_mgr=manager,
        abort_notified=True,
        conclude_state=None,
        init_time=None,
        bootstrap_infos=[{"rank_ip": "127.0.0.1", "rank_port": 8998}],
        _prealloc_abort_started=time.monotonic(),
        _prealloc_abort_last_sent=time.monotonic(),
    )
    receiver._connect_to_bootstrap_server = lambda info: (
        SimpleNamespace(send_multipart=lambda frames: None),
        threading.Lock(),
    )
    req = ReqDouble(
        rid="rejected",
        needs_expanded_input_len=True,
        expanded_input_len=None,
        bootstrap_room=room,
        finished_reason=FINISH_ABORT("invalid metadata", 400),
        return_logprob=False,
    )
    item = SimpleNamespace(req=req, kv_receiver=receiver, waiting_for_input=True)
    queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
    queue.pp_size = 1
    queue.tp_rank = 0
    queue.gloo_group = None
    queue.queue = [item]
    queue.pending_reqs = []
    queue.retracted_queue = []
    queue._resolve_pending_reqs = MagicMock()
    queue._sync_expanded_input_metadata = MagicMock()
    queue._update_handshake_waiters = MagicMock()
    queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
    queue.token_to_kv_pool_allocator = MagicMock()
    bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
    queue._allocatable_token_budgets = MagicMock(return_value=0)
    queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
    queue._pre_alloc = MagicMock(
        side_effect=AssertionError("rejected request allocated")
    )
    scheduler = MagicMock()
    scheduler.running_batch.reqs = []
    scheduler.enable_priority_scheduling = scheduler.enable_hisparse = (
        scheduler.enable_lora
    ) = False
    scheduler.max_req_input_len = scheduler.max_req_len = 1024
    outputs = []
    scheduler.output_streamer.stream_output = lambda reqs, logprob: outputs.extend(
        r.rid for r in reqs
    )
    queue.scheduler = scheduler
    return queue, item, manager, outputs


def _run_tp_abort_rank(rank, init_path, result_path):
    """Two real Gloo ranks observe ACKs on different scheduler ticks."""
    import datetime
    import json
    from pathlib import Path

    import torch.distributed as dist

    from sglang.srt.disaggregation.utils import poll_and_all_reduce

    dist.init_process_group(
        "gloo",
        init_method="file://" + init_path,
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=10),
    )
    try:
        queue, item, manager, outputs = _pending_abort_queue(acknowledged=rank == 0)
        queue.gloo_group = dist.group.WORLD
        healthy = SimpleNamespace(
            req=ReqDouble(rid="healthy", finished_reason=None),
            kv_receiver=SimpleNamespace(poll=lambda: KVPoll.Bootstrapping),
            waiting_for_input=False,
        )
        queue.queue.append(healthy)
        ready, failed = queue.pop_preallocated()
        assert not ready and not failed and len(queue.queue) == 2 and not outputs
        assert manager._prealloc_abort_ack_tracker[(83, "pair")] == (
            {0} if rank == 0 else set()
        )
        _ack(manager, 83, 0)
        ready, failed = queue.pop_preallocated()
        assert (
            not ready
            and failed == [item]
            and queue.queue == [healthy]
            and outputs == ["rejected"]
        )
        assert manager._prealloc_abort_ack_tracker == {}
        # This is the next scheduler collective. A rank-local removal would
        # change the poll vector lengths and fail/hang here.
        polls = poll_and_all_reduce(
            [r.kv_receiver for r in queue.queue], dist.group.WORLD
        )
        assert polls == [KVPoll.Bootstrapping]
        Path(result_path).write_text(
            json.dumps(
                {
                    "rank": rank,
                    "remaining": [r.req.rid for r in queue.queue],
                    "outputs": outputs,
                }
            )
        )
    finally:
        dist.destroy_process_group()


def _ack(manager, room, rank, pair="pair"):
    manager._handle_prealloc_abort_ack(
        [b"PREALLOC_ABORT_ACK", str(room).encode(), pair.encode(), str(rank).encode()]
    )


class TestDecodeQueueCleanup(CustomTestCase):
    def setUp(self):
        # The code under test reads its config from the bags.
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_paged_swa_retraction_resume_uses_physical_page_budget(self):
        # resume_retracted_reqs reads the retraction backend off the disagg
        # bag, so the case publishes a config instead of injecting one.
        override = get_context().override_server_args(
            disaggregation_decode_retraction_backup="cpu_tensor"
        )
        override.install()
        self.addCleanup(override.restore)

        page_size = 128
        fill_len = 574
        physical_tokens_per_req = 5 * page_size
        physical_available = 18 * page_size

        reqs = [
            ReqDouble(
                rid=f"req-{i}",
                origin_input_ids=[0] * fill_len,
                output_ids=[],
                is_retracted=True,
                retraction_backup=None,
                load_kv_cache=MagicMock(),
            )
            for i in range(4)
        ]

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.retracted_queue = reqs.copy()
        queue.num_reserved_decode_tokens = 0
        queue.req_to_token_pool = SimpleNamespace(available_size=lambda: len(reqs))
        queue.token_to_kv_pool_allocator = MagicMock(page_size=page_size)
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.scheduler = SimpleNamespace(
            sliding_window_size=2047,
            server_args=SimpleNamespace(disable_radix_cache=True),
        )
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(physical_available, physical_available)
        )
        queue._allocatable_token_budgets = MagicMock(
            side_effect=lambda **_: physical_available
        )
        queue._swa_tail_allocatable_token_budget = MagicMock(
            side_effect=lambda **_: physical_available
        )

        def pre_alloc(_req):
            nonlocal physical_available
            self.assertGreaterEqual(physical_available, physical_tokens_per_req)
            physical_available -= physical_tokens_per_req

        queue._pre_alloc = MagicMock(side_effect=pre_alloc)

        resumed = queue.resume_retracted_reqs()

        self.assertEqual(resumed, reqs[:3])
        self.assertEqual(queue.retracted_queue, reqs[3:])
        self.assertEqual(physical_available, 3 * page_size)
        self.assertEqual(queue._pre_alloc.call_count, 3)

    def test_prealloc_abort_clears_receiver_before_removing_request(self):
        receiver = FakeReceiver()
        req = ReqDouble(
            rid="abort-prealloc",
            bootstrap_room=42,
            finished_reason=None,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req, kv_receiver=receiver, waiting_for_input=True
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.tp_rank = 0
        queue.gloo_group = object()
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        # `_uses_swa_reservation` consults the allocator once tail prealloc is
        # off, so this abort path needs one even though it never allocates.
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.metrics_reporter.enable_metrics = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ) as poll:
            preallocated, failed = queue.pop_preallocated()

        poll.assert_called_once_with([receiver], queue.gloo_group)
        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertEqual(receiver.events, ["notify", "clear"])
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_local_reject_waits_for_bootstrap_before_notifying_and_retiring(self):
        """A metadata rejection racing bootstrap must not lose its peer abort."""
        import threading
        from collections import defaultdict

        from sglang.srt.disaggregation.mooncake.conn import (
            MooncakeKVManager,
            MooncakeKVReceiver,
        )

        room = 77
        events = []

        class Socket:
            def send_multipart(self, frames):
                events.append(("wire", frames))
                prefill._handle_prealloc_abort(frames)

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._init_prealloc_abort_state()
        manager.__dict__.update(
            dict(
                enable_deferred_decode_kv_release=False,
                local_ip="127.0.0.1",
                rank_port=9000,
                request_status={room: KVPoll.Bootstrapping},
                required_prefill_response_num_table={},
                prefill_response_tracker={},
                addr_to_rooms_tracker=defaultdict(set),
                _prealloc_abort_ack_tracker={},
                waiting_timeout=30,
            )
        )
        prefill = MooncakeKVManager.__new__(MooncakeKVManager)
        prefill._init_prealloc_abort_state()
        prefill.request_status = {}
        prefill._staging_outstanding = {}
        prefill._send_prealloc_abort_ack = lambda ip, port, room, pair: _ack(
            manager, room, 0
        )
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.prealloc_abort_pair_key = "pair"
        receiver.bootstrap_room = room
        receiver.bootstrap_addr = "127.0.0.1:8998"
        receiver.kv_mgr = manager
        receiver.abort_notified = False
        receiver.conclude_state = None
        receiver.init_time = None
        receiver._connect_to_bootstrap_server = lambda info: (
            Socket(),
            threading.Lock(),
        )
        req = ReqDouble(
            rid="early-reject",
            needs_expanded_input_len=True,
            finished_reason=FINISH_ABORT("invalid metadata", 400),
            return_logprob=False,
        )
        item = SimpleNamespace(req=req, kv_receiver=receiver, waiting_for_input=True)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [item]
        queue.pending_reqs = [item]
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._sync_expanded_input_metadata = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._pre_alloc = MagicMock(
            side_effect=AssertionError("rejected room allocated")
        )
        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = scheduler.enable_hisparse = (
            scheduler.enable_lora
        ) = False

        def output(reqs, logprob):
            events.append(("output", reqs[0].finished_reason.to_json()))
            reqs[0].finished_output = True

        scheduler.output_streamer.stream_output = output
        queue.scheduler = scheduler
        ready, failed = queue.pop_preallocated()
        self.assertEqual((ready, failed), ([], []))
        self.assertEqual(queue.queue, [item])
        self.assertEqual(events, [])
        # The asynchronous bootstrap lookup finishes on the next scheduler tick.
        receiver.bootstrap_infos = [{"rank_ip": "127.0.0.1", "rank_port": 8998}]
        queue.pending_reqs.clear()
        ready, failed = queue.pop_preallocated()
        self.assertEqual((ready, failed), ([], []))
        self.assertEqual(queue.queue, [item])
        self.assertEqual(prefill.request_status, {})  # early ABORT creates no tombstone
        self.assertEqual(manager._prealloc_abort_ack_tracker[(room, "pair")], set())
        # P registers only after publishing expanded-length metadata to D.
        prefill.update_status(room, KVPoll.Bootstrapping)
        prefill._prealloc_abort_attempts[room] = "pair"
        receiver._prealloc_abort_last_sent = 0
        ready, failed = queue.pop_preallocated()
        self.assertEqual((ready, failed), ([], []))
        self.assertEqual(prefill.check_status(room), KVPoll.Failed)
        self.assertEqual(manager._prealloc_abort_ack_tracker[(room, "pair")], {0})
        ready, failed = queue.pop_preallocated()
        self.assertEqual((ready, failed), ([], [item]))
        self.assertEqual(queue.queue, [])
        self.assertIsNone(item.kv_receiver)
        self.assertEqual(manager._prealloc_abort_ack_tracker, {})
        self.assertEqual(events[0][0], "wire")
        self.assertEqual(events[0][1][:2], [b"ABORT", b"77"])
        self.assertEqual(events[0][1][4:], [b"WAIT_FOR_ROOM", b"pair"])
        self.assertEqual(events[1][0], "wire")
        self.assertEqual(events[2], ("output", req.finished_reason.to_json()))
        self.assertEqual(req.finished_reason.to_json()["status_code"], 400)
        self.assertTrue(receiver.abort_notified)
        queue._pre_alloc.assert_not_called()

    def test_prealloc_abort_waits_for_all_ranks_and_bounds_retries(self):
        import threading

        from sglang.srt.disaggregation.mooncake.conn import (
            MooncakeKVManager,
            MooncakeKVReceiver,
        )

        room = 81
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._init_prealloc_abort_state()
        manager._prealloc_abort_ack_tracker = {}
        manager.waiting_timeout = 2
        manager.local_ip, manager.rank_port = "127.0.0.1", 9000
        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.prealloc_abort_pair_key = "pair"
        receiver.kv_mgr, receiver.bootstrap_room, receiver.init_time = (
            manager,
            room,
            None,
        )
        receiver.bootstrap_infos = [{"rank_port": 1}, {"rank_port": 2}]
        sent = []

        class Socket:
            def send_multipart(self, frames):
                sent.append(frames)

        receiver._connect_to_bootstrap_server = lambda info: (
            Socket(),
            threading.Lock(),
        )
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=10
        ):
            self.assertFalse(receiver.finish_prealloc_abort())
            self.assertFalse(receiver.finish_prealloc_abort())
        self.assertEqual(len(sent), 2)  # one send per rank, no busy-loop retransmit
        _ack(manager, room, 0)
        _ack(manager, room, 0)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=10.2
        ):
            self.assertFalse(receiver.finish_prealloc_abort())
        self.assertEqual(len(sent), 4)
        _ack(manager, room, 1)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=10.3
        ):
            self.assertTrue(receiver.finish_prealloc_abort())
        self.assertEqual(manager._prealloc_abort_ack_tracker, {(room, "pair"): {0, 1}})
        # A local readiness result must keep its ACKs until TP agrees.
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=10.4
        ):
            self.assertTrue(receiver.finish_prealloc_abort())
        manager._prealloc_abort_ack_tracker.pop((room, "pair"), None)
        # An unreachable peer cannot hold a preallocation rejection forever.
        del receiver._prealloc_abort_started
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=20
        ):
            self.assertFalse(receiver.finish_prealloc_abort())
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=22
        ):
            self.assertTrue(receiver.finish_prealloc_abort())
        self.assertEqual(manager._prealloc_abort_ack_tracker, {(room, "pair"): set()})
        receiver.init_time = 22
        with self.assertRaises(AssertionError):
            receiver.finish_prealloc_abort()

    def test_prealloc_abort_ack_respects_outstanding_transfer_drain(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        room = 82
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._init_prealloc_abort_state()
        manager.request_status = {room: KVPoll.WaitingForInput}
        manager._staging_outstanding = {room: 1}
        manager._deferred_ack_targets = {}
        manager._deferred_ack_fanout_snapshots = {}
        manager.transfer_infos = {}
        acks = []
        manager._send_prealloc_abort_ack = lambda ip, port, value, pair: acks.append(
            value
        )
        manager.waiting_timeout = 30
        manager._prealloc_abort_attempts[room] = "pair"
        manager._handle_prealloc_abort(
            [b"ABORT", b"82", b"127.0.0.1", b"9000", b"WAIT_FOR_ROOM", b"pair"]
        )
        self.assertEqual(manager.check_status(room), KVPoll.Failed)
        self.assertEqual(acks, [])
        manager._staging_outstanding[room] = 0
        manager._maybe_ack_drained_abort(room)
        self.assertEqual(acks, [room])
        self.assertTrue(manager._prealloc_abort_replays[(room, "pair")].drained)

    def test_capacity_rejection_keeps_owner_until_peer_ack(self):
        """Expanded capacity rejection must not publish completion before cleanup."""
        queue, item, manager, outputs = _pending_abort_queue()
        item.req.finished_reason = None
        item.req.needs_expanded_input_len = True
        item.req.expanded_input_len = 100
        queue._prealloc_required_tokens = lambda req: (128, 128)
        queue.token_to_kv_pool_allocator.prealloc_ceiling_fits = lambda full, swa: False
        ready, failed = queue.pop_preallocated()
        self.assertEqual((ready, failed), ([], []))
        self.assertEqual(outputs, [])
        self.assertEqual(queue.queue, [item])
        self.assertEqual(item.req.finished_reason.to_json()["status_code"], 400)
        _ack(manager, 83, 0)
        ready, failed = queue.pop_preallocated()
        self.assertEqual((ready, failed), ([], [item]))
        self.assertEqual(outputs, ["rejected"])
        self.assertEqual(queue.queue, [])

    def test_rejection_deadline_covers_missing_bootstrap_route(self):
        """An empty DP-rank query must not retain a terminal request forever."""
        queue, item, manager, outputs = _pending_abort_queue()
        receiver = item.kv_receiver
        receiver.bootstrap_infos = None
        del receiver._prealloc_abort_started
        manager.waiting_timeout = 1
        queue.pending_reqs = [item]
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=10
        ):
            self.assertEqual(queue.pop_preallocated(), ([], []))
        self.assertEqual(outputs, [])
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=11
        ):
            self.assertEqual(queue.pop_preallocated(), ([], [item]))
        self.assertEqual(outputs, ["rejected"])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        self.assertEqual(manager._prealloc_abort_ack_tracker, {})

    def test_tp_ranks_retire_together_when_abort_acks_arrive_on_different_ticks(self):
        """ACK skew must preserve identical queues and subsequent collectives."""
        import json
        import subprocess
        import sys
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            script = (
                "import importlib.util,sys; "
                "s=importlib.util.spec_from_file_location('tp_abort_regression',sys.argv[1]); "
                "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m); "
                "m._run_tp_abort_rank(int(sys.argv[2]),sys.argv[3],sys.argv[4])"
            )
            children = []
            try:
                for rank in range(2):
                    children.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                "-c",
                                script,
                                __file__,
                                str(rank),
                                directory + "/store",
                                directory + "/rank%d.json" % rank,
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                        )
                    )
                for child in children:
                    log, _ = child.communicate(timeout=60)
                    self.assertEqual(child.returncode, 0, log)
                for rank in range(2):
                    result = json.loads(
                        Path(directory + "/rank%d.json" % rank).read_text()
                    )
                    self.assertEqual(result["remaining"], ["healthy"])
                    self.assertEqual(result["outputs"], ["rejected"])
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait()

    def test_prealloc_abort_also_drops_from_pending_reqs(self):
        """Local validation and timeout failures must notify P before clearing."""
        for status in (400, 500, 504):
            with self.subTest(status=status):
                self._assert_local_prealloc_abort_is_notified(status)

    def _assert_local_prealloc_abort_is_notified(self, status):
        # Same DecodeRequest lives in both queue and pending_reqs (add() slow
        # path). Aborting must drop it from both, and compare by identity since
        # DecodeRequest's dataclass __eq__ would compare the tensor receiver.
        class BadEqReceiver(FakeReceiver):
            def __eq__(self, other):
                raise TypeError("use identity comparison, not value equality")

            __hash__ = object.__hash__

        receiver = BadEqReceiver()
        req = ReqDouble(
            rid="abort-shared",
            finished_reason=FINISH_ABORT("local metadata failure", status),
            return_logprob=False,
        )
        decode_req = SimpleNamespace(req=req, kv_receiver=receiver)

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]  # same object, dual ownership
        queue.retracted_queue = []
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=False)
        # `_uses_swa_reservation` consults the allocator once tail prealloc is
        # off, so this abort path needs one even though it never allocates.
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue._allocatable_token_budgets = MagicMock(return_value=0)
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        # Must not raise on the receiver __eq__ above.
        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertTrue(all(r is not decode_req for r in queue.pending_reqs))
        self.assertIsNone(decode_req.kv_receiver)

        self.assertEqual(receiver.events, ["notify", "clear"])
        self.assertEqual(req.finished_reason.to_json()["status_code"], status)
        self.assertEqual(
            req.finished_reason.to_json()["message"], "local metadata failure"
        )

    def _swa_rejection_queue(self, expanded=True):
        queue, item, manager, outputs = _pending_abort_queue()
        item.req.finished_reason = None
        item.req.needs_expanded_input_len = expanded
        if not expanded:
            del item.kv_receiver.prealloc_abort_pair_key
            del item.kv_receiver._prealloc_abort_started
            item.kv_receiver.kv_mgr._prealloc_abort_ack_tracker.clear()
        item.req.expanded_input_len = 3
        item.req._expanded_input_len_capacity_checked = True
        item.req.origin_input_ids = [1, 2, 3]
        item.req.output_ids = []
        item.req.sampling_params = SimpleNamespace(max_new_tokens=1)
        queue.num_reserved_decode_tokens = 0
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(1024, 1024)
        )
        queue._prealloc_required_tokens = MagicMock(return_value=(3, 3))
        queue._prealloc_kv_lens = MagicMock(return_value=(3, 3))
        queue._reclaim_swa_tail_capacity = MagicMock(
            return_value="SWA eviction insufficient"
        )
        queue.req_to_token_pool = SimpleNamespace(
            available_size=lambda: 1, mamba_allocator=None
        )
        queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
            available_size=lambda: 1
        )
        queue.tree_cache = MagicMock()
        queue.scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        return queue, item, manager, outputs

    def test_swa_reclaim_rejection_waits_for_registered_peer_ack(self):
        """A SWA eviction failure must not retire an unacknowledged room."""
        queue, item, manager, outputs = self._swa_rejection_queue()
        self.assertEqual(queue.pop_preallocated(), ([], []))
        self.assertEqual(outputs, [])
        self.assertEqual(queue.queue, [item])
        self.assertIsNotNone(item.kv_receiver)
        self.assertEqual(item.req.finished_reason.to_json()["status_code"], 503)
        self.assertEqual(queue.pop_preallocated(), ([], []))
        self.assertEqual(outputs, [])
        _ack(manager, 83, 0)
        self.assertEqual(queue.pop_preallocated(), ([], [item]))
        self.assertEqual(outputs, ["rejected"])
        self.assertEqual(queue.queue, [])
        self.assertEqual(manager._prealloc_abort_ack_tracker, {})
        queue._pre_alloc.assert_not_called()

    def test_ordinary_pd_swa_rejection_returns_503_without_waiting_for_ack(self):
        queue, item, manager, outputs = self._swa_rejection_queue(expanded=False)
        receiver = item.kv_receiver
        receiver.finish_prealloc_abort = MagicMock(
            side_effect=AssertionError("legacy SWA rejection waited")
        )
        self.assertEqual(queue.pop_preallocated(), ([], [item]))
        self.assertEqual(outputs, ["rejected"])
        self.assertEqual(item.req.finished_reason.to_json()["status_code"], 503)
        receiver.finish_prealloc_abort.assert_not_called()

    def test_local_cancel_retries_until_prefill_registers_and_acks(self):
        """Scheduler cancellation must not take the peer-failure cleanup path."""
        import threading

        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        queue, item, manager, outputs = _pending_abort_queue()
        receiver = item.kv_receiver
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        item.req.finished_reason = None
        receiver.abort_notified = False
        receiver.bootstrap_infos = None
        receiver.abort()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertFalse(receiver.abort_notified)
        queue.gloo_group = None
        # Exercise the real handshake waiter: failure_exception must not clear it.
        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ):
            DecodePreallocQueue._update_handshake_waiters(queue)
        self.assertIsInstance(item.req.finished_reason, FINISH_ABORT)
        self.assertEqual(queue.pop_preallocated(), ([], []))
        self.assertEqual(outputs, [])
        prefill = MooncakeKVManager.__new__(MooncakeKVManager)
        prefill._init_prealloc_abort_state()
        prefill.request_status = {}
        prefill._staging_outstanding = {}
        prefill._send_prealloc_abort_ack = lambda ip, port, room, pair: _ack(
            manager, room, 0, pair
        )
        receiver.bootstrap_infos = [{"rank_ip": "127.0.0.1", "rank_port": 8998}]
        receiver._connect_to_bootstrap_server = lambda info: (
            SimpleNamespace(send_multipart=prefill._handle_prealloc_abort),
            threading.Lock(),
        )
        receiver._prealloc_abort_last_sent = 0
        self.assertEqual(queue.pop_preallocated(), ([], []))
        prefill.request_status[83] = KVPoll.Bootstrapping
        prefill._prealloc_abort_attempts[83] = "pair"
        receiver._prealloc_abort_last_sent = 0
        self.assertEqual(queue.pop_preallocated(), ([], []))
        self.assertEqual(prefill.check_status(83), KVPoll.Failed)
        self.assertEqual(queue.pop_preallocated(), ([], [item]))
        self.assertEqual(outputs, ["rejected"])
        self.assertEqual(manager.failure_records, {})

    def test_peer_failure_still_clears_healthy_rank_after_reduced_failed_poll(self):
        """Only a local cancel can suppress the existing cross-rank failure path."""
        import threading

        queue, item, manager, outputs = _pending_abort_queue()
        item.req.finished_reason = None
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        receiver = item.kv_receiver
        receiver.clear = MagicMock(wraps=receiver.clear)
        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ):
            DecodePreallocQueue._update_handshake_waiters(queue)
        receiver.clear.assert_called_once()
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertIsInstance(item.req.finished_reason, FINISH_ABORT)
        self.assertFalse(queue._needs_prealloc_abort_ack(item))

    def test_failed_expanded_abort_cannot_cancel_reused_prefill_room(self):
        """Exercise real D failure cleanup and P's real bootstrap dispatcher."""
        import threading

        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        queue, item, manager, outputs = _pending_abort_queue()
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        manager.enable_deferred_decode_kv_release = False
        receiver = item.kv_receiver
        receiver.abort_notified = False
        receiver.finish_prealloc_abort = MagicMock(
            side_effect=AssertionError("peer failure restarted ACK wait")
        )
        item.req.finished_reason = None
        frames = []
        receiver._connect_to_bootstrap_server = lambda info: (
            SimpleNamespace(send_multipart=lambda value: frames.append(value)),
            threading.Lock(),
        )
        with patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ):
            DecodePreallocQueue._update_handshake_waiters(queue)
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)
        self.assertEqual(queue.pop_preallocated(), ([], [item]))
        receiver.finish_prealloc_abort.assert_not_called()
        self.assertEqual(outputs, ["rejected"])
        self.assertEqual(len(frames), 1)
        self.assertEqual(manager._prealloc_abort_ack_tracker, {})

        prefill = MooncakeKVManager.__new__(MooncakeKVManager)
        prefill._init_prealloc_abort_state()
        prefill.request_status = {83: KVPoll.Bootstrapping}
        prefill._prealloc_abort_attempts[83] = "replacement-pair"
        prefill._staging_outstanding = {}
        prefill.enable_deferred_decode_kv_release = False
        prefill.server_socket = object()
        prefill._stopped = False
        prefill._worker_threads = []
        prefill._send_multipart_locked = MagicMock()
        pending = list(frames)

        def recv():
            if pending:
                return pending.pop(0)
            prefill._stopped = True
            return None

        prefill._make_worker_recv = MagicMock(return_value=recv)
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.threading.Thread"
        ) as thread:
            prefill.start_prefill_thread()
            thread.call_args.kwargs["target"]()
        self.assertEqual(prefill.check_status(83), KVPoll.Bootstrapping)
        self.assertEqual(prefill._prealloc_abort_attempts[83], "replacement-pair")
        self.assertEqual(frames[0][4:], [b"WAIT_FOR_ROOM", b"pair"])

    def test_abort_notification_preserves_published_and_partial_drain_protocol(self):
        """Pairing must not replace the drain protocol after any KV publication."""
        import threading

        for published, force_arm, paired in (
            (True, False, True),
            (False, True, True),
            (False, False, False),
        ):
            with self.subTest(published=published, force_arm=force_arm, paired=paired):
                queue, item, manager, outputs = _pending_abort_queue()
                receiver = item.kv_receiver
                manager.enable_deferred_decode_kv_release = True
                manager._deferred_abort_ack_tracker = {}
                receiver.init_time = 1 if published else None
                receiver.abort_notified = False
                if not paired:
                    del receiver.prealloc_abort_pair_key
                sent = []
                receiver._connect_to_bootstrap_server = lambda info: (
                    SimpleNamespace(send_multipart=lambda frames: sent.append(frames)),
                    threading.Lock(),
                )
                receiver.ensure_abort_notified(force_arm=force_arm)
                self.assertEqual(len(sent), 1)
                self.assertEqual(len(sent[0]), 4)
                self.assertEqual(sent[0][0], b"ABORT")
                self.assertEqual(
                    83 in manager._deferred_abort_ack_tracker, published or force_arm
                )

    def test_ordinary_pd_abort_does_not_wait_for_prealloc_ack(self):
        """Feature-off requests keep immediate notification and retirement."""
        queue, item, manager, outputs = _pending_abort_queue()
        item.req.needs_expanded_input_len = False
        receiver = item.kv_receiver
        del receiver.prealloc_abort_pair_key
        del receiver._prealloc_abort_started
        manager._prealloc_abort_ack_tracker.clear()
        receiver.finish_prealloc_abort = MagicMock(
            side_effect=AssertionError("legacy request waited for opt-in ACK")
        )
        self.assertEqual(queue.pop_preallocated(), ([], [item]))
        self.assertEqual(outputs, ["rejected"])
        receiver.finish_prealloc_abort.assert_not_called()

    def test_cleared_prefill_replays_ack_to_later_decode_ranks(self):
        """The sender can be retired between notifications from two D ranks."""
        from sglang.srt.disaggregation.mooncake.conn import (
            MooncakeKVManager,
            MooncakeKVSender,
        )

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._init_prealloc_abort_state()
        self.assertFalse(hasattr(manager, "waiting_timeout"))
        manager.request_status = {83: KVPoll.Bootstrapping}
        manager._staging_outstanding = {}
        manager._deferred_ack_targets = {}
        acks = []
        manager._send_prealloc_abort_ack = lambda ip, port, room, pair: acks.append(
            (port, pair)
        )
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr, sender.bootstrap_room = manager, 83
        sender.conclude_state = None
        sender.set_prealloc_abort_pair_key("pair")
        frames = [b"ABORT", b"83", b"127.0.0.1", b"9000", b"WAIT_FOR_ROOM", b"pair"]
        manager._handle_prealloc_abort(frames)
        sender.clear()
        self.assertNotIn(83, manager.request_status)
        frames[3] = b"9001"
        manager._handle_prealloc_abort(frames)
        manager._handle_prealloc_abort(frames)
        self.assertEqual(acks, [(9000, "pair"), (9001, "pair"), (9001, "pair")])
        self.assertEqual(manager.request_status, {})
        # Old retries only replay their old ACK and cannot abort the new sender.
        manager.request_status[83] = KVPoll.Bootstrapping
        next_sender = MooncakeKVSender.__new__(MooncakeKVSender)
        next_sender.kv_mgr, next_sender.bootstrap_room = manager, 83
        next_sender.conclude_state = None
        next_sender.set_prealloc_abort_pair_key("next-pair")
        manager._handle_prealloc_abort(frames)
        self.assertEqual(manager.check_status(83), KVPoll.Bootstrapping)
        frames[5] = b"never-registered"
        manager._handle_prealloc_abort(frames)
        self.assertEqual(manager.check_status(83), KVPoll.Bootstrapping)
        next_sender.clear()
        self.assertEqual(manager._prealloc_abort_attempts, {})

    def test_propagated_prefill_failure_keeps_ack_before_local_cancel_arrives(self):
        """P TP reduction can retire a rank before its own WAIT_FOR_ROOM arrives."""
        import threading

        from sglang.srt.disaggregation.common.conn import KVTransferError
        from sglang.srt.disaggregation.mooncake.conn import (
            MooncakeKVManager,
            MooncakeKVSender,
        )

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._init_prealloc_abort_state()
        manager.request_status = {83: KVPoll.Bootstrapping}
        manager._staging_outstanding = {}
        manager._deferred_ack_targets = {}
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        acks = []
        manager._send_prealloc_abort_ack = lambda ip, port, room, pair: acks.append(
            (port, pair)
        )
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr, sender.bootstrap_room, sender.conclude_state = manager, 83, None
        sender.set_prealloc_abort_pair_key("pair")
        # The actual failure mixin latches Failed, clears and reports propagation.
        with self.assertRaises(KVTransferError):
            sender.failure_exception()
        self.assertNotIn(83, manager.request_status)
        manager._handle_prealloc_abort(
            [b"ABORT", b"83", b"127.0.0.1", b"9000", b"WAIT_FOR_ROOM", b"pair"]
        )
        self.assertEqual(acks, [(9000, "pair")])
        manager.request_status[83] = KVPoll.Bootstrapping
        manager._prealloc_abort_attempts[83] = "next-pair"
        manager._handle_prealloc_abort(
            [b"ABORT", b"83", b"127.0.0.1", b"9001", b"WAIT_FOR_ROOM", b"pair"]
        )
        self.assertEqual(manager.check_status(83), KVPoll.Bootstrapping)
        self.assertEqual(acks[-1], (9001, "pair"))

    def test_prealloc_ack_survives_sender_clear_until_pending_writes_drain(self):
        from sglang.srt.disaggregation.mooncake.conn import (
            MooncakeKVManager,
            MooncakeKVSender,
        )

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._init_prealloc_abort_state()
        self.assertFalse(hasattr(manager, "waiting_timeout"))
        manager.request_status = {83: KVPoll.WaitingForInput}
        manager._staging_outstanding = {83: 1}
        manager._deferred_ack_targets = {}
        acks = []
        manager._send_prealloc_abort_ack = lambda ip, port, room, pair: acks.append(
            (port, pair)
        )
        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.kv_mgr, sender.bootstrap_room = manager, 83
        sender.conclude_state = None
        sender.set_prealloc_abort_pair_key("pair")
        frames = [b"ABORT", b"83", b"127.0.0.1", b"9000", b"WAIT_FOR_ROOM", b"pair"]
        manager._handle_prealloc_abort(frames)
        sender.clear()
        self.assertEqual(manager._prealloc_abort_attempts, {})
        frames[3] = b"9001"
        manager._handle_prealloc_abort(frames)
        self.assertEqual(acks, [])
        manager._staging_outstanding[83] = 0
        # The owner is retired; a Decode retry observes the drained replay.
        manager._handle_prealloc_abort(frames)
        self.assertEqual(set(acks), {(9000, "pair"), (9001, "pair")})
        self.assertEqual(manager._prealloc_abort_attempts, {})

    def test_old_or_unscoped_ack_cannot_complete_new_attempt(self):
        queue, item, manager, outputs = _pending_abort_queue()
        receiver = item.kv_receiver
        manager._deferred_abort_ack_tracker = {}
        receiver.prealloc_abort_pair_key = "next-pair"
        receiver._prealloc_abort_last_sent = 0
        self.assertFalse(receiver.finish_prealloc_abort())
        _ack(manager, 83, 0, "pair")
        manager.note_abort_ack(83, 0)  # Legacy drain ACK is a different protocol.
        self.assertFalse(receiver.finish_prealloc_abort())
        _ack(manager, 83, 0, "next-pair")
        self.assertTrue(receiver.finish_prealloc_abort())

    def test_prealloc_abort_replay_expires_and_does_not_create_unknown_rooms(self):
        from sglang.srt.disaggregation.expanded_input import EXPANDED_INPUT_MAX_INFLIGHT
        from sglang.srt.disaggregation.mooncake.conn import (
            MooncakeKVManager,
            _PreallocAbortReplay,
        )

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager._init_prealloc_abort_state()
        manager.request_status = {83: KVPoll.Bootstrapping}
        manager._prealloc_abort_attempts[83] = "pair"
        manager._staging_outstanding = {}
        manager.waiting_timeout = 1
        acks = []
        manager._send_prealloc_abort_ack = lambda *args: acks.append(args)
        frames = [b"ABORT", b"83", b"127.0.0.1", b"9000", b"WAIT_FOR_ROOM", b"pair"]
        with (
            patch(
                "sglang.srt.disaggregation.mooncake.conn.time.monotonic",
                return_value=10,
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.conn.envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get",
                return_value=1,
            ),
        ):
            manager._handle_prealloc_abort(frames)
        self.assertEqual(len(acks), 1)
        manager.request_status.clear()
        with patch(
            "sglang.srt.disaggregation.mooncake.conn.time.monotonic", return_value=11
        ):
            manager._handle_prealloc_abort(frames)
        self.assertEqual(len(acks), 1)
        self.assertFalse(manager._prealloc_abort_replays)
        self.assertFalse(manager._prealloc_abort_attempts)
        self.assertFalse(manager.request_status)
        for room in range(EXPANDED_INPUT_MAX_INFLIGHT):
            manager._prealloc_abort_replays[(room, "old")] = _PreallocAbortReplay(
                float("inf"), drained=True
            )
        manager.request_status[83] = KVPoll.Bootstrapping
        manager._prealloc_abort_attempts[83] = "pair"
        manager._handle_prealloc_abort(frames)
        self.assertEqual(
            len(manager._prealloc_abort_replays), EXPANDED_INPUT_MAX_INFLIGHT
        )
        self.assertEqual(manager.check_status(83), KVPoll.Bootstrapping)

    def test_healthy_prealloc_queue_does_not_require_abort_collective(self):
        """Healthy scheduler ticks must not wait for an extra TP collective."""
        queue, item, manager, outputs = _pending_abort_queue()
        item.req.finished_reason = None
        item.waiting_for_input = False
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_reduce",
                side_effect=AssertionError("healthy queue performed abort collective"),
            ),
        ):
            self.assertEqual(queue.pop_preallocated(), ([], []))
        self.assertEqual(queue.queue, [item])
        self.assertEqual(outputs, [])

    def test_swa_reclaim_failure_rejects_only_request(self):
        receiver = FakeReceiver()
        req = ReqDouble(
            rid="swa-reclaim-failed",
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            finished_reason=None,
            return_logprob=False,
            sampling_params=SimpleNamespace(max_new_tokens=1),
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            waiting_for_input=True,
            is_rebootstrap=False,
        )

        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pp_size = 1
        queue.queue = [decode_req]
        queue.pending_reqs = [decode_req]
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._uses_swa_tail_prealloc = MagicMock(return_value=True)
        queue._swa_aware_allocatable_token_budgets = MagicMock(
            return_value=(1024, 1024)
        )
        queue._prealloc_required_tokens = MagicMock(return_value=(3, 3))
        queue._prealloc_kv_lens = MagicMock(return_value=(3, 3))
        queue._reclaim_swa_tail_capacity = MagicMock(
            return_value=(
                "SWA eviction insufficient: needed=64, available=0, "
                "req=swa-reclaim-failed"
            )
        )
        queue._hicache_pending_restore_tokens = MagicMock(return_value=0)
        queue._pre_alloc = MagicMock()
        queue.token_to_kv_pool_allocator = MagicMock()
        bind_separate_buffer_capacity(queue.token_to_kv_pool_allocator)
        queue.tree_cache = MagicMock()
        queue.req_to_token_pool = MagicMock()
        queue.req_to_token_pool.available_size.return_value = 1
        # Non-hybrid pools have no mamba allocator; MagicMock would otherwise
        # auto-create one and break the `available_size() <= 0` comparison in
        # pop_preallocated.
        queue.req_to_token_pool.mamba_allocator = None
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator.available_size.return_value = 1

        scheduler = MagicMock()
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hisparse = False
        scheduler.enable_lora = False
        scheduler.server_args.disaggregation_decode_enable_radix_cache = False
        scheduler.output_streamer = MagicMock()
        queue.scheduler = scheduler

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [])
        self.assertEqual(failed, [decode_req])
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.pending_reqs, [])
        self.assertTrue(receiver.clear_called)
        self.assertEqual(receiver.events, ["notify", "clear"])
        self.assertIsNone(decode_req.kv_receiver)
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        queue._pre_alloc.assert_not_called()
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )

    def test_ensure_prefill_info_tolerates_cleared_receiver(self):
        # A req whose kv_receiver was already cleared must not crash on .abort().
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue._max_ensure_retries = 1
        queue._ensure_retry_interval = 0
        queue._ensure_retry_count = {"127.0.0.1:11500": 0}
        queue._ensure_last_attempt_time = {}
        queue.kv_manager = MagicMock()
        queue.kv_manager.try_ensure_parallel_info.return_value = False

        cleared_req = SimpleNamespace(
            req=SimpleNamespace(rid="cleared"), kv_receiver=None
        )
        addr_to_reqs = {"127.0.0.1:11500": [cleared_req]}

        ready, remaining = queue._ensure_prefill_info(addr_to_reqs)

        self.assertEqual(ready, {})
        self.assertEqual(remaining, [])

    def test_prefetches_prefill_dp_rank_query(self):
        addr = "127.0.0.1:11500"
        executor = MagicMock()
        future = Future()
        future.set_result({"7": 1})
        executor.submit.return_value = future

        def decode_req(room):
            return SimpleNamespace(
                req=SimpleNamespace(
                    bootstrap_host="127.0.0.1",
                    bootstrap_port=11500,
                    bootstrap_room=room,
                ),
                kv_receiver=MagicMock(),
            )

        first = decode_req(7)
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.pending_reqs = [first]
        queue._prefill_dp_rank_queries = {}
        queue.kv_manager = SimpleNamespace(
            prefill_info_table={addr: object()},
            _ensure_prefill_recompute_executor=lambda: executor,
        )
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        queue._ensure_prefill_info = lambda groups: (groups, [])

        queue.prefetch_prefill_dp_rank_queries()
        tail = decode_req(8)
        queue.pending_reqs.append(tail)
        with patch(
            "sglang.srt.disaggregation.decode.CommonKVReceiver.query_prefill_dp_ranks",
            return_value={"8": 2},
        ) as query:
            queue._resolve_pending_reqs()

        _, called_addr, called_rooms = executor.submit.call_args.args
        self.assertEqual((called_addr, called_rooms), (addr, [7]))
        query.assert_called_once_with(addr, [8])
        first.kv_receiver.init.assert_called_once_with(1)
        tail.kv_receiver.init.assert_called_once_with(2)
        self.assertEqual(queue.pending_reqs, [])

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_transfer_failure_cleanup_respects_deferred_release_gates(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        receiver = FakeReceiver()
        req = ReqDouble(
            rid="failed-transfer",
            bootstrap_room=7,
            return_logprob=False,
        )
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_deferred_kv_release = False
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()

        scheduler = MagicMock()
        scheduler.enable_decode_hicache = False
        scheduler.enable_hisparse = False
        scheduler.output_streamer = MagicMock()
        scheduler.metrics_reporter.enable_metrics = False
        queue.scheduler = scheduler

        mock_poll.return_value = [KVPoll.Failed]

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        scheduler.output_streamer.stream_output.assert_called_once_with(
            [req], req.return_logprob
        )
        mock_prepare_abort.assert_called_once()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

        receiver = FakeReceiver()
        receiver.kv_mgr = FakeKVManager.__new__(FakeKVManager)
        decode_req.kv_receiver = receiver
        queue.queue = [decode_req]
        queue.enable_deferred_kv_release = True
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()

        transferred = queue.pop_transferred()

        self.assertEqual(transferred, [])
        self.assertEqual(queue.queue, [])
        self.assertTrue(receiver.clear_called)
        self.assertIsNone(decode_req.kv_receiver)
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

        receiver = MagicMock()
        decode_req.kv_receiver = receiver
        decode_req.host_staged = True
        queue.enable_host_receive = True
        queue.enable_deferred_kv_release = False
        queue._defer_release = MagicMock()
        queue.queue = [decode_req]
        queue.req_to_metadata_buffer_idx_allocator.reset_mock()
        mock_release_kv_cache.reset_mock()
        self.assertEqual(queue.pop_transferred(), [])
        receiver.abort.assert_called_once_with()
        queue._defer_release.assert_called_once_with(decode_req)
        receiver.clear.assert_not_called()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        mock_release_kv_cache.assert_not_called()

        queue.queue = [decode_req]
        decode_req.req.finished_reason = FINISH_ABORT("cancelled")
        with (
            patch.object(
                queue, "_poll_with_metadata_gate", return_value=[KVPoll.Success]
            ),
            patch(
                "sglang.srt.disaggregation.decode.discard_kv_cache_backup"
            ) as discard,
        ):
            self.assertEqual(queue.pop_transferred(), [])
            discard.assert_called_once_with(req, queue.tree_cache, "host_pool")
        receiver.clear.assert_called_once_with()
        queue.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        mock_release_kv_cache.assert_not_called()

    def test_fake_receiver_initializes_deferred_release_state(self):
        manager = MagicMock()
        receiver = FakeKVReceiver(manager, "")

        self.assertIs(receiver.kv_mgr, manager)
        self.assertFalse(receiver.abort_notified)

    @patch("sglang.srt.disaggregation.decode.release_kv_cache")
    @patch("sglang.srt.disaggregation.decode.prepare_abort")
    @patch("sglang.srt.disaggregation.decode.poll_and_all_reduce")
    def test_failed_fake_transfer_releases_at_once_under_deferred_release(
        self, mock_poll, mock_prepare_abort, mock_release_kv_cache
    ):
        # A health-check request gets a FakeKVReceiver over the real transfer
        # manager, so a failed one reaches the deferred-release path.
        receiver = FakeKVReceiver(MagicMock(enable_deferred_decode_kv_release=True), "")
        req = ReqDouble(rid="health-check", bootstrap_room=0, return_logprob=False)
        decode_req = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=3,
            hicache_restore_status=HiCacheRestoreResult.READY,
        )

        queue = DecodeTransferQueue.__new__(DecodeTransferQueue)
        queue.queue = [decode_req]
        queue.enable_staging = False
        queue.enable_host_receive = False
        queue.enable_deferred_kv_release = True
        queue._defer_release = MagicMock()
        queue.gloo_group = MagicMock()
        queue.req_to_metadata_buffer_idx_allocator = MagicMock()
        queue.tp_rank = 0
        queue.tree_cache = MagicMock()
        queue.metadata_buffers = SimpleNamespace(bootstrap_room=[None] * 4)
        queue.spec_algorithm = MagicMock()
        queue.spec_algorithm.is_none.return_value = True
        queue._clean_hicache_prefetch_resources = MagicMock()
        queue.scheduler = MagicMock(enable_decode_hicache=False, enable_hisparse=False)
        queue.scheduler.metrics_reporter.enable_metrics = False
        mock_poll.return_value = [KVPoll.Failed]

        self.assertEqual(queue.pop_transferred(), [])
        self.assertFalse(receiver.abort_notified)
        queue._defer_release.assert_not_called()
        mock_release_kv_cache.assert_called_once_with(
            req, queue.tree_cache, is_insert=False
        )

    def test_retracted_decode_requests_keep_scheduler_non_idle(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.cur_batch_for_debug = None
        scheduler.enable_overlap = False
        scheduler.running_mbs = []
        scheduler.waiting_queue = []
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.DECODE
        scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
            queue=[], retracted_queue=[object()]
        )
        scheduler.disagg_decode_transfer_queue = SimpleNamespace(queue=[])
        scheduler.decode_offload_manager = None
        scheduler.enable_hisparse = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_lmcache = False

        self.assertFalse(scheduler.is_fully_idle())


if __name__ == "__main__":
    unittest.main()
