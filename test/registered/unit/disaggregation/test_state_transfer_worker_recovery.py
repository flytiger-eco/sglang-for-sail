"""A real state-layout failure must leave the worker able to drain its queue."""

import threading
import unittest
from collections import defaultdict
from copy import copy
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from sglang.srt.disaggregation.base.conn import KVPoll, StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestStateTransferWorkerRecovery(CustomTestCase):
    def test_layout_error_fails_one_room_and_drains_next_on_same_session(self):
        self._assert_failure_is_local("runtime")

    def test_mamba_value_error_does_not_kill_the_transfer_worker(self):
        self._assert_failure_is_local("value")

    def _assert_failure_is_local(self, error_kind):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.enable_trace = manager.enable_staging = False
        manager.enable_deferred_decode_kv_release = False
        manager._staging_outstanding = defaultdict(int)
        manager.request_status = {101: KVPoll.Transferring, 102: KVPoll.Transferring}
        manager.check_status = lambda room: manager.request_status[room]
        manager.update_status = lambda room, value: manager.request_status.__setitem__(
            room, value
        )
        manager.failure_lock = threading.Lock()
        manager.failure_records = {}
        manager.session_lock = threading.Lock()
        manager.failed_sessions = set()
        manager.session_failures = defaultdict(int)
        manager.req_to_decode_prefix_len = {101: 0, 102: 0}
        manager._prefill_unique_rank = lambda: 0
        manager._mamba_state_cp_sharded = False
        manager.attn_tp_size = manager.pp_size = 1
        manager.kv_args = SimpleNamespace(
            kv_data_ptrs=[],
            state_types=[StateType.QSA_PENDING],
            state_data_ptrs=[[1]],
            state_item_lens=[[8]],
            state_dim_per_tensor=[[]],
            state_layer_ids=[[0]],
            engine_rank=0,
        )
        peer = SimpleNamespace(
            requires_dcp_relayout=False,
            dst_attn_tp_size=1,
            dst_tp_rank=0,
            dst_state_data_ptrs=[[2]],
            dst_state_item_lens=[[8]],
            dst_state_dim_per_tensor=[[]],
            dst_state_layer_ids=[[0]],
            dst_aux_ptrs=[],
        )
        manager.decode_kv_args_table = {"healthy-session": peer}
        if error_kind == "value":
            manager.kv_args.state_types = [StateType.MAMBA]
            manager.kv_args.state_dim_per_tensor = [[0]]
            peer.dst_attn_tp_size = 2
            peer.dst_state_dim_per_tensor = [[0]]
            bad_peer = copy(peer)
            bad_peer.dst_state_dim_per_tensor = [[1]]
            manager.decode_kv_args_table["bad-layout-session"] = bad_peer
            manager._transfer_data = Mock(return_value=0)
        manager._get_dsa_cache_transfer_skip_flags = lambda _: (False, False)
        manager._send_kvcache_generic = Mock(return_value=0)
        manager.send_aux = Mock(return_value=0)
        manager.send_kv_status_message = Mock()
        manager.transfer_infos = {}
        queue = Queue()
        bad_indices = [1] if error_kind == "value" else [1, 2]
        for room, indices in [(101, bad_indices), (102, [1])]:
            req = SimpleNamespace(
                room=room,
                is_dummy=False,
                mooncake_session_id=(
                    "bad-layout-session"
                    if error_kind == "value" and room == 101
                    else "healthy-session"
                ),
                dst_kv_indices=np.array([], dtype=np.int32),
                dst_device_kv_indices=None,
                dst_state_indices=[[3]],
                endpoint="127.0.0.1",
                dst_port=9000,
                required_dst_info_num=1,
            )
            manager.transfer_infos[room] = {0: req}
            queue.put(
                SimpleNamespace(
                    room=room,
                    staging_counted=False,
                    index_slice=slice(0, 0),
                    prefill_kv_indices=np.array([], dtype=np.int32),
                    is_last_chunk=True,
                    state_indices=[indices],
                    prefill_aux_index=0,
                )
            )
        queue.put(None)

        # The bad room executes the native maybe_send_extra cardinality check;
        # the good room must be processed by this same worker invocation.
        manager.transfer_worker(queue, None)

        self.assertTrue(queue.empty())
        self.assertEqual(
            manager.request_status, {101: KVPoll.Failed, 102: KVPoll.Success}
        )
        self.assertIn(
            "replication metadata differs"
            if error_kind == "value"
            else "state index length mismatch",
            manager.failure_records[101],
        )
        self.assertNotIn(102, manager.failure_records)
        self.assertEqual(manager.failed_sessions, set())
        self.assertEqual(dict(manager.session_failures), {})
        self.assertEqual(manager.transfer_infos, {})
        self.assertEqual(dict(manager._staging_outstanding), {})
        self.assertEqual(manager.req_to_decode_prefix_len, {})
        if error_kind == "value":
            manager._transfer_data.assert_called_once()
        else:
            manager._send_kvcache_generic.assert_called_once()
        manager.send_aux.assert_called_once()
        self.assertEqual(manager.send_aux.call_args.args[0].room, 102)
        notifications = manager.send_kv_status_message.call_args_list
        self.assertEqual(
            [call.kwargs["status"] for call in notifications],
            [KVPoll.Failed, KVPoll.Success],
        )
        self.assertEqual(
            [call.kwargs["bootstrap_room"] for call in notifications], [101, 102]
        )


if __name__ == "__main__":
    unittest.main()
