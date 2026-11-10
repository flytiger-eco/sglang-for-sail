"""CPU regressions for mixed KV dispatch, batching, and sender completion."""

import concurrent.futures
import threading
import unittest
from collections import defaultdict
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np

import sglang.srt.disaggregation.mooncake.conn as mc
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from test_mooncake_mixed_transfer import manager, request

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def mixed_manager(*, cap=7, compressed=False, rank=0):
    m = manager()
    m.enable_custom_mem_pool = False
    m.custom_mem_pool_type = ""
    m.max_transfer_batch_indices = cap
    m.kv_args.engine_rank = rank
    m._validate_envelope_kv_layout = Mock()
    if compressed:
        m.kv_args.kv_item_lens[0] = 1728
        m.kv_args.mla_compression_ratios = [128]
    info = NS(
        dst_kv_ptrs=[40000, 50000, 60000],
        dst_kv_item_lens=[m.kv_args.kv_item_lens[0], 2048, 2048],
        dst_kv_item_len=m.kv_args.kv_item_lens[0],
        dst_kv_layer_ids=[0, 1, 1],
        dst_draft_entry_count=2,
        dst_draft_token_major=True,
        dst_draft_head_num=2,
        dst_draft_total_head_num=16,
        dst_page_size=64,
        dst_dcp_size=1,
        dst_dcp_rank=0,
        dst_attn_tp_size=8,
        dst_tp_rank=0,
        requires_dcp_relayout=False,
        dst_aux_ptrs=[],
    )
    m.prepare_dcp_entry_geometry(info)
    return m, info


def send(m, info, executor=None):
    return m.send_kvcache_mixed(
        "s",
        np.array([1, 3]),
        np.array([4, 7]),
        info=info,
        src_page_offset=0,
        decode_prefix_len=0,
        num_kv_tokens=65,
        executor=executor,
    )


def byte_pairs(calls):
    return sorted(
        (src + offset, dst + offset)
        for blocks in calls
        for src, dst, length in blocks
        for offset in range(length)
    )


def worker_manager(*, compressed=False):
    m, info = mixed_manager(cap=0, compressed=compressed)
    m.enable_trace = False
    m.bootstrap_port = 8998
    m._staging_outstanding = defaultdict(int)
    m._transfer_data = Mock(return_value=0)
    m.send_kvcache_mixed = Mock(wraps=m.send_kvcache_mixed)
    m.maybe_send_extra = Mock(return_value=0)
    m.send_aux = Mock(return_value=0)
    return m, info


def add_peer(m, info, session, room, pages, *, device_pages=None, required=1):
    req = mc.TransferInfo.from_zmq(request(session, room))
    req.dst_kv_indices = np.array(pages, dtype=np.int32)
    req.dst_device_kv_indices = (
        np.array(device_pages, dtype=np.int32) if device_pages is not None else None
    )
    req.required_dst_info_num = required
    m.decode_kv_args_table[session] = info
    m.request_status[room] = KVPoll.WaitingForInput
    m.transfer_infos.setdefault(room, {})[session] = req
    return req


def worker_chunk(room, tokens):
    return mc.TransferKVChunk(
        room=room,
        prefill_kv_indices=np.array([1, 3], dtype=np.int32),
        index_slice=slice(0, 2),
        is_last_chunk=True,
        prefill_aux_index=0,
        state_indices=[],
        num_kv_tokens=tokens,
    )


class TestMooncakeMixedDispatch(CustomTestCase):
    def test_index_batches_preserve_every_byte_once(self):
        for compressed in (False, True):
            for rank in (0, 1):
                with self.subTest(compressed=compressed, rank=rank):
                    m, info = mixed_manager(compressed=compressed, rank=rank)
                    calls = []
                    m._transfer_data = lambda session, blocks: calls.append(blocks) or 0
                    self.assertEqual(send(m, info), 0)
                    self.assertEqual(len(calls), 10)
                    for blocks in calls:
                        for base in m.kv_args.kv_data_ptrs[1:]:
                            self.assertLessEqual(
                                sum(
                                    size
                                    for src, _, size in blocks
                                    if base <= src < base + 10000
                                ),
                                7 * 16,
                            )
                    # Independent expected mapping, including the partial final page.
                    expected = []
                    if rank == 0:
                        width = 1728 if compressed else 1
                        if compressed:
                            expected.extend(
                                [
                                    (10000 + width, 40000 + 4 * width, width),
                                    (
                                        10000 + 3 * width,
                                        40000 + 7 * width,
                                        width,
                                    ),
                                ]
                            )
                        else:
                            expected.extend([(10064, 40256, 64), (10192, 40448, 1)])
                    for src_ptr, dst_ptr in zip([20000, 30000], [50000, 60000]):
                        for src_index, dst_index in zip(
                            list(range(64, 128)) + [192], list(range(256, 320)) + [448]
                        ):
                            expected.append(
                                (
                                    src_ptr + src_index * 16,
                                    dst_ptr + dst_index * 32 + rank * 16,
                                    16,
                                )
                            )
                    self.assertEqual(byte_pairs(calls), byte_pairs([expected]))
                    self.assertEqual(
                        len({dst for _, dst in byte_pairs(calls)}),
                        len(byte_pairs(calls)),
                    )

    def test_unbounded_and_short_paths_remain_single_call(self):
        for cap in (0, 65, 100):
            m, info = mixed_manager(cap=cap)
            m._transfer_data = Mock(return_value=0)
            self.assertEqual(send(m, info), 0)
            self.assertEqual(m._transfer_data.call_count, 1)

    def test_first_failed_batch_stops_later_writes(self):
        m, info = mixed_manager()
        m._transfer_data = Mock(side_effect=[0, -17, 0])
        self.assertEqual(send(m, info), -17)
        self.assertEqual(m._transfer_data.call_count, 2)

    def test_empty_plan_does_not_dispatch(self):
        m, info = mixed_manager()
        m._transfer_data = Mock(return_value=0)
        self.assertEqual(
            m.send_kvcache_mixed(
                "s",
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                info=info,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=0,
            ),
            0,
        )
        m._transfer_data.assert_not_called()

    def test_compressed_hisparse_batches_keep_host_and_device_indices(self):
        m, info = mixed_manager(cap=1, compressed=True)
        a = m.kv_args
        a.mla_compression_ratios = [4, 128]
        a.prefill_start_layer, a.prefill_end_layer = 0, 2
        a.kv_data_ptrs = [100000, 200000, 300000, 400000]
        a.kv_item_lens = [37440, 1728, 1024, 1024]
        a.kv_layer_ids = [0, 1, 2, 2]
        info.dst_kv_ptrs = [500000, 600000, 700000, 800000]
        info.dst_kv_item_lens = [37440, 1728, 2048, 2048]
        info.dst_kv_item_len = 37440
        info.dst_kv_layer_ids = [0, 1, 2, 2]
        m.prepare_dcp_entry_geometry(info)
        calls = []
        m._transfer_data = lambda session, blocks: calls.append(blocks) or 0
        self.assertEqual(
            m.send_kvcache_mixed(
                "s",
                np.array([1, 3]),
                np.array([4, 7]),
                info=info,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=65,
                dst_device_kv_indices=np.array([8, 9]),
            ),
            0,
        )
        self.assertEqual(len(calls), 65)
        targets = [block for blocks in calls for block in blocks if block[0] < 300000]
        self.assertEqual(
            targets,
            [
                (100000 + 37440, 500000 + 4 * 37440, 37440),
                (200000 + 1728, 600000 + 8 * 1728, 1728),
                (100000 + 3 * 37440, 500000 + 7 * 37440, 37440),
                (200000 + 3 * 1728, 600000 + 9 * 1728, 1728),
            ],
        )
        drafts = [block for blocks in calls for block in blocks if block[0] >= 300000]
        self.assertEqual(len(drafts), 130)
        self.assertIn((300000 + 64 * 16, 700000 + 8 * 64 * 32, 16), drafts)
        self.assertIn((300000 + 192 * 16, 700000 + 9 * 64 * 32, 16), drafts)

    def test_custom_pool_dispatches_bounded_per_entry_tasks(self):
        m, info = mixed_manager()
        m.enable_custom_mem_pool = True
        m.custom_mem_pool_type = "NVLINK"
        calls = []
        m._transfer_data = lambda session, blocks: calls.append(blocks) or 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            self.assertEqual(send(m, info, executor), 0)
        self.assertEqual(len(calls), 30)
        for blocks in calls:
            entries = {
                next(
                    i
                    for i, base in enumerate(m.kv_args.kv_data_ptrs)
                    if base <= src < base + 10000
                )
                for src, _, _ in blocks
            }
            self.assertEqual(len(entries), 1)
            self.assertLessEqual(len(blocks), 7)
        expected, expected_info = mixed_manager(cap=0)
        expected._transfer_data = Mock(return_value=0)
        self.assertEqual(send(expected, expected_info), 0)
        self.assertEqual(
            byte_pairs(calls),
            byte_pairs([expected._transfer_data.call_args.args[1]]),
        )

    def test_intra_node_pool_preserves_combined_dispatch(self):
        m, info = mixed_manager()
        m.enable_custom_mem_pool = True
        m.custom_mem_pool_type = "INTRA_NODE_NVLINK"
        m._transfer_data = Mock(return_value=0)
        executor = Mock()
        self.assertEqual(send(m, info, executor), 0)
        executor.submit.assert_not_called()
        self.assertEqual(m._transfer_data.call_count, 10)

    def test_deferred_release_drains_running_tasks_after_first_error(self):
        m, info = mixed_manager()
        m.enable_custom_mem_pool = True
        m.custom_mem_pool_type = "NVLINK"
        m.enable_deferred_decode_kv_release = True
        running, failed, release, returned = [threading.Event() for _ in range(4)]
        result = []

        def transfer(session, blocks):
            src = blocks[0][0]
            if src < 20000:
                self.assertTrue(running.wait(5))
                failed.set()
                return -17
            running.set()
            self.assertTrue(release.wait(5))
            return 0

        m._transfer_data = transfer
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:

            def invoke():
                try:
                    result.append(send(m, info, executor))
                finally:
                    returned.set()

            thread = threading.Thread(target=invoke)
            thread.start()
            try:
                self.assertTrue(failed.wait(5))
                self.assertFalse(returned.wait(0.05))
            finally:
                release.set()
                thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(result, [-17])

    def test_padding_trim_is_peer_local_and_preserves_the_queued_chunk(self):
        m, info = worker_manager()
        add_peer(m, info, "short", 1, [4], required=2)
        add_peer(m, info, "full", 1, [7, 9], required=2)
        chunk = worker_chunk(1, 64)
        original_pages = chunk.prefill_kv_indices
        queue = NS(get=Mock(side_effect=[chunk, None]))
        with get_context().override_server_args(enable_unified_memory=False):
            m.transfer_worker(queue, Mock())

        calls = m.send_kvcache_mixed.call_args_list
        self.assertEqual([call.args[0] for call in calls], ["short", "full"])
        np.testing.assert_array_equal(calls[0].args[1], [1])
        np.testing.assert_array_equal(calls[1].args[1], [1, 3])
        self.assertIs(calls[1].args[1], original_pages)
        self.assertIs(chunk.prefill_kv_indices, original_pages)
        np.testing.assert_array_equal(original_pages, [1, 3])
        self.assertEqual(chunk.num_kv_tokens, 64)
        self.assertEqual([call.kwargs["num_kv_tokens"] for call in calls], [64, 64])
        writes = m._transfer_data.call_args_list
        self.assertEqual(len(writes), 2)
        self.assertIn((10064, 40256, 64), writes[0].args[1])
        self.assertIn((10064, 40448, 64), writes[1].args[1])
        for call in writes:
            self.assertEqual(sum(size for _, _, size in call.args[1]), 64 + 2 * 64 * 16)
        self.assertEqual(m.check_status(1), KVPoll.Success)
        self.assertFalse(m._staging_outstanding)

    def test_missing_peer_pages_fail_one_room_and_worker_serves_the_next(self):
        cases = (
            ([4], None, "tokens=65, capacity=64"),
            ([4, 7], [8], "Insufficient destination device KV pages"),
        )
        for pages, device_pages, reason in cases:
            with self.subTest(pages=pages, device_pages=device_pages):
                m, info = worker_manager(compressed=device_pages is not None)
                add_peer(m, info, "bad", 1, pages, device_pages=device_pages)
                add_peer(m, info, "healthy", 2, [7, 9])
                bad, healthy = worker_chunk(1, 65), worker_chunk(2, 65)
                original_pages = bad.prefill_kv_indices
                queue = NS(get=Mock(side_effect=[bad, healthy, None]))
                with get_context().override_server_args(enable_unified_memory=False):
                    m.transfer_worker(queue, Mock())

                self.assertEqual(m.check_status(1), KVPoll.Failed)
                self.assertIn(reason, m.failure_records[1])
                self.assertEqual(m.check_status(2), KVPoll.Success)
                self.assertNotIn(2, m.failure_records)
                self.assertEqual(m.send_kvcache_mixed.call_count, 1)
                self.assertEqual(m.send_kvcache_mixed.call_args.args[0], "healthy")
                self.assertEqual(
                    {call.args[0] for call in m._transfer_data.call_args_list},
                    {"healthy"},
                )
                self.assertIs(bad.prefill_kv_indices, original_pages)
                np.testing.assert_array_equal(original_pages, [1, 3])
                self.assertEqual(bad.num_kv_tokens, 65)
                statuses = {
                    (call.kwargs["bootstrap_room"], call.kwargs["status"])
                    for call in m.send_kv_status_message.call_args_list
                }
                self.assertEqual(statuses, {(1, KVPoll.Failed), (2, KVPoll.Success)})
                self.assertFalse(m._staging_outstanding)

    def test_staging_defer_requeues_original_chunk_after_peer_local_trim(self):
        m, info = worker_manager()
        m.is_mla_backend = m.is_hybrid_mla_backend = False
        m.enable_staging = True
        m._staging_ctx = NS(
            watermark_cv=threading.Condition(),
            prefetch_requested=set(),
            prefetched_rooms=set(),
        )
        info.staging_base_ptr = 1000000
        info.staging_total_size = 4096
        req = add_peer(m, info, "staging", 1, [4])
        strategy = NS(
            check_ready=Mock(side_effect=[(False, 0, 0, 0, 0), (True, 7, 16, 0, 0)]),
            transfer=Mock(return_value=0),
        )
        m._try_create_staging_strategy = Mock(return_value=strategy)
        m._send_chunk_ready = Mock()
        m.send_kvcache_slice = Mock()
        chunk = worker_chunk(1, 64)
        original_pages = chunk.prefill_kv_indices
        pending = [chunk]
        requeued = []

        def put(item):
            requeued.append(item)
            pending.append(item)

        queue = NS(get=lambda: pending.pop(0) if pending else None, put=put)
        with get_context().override_server_args(enable_unified_memory=False):
            m.transfer_worker(queue, Mock(), staging_buffer=object())

        self.assertEqual(len(requeued), 1)
        self.assertIs(requeued[0], chunk)
        self.assertIs(chunk.prefill_kv_indices, original_pages)
        np.testing.assert_array_equal(original_pages, [1, 3])
        self.assertEqual(chunk.num_kv_tokens, 64)
        self.assertEqual(strategy.check_ready.call_count, 2)
        for call in strategy.check_ready.call_args_list:
            self.assertIs(call.args[0], req)
            self.assertEqual(call.args[1:], (0, 1))
        strategy.transfer.assert_called_once()
        np.testing.assert_array_equal(strategy.transfer.call_args.args[1], [1])
        m._send_chunk_ready.assert_called_once()
        sent_chunk = m._send_chunk_ready.call_args.args[2]
        self.assertIsNot(sent_chunk, chunk)
        np.testing.assert_array_equal(sent_chunk.prefill_kv_indices, [1])
        m.send_kvcache_slice.assert_not_called()
        m.send_kvcache_mixed.assert_not_called()
        self.assertEqual(m.check_status(1), KVPoll.Success)
        self.assertFalse(m._staging_outstanding)

    def test_state_only_cp_rank_does_not_require_destination_kv_pages(self):
        for mode in ("replicated_target", "empty_pool"):
            with self.subTest(mode=mode):
                m, info = worker_manager()
                m.attn_cp_size, m.attn_cp_rank = 2, 1
                if mode == "replicated_target":
                    m.kv_args.draft_kv_entry_count = 0
                    m.kv_args.num_draft_entries = 0
                    m.kv_args.kv_data_ptrs = m.kv_args.kv_data_ptrs[:1]
                    m.kv_args.kv_item_lens = m.kv_args.kv_item_lens[:1]
                    m.kv_args.kv_layer_ids = [0]
                    m.kv_args.engine_rank = m.attn_tp_rank = 1
                else:
                    m.kv_args.kv_data_ptrs = []
                m._get_peer_chunk_indices = Mock(wraps=m._get_peer_chunk_indices)
                add_peer(m, info, "state_only", 1, [])
                chunk = worker_chunk(1, 65)
                chunk.state_indices = [0]
                queue = NS(get=Mock(side_effect=[chunk, None]))
                with get_context().override_server_args(
                    enable_unified_memory=False, enable_dsa_cache_layer_split=True
                ):
                    m.transfer_worker(queue, Mock())

                m._get_peer_chunk_indices.assert_not_called()
                m._transfer_data.assert_not_called()
                m.send_kvcache_mixed.assert_not_called()
                m.maybe_send_extra.assert_called_once()
                self.assertEqual(m.maybe_send_extra.call_args.args[1], [0])
                m.send_aux.assert_called_once()
                self.assertEqual(m.check_status(1), KVPoll.Success)
                self.assertNotIn(1, m.failure_records)
                self.assertFalse(m._staging_outstanding)

    def test_cp_without_local_draft_skips_target_not_state_or_completion(self):
        # Bootstrap keeps both TP senders on every CP rank when the last CP
        # owns drafts. CP0 must still send replicated targets only once.
        for hybrid in (False, True):
            for cp in (0, 1):
                for rank in (0, 1):
                    with self.subTest(hybrid=hybrid, cp=cp, rank=rank):
                        m, info = mixed_manager(rank=rank)
                        m.is_mla_backend = not hybrid
                        m.is_hybrid_mla_backend = hybrid
                        m.attn_cp_size, m.attn_cp_rank = 2, cp
                        m.attn_tp_rank = rank
                        info.dst_kv_ptrs = [40000, 45000, 50000, 60000]
                        info.dst_kv_item_lens = [64, 64, 2048, 2048]
                        info.dst_kv_layer_ids = [0, 1, 2, 2]
                        m.kv_args.kv_layer_ids = [1, 2, 2]
                        m._transfer_data = Mock(return_value=0)
                        if cp == 0:
                            m.kv_args.draft_kv_entry_count = 0
                            m.kv_args.kv_data_ptrs = m.kv_args.kv_data_ptrs[:1]
                            m.kv_args.kv_item_lens = m.kv_args.kv_item_lens[:1]
                            m.kv_args.kv_layer_ids = [0]
                        else:
                            m.prepare_dcp_entry_geometry(info)
                        m._should_skip_cp_replicated_state_transfer = Mock(
                            return_value=False
                        )
                        m.enable_trace = False
                        m.bootstrap_port = 8998
                        m._staging_outstanding = defaultdict(int)
                        m.send_kvcache = Mock(wraps=m.send_kvcache)
                        m.send_kvcache_mixed = Mock(wraps=m.send_kvcache_mixed)
                        m.maybe_send_extra = Mock(return_value=0)
                        m.send_aux = Mock(return_value=0)
                        m.decode_kv_args_table["s"] = info
                        m.request_status[1] = KVPoll.WaitingForInput
                        m.transfer_infos[1] = {
                            "s": mc.TransferInfo.from_zmq(request("s", 1))
                        }
                        chunk = NS(
                            room=1,
                            staging_counted=False,
                            is_last_chunk=True,
                            index_slice=slice(0, 1),
                            prefill_kv_indices=np.array([0]),
                            num_kv_tokens=1,
                            state_indices=[0],
                            prefill_aux_index=0,
                        )
                        queue = NS(get=Mock(side_effect=[chunk, None]))
                        executor = Mock()
                        with get_context().override_server_args(
                            enable_unified_memory=False
                        ):
                            m.transfer_worker(queue, executor)
                        self.assertEqual(
                            m.send_kvcache.call_count, int(cp == 0 and rank == 0)
                        )
                        self.assertEqual(m.send_kvcache_mixed.call_count, int(cp == 1))
                        if cp == 1:
                            self.assertIs(
                                m.send_kvcache_mixed.call_args.kwargs["executor"],
                                executor,
                            )
                        if cp == 0:
                            expected = (
                                [(10000, 40000 + 2 * 64, 64)] if rank == 0 else []
                            )
                        else:
                            expected = (
                                [(10000, 45000 + 2 * 64, 1)] if rank == 0 else []
                            ) + [
                                (src, dst + 2 * 64 * 32 + rank * 16, 16)
                                for src, dst in ((20000, 50000), (30000, 60000))
                            ]
                        self.assertEqual(
                            [
                                block
                                for call in m._transfer_data.call_args_list
                                for block in call.args[1]
                            ],
                            expected,
                        )
                        m.maybe_send_extra.assert_called_once()
                        m.send_aux.assert_called_once()
                        self.assertEqual(m.check_status(1), KVPoll.Success)
                        self.assertEqual(
                            m.send_kv_status_message.call_args.kwargs["status"],
                            KVPoll.Success,
                        )
                        self.assertFalse(m._staging_outstanding)


if __name__ == "__main__":
    unittest.main()
