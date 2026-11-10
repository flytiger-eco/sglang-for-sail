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
