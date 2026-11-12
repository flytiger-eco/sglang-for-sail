"""Registration-to-byte-copy regressions for target/draft layer mapping."""

import struct
import unittest
from collections import defaultdict
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import test_mooncake_entry_mapping as entry_mapping
from test_mooncake_mixed_transfer import manager, registration, request

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import TransferInfo

from sglang.srt.disaggregation.utils import build_kv_layer_ids
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeTargetLayerMapping(CustomTestCase):
    _copy_transport = entry_mapping.TestMooncakeEntryMapping._copy_transport

    @staticmethod
    def _pool(start, layers, groups=1, shard=False):
        pool_type = MHATokenToKVPool if groups == 2 else MLATokenToKVPool
        pool = pool_type.__new__(pool_type)
        pool.start_layer = start
        pool.layer_num = layers
        pool.layer_shard_enabled = shard
        count = layers * groups
        pool.get_contiguous_buf_infos = Mock(
            return_value=(list(range(count)), [128] * count, [64] * count)
        )
        return pool

    @classmethod
    def _ids(cls, start, layers, draft=False, shard=False):
        return build_kv_layer_ids(
            token_to_kv_pool=cls._pool(start, layers, shard=shard),
            draft_token_to_kv_pool=cls._pool(0, 1, groups=2) if draft else None,
            num_draft_entries=2 if draft else 0,
            num_hidden_layers=4,
        )

    @staticmethod
    def _manager(stage, src_ids, widths, draft=0, compressed=False):
        m = manager(draft=draft, tp=8)
        m.pp_size = 2
        m.pp_rank = stage
        a = m.kv_args
        a.prefill_start_layer = stage * 2
        a.prefill_end_layer = stage * 2 + 2
        a.mla_compression_ratios = [4, 128, 4, 128] if compressed else None
        a.kv_data_ptrs = [(i + 1) * 1000000 for i in range(len(widths))]
        a.kv_item_lens = widths
        a.kv_layer_ids = src_ids
        a.draft_kv_head_num = 2
        return m

    @staticmethod
    def _wire(dst_ids, widths, draft=2, old=False):
        msg = registration("session", count=draft)
        msg[4] = struct.pack(
            f"{len(widths)}Q", *[(i + 1) * 10000000 for i in range(len(widths))]
        )
        msg[9] = str(widths[0]).encode()
        msg[12] = struct.pack(f"{len(dst_ids)}I", *dst_ids) if dst_ids else b""
        msg[18] = struct.pack(f"{len(widths)}Q", *widths)
        if old:
            return (msg[:18] + [b""] + msg[18:])[:20]
        return msg

    def _register_and_copy(self, m, msg, expected_indices, worker=False):
        m._register_decode_peer(msg, "session")
        self.assertNotIn("session", m.rejected_registrations)
        info = m.decode_kv_args_table["session"]
        self.assertEqual(info.dcp_dst_entry_indices, expected_indices)
        sources = {
            base: (np.arange(width * 3, dtype=np.uint32) + i * 17).astype(np.uint8)
            for i, (base, width) in enumerate(
                zip(m.kv_args.kv_data_ptrs, m.kv_args.kv_item_lens)
            )
        }
        destinations = {
            base: np.full(width * 4, 205, dtype=np.uint8)
            for base, width in zip(info.dst_kv_ptrs, info.dst_kv_item_lens)
        }
        expected = {base: buf.copy() for base, buf in destinations.items()}
        records = []
        m._transfer_data = self._copy_transport(sources, destinations, records)
        with get_context().override_server_args(enable_unified_memory=False):
            if worker:
                m.enable_trace = False
                m.bootstrap_port = 8998
                m._staging_outstanding = defaultdict(int)
                m.send_aux = Mock(return_value=0)
                m.request_status[1] = KVPoll.WaitingForInput
                m.transfer_infos[1] = {
                    "session": TransferInfo.from_zmq(request("session", 1))
                }
                chunk = NS(
                    room=1,
                    staging_counted=False,
                    is_last_chunk=True,
                    index_slice=slice(0, 1),
                    prefill_kv_indices=np.array([1], dtype=np.int32),
                    num_kv_tokens=m.kv_args.page_size,
                    state_indices=[],
                    prefill_aux_index=0,
                )
                with patch.object(m, "send_kvcache", wraps=m.send_kvcache) as send:
                    m.transfer_worker(NS(get=Mock(side_effect=[chunk, None])), None)
                send.assert_called_once()
                self.assertEqual(
                    send.call_args.kwargs["dst_entry_indices"], expected_indices
                )
                self.assertEqual(m.request_status[1], KVPoll.Success)
                ret = 0
            else:
                ret = m.send_kvcache(
                    "session",
                    np.array([1], dtype=np.int32),
                    info.dst_kv_ptrs,
                    np.array([2], dtype=np.int32),
                    None,
                    dst_layer_ids=info.dst_kv_layer_ids,
                    dst_kv_item_len=info.dst_kv_item_len,
                    dst_attn_tp_size=info.dst_attn_tp_size,
                    dst_entry_indices=info.dcp_dst_entry_indices,
                )
        self.assertEqual(ret, 0)
        for i, j in enumerate(expected_indices):
            width = m.kv_args.kv_item_lens[i]
            expected[info.dst_kv_ptrs[j]][2 * width : 3 * width] = sources[
                m.kv_args.kv_data_ptrs[i]
            ][width : 2 * width]
        for base in destinations:
            np.testing.assert_array_equal(destinations[base], expected[base])
        self.assertEqual(len(records), len(expected_indices))

    def test_compressed_early_and_final_pp_stages_use_independent_draft_counts(self):
        for stage in (0, 1):
            with self.subTest(stage=stage):
                draft = 2 if stage else 0
                src = [37440, 9216, 1728] + [2048] * draft
                dst = [37440, 37440, 9216, 9216, 1728, 1728, 2048, 2048]
                m = self._manager(stage, [], src, draft=draft, compressed=True)
                m.kv_args.page_size = 256
                msg = self._wire([], dst)
                msg[22] = b"256"
                expected = [stage, stage + 2, stage + 4] + ([6, 7] if draft else [])
                self._register_and_copy(m, msg, expected, worker=True)

    def test_legacy_plain_mla_pp_with_draft_both_upgrade_directions(self):
        for old_peer in ("decode", "prefill"):
            with self.subTest(old_peer=old_peer):
                src_ids = self._ids(2, 2, draft=True)
                dst_ids = self._ids(0, 4, draft=True)
                self.assertEqual(src_ids, [2, 3, 4, 4])
                self.assertEqual(dst_ids, [0, 1, 2, 3, 4, 4])
                if old_peer == "decode":
                    dst_ids = []
                else:
                    src_ids = []
                m = self._manager(1, src_ids, [64, 64, 2048, 2048], draft=2)
                msg = self._wire(
                    dst_ids, [64] * 4 + [2048] * 2, old=old_peer == "decode"
                )
                self._register_and_copy(m, msg, [2, 3, 4, 5])

    def test_decode_only_speculation_and_draft_free_prefill_stage(self):
        dst_ids = self._ids(0, 4, draft=True)
        for stage in (0, 1):
            with self.subTest(stage=stage):
                src_ids = self._ids(stage * 2, 2)
                self.assertEqual(src_ids, [])
                m = self._manager(stage, src_ids, [64, 64])
                self._register_and_copy(
                    m,
                    self._wire(dst_ids, [64] * 4 + [2048] * 2),
                    [stage * 2, stage * 2 + 1],
                )

    def test_target_layer_shard_with_full_stage_span(self):
        # Pool metadata preserves a layer-sharded export, even without draft.
        src_ids = self._ids(2, 2, shard=True)
        self.assertEqual(src_ids, [2, 3])
        m = self._manager(0, src_ids, [64, 64])
        m.pp_size = 1
        m.attn_cp_size = 2
        m.kv_args.prefill_start_layer = 0
        m.kv_args.prefill_end_layer = 4
        self._register_and_copy(m, self._wire([], [64] * 4, draft=0), [2, 3])

    def test_bad_target_metadata_is_rejected_at_registration(self):
        m = self._manager(1, [0, 1], [64, 64])
        m._register_decode_peer(self._wire([], [64] * 4, draft=0), "session")
        self.assertIn("session", m.rejected_registrations)
        self.assertNotIn("session", m.decode_kv_args_table)

    def test_native_mla_draft_old_peer_keeps_whole_pages(self):
        m = self._manager(1, [2, 3, 4], [64, 64, 64])
        m.kv_args.num_draft_entries = 1
        self._register_and_copy(
            m, self._wire([], [64] * 5, draft=0, old=True), [2, 3, 4]
        )

    def test_decode_only_native_draft_uses_reserved_layer_band(self):
        dst_ids = build_kv_layer_ids(
            token_to_kv_pool=self._pool(0, 4),
            draft_token_to_kv_pool=self._pool(0, 1),
            num_draft_entries=1,
            num_hidden_layers=4,
        )
        self.assertEqual(dst_ids, [0, 1, 2, 3, 4])
        m = self._manager(0, [], [64] * 4)
        m.pp_size = 1
        m.kv_args.prefill_end_layer = 4
        self._register_and_copy(m, self._wire(dst_ids, [64] * 5, draft=0), [0, 1, 2, 3])

    def test_invalid_draft_band_is_rejected_before_transfer(self):
        for src_ids, dst_ids, old in (
            ([2, 3, 99, 99], [], True),
            ([], [0, 1, 2, 3, 99, 99], False),
        ):
            with self.subTest(src_ids=src_ids, dst_ids=dst_ids):
                m = self._manager(1, src_ids, [64, 64, 2048, 2048], draft=2)
                msg = self._wire(dst_ids, [64] * 4 + [2048] * 2, old=old)
                m._register_decode_peer(msg, "session")
                self.assertIn("session", m.rejected_registrations)
                self.assertNotIn("session", m.decode_kv_args_table)


if __name__ == "__main__":
    unittest.main()
