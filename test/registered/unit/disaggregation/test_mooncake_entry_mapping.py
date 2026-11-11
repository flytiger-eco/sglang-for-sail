"""CPU regressions for Mooncake registration and transfer entry pairing."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import numpy as np
from test_mooncake_mixed_transfer import manager, registration

from sglang.srt.disaggregation.mooncake.conn import KVArgsRegisterInfo
from sglang.srt.disaggregation.utils import build_kv_layer_ids
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeEntryMapping(CustomTestCase):
    @staticmethod
    def _plain_manager(src_layer_ids):
        m = manager(draft=0, tp=8)
        m.kv_args.page_size = 4
        m.kv_args.kv_item_lens = [12, 20]
        m.kv_args.kv_data_ptrs = [1000000, 2000000]
        m.kv_args.kv_layer_ids = src_layer_ids
        m.kv_args.prefill_start_layer = 0
        m.kv_args.prefill_end_layer = 2
        m.kv_args.mla_compression_ratios = None
        return m

    @staticmethod
    def _plain_peer(dst_layer_ids, dcp_rank=0):
        info = KVArgsRegisterInfo.from_zmq(registration(count=0))
        info.dst_kv_ptrs = [10000000, 20000000]
        info.dst_kv_item_lens = [12, 20]
        info.dst_kv_item_len = 12
        info.dst_kv_layer_ids = dst_layer_ids
        info.dst_page_size = 4
        info.dst_dcp_size = 2
        info.dst_dcp_rank = dcp_rank
        return info

    def _copy_transport(self, sources, destinations, records):
        # Mock only the transport. Real production mapping builds every block;
        # symbolic addresses keep invalid descriptors from accessing host memory.
        def transfer(session, blocks):
            self.assertEqual(session, "session")
            for src, dst, size in blocks:
                src_matches = [
                    (base, buf)
                    for base, buf in sources.items()
                    if base <= src and src + size <= base + buf.size
                ]
                dst_matches = [
                    (base, buf)
                    for base, buf in destinations.items()
                    if base <= dst and dst + size <= base + buf.size
                ]
                self.assertGreater(size, 0)
                self.assertEqual(len(src_matches), 1)
                self.assertEqual(len(dst_matches), 1)
                src_base, src_buffer = src_matches[0]
                dst_base, dst_buffer = dst_matches[0]
                dst_buffer[dst - dst_base : dst - dst_base + size] = src_buffer[
                    src - src_base : src - src_base + size
                ]
                records.append((src, dst, size))
            return 0

        return transfer

    def test_compressed_pp_stage_maps_targets_and_draft_suffix(self):
        # Actual compressed layout has no layer IDs. Target buffers are grouped
        # as C4 KV, C4 indexer KV, C128 KV, followed by MHA draft K and V.
        # P stage 1 owns layers 2/3; D PP1 owns all four layers.
        src_widths = [37440, 9216, 1728, 4096, 4096]
        dst_widths = [37440, 37440, 9216, 9216, 1728, 1728, 8192, 8192]
        expected_mapping = [1, 3, 5, 6, 7]
        src_bases = [(i + 1) * 1000000 for i in range(5)]
        dst_bases = [(i + 1) * 10000000 for i in range(8)]
        destinations = {
            base: np.full(7 * width, 205, dtype=np.uint8)
            for base, width in zip(dst_bases, dst_widths)
        }
        expected = {base: data.copy() for base, data in destinations.items()}
        src_pages, dst_pages = [1, 3], [4, 6]
        for rank in (0, 1):
            with self.subTest(source_rank=rank):
                m = manager()
                m.pp_size = 2
                m.pp_rank = 1
                m.attn_tp_rank = rank
                a = m.kv_args
                a.engine_rank = rank + 16
                a.page_size = 256
                a.prefill_start_layer = 2
                a.prefill_end_layer = 4
                a.mla_compression_ratios = [4, 128, 4, 128]
                a.kv_layer_ids = []
                a.kv_item_lens = src_widths
                a.kv_data_ptrs = src_bases
                info = KVArgsRegisterInfo.from_zmq(registration())
                info.dst_page_size = 256
                info.dst_kv_layer_ids = []
                info.dst_kv_item_lens = dst_widths
                info.dst_kv_item_len = dst_widths[0]
                info.dst_kv_ptrs = dst_bases
                self.assertEqual(
                    m._resolve_kv_entry_indices([], len(dst_bases), 2),
                    expected_mapping,
                )
                m.prepare_dcp_entry_geometry(info)
                self.assertTrue(info.target_page_transfer)
                self.assertEqual(info.dcp_token_item_lens, [0, 0, 0, 16, 16])
                self.assertEqual(info.dcp_dst_token_item_lens, [0, 0, 0, 32, 32])
                sources = {
                    base: (
                        np.arange(4 * width, dtype=np.uint32) + rank * 71 + i * 13
                    ).astype(np.uint8)
                    for i, (base, width) in enumerate(zip(src_bases, src_widths))
                }
                records = []
                m._transfer_data = self._copy_transport(sources, destinations, records)
                # Envelope runtime config is unrelated to entry addressing.
                m._validate_envelope_kv_layout = Mock()
                result = m.send_kvcache_mixed(
                    "session",
                    np.array(src_pages, dtype=np.int32),
                    np.array(dst_pages, dtype=np.int32),
                    info=info,
                    src_page_offset=0,
                    decode_prefix_len=0,
                    num_kv_tokens=257,
                )
                self.assertEqual(result, 0)
                for i, j in enumerate(expected_mapping):
                    source = sources[src_bases[i]]
                    destination = expected[dst_bases[j]]
                    if i < 3:
                        if rank == 0:
                            width = src_widths[i]
                            for src_page, dst_page in zip(src_pages, dst_pages):
                                # Include padding and the whole final partial page.
                                destination[
                                    dst_page * width : (dst_page + 1) * width
                                ] = source[src_page * width : (src_page + 1) * width]
                    else:
                        for token in range(257):
                            src_slot = src_pages[token // 256] * 256 + token % 256
                            dst_slot = dst_pages[token // 256] * 256 + token % 256
                            dst_offset = dst_slot * 32 + rank * 16
                            destination[dst_offset : dst_offset + 16] = source[
                                src_slot * 16 : (src_slot + 1) * 16
                            ]
                for base in dst_bases:
                    np.testing.assert_array_equal(destinations[base], expected[base])
                if rank == 1:
                    self.assertTrue(records)
                    self.assertTrue(all(dst >= dst_bases[6] for _, dst, _ in records))

    def test_compressed_pp_whole_pages_keep_draft_suffix_outside_target_groups(self):
        m = manager(tp=16)
        m.pp_size = 2
        m.pp_rank = 1
        a = m.kv_args
        a.engine_rank = 16
        a.page_size = 256
        a.prefill_start_layer = 2
        a.prefill_end_layer = 4
        a.mla_compression_ratios = [4, 128, 4, 128]
        a.kv_layer_ids = []
        a.kv_item_lens = [37440, 9216, 1728, 4096, 4096]
        a.kv_data_ptrs = [(i + 1) * 1000000 for i in range(5)]
        info = KVArgsRegisterInfo.from_zmq(registration())
        info.dst_attn_tp_size = 16
        info.dst_draft_head_num = 1
        info.dst_page_size = 256
        info.dst_kv_layer_ids = []
        info.dst_kv_item_lens = [37440, 37440, 9216, 9216, 1728, 1728, 4096, 4096]
        info.dst_kv_item_len = 37440
        info.dst_kv_ptrs = [(i + 1) * 10000000 for i in range(8)]
        expected_mapping = [1, 3, 5, 6, 7]
        m.prepare_dcp_entry_geometry(info)
        self.assertEqual(info.dcp_dst_entry_indices, expected_mapping)
        self.assertFalse(m._uses_mixed_kv_transfer(info))
        sources = {
            base: (np.arange(4 * width, dtype=np.uint32) + i * 13).astype(np.uint8)
            for i, (base, width) in enumerate(zip(a.kv_data_ptrs, a.kv_item_lens))
        }
        destinations = {
            base: np.full(6 * width, 205, dtype=np.uint8)
            for base, width in zip(info.dst_kv_ptrs, info.dst_kv_item_lens)
        }
        expected = {base: data.copy() for base, data in destinations.items()}
        records = []
        m._transfer_data = self._copy_transport(sources, destinations, records)
        # Exercise the real main-KV sender, including its envelope configuration
        # check. Only transport is replaced; mapping is never mocked.
        with get_context().override_server_args(enable_unified_memory=False):
            result = m.send_kvcache(
                "session",
                np.array([1, 3], dtype=np.int32),
                info.dst_kv_ptrs,
                np.array([2, 5], dtype=np.int32),
                None,
                dst_layer_ids=[],
                dst_kv_item_len=info.dst_kv_item_len,
                dst_attn_tp_size=16,
            )
        self.assertEqual(result, 0)
        self.assertEqual(len(records), 10)
        for i, j in enumerate(expected_mapping):
            width = a.kv_item_lens[i]
            for src_page, dst_page in ((1, 2), (3, 5)):
                expected[info.dst_kv_ptrs[j]][
                    dst_page * width : (dst_page + 1) * width
                ] = sources[a.kv_data_ptrs[i]][
                    src_page * width : (src_page + 1) * width
                ]
        # Full arrays include target-page padding, both complete draft pages,
        # untouched P-stage-0 buffers, and untouched destination pages.
        for base in info.dst_kv_ptrs:
            np.testing.assert_array_equal(destinations[base], expected[base])

    def test_plain_mla_one_sided_ids_preserve_dcp_bytes(self):
        for src_ids, dst_ids in (([0, 1], []), ([], [0, 1])):
            for dcp_rank in (0, 1):
                with self.subTest(src_ids=src_ids, dst_ids=dst_ids, dcp_rank=dcp_rank):
                    m = self._plain_manager(src_ids)
                    info = self._plain_peer(dst_ids, dcp_rank)
                    m.prepare_dcp_entry_geometry(info)
                    self.assertEqual(info.dcp_token_item_lens, [3, 5])
                    sources = {
                        base: (np.arange(4 * width, dtype=np.uint32) + 7).astype(
                            np.uint8
                        )
                        for base, width in zip(m.kv_args.kv_data_ptrs, [12, 20])
                    }
                    destinations = {
                        base: np.full(3 * width, 205, dtype=np.uint8)
                        for base, width in zip(info.dst_kv_ptrs, [12, 20])
                    }
                    expected = {
                        base: data.copy() for base, data in destinations.items()
                    }
                    records = []
                    m._transfer_data = self._copy_transport(
                        sources, destinations, records
                    )
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = m.send_kvcache_dcp(
                            "session",
                            np.array([1, 3], dtype=np.int32),
                            info.dst_kv_ptrs,
                            np.array([2], dtype=np.int32),
                            dcp_token_item_lens=info.dcp_token_item_lens,
                            dst_dcp_size=2,
                            dst_dcp_rank=dcp_rank,
                            src_page_offset=0,
                            decode_prefix_len=0,
                            num_kv_tokens=5,
                            executor=executor,
                            dst_layer_ids=dst_ids,
                            dst_kv_item_lens=info.dst_kv_item_lens,
                            dst_attn_tp_size=8,
                        )
                    self.assertEqual(result, 0)
                    self.assertTrue(records)
                    for i, width in enumerate((3, 5)):
                        for token in range(dcp_rank, 5, 2):
                            src_slot = [1, 3][token // 4] * 4 + token % 4
                            dst_slot = 2 * 4 + token // 2
                            expected[info.dst_kv_ptrs[i]][
                                dst_slot * width : (dst_slot + 1) * width
                            ] = sources[m.kv_args.kv_data_ptrs[i]][
                                src_slot * width : (src_slot + 1) * width
                            ]
                    for base in info.dst_kv_ptrs:
                        np.testing.assert_array_equal(
                            destinations[base], expected[base]
                        )

    def test_one_sided_ids_reject_ambiguous_layouts(self):
        cases = (
            ("pp_bad_span", {"pp_size": 2}, {"prefill_start_layer": 1}, 0),
            ("cp_layer_split", {"attn_cp_size": 2}, {}, 0),
            ("hybrid", {"is_hybrid_mla_backend": True}, {}, 0),
            ("not_mla", {"is_mla_backend": False}, {}, 0),
            ("native_draft", {}, {"num_draft_entries": 1}, 0),
            ("mha_draft", {}, {"draft_kv_entry_count": 1, "num_draft_entries": 1}, 1),
            ("dst_draft", {}, {}, 1),
            ("compressed", {}, {"mla_compression_ratios": [4, 128]}, 0),
            ("partial_start", {}, {"prefill_start_layer": 1}, 0),
            ("partial_end", {}, {"prefill_end_layer": 1}, 0),
        )
        for name, manager_changes, arg_changes, dst_draft in cases:
            for src_ids, dst_ids in (([0, 1], []), ([], [0, 1])):
                with self.subTest(case=name, src_ids=src_ids):
                    m = self._plain_manager(src_ids)
                    for key, value in manager_changes.items():
                        setattr(m, key, value)
                    for key, value in arg_changes.items():
                        setattr(m.kv_args, key, value)
                    with self.assertRaisesRegex(RuntimeError, "Layer metadata"):
                        m._resolve_kv_entry_indices(dst_ids, 2, dst_draft)
        for src_ids, dst_ids in (([1, 0], []), ([], [1, 0])):
            with self.subTest(unordered=(src_ids, dst_ids)):
                m = self._plain_manager(src_ids)
                with self.assertRaisesRegex(RuntimeError, "Layer metadata"):
                    m._resolve_kv_entry_indices(dst_ids, 2)
        for src_ids, dst_ids in (([0, 1], []), ([], [0, 1, 2])):
            with self.subTest(extra_destination_entry=(src_ids, dst_ids)):
                m = self._plain_manager(src_ids)
                with self.assertRaisesRegex(RuntimeError, "Layer metadata"):
                    m._resolve_kv_entry_indices(dst_ids, 3)

    def test_invalid_draft_entry_counts_rejected(self):
        for src_draft, dst_draft in ((-1, 0), (3, 0), (0, -1), (0, 3), (1, 0), (1, 2)):
            with self.subTest(src_draft=src_draft, dst_draft=dst_draft):
                m = self._plain_manager([])
                m.kv_args.draft_kv_entry_count = src_draft
                with self.assertRaisesRegex(ValueError, "draft entry count"):
                    m._resolve_kv_entry_indices([], 2, dst_draft)

    def test_destination_must_cover_every_source_entry(self):
        m = self._plain_manager([])
        with self.assertRaisesRegex(ValueError, "Destination entries"):
            m._resolve_kv_entry_indices([], 1)
        m = self._plain_manager([0, 1])
        with self.assertRaisesRegex(RuntimeError, "missing a transfer entry"):
            m._resolve_kv_entry_indices([0], 1)

    def test_plain_mla_pp_one_sided_ids_copy_the_correct_stage(self):
        # The old wire has no IDs. The upgraded peer must use the existing PP
        # slice, not positional zip(), for both upgrade directions.
        for stage in (0, 1):
            for old_peer in ("prefill", "decode"):
                with self.subTest(stage=stage, old_peer=old_peer):
                    start = stage * 2
                    src_ids = [] if old_peer == "prefill" else [start, start + 1]
                    dst_ids = [] if old_peer == "decode" else [0, 1, 2, 3]
                    m = self._plain_manager(src_ids)
                    m.pp_size = 2
                    m.pp_rank = stage
                    m.kv_args.prefill_start_layer = start
                    m.kv_args.prefill_end_layer = start + 2
                    dst_ptrs = [10000000, 20000000, 30000000, 40000000]
                    sources = {
                        base: (np.arange(width * 3, dtype=np.uint32) + 19).astype(
                            np.uint8
                        )
                        for base, width in zip(m.kv_args.kv_data_ptrs, [12, 20])
                    }
                    destinations = {
                        base: np.full(width * 4, 205, dtype=np.uint8)
                        for base, width in zip(dst_ptrs, [12, 20, 12, 20])
                    }
                    expected = {
                        base: data.copy() for base, data in destinations.items()
                    }
                    records = []
                    m._transfer_data = self._copy_transport(
                        sources, destinations, records
                    )
                    result = m._send_kvcache_generic(
                        "session",
                        m.kv_args.kv_data_ptrs,
                        dst_ptrs,
                        m.kv_args.kv_item_lens,
                        np.array([1], dtype=np.int32),
                        np.array([2], dtype=np.int32),
                        None,
                        src_layer_ids=src_ids,
                        dst_layer_ids=dst_ids,
                    )
                    self.assertEqual(result, 0)
                    for i, width in enumerate((12, 20)):
                        expected[dst_ptrs[start + i]][2 * width : 3 * width] = sources[
                            m.kv_args.kv_data_ptrs[i]
                        ][width : 2 * width]
                    for base in dst_ptrs:
                        np.testing.assert_array_equal(
                            destinations[base], expected[base]
                        )

    def test_pp_legacy_mapping_rejects_wrong_known_layer_span(self):
        for src_ids, dst_ids in (([0, 1], []), ([], [1, 2, 3, 4])):
            with self.subTest(src_ids=src_ids, dst_ids=dst_ids):
                m = self._plain_manager(src_ids)
                m.pp_size = 2
                m.kv_args.prefill_start_layer = 2
                m.kv_args.prefill_end_layer = 4
                with self.assertRaisesRegex(RuntimeError, "Layer metadata"):
                    m._resolve_kv_entry_indices(dst_ids, 4)

    def test_plain_mla_metadata_keeps_old_wire_but_preserves_layer_shards(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        for layer_shard, groups in ((False, 1), (True, 1), (False, 3)):
            with self.subTest(layer_shard=layer_shard, groups=groups):
                pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
                pool.start_layer = 2
                pool.layer_num = 2
                pool.layer_shard_enabled = layer_shard
                count = 2 * groups
                pool.get_contiguous_buf_infos = Mock(
                    return_value=(list(range(count)), [1024] * count, [256] * count)
                )
                with patch(
                    "sglang.srt.disaggregation.utils.is_npu", return_value=False
                ):
                    ids = build_kv_layer_ids(
                        token_to_kv_pool=pool,
                        draft_token_to_kv_pool=None,
                        num_draft_entries=0,
                        num_hidden_layers=4,
                    )
                self.assertEqual(
                    ids, [2, 3] * groups if layer_shard or groups > 1 else []
                )

    def test_complete_metadata_retains_strict_layer_mapping(self):
        m = self._plain_manager([1, 3])
        m.pp_size = 2
        m.kv_args.prefill_start_layer = 1
        m.kv_args.prefill_end_layer = 4
        self.assertEqual(m._resolve_kv_entry_indices([0, 1, 2, 3], 4), [1, 3])


if __name__ == "__main__":
    unittest.main()
