"""Whole-page Mooncake transfers across matching and legacy PD layouts."""

import struct
import unittest
from unittest.mock import Mock

import numpy as np
from test_mooncake_mixed_transfer import manager, registration

from sglang.srt.disaggregation.mooncake.conn import KVArgsRegisterInfo
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMooncakeWholePageCompat(CustomTestCase):
    @staticmethod
    def _manager(dcp=1, rank=0, token_major=True):
        m = manager(tp=8)
        m.dcp_size = dcp
        m.dcp_rank = rank
        m.kv_args.draft_kv_head_num = 2
        m.kv_args.prefill_start_layer = 0
        m.kv_args.prefill_end_layer = 1
        m.kv_args.draft_kv_token_major = token_major
        m.kv_args.kv_item_lens = [64, 2048 * dcp, 2048 * dcp]
        m.kv_args.kv_data_ptrs = [1000000, 2000000, 3000000]
        m._init_dcp_pack_buffers_once = Mock()
        return m

    @staticmethod
    def _wire(m, version):
        wire = registration("whole", token_major=m.kv_args.draft_kv_token_major)
        wire[4] = struct.pack("3Q", 10000000, 20000000, 30000000)
        wire[16:18] = [str(m.dcp_size).encode(), str(m.dcp_rank).encode()]
        wire[18] = struct.pack("3Q", *m.kv_args.kv_item_lens)
        if version == "legacy23":
            return wire[:23]
        if version == "legacy24":
            return wire
        current = wire[:18] + [b""] + wire[18:]
        if version == "old19":
            return current[:19]
        if version == "old20":
            return current[:20]
        return current

    def test_matching_layout_registration_and_page_bytes(self):
        for dcp, rank in ((1, 0), (8, 0), (8, 7)):
            for token_major in (True, False):
                for version in ("old19", "old20", "legacy23", "legacy24", "current25"):
                    with self.subTest(
                        dcp=dcp, rank=rank, layout=token_major, wire=version
                    ):
                        m = self._manager(dcp, rank, token_major)
                        wire = self._wire(m, version)
                        m._register_decode_peer(wire, "whole")
                        self.assertNotIn("whole", m.rejected_registrations)
                        info = m.decode_kv_args_table["whole"]
                        self.assertFalse(info.requires_dcp_relayout)
                        self.assertFalse(m._uses_mixed_kv_transfer(info))
                        m._init_dcp_pack_buffers_once.assert_not_called()
                        widths = m.kv_args.kv_item_lens
                        sources = {
                            base: (
                                np.arange(4 * width, dtype=np.uint32) + i * 11
                            ).astype(np.uint8)
                            for i, (base, width) in enumerate(
                                zip(m.kv_args.kv_data_ptrs, widths)
                            )
                        }
                        destinations = {
                            base: np.full(6 * width, 205, dtype=np.uint8)
                            for base, width in zip(info.dst_kv_ptrs, widths)
                        }
                        expected = {
                            base: data.copy() for base, data in destinations.items()
                        }

                        def transfer(session, blocks):
                            self.assertEqual(session, "whole")
                            for src, dst, size in blocks:
                                src_base = max(base for base in sources if base <= src)
                                dst_base = max(
                                    base for base in destinations if base <= dst
                                )
                                self.assertGreater(size, 0)
                                self.assertLessEqual(
                                    src - src_base + size, sources[src_base].size
                                )
                                self.assertLessEqual(
                                    dst - dst_base + size, destinations[dst_base].size
                                )
                                destinations[dst_base][
                                    dst - dst_base : dst - dst_base + size
                                ] = sources[src_base][
                                    src - src_base : src - src_base + size
                                ]
                            return 0

                        m._transfer_data = transfer
                        result = m._send_kvcache_generic(
                            "whole",
                            m.kv_args.kv_data_ptrs,
                            info.dst_kv_ptrs,
                            widths,
                            np.array([1, 3], dtype=np.int32),
                            np.array([2, 5], dtype=np.int32),
                            None,
                            src_layer_ids=m.kv_args.kv_layer_ids,
                            dst_layer_ids=info.dst_kv_layer_ids,
                        )
                        self.assertEqual(result, 0)
                        for i, width in enumerate(widths):
                            for src_page, dst_page in ((1, 2), (3, 5)):
                                expected[info.dst_kv_ptrs[i]][
                                    dst_page * width : (dst_page + 1) * width
                                ] = sources[m.kv_args.kv_data_ptrs[i]][
                                    src_page * width : (src_page + 1) * width
                                ]
                        for base in info.dst_kv_ptrs:
                            np.testing.assert_array_equal(
                                destinations[base], expected[base]
                            )

    def test_old_wire_one_sided_ids_keep_plain_mla_draft_page_mapping(self):
        m = self._manager()
        wire = self._wire(m, "old20")
        wire[12] = b""
        m._register_decode_peer(wire, "whole")
        self.assertNotIn("whole", m.rejected_registrations)
        info = m.decode_kv_args_table["whole"]
        m._transfer_data = Mock(return_value=0)
        self.assertEqual(
            m._send_kvcache_generic(
                "whole",
                m.kv_args.kv_data_ptrs,
                info.dst_kv_ptrs,
                m.kv_args.kv_item_lens,
                np.array([1], dtype=np.int32),
                np.array([2], dtype=np.int32),
                None,
                src_layer_ids=m.kv_args.kv_layer_ids,
                dst_layer_ids=[],
            ),
            0,
        )
        self.assertEqual(
            m._transfer_data.call_args.args[1],
            [
                (1000064, 10000128, 64),
                (2002048, 20004096, 2048),
                (3002048, 30004096, 2048),
            ],
        )

    def test_old_compressed_peer_keeps_one_sided_pp1_whole_page_mapping(self):
        m = self._manager()
        m.kv_args.mla_compression_ratios = [4]
        wire = self._wire(m, "old20")
        wire[12] = b""
        m._register_decode_peer(wire, "whole")
        self.assertNotIn("whole", m.rejected_registrations)
        info = m.decode_kv_args_table["whole"]
        m._transfer_data = Mock(return_value=0)
        with get_context().override_server_args(enable_unified_memory=False):
            result = m.send_kvcache(
                "whole",
                np.array([1], dtype=np.int32),
                info.dst_kv_ptrs,
                np.array([2], dtype=np.int32),
                None,
                dst_layer_ids=[],
                dst_kv_item_len=info.dst_kv_item_len,
                dst_attn_tp_size=8,
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            m._transfer_data.call_args.args[1],
            [
                (1000064, 10000128, 64),
                (2002048, 20004096, 2048),
                (3002048, 30004096, 2048),
            ],
        )

    def test_full_metadata_mismatch_is_not_legacy(self):
        for field, value in (
            (20, b"1"),
            (21, b"1"),
            (22, b"32"),
            (23, b"32"),
            (24, b"0"),
            (19, struct.pack("3Q", 64, 4096, 2048)),
        ):
            with self.subTest(field=field):
                m = self._manager()
                wire = self._wire(m, "current25")
                wire[field] = value
                m._register_decode_peer(wire, "bad")
                self.assertIn("bad", m.rejected_registrations)
                self.assertNotIn("bad", m.decode_kv_args_table)

    def test_missing_or_partial_metadata_does_not_enable_remapping(self):
        for version in ("old19", "old20", "legacy23"):
            with self.subTest(version=version):
                m = self._manager()
                wire = self._wire(m, version)
                wire[8] = b"4"
                m._register_decode_peer(wire, "bad")
                self.assertIn("bad", m.rejected_registrations)
        for fields in (1, 2, 3):
            with self.subTest(partial_fields=fields):
                m = self._manager()
                wire = self._wire(m, "current25")[: 20 + fields]
                m._register_decode_peer(wire, "bad")
                self.assertIn("bad", m.rejected_registrations)

    def test_dcp_rank_must_match_for_whole_page_transfer(self):
        m = self._manager(8, 7)
        wire = self._wire(m, "current25")
        wire[17] = b"6"
        m._register_decode_peer(wire, "bad")
        self.assertIn("bad", m.rejected_registrations)

    def test_legacy_layout_presence_is_distinct_from_false(self):
        m = self._manager(token_major=False)
        old = KVArgsRegisterInfo.from_zmq(self._wire(m, "old20"))
        legacy = KVArgsRegisterInfo.from_zmq(self._wire(m, "legacy23"))
        current = KVArgsRegisterInfo.from_zmq(self._wire(m, "current25"))
        self.assertFalse(old.dst_draft_metadata_present)
        self.assertTrue(legacy.dst_draft_metadata_present)
        self.assertIsNone(old.dst_draft_token_major)
        self.assertIsNone(legacy.dst_draft_token_major)
        self.assertIs(current.dst_draft_token_major, False)


if __name__ == "__main__":
    unittest.main()
