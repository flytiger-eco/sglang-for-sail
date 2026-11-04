"""CPU regressions for independent target/draft Mooncake geometry."""

import struct
import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import KVArgs
from sglang.srt.disaggregation.common.utils import (
    DCPTokenTransferPlan,
    build_dcp_token_transfer_plan,
)
from sglang.srt.disaggregation.mooncake.conn import (
    KVArgsRegisterInfo,
    MooncakeKVManager,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPDDraftGeometry(CustomTestCase):
    def test_contiguous_full_head_rows_are_coalesced(self):
        from sglang.srt.disaggregation.common.utils import (
            build_draft_head_transfer_blocks,
        )

        for count in (0, 1, 65, 8192):
            with self.subTest(count=count):
                plan = DCPTokenTransferPlan(np.arange(count), np.arange(count) + 64)
                blocks = build_draft_head_transfer_blocks(
                    plan,
                    src_ptr=10000,
                    dst_ptr=1000000,
                    src_token_bytes=32,
                    dst_token_bytes=32,
                    total_heads=16,
                    src_tp_size=8,
                    src_tp_rank=0,
                    dst_tp_size=8,
                    dst_tp_rank=0,
                )
                self.assertEqual(
                    blocks, [(10000, 1000000 + 64 * 32, count * 32)] if count else []
                )

    def test_coalescing_preserves_source_and_destination_gaps(self):
        from sglang.srt.disaggregation.common.utils import (
            build_draft_head_transfer_blocks,
        )

        plan = DCPTokenTransferPlan(
            np.array([0, 1, 3, 4, 5]), np.array([4, 5, 6, 8, 7])
        )
        blocks = build_draft_head_transfer_blocks(
            plan,
            src_ptr=10000,
            dst_ptr=20000,
            src_token_bytes=32,
            dst_token_bytes=32,
            total_heads=16,
            src_tp_size=8,
            src_tp_rank=0,
            dst_tp_size=8,
            dst_tp_rank=0,
        )
        self.assertEqual(
            blocks,
            [
                (10000, 20128, 64),
                (10096, 20192, 32),
                (10128, 20256, 32),
                (10160, 20224, 32),
            ],
        )

    def test_replicated_draft_plan_includes_empty_target_shard(self):
        args = dict(physical_page_size=4, dcp_size=8, dcp_rank=7, num_kv_tokens=1)
        self.assertEqual(
            build_dcp_token_transfer_plan(
                np.array([3]), np.array([2]), **args
            ).src_token_indices.size,
            0,
        )
        plan = build_dcp_token_transfer_plan(
            np.array([3]), np.array([2]), replicate_to_virtual=True, **args
        )
        np.testing.assert_array_equal(plan.src_token_indices, [12])
        np.testing.assert_array_equal(plan.dst_token_indices, [64])

    def test_metadata_preserves_total_heads_under_replication(self):
        from sglang.srt.disaggregation.utils import set_draft_kv_transfer_metadata

        args = KVArgs()
        set_draft_kv_transfer_metadata(
            args,
            SimpleNamespace(head_num=1),
            draft_kv_entry_count=2,
            tp_size=32,
            total_head_num=16,
        )
        self.assertEqual(
            (
                args.draft_kv_entry_count,
                args.draft_kv_head_num,
                args.draft_total_kv_head_num,
            ),
            (2, 1, 16),
        )

    def test_invalid_token_geometry(self):
        for changes in (
            dict(dcp_size=0),
            dict(dcp_rank=8),
            dict(physical_page_size=0),
            dict(src_page_offset=-1),
        ):
            args = dict(physical_page_size=4, dcp_size=8, dcp_rank=0)
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_dcp_token_transfer_plan(np.array([1]), np.array([2]), **args)

    def test_draft_heads_byte_exact_once_for_heterogeneous_tp(self):
        from sglang.srt.disaggregation.common.utils import (
            build_draft_head_transfer_blocks,
        )

        # Scalar reference covers replication, gathering, scattering and holes.
        for src_tp, dst_tp in (
            (16, 8),
            (32, 8),
            (32, 16),
            (64, 32),
            (8, 32),
            (8, 8),
            (16, 32),
        ):
            for dst_rank in range(dst_tp):
                src_heads, dst_heads = max(1, 16 // src_tp), max(1, 16 // dst_tp)
                out = {}
                plan = DCPTokenTransferPlan(np.array([3, 9, 4]), np.array([5, 1, 7]))
                for src_rank in range(src_tp):
                    blocks = build_draft_head_transfer_blocks(
                        plan,
                        src_ptr=10000,
                        dst_ptr=20000,
                        src_token_bytes=src_heads * 3,
                        dst_token_bytes=dst_heads * 3,
                        total_heads=16,
                        src_tp_size=src_tp,
                        src_tp_rank=src_rank,
                        dst_tp_size=dst_tp,
                        dst_tp_rank=dst_rank,
                    )
                    src_begin = src_rank // max(1, src_tp // 16) * src_heads
                    for src, dst, length in blocks:
                        for byte in range(length):
                            source_byte = src + byte - 10000
                            slot, column = divmod(source_byte, src_heads * 3)
                            value = (slot, src_begin + column // 3, column % 3)
                            self.assertNotIn(dst + byte, out)
                            out[dst + byte] = value
                expected = {}
                dst_begin = dst_rank // max(1, dst_tp // 16) * dst_heads
                for src, dst in zip([3, 9, 4], [5, 1, 7]):
                    for head in range(dst_heads):
                        for byte in range(3):
                            expected[20000 + dst * dst_heads * 3 + head * 3 + byte] = (
                                src,
                                dst_begin + head,
                                byte,
                            )
                self.assertEqual(out, expected)

    def test_invalid_head_geometry_rejected_even_for_empty_plan(self):
        from sglang.srt.disaggregation.common.utils import (
            build_draft_head_transfer_blocks,
        )

        plan = DCPTokenTransferPlan(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        )
        for changes in (
            dict(total_heads=0),
            dict(src_tp_rank=16),
            dict(dst_tp_size=3),
            dict(dst_token_bytes=7),
        ):
            args = dict(
                src_ptr=0,
                dst_ptr=0,
                src_token_bytes=3,
                dst_token_bytes=6,
                total_heads=16,
                src_tp_size=16,
                src_tp_rank=0,
                dst_tp_size=8,
                dst_tp_rank=0,
            )
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                build_draft_head_transfer_blocks(plan, **args)

    def test_registration_rejects_invalid_mixed_geometry(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.attn_tp_size = 16
        mgr.kv_args = SimpleNamespace(
            page_size=4,
            kv_data_ptrs=[10, 20],
            kv_item_lens=[24, 12],
            kv_layer_ids=[],
            draft_kv_token_major=True,
            draft_kv_entry_count=1,
            draft_kv_head_num=1,
            draft_total_kv_head_num=16,
            engine_rank=0,
        )
        base = dict(
            dst_kv_ptrs=[30, 40],
            dst_kv_item_lens=[24, 192],
            dst_draft_token_major=True,
            dst_draft_entry_count=1,
            dst_page_size=4,
            dst_kv_layer_ids=[],
            dst_dcp_size=8,
            dst_dcp_rank=0,
            dst_attn_tp_size=8,
            dst_tp_rank=0,
            dst_draft_head_num=2,
            dst_draft_total_head_num=16,
        )
        for changes in (
            dict(dst_kv_item_lens=None),
            dict(dst_kv_item_lens=[24]),
            dict(dst_kv_item_lens=[24, 0]),
            dict(dst_kv_item_lens=[24, 191]),
            dict(dst_draft_entry_count=0),
            dict(dst_page_size=8),
            dict(dst_draft_total_head_num=32),
            dict(dst_draft_head_num=1),
            dict(dst_kv_item_lens=[20, 192]),
            dict(dst_tp_rank=8),
            dict(dst_dcp_rank=8),
        ):
            info = SimpleNamespace(**dict(base, **changes))
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                mgr.prepare_dcp_entry_geometry(info)

    def test_target_dedup_does_not_skip_required_draft_sender(self):
        # Exercises the actual queue-entry gate that used to drop the whole
        # envelope before the draft sender could be called.
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.attn_tp_size = 16
        mgr.attn_cp_size = 1
        mgr.is_hybrid_mla_backend = True
        mgr.kv_args = SimpleNamespace(engine_rank=1, draft_kv_entry_count=1)
        info = SimpleNamespace(dst_attn_tp_size=8)
        self.assertEqual(mgr._get_dsa_cache_transfer_skip_flags(info), (False, False))
        mgr.kv_args.draft_kv_entry_count = 0
        self.assertEqual(mgr._get_dsa_cache_transfer_skip_flags(info), (True, False))
        mgr.is_hybrid_mla_backend = False
        self.assertEqual(mgr._get_dsa_cache_transfer_skip_flags(info), (False, False))

    def test_source_dcp_mixed_layout_is_rejected(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.dcp_size = 8
        mgr.kv_args = SimpleNamespace(kv_item_lens=[24, 12], draft_kv_entry_count=1)
        with self.assertRaisesRegex(ValueError, "Prefill DCP1"):
            mgr.kv_args.draft_kv_token_major = True
            mgr.prepare_dcp_entry_geometry(SimpleNamespace(dst_draft_token_major=True))

    def test_failure_status_reaches_transfer_caller(self):
        from unittest.mock import Mock

        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.attn_tp_size = 16
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = True
        mgr.kv_args = SimpleNamespace(
            engine_rank=1,
            page_size=4,
            kv_data_ptrs=[10000, 20000],
            kv_layer_ids=[],
            draft_kv_token_major=True,
            draft_kv_entry_count=1,
            draft_total_kv_head_num=16,
        )
        info = SimpleNamespace(
            dst_kv_ptrs=[200000, 300000],
            dst_kv_item_len=24,
            dst_attn_tp_size=8,
            dst_tp_rank=0,
            dst_dcp_size=8,
            dst_dcp_rank=7,
            dst_kv_layer_ids=[],
            dcp_token_item_lens=[6, 3],
            dcp_dst_token_item_lens=[6, 6],
        )
        mgr._validate_envelope_kv_layout = Mock()
        mgr._transfer_data = Mock(return_value=17)
        self.assertEqual(
            mgr.send_kvcache_mixed(
                "s",
                np.array([1]),
                np.array([2]),
                info=info,
                src_page_offset=0,
                decode_prefix_len=0,
                num_kv_tokens=1,
            ),
            17,
        )
        mgr._transfer_data.assert_called_once()
        self.assertTrue(
            all(src >= 20000 for src, _, _ in mgr._transfer_data.call_args.args[1])
        )

    def test_legacy_registration_and_malformed_extension(self):
        msg = [
            b"None",
            b"host",
            b"1",
            b"s",
            struct.pack("Q", 20),
            b"",
            b"",
            b"0",
            b"8",
            b"24",
            b"",
            b"",
            b"",
            b"",
            b"",
            b"",
            b"8",
            b"0",
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertIsNone(info.dst_kv_item_lens)
        with self.assertRaises(struct.error):
            KVArgsRegisterInfo.from_zmq(msg + [b"bad"])
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.attn_tp_size = 16
        mgr.kv_args = SimpleNamespace(
            page_size=4,
            kv_data_ptrs=[10],
            kv_item_lens=[24],
            kv_layer_ids=[],
            draft_kv_entry_count=0,
        )
        mgr.prepare_dcp_entry_geometry(info)
        self.assertEqual(info.dcp_token_item_lens, [6])

    def test_layer_ids_disambiguate_draft_and_target(self):
        from sglang.srt.disaggregation.utils import get_transfer_kv_layer_ids

        target = SimpleNamespace(get_kv_layer_ids=lambda: [3, 7])
        draft = SimpleNamespace(
            start_layer=0,
            layer_num=2,
            get_contiguous_buf_infos=lambda: ([1, 2, 3, 4], [], []),
        )
        self.assertEqual(
            get_transfer_kv_layer_ids(target, draft, 94), [3, 7, 94, 95, 94, 95]
        )
        self.assertEqual(get_transfer_kv_layer_ids(target, None, 94), [3, 7])

    def test_metadata_does_not_guess_replicated_total_heads(self):
        from sglang.srt.disaggregation.utils import set_draft_kv_transfer_metadata

        with self.assertRaises(ValueError):
            set_draft_kv_transfer_metadata(
                KVArgs(),
                SimpleNamespace(head_num=1),
                draft_kv_entry_count=2,
                tp_size=32,
            )

    def test_metadata_through_wire_and_actual_mixed_transfer(self):
        for count in (1, 9, 33, 65):
            for offset in (0, 1):
                with self.subTest(count=count, offset=offset):
                    self._exercise_mixed(count, offset)

    def _exercise_mixed(self, count, offset):
        from sglang.srt.disaggregation.utils import set_draft_kv_transfer_metadata

        # P TP16 -> D attention TP8 (global TP32/DP4), DCP1 and DCP8.
        # Each D rank contacts two P ranks; each contributes one draft head.
        src_pages = [1 + 2 * i for i in range((count + 3) // 4)]
        dst_pages = [2 + 3 * i for i in range((count + offset * 4 + 3) // 4)]
        for dcp in (1, 8):
            for dst_rank in range(8):
                target_writes, draft_writes = {}, {}
                for src_rank in range(dst_rank * 2, dst_rank * 2 + 2):
                    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
                    mgr.attn_tp_size = 16
                    mgr.attn_cp_size = 1
                    mgr.is_mla_backend = False
                    mgr.is_hybrid_mla_backend = True
                    # The memory envelope gate requires a published server bag;
                    # this CPU fixture supplies the supported non-envelope boundary.
                    mgr._validate_envelope_kv_layout = lambda *args: None
                    mgr.kv_args = KVArgs()
                    args = mgr.kv_args
                    args.engine_rank = src_rank
                    args.page_size = 4
                    args.kv_data_ptrs = [10000, 20000]
                    args.kv_item_lens = [24, 12]
                    args.kv_layer_ids = [3, 94]
                    set_draft_kv_transfer_metadata(
                        args,
                        SimpleNamespace(head_num=1),
                        draft_kv_entry_count=1,
                        tp_size=16,
                        total_head_num=16,
                    )
                    args.draft_kv_token_major = (
                        True  # Synthetic contiguous address fixture.
                    )
                    msg = [
                        b"None",
                        b"127.0.0.1",
                        b"1234",
                        b"s",
                        struct.pack("2Q", 200000, 300000),
                        b"",
                        b"",
                        str(dst_rank).encode(),
                        b"8",
                        b"24",
                        b"",
                        b"",
                        struct.pack("2I", 3, 94),
                        b"",
                        b"",
                        b"",
                        str(dcp).encode(),
                        str(dst_rank % dcp).encode(),
                        struct.pack("2Q", 24, 24 * dcp),
                        b"1",
                        b"2",
                        b"16",
                        b"4",
                        b"1",
                    ]
                    info = KVArgsRegisterInfo.from_zmq(msg)
                    mgr.prepare_dcp_entry_geometry(info)
                    self.assertEqual(info.dcp_token_item_lens, [6, 3])
                    self.assertEqual(info.dcp_dst_token_item_lens, [6, 6])
                    self.assertFalse(mgr._get_dsa_cache_transfer_skip_flags(info)[0])

                    def transfer(session, blocks):
                        for src, dst, length in blocks:
                            self.assertGreater(length, 0)
                            is_target = src < 20000
                            out = target_writes if is_target else draft_writes
                            base, width = (10000, 6) if is_target else (20000, 3)
                            for byte in range(length):
                                slot = (src + byte - base) // width
                                token = src_pages.index(slot // 4) * 4 + slot % 4
                                self.assertLess(token, count)
                                self.assertNotIn(dst + byte, out)
                                out[dst + byte] = (17 if is_target else src_rank, token)
                        return 0

                    mgr._transfer_data = transfer
                    ret = mgr.send_kvcache_mixed(
                        "s",
                        np.array(src_pages),
                        np.array(dst_pages),
                        info=info,
                        src_page_offset=offset,
                        decode_prefix_len=64,
                        num_kv_tokens=count,
                    )
                    self.assertEqual(ret, 0)
                expected_draft, expected_target = {}, {}
                for token in range(count):
                    relative = offset * 4 + token
                    draft_slot = dst_pages[relative // (4 * dcp)] * (
                        4 * dcp
                    ) + relative % (4 * dcp)
                    for head in range(2):
                        for byte in range(3):
                            expected_draft[
                                300000 + draft_slot * 6 + head * 3 + byte
                            ] = (dst_rank * 2 + head, token)
                    if (64 + relative) % dcp == dst_rank % dcp:
                        local = relative // dcp
                        target_slot = dst_pages[local // 4] * 4 + local % 4
                        for byte in range(6):
                            expected_target[200000 + target_slot * 6 + byte] = (
                                17,
                                token,
                            )
                self.assertEqual(draft_writes, expected_draft)
                self.assertEqual(target_writes, expected_target)


if __name__ == "__main__":
    unittest.main()
