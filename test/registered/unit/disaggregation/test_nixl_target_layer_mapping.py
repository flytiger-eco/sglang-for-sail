"""NIXL target mappings from the metadata emitted by real KV pool types."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.srt.disaggregation.utils import build_kv_layer_ids
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pool(start, layers, *, native=True):
    cls = MLATokenToKVPool if native else MHATokenToKVPool
    pool = cls.__new__(cls)
    pool.start_layer = start
    pool.layer_num = layers
    pool.layer_shard_enabled = False
    count = layers if native else 2 * layers
    # No tensors or accelerator allocation: mock only exported storage, not IDs.
    pool.get_contiguous_buf_infos = Mock(
        return_value=(list(range(count)), [1024] * count, [64] * count)
    )
    return pool


def _ids(pool, draft=None):
    draft_entries = len(draft.get_contiguous_buf_infos()[0]) if draft else 0
    with patch("sglang.srt.disaggregation.utils.is_npu", return_value=False):
        return build_kv_layer_ids(
            token_to_kv_pool=pool,
            draft_token_to_kv_pool=draft,
            num_draft_entries=draft_entries,
            num_hidden_layers=4,
        )


class TestNixlTargetLayerMapping(CustomTestCase):
    def _fixture(
        self,
        *,
        start=0,
        layers=4,
        pp=1,
        dcp=1,
        decode_start=0,
        decode_layers=4,
        source_draft=None,
        decode_draft=None,
    ):
        source_pool = _pool(start, layers)
        decode_pool = _pool(decode_start, decode_layers)
        source_ids = _ids(source_pool, source_draft)
        decode_ids = _ids(decode_pool, decode_draft)
        source_draft_entries = (
            len(source_draft.get_contiguous_buf_infos()[0]) if source_draft else 0
        )
        decode_draft_entries = (
            len(decode_draft.get_contiguous_buf_infos()[0]) if decode_draft else 0
        )
        n_src = layers + source_draft_entries
        n_dst = decode_layers + decode_draft_entries
        manager = NixlKVManager.__new__(NixlKVManager)
        manager.pp_size = pp
        manager.attn_cp_size = 1
        manager.is_mla_backend = True
        manager.is_hybrid_mla_backend = False
        manager.kv_args = SimpleNamespace(
            prefill_start_layer=start,
            prefill_end_layer=start + layers,
            num_draft_entries=source_draft_entries,
            kv_layer_ids=source_ids,
            kv_item_lens=[64] * n_src,
            kv_data_ptrs=[1000 + 100 * i for i in range(n_src)],
            mla_compression_ratios=None,
            page_size=32,
            gpu_id=0,
        )
        manager.src_mem_kind = "VRAM"
        manager.prep_handles = {}
        manager._num_slots_src = 16
        # Only the external NIXL descriptor-registration boundary is mocked.
        manager._init_equal_tp_prep_handle = Mock()
        peer = SimpleNamespace(
            dst_kv_layer_ids=decode_ids,
            dst_kv_item_lens=[64] * decode_layers + [64 * dcp] * decode_draft_entries,
            dst_kv_ptrs=[10000 + 100 * i for i in range(n_dst)],
            dst_kv_mem_kinds=["VRAM"] * n_dst,
            requires_dcp_relayout=dcp > 1,
            dst_dcp_size=dcp,
            dst_num_slots=16,
            agent_name="decode",
            gpu_id=1,
        )
        return manager, peer

    def _assert_prepared(self, manager, peer, expected):
        manager._prepare_payload_xfer(peer)
        if peer.requires_dcp_relayout:
            self.assertEqual(peer.dcp_dst_region_indices, expected)
            self.assertEqual(peer.dcp_token_item_lens, [2] * len(expected))
            manager._init_equal_tp_prep_handle.assert_not_called()
        else:
            call = manager._init_equal_tp_prep_handle.call_args
            self.assertEqual(call.args[0], "decode")
            self.assertEqual(call.args[1], [peer.dst_kv_ptrs[i] for i in expected])
            self.assertEqual(
                call.kwargs["kv_item_lens"],
                [peer.dst_kv_item_lens[i] for i in expected],
            )
            self.assertEqual(
                call.kwargs["kv_data_lens"],
                [peer.dst_kv_item_lens[i] * 16 for i in expected],
            )
            self.assertEqual(call.kwargs["kv_xfer_lens"], manager.kv_args.kv_item_lens)

    def test_decode_only_speculation_maps_target_without_draft_regions(self):
        for dcp in (1, 4):
            with self.subTest(dcp=dcp):
                manager, peer = self._fixture(
                    dcp=dcp, decode_draft=_pool(0, 1, native=False)
                )
                self.assertEqual(manager.kv_args.kv_layer_ids, [])
                self.assertEqual(peer.dst_kv_layer_ids, [0, 1, 2, 3, 4, 4])
                self._assert_prepared(manager, peer, [0, 1, 2, 3])

    def test_draft_free_prefill_stages_map_to_full_decode_target(self):
        for start in (0, 2):
            for dcp in (1, 4):
                with self.subTest(start=start, dcp=dcp):
                    manager, peer = self._fixture(
                        start=start,
                        layers=2,
                        pp=2,
                        dcp=dcp,
                        decode_draft=_pool(0, 1, native=False),
                    )
                    self.assertEqual(manager.kv_args.kv_layer_ids, [])
                    self._assert_prepared(manager, peer, [start, start + 1])

    def test_matching_pp_decode_with_draft_uses_local_target_entries(self):
        for dcp in (1, 4):
            for native in (False, True):
                with self.subTest(dcp=dcp, native_draft=native):
                    manager, peer = self._fixture(
                        start=2,
                        layers=2,
                        pp=2,
                        dcp=dcp,
                        decode_start=2,
                        decode_layers=2,
                        decode_draft=_pool(0, 1, native=native),
                    )
                    expected_ids = [2, 3, 4] if native else [2, 3, 4, 4]
                    self.assertEqual(peer.dst_kv_layer_ids, expected_ids)
                    self._assert_prepared(manager, peer, [0, 1])

    def test_matching_pp_draft_does_not_allow_invalid_target_span(self):
        for dcp in (1, 4):
            for invalid_ids in (
                [3, 2, 4, 4],
                [2, 3, 3, 4],
                [1, 2, 4, 4],
                [2, 4, 4, 4],
                [2, 3, 4, 99],
                [2, 3, 99, 99],
            ):
                with self.subTest(dcp=dcp, ids=invalid_ids):
                    manager, peer = self._fixture(
                        start=2,
                        layers=2,
                        pp=2,
                        dcp=dcp,
                        decode_start=2,
                        decode_layers=2,
                        decode_draft=_pool(0, 1, native=False),
                    )
                    peer.dst_kv_layer_ids = invalid_ids
                    with self.assertRaises(RuntimeError):
                        manager._prepare_payload_xfer(peer)
                    # Preparing our own source descriptor is harmless; reject
                    # the peer before registering any destination descriptor.
                    self.assertFalse(
                        any(
                            call.args[0] == "decode"
                            for call in manager._init_equal_tp_prep_handle.call_args_list
                        )
                    )

    def test_empty_id_legacy_whole_pages_keep_existing_pp_slice(self):
        manager, peer = self._fixture(start=2, layers=2, pp=2)
        self.assertEqual(manager.kv_args.kv_layer_ids, [])
        self.assertEqual(peer.dst_kv_layer_ids, [])
        self._assert_prepared(manager, peer, [2, 3])

    def test_native_draft_dcp_virtual_page_width_is_not_rejected(self):
        manager, peer = self._fixture(
            dcp=4, source_draft=_pool(0, 1), decode_draft=_pool(0, 1)
        )
        self.assertEqual(manager.kv_args.kv_layer_ids, [0, 1, 2, 3, 4])
        self.assertEqual(peer.dst_kv_layer_ids, [0, 1, 2, 3, 4])
        self.assertEqual(peer.dst_kv_item_lens, [64, 64, 64, 64, 256])
        self._assert_prepared(manager, peer, [0, 1, 2, 3, 4])

    def test_missing_target_id_and_wrong_target_width_are_rejected(self):
        for dcp in (1, 4):
            for invalid in ("missing_layer", "width"):
                with self.subTest(dcp=dcp, invalid=invalid):
                    manager, peer = self._fixture(
                        dcp=dcp, decode_draft=_pool(0, 1, native=False)
                    )
                    if invalid == "missing_layer":
                        peer.dst_kv_layer_ids[2] = 99
                    else:
                        peer.dst_kv_item_lens[2] = 96
                    with self.assertRaises(RuntimeError):
                        manager._prepare_payload_xfer(peer)
                    self.assertFalse(
                        any(
                            call.args[0] == "decode"
                            for call in manager._init_equal_tp_prep_handle.call_args_list
                        )
                    )


if __name__ == "__main__":
    unittest.main()
