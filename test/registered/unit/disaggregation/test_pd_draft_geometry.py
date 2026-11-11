"""CPU regressions for independent target/draft Mooncake geometry."""

import struct
import unittest
from unittest.mock import Mock, patch
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
from sglang.srt.disaggregation.utils import (
    build_kv_layer_ids,
    get_draft_total_num_kv_heads,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPDDraftModelRunners(CustomTestCase):
    @staticmethod
    def _runner(total_heads):
        return SimpleNamespace(
            model_config=SimpleNamespace(get_total_num_kv_heads=lambda: total_heads)
        )

    def _scheduler(self, algorithm, draft_worker):
        worker = BaseSpecWorker()
        target_runner = self._runner(128)
        target_runner.spec_algorithm = algorithm
        worker._target_worker = SimpleNamespace(model_runner=target_runner)
        worker._draft_worker = draft_worker
        return SimpleNamespace(draft_worker=worker)

    def test_no_speculative_worker(self):
        for scheduler in (SimpleNamespace(), SimpleNamespace(draft_worker=None)):
            with self.subTest(scheduler=scheduler):
                self.assertIsNone(get_draft_total_num_kv_heads(scheduler))

    def test_dflash_family_uses_draft_model_runner(self):
        for algorithm in (SpeculativeAlgorithm.DSPARK, SpeculativeAlgorithm.DFLASH):
            for compatibility_alias in (False, True):
                with self.subTest(
                    algorithm=algorithm, compatibility_alias=compatibility_alias
                ):
                    runner = self._runner(16)
                    draft_worker = SimpleNamespace(model_runner=runner)
                    # The current builder installs this alias. The shared runner
                    # interface must not require it to determine draft geometry.
                    if compatibility_alias:
                        draft_worker.draft_runner = runner
                    scheduler = self._scheduler(algorithm, draft_worker)
                    self.assertEqual(get_draft_total_num_kv_heads(scheduler), 16)

    def test_eagle_uses_resolved_draft_runners(self):
        for algorithm in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.EAGLE3):
            for num_layers in (1, 3):
                with self.subTest(algorithm=algorithm, num_layers=num_layers):
                    draft_worker = SimpleNamespace(
                        draft_runners=tuple(self._runner(16) for _ in range(num_layers))
                    )
                    scheduler = self._scheduler(algorithm, draft_worker)
                    self.assertEqual(get_draft_total_num_kv_heads(scheduler), 16)

    def test_no_draft_kv_runners(self):
        cases = (
            (SpeculativeAlgorithm.DSPARK, None),
            (SpeculativeAlgorithm.DFLASH, None),
            (SpeculativeAlgorithm.NGRAM, None),
            (
                SpeculativeAlgorithm.FROZEN_KV_MTP,
                SimpleNamespace(model_runner=self._runner(16)),
            ),
        )
        for algorithm, draft_worker in cases:
            with self.subTest(algorithm=algorithm):
                scheduler = self._scheduler(algorithm, draft_worker)
                self.assertIsNone(get_draft_total_num_kv_heads(scheduler))


class TestPDLayerMetadataScope(CustomTestCase):
    def test_non_hybrid_npu_keeps_legacy_metadata(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        for has_draft in (False, True):
            with self.subTest(has_draft=has_draft):
                pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
                # A packed FP8 DSA pool can export six buffers while its
                # backend-specific getter still describes eight entries.
                pool.get_kv_layer_ids = Mock(return_value=[0, 1] * 4)
                pool.get_contiguous_buf_infos = Mock(
                    return_value=(list(range(6)), [1024] * 6, [256] * 6)
                )
                draft = (
                    SimpleNamespace(get_kv_layer_ids=lambda: [0, 0])
                    if has_draft
                    else None
                )
                with patch("sglang.srt.disaggregation.utils.is_npu", return_value=True):
                    ids = build_kv_layer_ids(
                        token_to_kv_pool=pool,
                        draft_token_to_kv_pool=draft,
                        num_draft_entries=2 if has_draft else 0,
                        num_hidden_layers=2,
                    )
                self.assertEqual(ids, [])
                pool.get_kv_layer_ids.assert_not_called()
                pool.get_contiguous_buf_infos.assert_not_called()

    def test_non_npu_mla_keeps_target_and_draft_metadata(self):
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

        pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
        pool.start_layer = 2
        pool.layer_num = 2
        pool.get_contiguous_buf_infos = Mock(
            return_value=(list(range(6)), [1024] * 6, [256] * 6)
        )
        draft = SimpleNamespace(
            start_layer=0,
            layer_num=1,
            get_contiguous_buf_infos=lambda: ([20, 21], [1024, 1024], [256, 256]),
        )
        with patch("sglang.srt.disaggregation.utils.is_npu", return_value=False):
            ids = build_kv_layer_ids(
                token_to_kv_pool=pool,
                draft_token_to_kv_pool=draft,
                num_draft_entries=2,
                num_hidden_layers=4,
            )
        self.assertEqual(ids, [2, 3] * 3 + [4, 4])

    def test_hybrid_npu_keeps_existing_metadata(self):
        from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

        pool = HybridLinearKVPool.__new__(HybridLinearKVPool)
        pool.get_kv_layer_ids = Mock(return_value=[2, 3, 2, 3])
        with patch("sglang.srt.disaggregation.utils.is_npu", return_value=True):
            ids = build_kv_layer_ids(
                token_to_kv_pool=pool,
                draft_token_to_kv_pool=None,
                num_draft_entries=0,
                num_hidden_layers=4,
            )
        self.assertEqual(ids, [2, 3, 2, 3])


class TestPDDraftGeometry(CustomTestCase):
    def test_compressed_target_pages_and_draft_rows(self):
        import ctypes

        import torch

        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4SingleKVPool,
            KVLayout,
        )

        for ratio, expected_bytes in ((4, 37440), (128, 1728)):
            for src_rank in (0, 1):
                with self.subTest(ratio=ratio, src_rank=src_rank):
                    pool = DeepSeekV4SingleKVPool.__new__(DeepSeekV4SingleKVPool)
                    pool.page_size = 256 // ratio
                    pool.kv_layout = KVLayout.V4
                    pool.qk_nope_head_dim = 448
                    pool.qk_rope_head_dim = 64
                    pool.rope_storage_dtype = torch.bfloat16
                    pool.quantize_block_size = KVLayout.V4.tile_size
                    pool.scale_pad = 1
                    pool.store_dtype = torch.uint8
                    pool.device = "cpu"
                    target = pool.create_buffer(num_pages=5)
                    for page_index in range(5):
                        target[page_index].fill_(17 + page_index)
                    dest = torch.zeros_like(target)
                    page_bytes = target.stride(0)
                    self.assertEqual(page_bytes, expected_bytes)
                    draft = (
                        torch.arange(2560, dtype=torch.int32)
                        .remainder(251)
                        .to(torch.uint8)
                        .view(1280, 1, 2)
                    )
                    draft_dest = torch.zeros((1280, 2, 2), dtype=torch.uint8)
                    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
                    mgr.enable_custom_mem_pool = False
                    mgr.custom_mem_pool_type = ""
                    mgr.max_transfer_batch_indices = 0
                    mgr.attn_tp_size = 16
                    mgr.dcp_size = 1
                    mgr.dcp_rank = 0
                    mgr.is_mla_backend = True
                    mgr.is_hybrid_mla_backend = False
                    mgr._validate_envelope_kv_layout = lambda *args: None
                    a = mgr.kv_args = KVArgs()
                    a.page_size = 256
                    a.engine_rank = src_rank
                    a.kv_data_ptrs = [target.data_ptr(), draft.data_ptr()]
                    a.kv_item_lens = [page_bytes, 512]
                    a.kv_layer_ids = [0, 1]
                    a.mla_compression_ratios = [ratio]
                    a.draft_kv_entry_count = 1
                    a.draft_kv_head_num = 1
                    a.draft_total_kv_head_num = 16
                    a.draft_kv_token_major = True
                    info = SimpleNamespace(
                        dst_kv_ptrs=[dest.data_ptr(), draft_dest.data_ptr()],
                        dst_kv_item_lens=[page_bytes, 1024],
                        dst_kv_item_len=page_bytes,
                        dst_kv_layer_ids=[0, 1],
                        dst_draft_entry_count=1,
                        dst_draft_metadata_present=True,
                        dst_draft_token_major=True,
                        dst_draft_head_num=2,
                        dst_draft_total_head_num=16,
                        dst_page_size=256,
                        dst_dcp_size=1,
                        dst_dcp_rank=0,
                        dst_attn_tp_size=8,
                        dst_tp_rank=0,
                    )
                    mgr.prepare_dcp_entry_geometry(info)

                    def transfer(session, blocks):
                        for src, dst, size in blocks:
                            ctypes.memmove(dst, src, size)
                        return 0

                    mgr._transfer_data = transfer
                    mgr.send_kvcache_mixed(
                        "s",
                        np.array([1, 3]),
                        np.array([2, 4]),
                        info=info,
                        src_page_offset=0,
                        decode_prefix_len=0,
                        num_kv_tokens=257,
                    )
                    expected = torch.zeros_like(dest)
                    if src_rank == 0:
                        expected[2] = target[1]
                        expected[4] = target[3]
                    torch.testing.assert_close(dest, expected)
                    expected_draft = torch.zeros_like(draft_dest)
                    expected_draft[512:768, src_rank] = draft[256:512, 0]
                    expected_draft[1024, src_rank] = draft[768, 0]
                    torch.testing.assert_close(draft_dest, expected_draft)
                    mgr.attn_tp_size = 8
                    a.engine_rank = 0
                    a.draft_kv_head_num = 2
                    a.kv_item_lens[1] = 1024
                    equal_tp_draft = torch.full_like(draft_dest, 31)
                    a.kv_data_ptrs[1] = equal_tp_draft.data_ptr()
                    mgr.prepare_dcp_entry_geometry(info)
                    dest.zero_()
                    draft_dest.zero_()
                    mgr.send_kvcache_mixed(
                        "s",
                        np.array([1, 3]),
                        np.array([2, 4]),
                        info=info,
                        src_page_offset=0,
                        decode_prefix_len=0,
                        num_kv_tokens=257,
                    )
                    expected[2] = target[1]
                    expected[4] = target[3]
                    torch.testing.assert_close(dest, expected)
                    expected_draft.zero_()
                    expected_draft[512:768] = 31
                    expected_draft[1024] = 31
                    torch.testing.assert_close(draft_dest, expected_draft)
                    info.dst_kv_item_lens[0] += 576
                    with self.assertRaisesRegex(
                        ValueError, "^Source/destination KV page geometry differs$"
                    ):
                        mgr.prepare_dcp_entry_geometry(info)
                    info.dst_kv_item_lens[0] -= 576
                    info.dst_dcp_size = 8
                    with self.assertRaisesRegex(ValueError, "Decode DCP1"):
                        mgr.prepare_dcp_entry_geometry(info)

    def test_hisparse_target_and_draft_use_independent_page_indices(self):
        import ctypes

        import torch

        from sglang.srt.disaggregation.mooncake import conn as mc

        for src_tp, src_rank in ((8, 0), (16, 0), (16, 1)):
            with self.subTest(src_tp=src_tp, src_rank=src_rank):
                # Wire order: C4 KV (host), C4 indexer, C128 KV, draft (device).
                # C4/C128 page strides include the production layout's padding.
                target_widths = [37440, 64 * 132, 1728]
                sources = [
                    torch.arange(4 * width)
                    .remainder(251)
                    .to(torch.uint8)
                    .view(4, width)
                    for width in target_widths
                ]
                destinations = [
                    torch.zeros((8, width), dtype=torch.uint8)
                    for width in target_widths
                ]
                src_heads = 16 // src_tp
                draft = (
                    torch.arange(4 * 256 * src_heads * 2)
                    .remainder(251)
                    .to(torch.uint8)
                    .view(4 * 256, src_heads, 2)
                )
                draft_dest = torch.zeros((8 * 256, 2, 2), dtype=torch.uint8)
                sources.append(draft)
                destinations.append(draft_dest)
                mgr = MooncakeKVManager.__new__(MooncakeKVManager)
                mgr.enable_custom_mem_pool = False
                mgr.custom_mem_pool_type = ""
                mgr.max_transfer_batch_indices = 0
                mgr.attn_tp_size = src_tp
                mgr.dcp_size = 1
                mgr.dcp_rank = 0
                mgr.is_mla_backend = True
                mgr.is_hybrid_mla_backend = False
                mgr._validate_envelope_kv_layout = Mock()
                args = mgr.kv_args = KVArgs()
                args.page_size = 256
                args.engine_rank = src_rank
                args.kv_data_ptrs = [t.data_ptr() for t in sources]
                args.kv_item_lens = target_widths + [256 * src_heads * 2]
                args.kv_layer_ids = list(range(4))
                args.mla_compression_ratios = [4, 128]
                args.prefill_start_layer = 0
                args.prefill_end_layer = 2
                args.draft_kv_entry_count = 1
                args.draft_kv_head_num = src_heads
                args.draft_total_kv_head_num = 16
                args.draft_kv_token_major = True
                info = SimpleNamespace(
                    dst_kv_ptrs=[t.data_ptr() for t in destinations],
                    dst_kv_item_lens=target_widths + [1024],
                    dst_kv_item_len=target_widths[0],
                    dst_kv_layer_ids=list(range(4)),
                    dst_draft_entry_count=1,
                    dst_draft_metadata_present=True,
                    dst_draft_token_major=True,
                    dst_draft_head_num=2,
                    dst_draft_total_head_num=16,
                    dst_page_size=256,
                    dst_dcp_size=1,
                    dst_dcp_rank=0,
                    dst_attn_tp_size=8,
                    dst_tp_rank=0,
                )
                mgr.prepare_dcp_entry_geometry(info)

                def transfer(session, blocks):
                    for src, dst, size in blocks:
                        ctypes.memmove(dst, src, size)
                    return 0

                mgr._transfer_data = Mock(side_effect=transfer)
                kwargs = dict(
                    info=info,
                    src_page_offset=0,
                    decode_prefix_len=0,
                    num_kv_tokens=257,
                )
                with patch.object(
                    mc,
                    "build_dcp_token_transfer_plan",
                    wraps=mc.build_dcp_token_transfer_plan,
                ) as build_plan:
                    mgr.send_kvcache_mixed(
                        "s",
                        np.array([1, 3]),
                        np.array([2, 4]),
                        dst_device_kv_indices=np.array([5, 7]),
                        **kwargs,
                    )
                self.assertEqual(build_plan.call_count, 1)
                for entry, (source, dest) in enumerate(
                    zip(sources[:-1], destinations[:-1])
                ):
                    expected = torch.zeros_like(dest)
                    if src_rank == 0:
                        pages = (2, 4) if entry == 0 else (5, 7)
                        expected[pages[0]] = source[1]
                        expected[pages[1]] = source[3]
                    torch.testing.assert_close(dest, expected)
                expected_draft = torch.zeros_like(draft_dest)
                heads = slice(src_rank * src_heads, (src_rank + 1) * src_heads)
                expected_draft[5 * 256 : 6 * 256, heads] = draft[256:512]
                expected_draft[7 * 256, heads] = draft[3 * 256]
                torch.testing.assert_close(draft_dest, expected_draft)

                # Bad page vectors must fail before issuing any transfer.
                for src, host, device in (
                    ([1, 3], [2, 4], []),
                    ([1, 3], [2, 4], [5]),
                    ([1, 3], [2, 4], [5, 7, 9]),
                    ([1, 3], [2, 4], [-1, 7]),
                    ([1, 3], [2, 4], [[5, 7]]),
                    ([1, 3], [-1, 4], [5, 7]),
                    ([1, 3], [2], [5, 7]),
                    ([-1, 3], [2, 4], [5, 7]),
                ):
                    with self.subTest(src=src, host=host, device=device):
                        mgr._transfer_data.reset_mock()
                        with self.assertRaises(ValueError):
                            mgr.send_kvcache_mixed(
                                "s",
                                np.array(src),
                                np.array(host),
                                dst_device_kv_indices=np.array(device),
                                **kwargs,
                            )
                        mgr._transfer_data.assert_not_called()
                info.dst_dcp_size = 8
                with self.assertRaisesRegex(ValueError, "HiSparse.*DCP1"):
                    mgr.send_kvcache_mixed(
                        "s",
                        np.array([1, 3]),
                        np.array([2, 4]),
                        dst_device_kv_indices=np.array([5, 7]),
                        **kwargs,
                    )
                mgr._transfer_data.assert_not_called()

    def test_token_major_draft_metadata_matches_registered_buffers(self):
        import torch

        from sglang.srt.disaggregation.utils import set_draft_kv_transfer_metadata

        for mismatch in (None, "pointer", "length", "page_width"):
            with self.subTest(mismatch=mismatch):
                buffers = [torch.zeros((32, 2, 3)) for _ in range(2)]
                ptrs = [t.data_ptr() for t in buffers]
                lengths = [t.nbytes for t in buffers]
                item_lens = [96, 96]
                if mismatch == "pointer":
                    ptrs[0] += 4
                elif mismatch == "length":
                    lengths[0] += 4
                elif mismatch == "page_width":
                    item_lens[0] += 4
                pool = SimpleNamespace(
                    head_num=2,
                    page_size=4,
                    k_buffer=buffers[:1],
                    v_buffer=buffers[1:],
                    get_contiguous_buf_infos=lambda: (ptrs, lengths, item_lens),
                )
                args = KVArgs()
                set_draft_kv_transfer_metadata(
                    args, pool, draft_kv_entry_count=2, tp_size=8, total_head_num=16
                )
                self.assertEqual(args.draft_kv_token_major, mismatch is None)

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
        mgr.dcp_size = 1
        mgr.dcp_rank = 0
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
            dst_draft_metadata_present=True,
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
        mgr.is_mla_backend = False
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
        mgr.dcp_rank = 0
        mgr.attn_tp_size = 16
        mgr.kv_args = SimpleNamespace(kv_item_lens=[24, 12], draft_kv_entry_count=1)
        with self.assertRaisesRegex(ValueError, "Prefill DCP1"):
            mgr.kv_args.draft_kv_token_major = True
            mgr.prepare_dcp_entry_geometry(
                SimpleNamespace(
                    dst_draft_token_major=True,
                    dst_attn_tp_size=8,
                    dst_dcp_size=1,
                    dst_dcp_rank=0,
                )
            )

    def test_failure_status_reaches_transfer_caller(self):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.enable_custom_mem_pool = False
        mgr.custom_mem_pool_type = ""
        mgr.max_transfer_batch_indices = 0
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
            target_page_transfer=False,
            dcp_dst_entry_indices=[0, 1],
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
        self.assertEqual(info.dst_kv_item_lens, [])
        with self.assertRaises(struct.error):
            KVArgsRegisterInfo.from_zmq(msg + [b"bad"])
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.attn_tp_size = 16
        mgr.dcp_size = 1
        mgr.dcp_rank = 0
        mgr.kv_args = SimpleNamespace(
            page_size=4,
            kv_data_ptrs=[10],
            kv_item_lens=[24],
            kv_layer_ids=[],
            draft_kv_entry_count=0,
            num_draft_entries=0,
        )
        mgr.prepare_dcp_entry_geometry(info)
        self.assertEqual(info.dcp_token_item_lens, [6])

    def test_registration_with_staging_slot_field_and_draft_geometry(self):
        msg = [
            b"None",
            b"host",
            b"1",
            b"s",
            struct.pack("2Q", 20, 30),
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
            b"1",
            b"0",
            b"",  # v0.5.21 staging slot layer IDs
            struct.pack("2Q", 24, 192),
            b"2",
            b"1",
            b"16",
            b"4",
            b"1",
        ]
        info = KVArgsRegisterInfo.from_zmq(msg)
        self.assertEqual(info.dst_kv_item_lens, [24, 192])
        self.assertEqual(info.dst_draft_entry_count, 2)
        self.assertEqual(info.dst_draft_head_num, 1)
        self.assertEqual(info.dst_draft_total_head_num, 16)
        self.assertEqual(info.dst_page_size, 4)
        self.assertTrue(info.dst_draft_token_major)

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
                    mgr.enable_custom_mem_pool = False
                    mgr.custom_mem_pool_type = ""
                    mgr.max_transfer_batch_indices = 0
                    mgr.attn_tp_size = 16
                    mgr.attn_cp_size = 1
                    mgr.dcp_size = 1
                    mgr.dcp_rank = 0
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
