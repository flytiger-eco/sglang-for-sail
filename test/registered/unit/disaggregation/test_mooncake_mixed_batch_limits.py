"""Bound default head-sliced draft descriptors without splitting full rows."""

import concurrent.futures
import unittest
from unittest.mock import Mock

import numpy as np
from test_mooncake_mixed_dispatch import mixed_manager

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_transfer(tokens=8193, *, cap=0, full_rows=False, rank=0):
    m, info = mixed_manager(cap=cap, rank=rank)
    m.kv_args.kv_data_ptrs = [100000000, 200000000, 300000000]
    info.dst_kv_ptrs = [1000000000, 2000000000, 3000000000]
    if full_rows:
        # Different TP sizes, but replicated heads still copy complete rows.
        m.kv_args.draft_total_kv_head_num = 8
        info.dst_draft_total_head_num = 8
        info.dst_draft_head_num = 1
        info.dst_kv_item_lens[1:] = [1024, 1024]
    m.prepare_dcp_entry_geometry(info)
    pages = (tokens + 63) // 64
    kwargs = dict(
        session="s",
        src_pages=np.arange(1, pages + 1, dtype=np.int32),
        dst_pages=np.arange(3, pages + 3, dtype=np.int32),
        info=info,
        src_page_offset=0,
        decode_prefix_len=0,
        num_kv_tokens=tokens,
    )
    return m, info, kwargs


class TestMooncakeMixedBatchLimits(CustomTestCase):
    def test_default_bound_preserves_bytes_and_whole_target_rows(self):
        tokens = 8193
        for custom_pool in (False, True):
            with self.subTest(custom_pool=custom_pool):
                m, info, kwargs = make_transfer(tokens)
                m.enable_custom_mem_pool = custom_pool
                m.custom_mem_pool_type = "NVLINK" if custom_pool else ""
                src_widths, dst_widths = (1, 16, 16), (1, 32, 32)
                source = [
                    (np.arange((tokens + 64) * width, dtype=np.uint32) % 251).astype(
                        np.uint8
                    )
                    for width in src_widths
                ]
                destination = [
                    np.full((tokens + 192) * width, 205, dtype=np.uint8)
                    for width in dst_widths
                ]
                expected = [buf.copy() for buf in destination]
                expected[0][192 : 192 + tokens] = source[0][64 : 64 + tokens]
                for i in (1, 2):
                    expected[i].reshape(-1, 32)[192 : 192 + tokens, :16] = source[
                        i
                    ].reshape(-1, 16)[64 : 64 + tokens]
                calls = []

                def transfer(session, blocks):
                    self.assertEqual(session, "s")
                    for src, dst, size in blocks:
                        entry = next(
                            i
                            for i, base in enumerate(m.kv_args.kv_data_ptrs)
                            if base <= src < base + source[i].size
                        )
                        src_offset = src - m.kv_args.kv_data_ptrs[entry]
                        dst_offset = dst - info.dst_kv_ptrs[entry]
                        self.assertGreater(size, 0)
                        self.assertGreaterEqual(dst_offset, 0)
                        self.assertLessEqual(src_offset + size, source[entry].size)
                        self.assertLessEqual(dst_offset + size, destination[entry].size)
                        destination[entry][dst_offset : dst_offset + size] = source[
                            entry
                        ][src_offset : src_offset + size]
                    calls.append(blocks)
                    return 0

                m._transfer_data = transfer
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    self.assertEqual(
                        m.send_kvcache_mixed(**kwargs, executor=executor), 0
                    )
                self.assertEqual(len(calls), 7 if custom_pool else 3)
                for blocks in calls:
                    for base in m.kv_args.kv_data_ptrs[1:]:
                        self.assertLessEqual(
                            sum(base <= src < base + 10000000 for src, _, _ in blocks),
                            4096,
                        )
                target_blocks = [
                    block
                    for blocks in calls
                    for block in blocks
                    if block[0] < 200000000
                ]
                self.assertEqual(target_blocks, [(100000064, 1000000192, tokens)])
                self.assertEqual(
                    sum(size for blocks in calls for _, _, size in blocks), tokens * 33
                )
                for actual, reference in zip(destination, expected):
                    np.testing.assert_array_equal(actual, reference)

    def test_full_draft_rows_remain_coalesced_with_different_tp(self):
        m, info, kwargs = make_transfer(full_rows=True)
        m._transfer_data = Mock(return_value=0)
        self.assertEqual(m.send_kvcache_mixed(**kwargs), 0)
        m._transfer_data.assert_called_once_with(
            "s",
            [
                (100000064, 1000000192, 8193),
                (200001024, 2000003072, 8193 * 16),
                (300001024, 3000003072, 8193 * 16),
            ],
        )

    def test_replicated_non_owner_does_not_dispatch(self):
        m, info, kwargs = make_transfer(full_rows=True, rank=1)
        m._transfer_data = Mock(return_value=0)
        self.assertEqual(m.send_kvcache_mixed(**kwargs), 0)
        m._transfer_data.assert_not_called()

    def test_four_k_tokens_remain_one_transfer(self):
        m, info, kwargs = make_transfer(4096)
        m._transfer_data = Mock(return_value=0)
        self.assertEqual(m.send_kvcache_mixed(**kwargs), 0)
        self.assertEqual(m._transfer_data.call_count, 1)
        blocks = m._transfer_data.call_args.args[1]
        self.assertEqual(len(blocks), 1 + 4096 * 2)
        self.assertEqual(sum(size for _, _, size in blocks), 4096 * 33)

    def test_explicit_positive_limits_keep_their_existing_semantics(self):
        for cap, num_calls in ((1024, 9), (20000, 1)):
            with self.subTest(cap=cap):
                m, info, kwargs = make_transfer(cap=cap)
                m._transfer_data = Mock(return_value=0)
                self.assertEqual(m.send_kvcache_mixed(**kwargs), 0)
                self.assertEqual(m._transfer_data.call_count, num_calls)
                blocks = [
                    block
                    for call in m._transfer_data.call_args_list
                    for block in call.args[1]
                ]
                self.assertEqual(sum(size for _, _, size in blocks), 8193 * 33)
                # Explicit limits also split target rows, as before this change.
                self.assertEqual(
                    sum(src < 200000000 for src, _, _ in blocks), num_calls
                )

    def test_default_sliced_batches_stop_after_first_failure(self):
        m, info, kwargs = make_transfer()
        m._transfer_data = Mock(side_effect=[0, -17, 0])
        self.assertEqual(m.send_kvcache_mixed(**kwargs), -17)
        self.assertEqual(m._transfer_data.call_count, 2)
        self.assertTrue(
            all(
                src >= 200000000
                for src, _, _ in m._transfer_data.call_args_list[1].args[1]
            )
        )


if __name__ == "__main__":
    unittest.main()
