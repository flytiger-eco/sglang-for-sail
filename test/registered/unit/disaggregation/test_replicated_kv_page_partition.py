"""Verify multi-sender destination coverage against the single-sender path."""

from unittest.mock import patch

import numpy as np
import pytest
from test_replicated_kv_page_transfer import make_manager, send

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

SWITCH = envs.SGLANG_MOONCAKE_REPLICATED_KV_PAGE_TRANSFER_ALL_RANKS


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_multi_rank_exact_coverage_for_uneven_fragmented_pages(tp, count):
    manager, sources, targets = make_manager()
    manager.attn_tp_size = tp
    pages = (3, 0, 2, 1)[:count]
    dst_pages = (4, 1, 3, 0)[:count]
    intervals = []
    per_rank_pages = []
    with patch.object(SWITCH, "get", return_value=True):
        for rank in range(tp):
            manager.kv_args.engine_rank = rank
            manager.engine.batch_transfer_sync.reset_mock()
            assert send(manager, targets, pages=pages, dst_pages=dst_pages) == 0
            calls = manager.engine.batch_transfer_sync.call_args_list
            copied = sum(sum(c.args[3]) for c in calls)
            assert copied % (2 * 32) == 0
            per_rank_pages.append(copied // (2 * 32))
            for call in calls:
                intervals.extend(
                    (dst, dst + size)
                    for dst, size in zip(call.args[2], call.args[3], strict=True)
                )
    assert sum(per_rank_pages) == count
    assert max(per_rank_pages) - min(per_rank_pages) <= 1
    intervals.sort()
    assert all(b <= c for (_, b), (c, _) in zip(intervals, intervals[1:]))
    for source, target in zip(sources, targets):
        expected = np.full_like(target, -1)
        for src, dst in zip(pages, dst_pages, strict=True):
            expected[dst * 8 : (dst + 1) * 8] = source[src * 8 : (src + 1) * 8]
        np.testing.assert_array_equal(target, expected)


def test_multi_rank_equals_single_sender_across_chunks():
    manager, sources, reference = make_manager()
    chunks = [((3,), (0,)), ((0, 2, 1), (4, 1, 3))]
    with patch.object(SWITCH, "get", return_value=False):
        for pages, dst_pages in chunks:
            assert send(manager, reference, pages=pages, dst_pages=dst_pages) == 0
    targets = [np.full_like(t, -1) for t in reference]
    with patch.object(SWITCH, "get", return_value=True):
        for rank in range(4):
            manager.kv_args.engine_rank = rank
            for pages, dst_pages in chunks:
                assert send(manager, targets, pages=pages, dst_pages=dst_pages) == 0
    for actual, expected in zip(targets, reference):
        np.testing.assert_array_equal(actual, expected)


def test_multi_rank_supports_different_source_page_allocations():
    manager, source, target = make_manager()
    original = [a.copy() for a in source]
    with patch.object(SWITCH, "get", return_value=True):
        for rank in range(4):
            manager.kv_args.engine_rank = rank
            allocation = np.roll(np.arange(4), rank)
            alternate = [
                a.reshape(4, 8)[allocation].copy().reshape(-1) for a in original
            ]
            manager.kv_args.kv_data_ptrs = [a.ctypes.data for a in alternate]
            pages = [int(np.where(allocation == logical)[0][0]) for logical in range(4)]
            assert send(manager, target, pages=pages, dst_pages=(0, 1, 3, 4)) == 0
    for original_array, actual in zip(original, target):
        expected = np.full_like(actual, -1)
        for src, dst in zip(range(4), (0, 1, 3, 4)):
            expected[dst * 8 : (dst + 1) * 8] = original_array[src * 8 : (src + 1) * 8]
        np.testing.assert_array_equal(actual, expected)


def test_each_rank_propagates_its_engine_error():
    manager, _, target = make_manager(rank=1)
    manager.engine.batch_transfer_sync.side_effect = None
    manager.engine.batch_transfer_sync.return_value = -8
    with patch.object(SWITCH, "get", return_value=True):
        assert send(manager, target, pages=(0, 1, 2, 3), dst_pages=(0, 1, 3, 4)) == -8


def test_rank_partition_rejects_page_count_mismatch():
    manager, _, target = make_manager(rank=1)
    with patch.object(SWITCH, "get", return_value=True):
        with pytest.raises(ValueError, match="page counts differ"):
            send(manager, target, pages=(0, 1), dst_pages=(2,))
    manager.engine.batch_transfer_sync.assert_not_called()


def test_switch_does_not_override_draft_head_slicing():
    manager, source, targets = make_manager(draft_entries=2)
    with patch.object(SWITCH, "get", return_value=True):
        for rank in range(4):
            manager.kv_args.engine_rank = rank
            assert send(manager, targets) == 0
    for src, target in zip(source, targets):
        expected = np.full_like(target, -1)
        expected[16:24] = src[8:16]
        expected[32:40] = src[16:24]
        np.testing.assert_array_equal(target, expected)


def test_switch_does_not_enable_page_fast_path_by_itself():
    manager, source, targets = make_manager()
    with patch.object(SWITCH, "get", return_value=True):
        for rank in range(4):
            manager.kv_args.engine_rank = rank
            assert send(manager, targets, enabled=False) == 0
    for src, target in zip(source, targets):
        expected = np.full_like(target, -1)
        expected[16:24] = src[8:16]
        expected[32:40] = src[16:24]
        np.testing.assert_array_equal(target, expected)


def test_partition_uses_local_rank_with_global_rank_offset():
    manager, sources, targets = make_manager()
    with patch.object(SWITCH, "get", return_value=True):
        for local_rank in range(4):
            manager.kv_args.engine_rank = 4 + local_rank
            assert (
                send(manager, targets, pages=(0, 1, 2, 3), dst_pages=(0, 1, 3, 4)) == 0
            )
    for src, target in zip(sources, targets):
        expected = np.full_like(target, -1)
        for src_page, dst_page in zip(range(4), (0, 1, 3, 4)):
            expected[dst_page * 8 : (dst_page + 1) * 8] = src[
                src_page * 8 : (src_page + 1) * 8
            ]
        np.testing.assert_array_equal(target, expected)


def test_nonreplicated_source_keeps_head_slice_layout():
    manager, sources, targets = make_manager()
    manager.kv_args.kv_head_num = 1
    manager.kv_args.kv_item_lens = [8, 8]
    expected = [np.full_like(t, -1) for t in targets]
    with patch.object(SWITCH, "get", return_value=True):
        for rank in range(4):
            manager.kv_args.engine_rank = rank
            local = [s + 1000 * rank for s in sources]
            manager.kv_args.kv_data_ptrs = [s.ctypes.data for s in local]
            assert send(manager, targets) == 0
            for source, dest in zip(local, expected):
                for src_page, dst_page in [(1, 2), (2, 4)]:
                    for token in range(2):
                        dest[dst_page * 8 + token * 4 + rank] = source[
                            src_page * 2 + token
                        ]
    for actual, reference in zip(targets, expected):
        np.testing.assert_array_equal(actual, reference)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
