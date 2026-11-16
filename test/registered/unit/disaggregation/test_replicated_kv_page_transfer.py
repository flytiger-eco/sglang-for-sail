import concurrent.futures
import ctypes
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake import conn as mooncake_conn
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager, MooncakeKVSender
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def single_sender_mode():
    # This module specifically verifies the legacy single-sender page mode.
    with patch.object(
        envs.SGLANG_MOONCAKE_REPLICATED_KV_PAGE_TRANSFER_ALL_RANKS,
        "get",
        return_value=False,
    ):
        yield


def make_manager(rank=0, draft_entries=0):
    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    # Two K/V entries, four pages, two tokens/page, four replicated heads.
    source = [np.arange(32, dtype=np.int32) + 100 * i for i in range(2)]
    target = [np.full(40, -1, dtype=np.int32) for _ in range(2)]
    manager.kv_args = SimpleNamespace(
        kv_item_lens=[32, 32],
        page_size=2,
        total_kv_head_num=4,
        kv_head_num=4,
        engine_rank=rank,
        num_draft_entries=draft_entries,
        kv_data_ptrs=[a.ctypes.data for a in source],
        kv_layer_ids=[0, 1],
    )
    manager.attn_tp_size = 4
    manager.pp_size = 1
    manager.dcp_size = 1
    manager.enable_custom_mem_pool = False
    manager.custom_mem_pool_type = None
    manager.is_mla_backend = False
    manager.is_hybrid_mla_backend = False
    manager.max_transfer_batch_indices = 0

    def transfer(session, sources, destinations, lengths):
        for a, b, n in zip(sources, destinations, lengths, strict=True):
            ctypes.memmove(b, a, n)
        return 0

    manager.engine = SimpleNamespace(
        batch_transfer_sync=MagicMock(side_effect=transfer)
    )
    return manager, source, target


def send(manager, target, enabled=True, pages=(1, 2), dst_pages=(2, 4)):
    with patch.object(
        envs.SGLANG_MOONCAKE_REPLICATED_KV_PAGE_TRANSFER, "get", return_value=enabled
    ):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            return manager.send_kvcache_slice(
                "cpu",
                np.array(pages, dtype=np.int32),
                [a.ctypes.data for a in target],
                np.array(dst_pages, dtype=np.int32),
                0,
                1,
                32,
                executor,
                [0, 1],
            )


def test_fast_pages_copy_every_head_at_exact_destinations():
    manager, source, target = make_manager()
    assert send(manager, target) == 0
    for src, dst in zip(source, target):
        expected = np.full_like(dst, -1)
        expected[16:24] = src[8:16]
        expected[32:40] = src[16:24]
        np.testing.assert_array_equal(dst, expected)
    calls = manager.engine.batch_transfer_sync.call_args_list
    assert len(calls) == 1 and calls[0].args[3] == [32, 32, 32, 32]


def test_coalesces_adjacent_pages_without_crossing_allocations():
    manager, source, target = make_manager()
    assert send(manager, target, dst_pages=(2, 3)) == 0
    assert manager.engine.batch_transfer_sync.call_args.args[3] == [64, 64]


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_other_replicated_senders_skip_main_kv_only(rank):
    manager, source, target = make_manager(rank)
    assert send(manager, target) == 0
    manager.engine.batch_transfer_sync.assert_not_called()
    for dst in target:
        assert (dst == -1).all()


def test_fast_transfer_equals_all_four_sliced_senders():
    manager, source, fast = make_manager()
    assert send(manager, fast) == 0
    sliced = [np.full_like(x, -1) for x in fast]
    for rank in range(4):
        manager.kv_args.engine_rank = rank
        assert send(manager, sliced, enabled=False) == 0
    for a, b in zip(fast, sliced):
        np.testing.assert_array_equal(a, b)


def test_draft_registration_retains_original_head_slices():
    manager, source, target = make_manager(draft_entries=2)
    assert send(manager, target) == 0
    # Rank zero may fill only head zero; it must not claim all draft heads.
    for src, dst in zip(source, target):
        expected = np.full_like(dst, -1)
        expected[[16, 20]] = src[[8, 12]]
        expected[[32, 36]] = src[[16, 20]]
        np.testing.assert_array_equal(dst, expected)


def test_fast_path_propagates_engine_failure():
    manager, source, target = make_manager()
    manager.engine.batch_transfer_sync.side_effect = None
    manager.engine.batch_transfer_sync.return_value = -7
    assert send(manager, target) == -7


def test_rejects_incompatible_byte_layout_before_fast_path():
    manager, source, target = make_manager()
    manager.kv_args.kv_item_lens = [40, 40]
    with pytest.raises(RuntimeError, match="bytes per head"):
        send(manager, target)
    manager.engine.batch_transfer_sync.assert_not_called()


def test_page_transfer_builds_layer_mapping_once():
    manager, source, target = make_manager()
    with patch.object(
        mooncake_conn,
        "build_transfer_entry_pairs",
        wraps=mooncake_conn.build_transfer_entry_pairs,
    ) as pair_entries:
        assert send(manager, target) == 0
    pair_entries.assert_called_once()


@pytest.mark.parametrize("enabled", [False, True])
def test_all_paths_reject_inconsistent_entry_lengths(enabled):
    manager, source, target = make_manager()
    manager.kv_args.kv_item_lens = [32, 40]
    assert send(manager, target, enabled=enabled) == -1
    manager.engine.batch_transfer_sync.assert_not_called()


@pytest.mark.parametrize("last", [False, True])
def test_early_prefix_completion_event_is_forwarded_and_consumed(last):
    sender = MooncakeKVSender.__new__(MooncakeKVSender)
    event = object()
    sender._early_send_wait_event = event
    sender.bootstrap_room = 3
    sender.aux_index = 9
    sender.trace_ctx = MagicMock()
    sender.kv_mgr = SimpleNamespace(add_transfer_request=MagicMock())
    pages = np.array([1], dtype=np.int32)
    sender._prepare_send_indices = MagicMock(
        return_value=(pages, slice(0, 1), last, False)
    )
    sender._record_transfer_indices = MagicMock()
    sender.send(pages, num_kv_tokens=2)
    assert sender._early_send_wait_event is None
    assert sender.kv_mgr.add_transfer_request.call_args.kwargs["wait_event"] is event
    # Last/non-last semantics stay unchanged, including each sender's aux.
    assert sender.kv_mgr.add_transfer_request.call_args.args[3] is last


@pytest.mark.parametrize("failed", [False, True])
def test_worker_waits_for_writes_before_reading_live_room(failed):
    from collections import defaultdict
    from queue import Queue

    from sglang.srt.disaggregation.common.utils import TransferKVChunk

    order = []
    event = SimpleNamespace(synchronize=lambda: order.append("writes_done"))

    class Infos(dict):
        def values(self):
            order.append("read_room")
            return super().values()

    manager = MooncakeKVManager.__new__(MooncakeKVManager)
    manager.enable_trace = False
    manager.enable_staging = False
    manager.enable_deferred_decode_kv_release = False
    manager._staging_outstanding = defaultdict(int)
    manager.request_status = {3: KVPoll.Failed if failed else KVPoll.Transferring}
    manager.transfer_infos = {3: Infos(dummy=SimpleNamespace(is_dummy=True, room=3))}
    manager._prefill_unique_rank = lambda: 0
    q = Queue()
    q.put(
        TransferKVChunk(
            3,
            np.array([1], dtype=np.int32),
            slice(0, 1),
            False,
            None,
            None,
            wait_event=event,
        )
    )
    q.put(None)
    manager.transfer_worker(q, None)
    assert order == ([] if failed else ["writes_done", "read_room"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
