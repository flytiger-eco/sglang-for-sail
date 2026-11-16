import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

spec = importlib.util.spec_from_file_location(
    "prefill_abort_helpers",
    Path(__file__).with_name("test_prefill_abort_result_cleanup.py"),
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.mark.parametrize(
    "prepared,pending", [(True, False), (True, True), (False, False)]
)
@patch("sglang.srt.disaggregation.prefill.maybe_cache_unfinished_req")
def test_snapshot_keeps_pages_owned_until_transfer_finishes(
    cache_unfinished, prepared, pending
):
    scheduler = helpers._Scheduler()
    req = helpers._Req(inflight_middle_chunks=0)
    req.to_finish = None
    req.pending_bootstrap = pending
    req.time_stats.set_prefill_transfer_queue_entry_time = Mock()
    result = GenerationBatchResult(next_token_ids=torch.tensor([11]))
    if prepared:
        result.disagg_mamba_indices = torch.tensor([7], dtype=torch.int32)
        result.disagg_kv_page_indices = torch.tensor([[10, 20, 30]], dtype=torch.int32)
    scheduler.process_batch_result_disagg_prefill(helpers._batch(req), result)
    assert cache_unfinished.call_count == int(not prepared or pending)
    assert req.kv.holds_kv and req.kv.holds_mamba
    assert scheduler.disagg_prefill_inflight_queue == [req]
    assert scheduler.send_kv_chunk.call_count == int(not pending)


@patch("sglang.srt.disaggregation.prefill.release_kv_cache")
@patch(
    "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
    return_value=[KVPoll.Success],
)
def test_completed_transfer_still_inserts_and_releases_the_request(poll, release):
    scheduler = helpers._Scheduler()
    scheduler.attn_cp_cpu_group = scheduler.attn_tp_cpu_group = None
    scheduler.scheduler_stage_metrics = None
    req = helpers._Req(inflight_middle_chunks=0)
    req.to_finish = None
    req.pending_bootstrap = False
    req.bootstrap_host = FAKE_BOOTSTRAP_HOST
    req.time_stats.set_prefill_kv_transfer_finish_time = Mock()
    scheduler.disagg_prefill_inflight_queue = [req]
    done = scheduler.process_disagg_prefill_inflight_queue()
    assert done == [req]
    release.assert_called_once_with(req, scheduler.tree_cache)
    req.disagg_kv_sender.clear.assert_called_once()
    assert not scheduler.disagg_prefill_inflight_queue


@patch(
    "sglang.srt.disaggregation.prefill.get_qsa_pending_state_indices",
    return_value=np.array([3, 0], dtype=np.int32),
)
def test_segmented_last_chunk_uses_snapshot_without_reading_device_indices(pending):
    sender = Mock()
    sender.get_max_transfer_tokens.return_value = 8
    sender.should_send_kv_chunk.return_value = True
    req = SimpleNamespace(
        rid="snapshot-test",
        start_send_idx=0,
        origin_input_ids=list(range(11)),
        extend_range=SimpleNamespace(end=11),
        disagg_decode_prefix_len=0,
        kv=SimpleNamespace(req_pool_idx=3),
        disagg_kv_sender=sender,
    )
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(page_size=4),
        enable_staging=False,
        req_to_token_pool=SimpleNamespace(),
        disagg_metadata_buffers=Mock(),
        disagg_prefill_pending_chunk_rids={req.rid},
        disagg_prefill_bootstrap_queue=SimpleNamespace(
            kv_manager=SimpleNamespace(
                kv_args=SimpleNamespace(
                    state_types=[
                        StateType.MAMBA,
                        StateType.QSA_PENDING,
                        StateType.QSA_COMPRESSED,
                    ]
                )
            )
        ),
    )
    pages = np.array([10, 20, 30], dtype=np.int32)
    SchedulerDisaggregationPrefillMixin._send_kv_chunk(
        scheduler, req, last_chunk=True, mamba_index=np.int32(7), kv_page_indices=pages
    )
    assert sender.send.call_count == 2
    first, last = sender.send.call_args_list
    np.testing.assert_array_equal(first.args[0], [10, 20])
    assert first.args[1] is None and first.kwargs["num_kv_tokens"] == 8
    np.testing.assert_array_equal(last.args[0], [30])
    np.testing.assert_array_equal(last.args[1][0][0], np.array(7, dtype=np.int32))
    np.testing.assert_array_equal(last.args[1][2], pages)
    assert last.kwargs["num_kv_tokens"] == 3
    assert req.start_send_idx == 11 and not scheduler.disagg_prefill_pending_chunk_rids


def test_overlap_polls_completed_transfers_before_waiting_for_the_previous_result():
    events = []

    class Loop:
        def __init__(self):
            self.gracefully_exit = False
            self._engine_paused = False
            self.waiting_queue = []
            self.disagg_prefill_bootstrap_queue = SimpleNamespace(
                pop_bootstrapped=lambda: []
            )
            self.running_batch = self.last_batch = self.chunked_req = None
            self.enable_staging = False
            self.iteration = 0
            self.ngram_embedding_manager = SimpleNamespace(
                prepare_for_forward=lambda batch, **kwargs: batch
            )

        def ingest_requests(self):
            self.iteration += 1
            self.gracefully_exit = self.iteration == 3

        def get_next_disagg_prefill_batch_to_run(self, **kwargs):
            batch = SimpleNamespace(name=self.iteration)
            batch.copy = lambda: batch
            return SimpleNamespace(running_batch=None, batch_to_run=batch)

        def run_batch(self, batch):
            events.append(("launch", batch.name))
            return batch.name

        def _apply_war_barrier(self):
            pass

        def process_batch_result(self, batch, result):
            events.append(("result", result, self.iteration))

        def process_disagg_prefill_inflight_queue(self):
            events.append(("poll", self.iteration))

        def launch_batch_sample_if_needed(self, result, batch):
            pass

    SchedulerDisaggregationPrefillMixin.event_loop_overlap_disagg_prefill(Loop())
    # Batch 1's RDMA can finish while batch 2 computes. Polling before batch 2's
    # result is consumed lets batch 1 complete without waiting for that compute.
    assert events.index(("poll", 3)) < events.index(("result", 2, 3))


def test_empty_transfer_batch_does_not_read_pool_mappings():
    scheduler = SimpleNamespace(enable_unified_memory=False)
    for rows in [None, torch.empty(0, dtype=torch.int32)]:
        result = GenerationBatchResult()
        SchedulerDisaggregationPrefillMixin.prepare_disagg_transfer_indices(
            scheduler, SimpleNamespace(req_pool_indices=rows), result
        )
        assert result.disagg_mamba_indices is None
        assert result.disagg_kv_page_indices is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
