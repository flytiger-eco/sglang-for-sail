import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.load_snapshot import (
    ShmLoadSnapshotReader,
    ShmLoadSnapshotWriter,
)
from sglang.srt.managers.scheduler_components.load_inquirer import SchedulerLoadInquirer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def inquirer(mode, pending, *, chunked_req=None, scheduled_end=0, waiting_queue=None):
    empty = lambda: []
    queues = NS(queue=[], retracted_queue=[])
    stats = NS(
        num_grammar_queue_reqs=0,
        num_paused_reqs=0,
        num_retracted_reqs=0,
        kv_transfer_speed_gb_s=0,
        kv_transfer_latency_ms=0,
        gen_throughput=0,
        cache_hit_rate=0,
        utilization=0,
    )
    return SchedulerLoadInquirer(
        disaggregation_mode=mode,
        server_args=NS(),
        max_total_num_tokens=4096,
        max_running_requests=64,
        pool_stats_observer=NS(
            get_pool_stats=lambda: NS(get_kv_token_stats=lambda: (0, 0))
        ),
        tp_worker=NS(),
        token_to_kv_pool_allocator=NS(),
        spec_algorithm=NS(is_none=lambda: True),
        get_running_batch=lambda: NS(reqs=[]),
        get_waiting_queue=lambda: (
            [NS(seqlen=100, num_matched_prefix_tokens=20)]
            if waiting_queue is None
            else waiting_queue
        ),
        waiting_queue_prefix_matched=lambda: True,
        get_recent_cache_hit_rate=lambda: 0,
        get_stats=lambda: stats,
        # Of 1,000 input tokens, 600 have been allocated to completed or
        # launched chunks. Only the remaining 400 belong in waiting work.
        get_chunked_req=lambda: (
            NS(seqlen=1000, prefix_indices=range(600))
            if chunked_req is None
            else chunked_req
        ),
        get_disagg_prefill_bootstrap_queue=lambda: queues,
        get_disagg_prefill_inflight_queue=empty,
        get_disagg_decode_prealloc_queue=lambda: queues,
        get_disagg_decode_transfer_queue=lambda: queues,
        get_spec_total_num_accept_tokens=lambda: 0,
        get_spec_total_num_forward_ct=lambda: 0,
        get_total_prefill_uncached_tokens=lambda: 0,
        get_total_prefill_busy_us=lambda: 0,
        get_decode_moment_totals=lambda: [0] * 6,
        get_prefill_gpu_pending_tokens=pending,
        get_prefill_scheduled_chunk_end=lambda: scheduled_end,
    )


class TestPrefillGpuPendingLoad(unittest.TestCase):
    def snapshot(self, mode, pending, **kwargs):
        module = "sglang.srt.managers.scheduler_components.load_inquirer"
        with patch(module + ".get_parallel", return_value=NS(dp_rank=0)), patch(
            module + ".get_lora", return_value=NS(enable_lora=False)
        ):
            return inquirer(mode, pending, **kwargs).get_loads()

    def test_launched_chunk_and_unlaunched_tail_are_separate(self):
        load = self.snapshot(DisaggregationMode.PREFILL, lambda: 200)
        self.assertEqual(load.num_waiting_uncached_tokens, 480)
        self.assertEqual(load.num_prefill_gpu_pending_tokens, 200)
        self.assertEqual(load.to_dict({"core"})["num_prefill_gpu_pending_tokens"], 200)

    def test_decode_does_not_read_prefill_gpu_state(self):
        def unexpected():
            raise AssertionError("Decode has no Prefill GPU tracker")

        load = self.snapshot(DisaggregationMode.DECODE, unexpected)
        self.assertEqual(load.num_prefill_gpu_pending_tokens, 0)

    def test_launch_before_prefix_update_counts_40k_once(self):
        load = self.snapshot(
            DisaggregationMode.PREFILL,
            lambda: 32768,
            chunked_req=NS(seqlen=40960, prefix_indices=[]),
            scheduled_end=32768,
            waiting_queue=[],
        )
        self.assertEqual(load.num_waiting_uncached_tokens, 40960)
        self.assertEqual(load.num_prefill_unscheduled_uncached_tokens, 8192)
        self.assertEqual(
            load.num_prefill_unscheduled_uncached_tokens
            + load.num_prefill_gpu_pending_tokens,
            40960,
        )

    def test_completed_chunk_is_not_recharged_before_prefix_update(self):
        load = self.snapshot(
            DisaggregationMode.PREFILL,
            lambda: 0,
            chunked_req=NS(seqlen=40960, prefix_indices=[]),
            scheduled_end=32768,
            waiting_queue=[],
        )
        self.assertEqual(load.num_waiting_uncached_tokens, 40960)
        self.assertEqual(load.num_prefill_unscheduled_uncached_tokens, 8192)
        self.assertEqual(load.num_prefill_gpu_pending_tokens, 0)

    def test_prefix_catching_up_does_not_deduct_launched_chunk_twice(self):
        load = self.snapshot(
            DisaggregationMode.PREFILL,
            lambda: 32768,
            chunked_req=NS(seqlen=40960, prefix_indices=range(32768)),
            scheduled_end=32768,
            waiting_queue=[],
        )
        self.assertEqual(load.num_waiting_uncached_tokens, 8192)
        self.assertEqual(load.num_prefill_unscheduled_uncached_tokens, 8192)

    def test_pending_chunk_visibility_survives_shm_transport(self):
        load = self.snapshot(DisaggregationMode.PREFILL, lambda: 200)
        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "loads.shm")
            writer = ShmLoadSnapshotWriter(path, dp_size=1, dp_rank=0)
            reader = ShmLoadSnapshotReader(path, dp_size=1)
            try:
                writer.write(load)
                received = reader.read(0)
                self.assertEqual(received.num_waiting_uncached_tokens, 480)
                self.assertEqual(received.num_prefill_gpu_pending_tokens, 200)
            finally:
                reader.close()
                writer.close()


if __name__ == "__main__":
    unittest.main()
