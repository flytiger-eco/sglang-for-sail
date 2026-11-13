import multiprocessing
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist

from sglang.srt.mem_cache.layer_split_batch_pages import (
    BatchPageBroadcastPlan,
    build_batch_page_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def token_table(rows):
    return torch.tensor(rows).repeat_interleave(4, dim=1) * 4 + torch.arange(
        4
    ).repeat(len(rows[0]))


def check_rank(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=25),
    )
    try:
        pages = build_batch_page_ids(
            token_table([[2, 5]] if rank == 0 else [[7, 2]]),
            torch.tensor([5]),
            page_size=4,
            num_pages=10,
            group=dist.group.WORLD,
        )
        torch.testing.assert_close(pages, torch.tensor([0, 2, 5, 7]))
        plan = BatchPageBroadcastPlan(pages)
        for owner in (1, 0):
            source = torch.arange(80).reshape(10, 8) + owner * 1000
            result = torch.full_like(source, -1)
            expected = result.clone()
            expected[pages] = source[pages]
            plan.broadcast(
                result,
                source if rank == owner else None,
                lambda packed: dist.broadcast(packed, src=owner),
            )
            torch.testing.assert_close(result, expected)
    finally:
        dist.destroy_process_group()


class TestLayerSplitBatchPages(unittest.TestCase):
    def test_plan_keeps_prefix_partial_tail_and_null_page(self):
        pages = build_batch_page_ids(
            token_table([[2, 7, 9, 99], [2, 4, 88, 88]]),
            torch.tensor([9, 5]),
            page_size=4,
            num_pages=12,
        )
        torch.testing.assert_close(pages, torch.tensor([0, 2, 4, 7, 9]))

    @unittest.skipUnless(dist.is_gloo_available(), "Gloo is unavailable")
    def test_union_across_two_ranks_and_rotating_owners(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = str(Path(directory) / "rendezvous")
            workers = [
                context.Process(target=check_rank, args=(rank, rendezvous))
                for rank in range(2)
            ]
            started = []
            try:
                for worker in workers:
                    worker.start()
                    started.append(worker)
                for worker in started:
                    worker.join(timeout=40)
                    self.assertEqual(worker.exitcode, 0)
            finally:
                for worker in started:
                    if worker.is_alive():
                        worker.terminate()
                    worker.join()

    def test_pool_scatters_in_place_and_resets_to_full_transport(self):
        from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool

        pool = LayerSplitDSATokenToKVPool.__new__(LayerSplitDSATokenToKVPool)
        pool.page_size = 4
        pool.remote_kv_buffer = torch.zeros(40, 1, 6, dtype=torch.bfloat16)
        pool.index_key_cache = SimpleNamespace(
            remote_buffer=torch.zeros(10, 132, dtype=torch.uint8), remote_layer_id=4
        )
        pool.batch_page_broadcast_enabled = True
        pool.remote_kv_layer_id = pool.pending_remote_kv_layer_id = 4
        pool._finalize_pending_kv_broadcast = Mock()
        pages = torch.tensor([0, 2, 7])
        pool.batch_page_broadcast_plan = BatchPageBroadcastPlan(pages)
        transport = Mock(side_effect=lambda tensor, *args, **kwargs: tensor)
        pool._broadcast_full_tensor_from_owner = transport
        for buffer, page_view in (
            (pool.remote_kv_buffer, pool.remote_kv_buffer.view(10, 4, 1, 6)),
            (pool.index_key_cache.remote_buffer, pool.index_key_cache.remote_buffer),
        ):
            source = torch.arange(buffer.numel()).reshape_as(buffer).to(buffer.dtype)
            pool._broadcast_tensor_from_owner(buffer, 4, source)
            expected = torch.zeros_like(page_view)
            expected[pages] = source.view_as(page_view)[pages]
            torch.testing.assert_close(page_view, expected)
            self.assertEqual(transport.call_args.args[0].shape[0], len(pages))

        pool.reset_batch_page_broadcast()
        pool._finalize_pending_kv_broadcast.assert_called_once_with(
            set_remote_layer_id=False
        )
        self.assertIsNone(pool.batch_page_broadcast_plan)
        self.assertIsNone(pool.remote_kv_layer_id)
        self.assertIsNone(pool.pending_remote_kv_layer_id)
        self.assertIsNone(pool.index_key_cache.remote_layer_id)
        pool._broadcast_tensor_from_owner(pool.remote_kv_buffer, 4)
        self.assertIs(transport.call_args.args[0], pool.remote_kv_buffer)


if __name__ == "__main__":
    unittest.main()
