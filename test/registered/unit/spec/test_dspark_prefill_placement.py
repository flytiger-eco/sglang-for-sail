import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    PrefillPlacement,
    TargetHiddenKvInjector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakePlan:
    def __init__(self, write_ids: torch.Tensor):
        self._write_ids = write_ids

    def write_ids(self, translator, cols=None):
        return self._write_ids


class _FakeTranslator:
    def __init__(self, write_ids: torch.Tensor):
        self._write_ids = write_ids

    def own_plan(self, batch):
        return _FakePlan(self._write_ids)


class _FakePool:
    """Only the ring size the injector reads; no fused SWA-norm-rope entry."""

    def __init__(self, swa_req_ring_size=None):
        if swa_req_ring_size is not None:
            self.swa_req_ring_size = swa_req_ring_size


def _make_injector(*, write_ids: torch.Tensor, swa_req_ring_size=None):
    injector = object.__new__(TargetHiddenKvInjector)
    injector.model_runner = SimpleNamespace(
        device=torch.device("cpu"),
        prefill_attention_backend_str="torch_native",
        kv_index_translator=_FakeTranslator(write_ids),
    )
    injector.draft_model_runner = SimpleNamespace(
        token_to_kv_pool=_FakePool(swa_req_ring_size),
        kv_index_translator=object(),
    )
    return injector


def _make_batch(*, extend_lens, prefix_lens, req_pool_indices, out_cache_loc):
    return SimpleNamespace(
        extend_lens=extend_lens,
        prefix_lens=prefix_lens,
        req_pool_indices=torch.tensor(req_pool_indices),
        out_cache_loc=out_cache_loc,
    )


class TestPrefillPlacement(CustomTestCase):
    def test_select_indexes_tensors_and_preserves_none(self):
        placement = PrefillPlacement(
            cache_loc=torch.tensor([10, 11, 12]),
            positions=torch.tensor([0, 1, 2]),
            state_slot=None,
            final_pos=None,
            swa_loc=torch.tensor([7, 8, 9]),
        )

        picked = placement.select(torch.tensor([2, 0]))

        self.assertEqual(picked.cache_loc.tolist(), [12, 10])
        self.assertEqual(picked.positions.tolist(), [2, 0])
        self.assertEqual(picked.swa_loc.tolist(), [9, 7])
        self.assertIsNone(picked.state_slot)
        self.assertIsNone(picked.final_pos)

    def test_place_prefill_without_swa_ring(self):
        write_ids = torch.tensor([100, 101, 102, 103, 104])
        injector = _make_injector(write_ids=write_ids)
        batch = _make_batch(
            extend_lens=[3, 2],
            prefix_lens=[0, 4],
            req_pool_indices=[5, 7],
            out_cache_loc=write_ids,
        )

        placement = injector.place_prefill(batch)

        self.assertEqual(placement.positions.tolist(), [0, 1, 2, 4, 5])
        self.assertEqual(placement.cache_loc.tolist(), write_ids.tolist())
        self.assertIsNone(placement.state_slot)
        self.assertIsNone(placement.final_pos)
        self.assertIsNone(placement.swa_loc)

    def test_place_prefill_threads_swa_ring_slot_and_final_pos(self):
        write_ids = torch.tensor([100, 101, 102, 103, 104])
        injector = _make_injector(write_ids=write_ids, swa_req_ring_size=128)
        batch = _make_batch(
            extend_lens=[3, 2],
            prefix_lens=[0, 4],
            req_pool_indices=[5, 7],
            out_cache_loc=write_ids,
        )

        placement = injector.place_prefill(batch)

        self.assertEqual(placement.state_slot.tolist(), [5, 5, 5, 7, 7])
        self.assertEqual(placement.final_pos.tolist(), [2, 2, 2, 5, 5])
        self.assertIsNone(placement.swa_loc)

    def test_place_prefill_requires_scheduler_tables(self):
        injector = _make_injector(write_ids=torch.tensor([1]))
        with self.assertRaises(RuntimeError):
            injector.place_prefill(
                _make_batch(
                    extend_lens=None,
                    prefix_lens=[0],
                    req_pool_indices=[0],
                    out_cache_loc=torch.tensor([1]),
                )
            )
        with self.assertRaises(RuntimeError):
            injector.place_prefill(
                _make_batch(
                    extend_lens=[1],
                    prefix_lens=[0],
                    req_pool_indices=[0],
                    out_cache_loc=None,
                )
            )


if __name__ == "__main__":
    unittest.main()
