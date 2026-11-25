import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")
register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


class TestMambaSlotIndirectionCommit(CustomTestCase):
    def test_verify_commit_updates_mapping_without_persistent_copy(self):
        req_pool_indices = torch.tensor([5, 3], dtype=torch.int32)
        accepted_steps = torch.tensor([2, 1], dtype=torch.int64)
        state_indices = torch.tensor([19, 17], dtype=torch.int32)
        source_groups = torch.tensor([5, 3], dtype=torch.int32)
        track_indices = torch.tensor([7, 8], dtype=torch.int32)
        track_steps = torch.tensor([1, -1], dtype=torch.int64)

        req_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(replayssm_is_kda=False),
            commit_mamba_accept_steps=MagicMock(),
            get_mamba_indices=MagicMock(return_value=state_indices),
            get_speculative_mamba2_params_all_layers=MagicMock(return_value=object()),
        )
        linear_backend = SimpleNamespace(
            mamba_slot_indirection=True,
            accept_lens_pool=None,
            req_to_token_pool=req_pool,
            forward_metadata=ForwardMetadata(
                query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
                mamba_cache_indices=state_indices,
                mamba_intermediate_indices=source_groups,
            ),
            _translate_mamba_indices=lambda indices: indices,
        )
        backend = object.__new__(HybridLinearAttnBackend)
        backend.linear_attn_backend = linear_backend
        backend._update_ple_state_after_mtp_verify = MagicMock()

        scatter_path = (
            "sglang.srt.layers.attention.hybrid_linear_attn_backend."
            "scatter_mamba_states_after_mtp_verify"
        )
        with patch(scatter_path) as scatter:
            backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=accepted_steps,
                mamba_track_indices=track_indices,
                mamba_steps_to_track=track_steps,
                model=object(),
                req_pool_indices=req_pool_indices,
            )

        commit_args = req_pool.commit_mamba_accept_steps.call_args.args
        self.assertTrue(torch.equal(commit_args[0], req_pool_indices))
        self.assertTrue(torch.equal(commit_args[1], accepted_steps))

        scatter.assert_called_once()
        call = scatter.call_args
        self.assertIs(
            call.args[0],
            req_pool.get_speculative_mamba2_params_all_layers.return_value,
        )
        self.assertTrue(torch.equal(call.args[1], state_indices))
        self.assertTrue(torch.equal(call.args[2], accepted_steps))
        self.assertTrue(torch.equal(call.args[3], track_indices))
        self.assertTrue(torch.equal(call.args[4], track_steps))
        self.assertTrue(
            torch.equal(call.kwargs["intermediate_src_indices"], source_groups)
        )
        self.assertFalse(call.kwargs["commit_persistent_state"])

    def test_verify_commit_requires_explicit_request_rows(self):
        req_pool = SimpleNamespace(
            mamba_pool=SimpleNamespace(replayssm_is_kda=False),
            get_speculative_mamba2_params_all_layers=lambda: object(),
        )
        linear_backend = SimpleNamespace(
            mamba_slot_indirection=True,
            accept_lens_pool=None,
            req_to_token_pool=req_pool,
            forward_metadata=ForwardMetadata(
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                mamba_cache_indices=torch.tensor([4], dtype=torch.int32),
                mamba_intermediate_indices=torch.tensor([2], dtype=torch.int32),
            ),
        )
        backend = object.__new__(HybridLinearAttnBackend)
        backend.linear_attn_backend = linear_backend

        with self.assertRaisesRegex(ValueError, "explicit request-pool indices"):
            backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=torch.tensor([0]),
                mamba_track_indices=None,
                mamba_steps_to_track=None,
                model=object(),
                req_pool_indices=None,
            )

    def test_intermediate_groups_keep_live_zero_and_poison_graph_padding(self):
        backend = SimpleNamespace(mamba_slot_indirection=True)
        output = torch.full((3,), 99, dtype=torch.int32)
        actual = MambaAttnBackendBase._mamba_intermediate_indices(
            backend,
            torch.tensor([0, 5, 0], dtype=torch.int32),
            torch.tensor([7, 8, -1], dtype=torch.int32),
            output,
        )
        self.assertIs(actual, output)
        self.assertEqual(output.tolist(), [0, 5, -1])



class TestMambaSlotIndirectionStreamOrder(CustomTestCase):
    def test_shared_draft_does_not_replace_mapping_owner(self):
        pool = object.__new__(HybridReqToTokenPool)
        main_stream, draft_stream = object(), object()
        pool.register_forward_stream(main_stream)
        pool.register_forward_stream(draft_stream)
        self.assertIs(pool.mamba_mapping_stream, main_stream)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_all_mapping_writers_share_forward_stream_order(self):
        pool = object.__new__(HybridReqToTokenPool)
        pool.device = "cuda"
        pool.mamba_pool = SimpleNamespace(slot_indirection=True)
        pool._owns_mamba_index_mapping = True
        pool.req_index_to_mamba_index_mapping = torch.zeros(
            1, dtype=torch.int32, device="cuda"
        )

        mapping_stream = torch.cuda.Stream()
        pool.register_forward_stream(mapping_stream)
        gate = torch.cuda.Event()

        # Hold an old request's in-flight commit on the mapping stream. A new
        # request's publish must enqueue behind it; a write on the scheduler
        # stream would land first and then be overwritten by the old owner.
        with torch.cuda.stream(mapping_stream):
            mapping_stream.wait_event(gate)
            pool.req_index_to_mamba_index_mapping[0] = 11

        req_indices = torch.tensor([0], dtype=torch.int64, device="cuda")
        new_row = torch.tensor([3], dtype=torch.int32, device="cuda")
        pool._write_mamba_index_mapping(req_indices, new_row)
        gate.record(torch.cuda.current_stream())
        torch.cuda.synchronize()

        self.assertEqual(pool.req_index_to_mamba_index_mapping[0].item(), 3)


if __name__ == "__main__":
    unittest.main()
