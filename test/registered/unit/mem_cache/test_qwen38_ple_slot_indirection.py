"""Persistent PLE ownership while speculative Mamba rows move."""

from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import _conv_multi_eligible
from sglang.srt.layers.attention import hybrid_linear_attn_backend as hybrid
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache import memory_pool as pools
from sglang.srt.mem_cache.ple_state_pool import ShortConvPool
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _pool():
    pool = pools.HybridReqToTokenPool.__new__(pools.HybridReqToTokenPool)
    pool.device = "cpu"
    pool.mamba_mapping_stream = None
    pool.req_index_to_mamba_index_mapping = torch.zeros(4, dtype=torch.int32)
    pool.req_index_to_ple_index_mapping = torch.zeros(4, dtype=torch.int32)
    pool.mamba_pool = NS(
        slot_indirection=True, spec_group_row=lambda rows, steps: 8 + rows * 4 + steps
    )
    return pool


def test_persistent_mapping_changes_only_at_allocation():
    pool = _pool()
    rows = torch.tensor([0, 2])
    pool._publish_mamba_index_mapping([0, 2], torch.tensor([1, 3], dtype=torch.int32))
    pool.commit_mamba_accept_steps(rows, torch.tensor([1, 2]))
    assert pool.get_mamba_indices(rows).tolist() == [9, 18]
    assert pool.get_ple_state_indices(rows).tolist() == [1, 3]
    pool._publish_mamba_index_mapping([0], torch.tensor([2], dtype=torch.int32))
    assert pool.get_mamba_indices(rows).tolist() == [2, 18]
    assert pool.get_ple_state_indices(rows).tolist() == [2, 3]
    pool.req_index_to_ple_index_mapping = None
    assert pool.get_ple_state_indices(rows).tolist() == [2, 18]



def test_live_mamba_and_persistent_ple_host_roundtrip():
    req_pool = _pool()
    req_pool.req_to_token = torch.arange(8).reshape(1, 8)
    req_pool.req_index_to_mamba_index_mapping[0] = 7
    req_pool.req_index_to_ple_index_mapping[0] = 2
    mamba = pools.MambaPool.__new__(pools.MambaPool)
    mamba.slot_indirection = True
    mamba.mamba_cache = NS(
        conv=[torch.zeros((1, 10, 2, 3))], temporal=torch.zeros((1, 10, 2, 2))
    )
    mamba.mamba_cache.conv[0][:, 7] = 11
    mamba.mamba_cache.temporal[:, 7] = 22
    short = ShortConvPool.__new__(ShortConvPool)
    short.conv_state = torch.zeros((1, 4, 2, 3))
    short.conv_state[:, 2] = 33
    mamba.register_slot_state(short)
    req_pool.mamba_pool = mamba
    kv_pool = pools.HybridLinearKVPool.__new__(pools.HybridLinearKVPool)
    kv_pool.mamba_pool = mamba
    kv_pool._mamba_translate = lambda ids: ids
    kv_pool.full_kv_pool = NS(
        get_cpu_copy=Mock(return_value="kv"), load_cpu_copy=Mock()
    )
    allocator = NS(
        get_kvcache=lambda: kv_pool,
        get_cpu_copy=kv_pool.get_cpu_copy,
        load_cpu_copy=kv_pool.load_cpu_copy,
    )
    req = Req.__new__(Req)
    req.kv = ReqKvInfo(req_pool_idx=0, mamba_pool_idx=torch.tensor(2))
    req.origin_input_ids, req.output_ids = [1, 2], [3]
    req.expanded_input_len = None
    with patch.object(pools.current_platform, "synchronize"):
        req.offload_kv_cache(req_pool, allocator)
        backup = req.kv.retraction_backup
        assert backup.cpu_tensors == ("kv", None)
        assert len(backup.mamba_cpu) == 3
        req.kv.mamba_pool_idx = torch.tensor(1)
        req.load_kv_cache(req_pool, allocator)
    assert torch.all(mamba.mamba_cache.conv[0][:, 1] == 11)
    assert torch.all(mamba.mamba_cache.temporal[:, 1] == 22)
    assert torch.all(short.conv_state[:, 1] == 33)
    assert req.kv.retraction_backup is None
    kv_pool.full_kv_pool.load_cpu_copy.assert_called_once()


@pytest.mark.parametrize("verify", [False, True])
def test_qwen_ple_batch_keeps_persistent_indices_and_masks_graph_padding(verify):
    from sglang.srt.models import qwen4_exp
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    pool = _pool()
    pool.req_index_to_mamba_index_mapping[:] = torch.tensor([9, 10, 11, 12])
    pool.req_index_to_ple_index_mapping[:] = torch.tensor([1, 2, 3, 4])
    mode = ForwardMode.TARGET_VERIFY if verify else ForwardMode.DECODE
    batch = NS(
        tbo_parent_token_range=None,
        spec_algorithm=None,
        spec_info=NS(topk=1, draft_token_num=2) if verify else None,
        _original_forward_mode=None,
        forward_mode=mode,
        global_num_token_non_padded_cpu=4 if verify else 2,
        extend_seq_lens_cpu=None,
        extend_seq_lens=None,
        req_pool_indices=torch.tensor([0, 2]),
        out_cache_loc=torch.tensor([1, 2, 0, 0]) if verify else torch.tensor([1, 0]),
    )
    with patch.object(qwen4_exp, "get_req_to_token_pool", return_value=pool):
        prepared = qwen4_exp._prepare_ple_batch(
            torch.arange(4 if verify else 2),
            batch,
            ngram_size=None,
            ngram_eos_token_id=None,
        )
    assert prepared.state_indices.tolist() == [1, 0]
