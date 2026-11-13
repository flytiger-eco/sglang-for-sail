from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.attention.qsa.cp import QSAOnlyCPContext, qsa_only_cp_gather_rows
from sglang.srt.layers.attention.qsa.kernel import (
    QSAPrefillPhysicalSelection,
    expand_map_qsa_physical_slots,
    torch_expand_qsa_block_indices,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.utils.hc_seq_parallel import (
    hc_sp_pack_interleaved,
    hc_sp_scatter_interleaved,
    hc_sp_unpack_interleaved,
)


@pytest.mark.parametrize("rows", [0, 1, 7, 16, 17])
@pytest.mark.parametrize("world", [1, 2, 4])
def test_interleaved_ownership_roundtrip(rows, world):
    local = (rows + world - 1) // world
    x = torch.arange(rows * 3).reshape(rows, 3).float()
    packed = hc_sp_pack_interleaved(
        x, num_tokens=rows, local_tokens=local, world_size=world
    )
    restored = hc_sp_unpack_interleaved(
        packed, num_tokens=rows, local_tokens=local, world_size=world
    )
    torch.testing.assert_close(restored, x)
    for rank in range(world):
        shard = hc_sp_scatter_interleaved(
            x, num_tokens=rows, local_tokens=local, world_size=world, rank=rank
        )
        expected = torch.zeros(local, 3)
        owned = x[rank::world]
        expected[: len(owned)] = owned
        torch.testing.assert_close(shard, expected)
        torch.testing.assert_close(packed[rank * local : (rank + 1) * local], expected)


def test_qsa_gather_restores_request_order_without_padding():
    x = torch.arange(30).reshape(10, 3).float()
    world, local = 4, 3
    rank_major = hc_sp_pack_interleaved(
        x, num_tokens=10, local_tokens=local, world_size=world
    )

    def gather(out, source):
        out.copy_(rank_major)

    parallel = SimpleNamespace(
        attn_tp_group=SimpleNamespace(all_gather_into_tensor=gather)
    )
    with patch(
        "sglang.srt.layers.attention.qsa.cp.get_parallel", return_value=parallel
    ):
        for rank in range(world):
            context = QSAOnlyCPContext(10, local, rank, world, True)
            result = qsa_only_cp_gather_rows(x[rank::world], context)
            torch.testing.assert_close(result, x)


def test_physical_selection_marker_survives_query_padding_trim():
    indices = torch.arange(30).reshape(5, 6).int()
    result = QSAPrefillPhysicalSelection(indices)[:3]
    assert isinstance(result, QSAPrefillPhysicalSelection)
    torch.testing.assert_close(result.slots, indices[:3])


def test_fused_physical_mapping_matches_logical_selection_for_mixed_requests():
    blocks = torch.tensor([[0, -1], [-1, -1], [0, 1]], dtype=torch.int32)
    positions = torch.tensor([4, 0, 7], dtype=torch.int32)
    lengths = torch.tensor([5, 1, 8], dtype=torch.int32)
    sequence_ids = torch.tensor([0, 1, 0], dtype=torch.int32)
    slots = torch.stack([torch.arange(200, 208), torch.arange(300, 308)]).int()
    logical = torch_expand_qsa_block_indices(blocks, positions, lengths, 4, 8)
    mapped = expand_map_qsa_physical_slots(
        blocks, positions, lengths, sequence_ids, slots, compress_ratio=4, token_topk=8
    )
    expected = torch.where(
        logical >= 0,
        slots[sequence_ids.long()[:, None], logical.clamp_min(0).long()],
        -1,
    )
    torch.testing.assert_close(mapped, expected)


def test_cp_logical_mapping_uses_each_ranks_request_ids():
    metadata = SimpleNamespace(
        token_to_batch_idx=torch.tensor([0, 0, 0, 1, 1, 1]),
        sequence_lengths=torch.tensor([3, 3]),
        token_slot_table=torch.tensor([[40, 41, 42], [70, 71, 72]]),
    )
    logical = torch.tensor([[0, 1, -1], [2, -1, -1], [0, 2, -1]], dtype=torch.int32)
    context = QSAOnlyCPContext(6, 3, 0, 2, True)
    result = QwenSparseAttnBackend._logical_to_physical(logical, metadata, context)
    expected = torch.tensor(
        [[40, 41, -1], [42, -1, -1], [70, 72, -1]], dtype=torch.int32
    )
    torch.testing.assert_close(result, expected)


def test_cp_compression_uses_global_image_mrope_not_logical_cache_positions():
    from unittest.mock import Mock

    from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer

    class CompressionReached(Exception):
        pass

    world, rank, rows = 2, 1, 6
    logical = torch.arange(rows)
    mrope = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 1, 1, 2, 3], [0, 1, 2, 1, 2, 3]])
    context = QSAOnlyCPContext(rows, 3, rank, world, True)
    local_rope = mrope[:, rank::world].contiguous()
    global_keys = torch.randn(rows, 1, 8)
    metadata = SimpleNamespace(
        get_token_to_batch_idx=lambda: torch.zeros(rows, dtype=torch.long),
        token_to_kv_pool=SimpleNamespace(
            get_qsa_compressed_k_buffer=lambda _: torch.empty(0, dtype=torch.bfloat16)
        ),
        pending_ring_slots=None,
    )
    mode = SimpleNamespace(
        is_decode=lambda: False,
        is_target_verify=lambda: False,
        is_draft_extend_v2=lambda: False,
    )
    batch = SimpleNamespace(positions=logical, mrope_positions=mrope, forward_mode=mode)

    def compress(keys, positions, rope, metadata, **kwargs):
        torch.testing.assert_close(keys, global_keys)
        torch.testing.assert_close(positions, logical)
        torch.testing.assert_close(rope, mrope)
        raise CompressionReached()

    indexer = SimpleNamespace(
        layer_id=0,
        project_qk=Mock(
            return_value=(torch.zeros(3, 1, 8), global_keys[rank::world], False)
        ),
        update_key_state_and_compress=compress,
    )
    with patch(
        "sglang.srt.layers.attention.qsa.qsa_indexer.qsa_only_cp_gather_rows",
        return_value=global_keys,
    ):
        with pytest.raises(CompressionReached):
            QSAIndexer.forward_cuda(
                indexer,
                torch.randn(3, 8),
                local_rope,
                batch,
                metadata,
                qsa_cp_context=context,
            )
    torch.testing.assert_close(indexer.project_qk.call_args.args[1], local_rope)
