"""QSA prefill keeps request lengths and compressed-key ranges aligned to queries."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention.qsa import qsa_indexer as indexer_module
from sglang.srt.layers.attention.qsa.cp import QSAOnlyCPContext
from sglang.srt.layers.attention.qsa.kernel import QSAPrefillPhysicalSelection
from sglang.srt.layers.attention.qsa.metadata import (
    QSAIndexerMetadata,
    build_qsa_prefill_mqa_plan,
)
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "requests",
    [((4, 0), (8, 0)), ((11, 4), (5, 1), (17, 8)), ((1, 0),)],
    ids=["mixed-lengths", "cached-prefixes", "empty-cp-ranks"],
)
@pytest.mark.parametrize(
    "world,rank", [(1, 0), (2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)]
)
@pytest.mark.parametrize("cached_plan", [False, True])
@pytest.mark.parametrize("physical", [False, True])
def test_prefill_metadata_and_selected_tokens_match_request_ownership(
    monkeypatch, requests, world, rank, cached_plan, physical
):
    ratio, token_topk, slot_width = 4, 8, 32
    lengths = torch.tensor([length for length, _ in requests], dtype=torch.int32)
    owners = torch.tensor(
        [
            sid
            for sid, (length, prefix) in enumerate(requests)
            for _ in range(prefix, length)
        ],
        dtype=torch.int32,
    )
    positions = torch.tensor(
        [pos for length, prefix in requests for pos in range(prefix, length)]
    )
    total_rows = len(positions)
    capacity = (total_rows + world - 1) // world
    local_indices = list(range(rank, total_rows, world))
    local_positions = positions[local_indices]
    context = (
        QSAOnlyCPContext(total_rows, capacity, rank, world, True)
        if world > 1
        else None
    )
    slots = torch.arange(len(requests) * slot_width).reshape(len(requests), slot_width)
    compressed_cache = torch.arange(len(requests) * slot_width).float().view(-1, 1, 1)
    pool = SimpleNamespace(
        get_qsa_compressed_k_buffer=lambda _: compressed_cache,
        qsa_index_kv_heads=1,
        qsa_index_head_dim=1,
    )
    plan = (
        build_qsa_prefill_mqa_plan(
            sequence_lengths=lengths,
            sequence_lengths_cpu=lengths.tolist(),
            token_slot_table=slots,
            query_positions=positions,
            query_sequence_ids=owners,
            compress_ratio=ratio,
        )
        if cached_plan
        else None
    )
    metadata = QSAIndexerMetadata(
        sequence_lengths=lengths,
        token_to_batch_idx=owners,
        token_slot_table=slots,
        out_cache_loc=torch.arange(total_rows),
        token_to_kv_pool=pool,
        compress_ratio=ratio,
        block_topk=token_topk // ratio,
        pending_ring_slots=torch.arange(total_rows),
        prefill_mqa_plan=plan,
    )
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, positions=positions)
    indexer = SimpleNamespace(
        layer_id=0,
        compress_ratio=ratio,
        token_topk=token_topk,
        block_topk=token_topk // ratio,
        project_qk=lambda hidden, *args, **kwargs: (
            torch.ones(len(hidden), 1, 1),
            torch.ones(len(hidden), 1, 1),
            False,
        ),
        update_key_state_and_compress=Mock(),
    )
    expected_starts, expected_ends, expected_lengths, expected_tokens = [], [], [], []
    for row in local_indices:
        sid, pos = int(owners[row]), int(positions[row])
        length = requests[sid][0]
        start = sum(previous_length // ratio for previous_length, _ in requests[:sid])
        visible_blocks = (pos + 1) // ratio
        expected_starts.append(start)
        expected_ends.append(start + visible_blocks)
        expected_lengths.append(length)
        # Scores increase with compressed slot ID: select the newest complete
        # blocks, followed by the incomplete causal tail for this query.
        first_token = max(0, visible_blocks - token_topk // ratio) * ratio
        tokens = list(range(first_token, pos + 1))
        if physical:
            tokens = [int(slots[sid, token]) for token in tokens]
        padding = [-1] * (token_topk + ratio - 1 - len(tokens))
        expected_tokens.append(sorted(tokens + padding))

    def select(q, keys, starts, ends, query_positions, row_lengths, **kwargs):
        assert starts.tolist() == expected_starts
        assert ends.tolist() == expected_ends
        assert row_lengths.tolist() == expected_lengths
        torch.testing.assert_close(query_positions, local_positions)
        return QSAIndexer.select_prefill_tokens(
            indexer, q, keys, starts, ends, query_positions, row_lengths, **kwargs
        )

    indexer.select_prefill_tokens = select
    monkeypatch.setattr(
        indexer_module,
        "qsa_only_cp_gather_rows",
        lambda *args: torch.ones(total_rows, 1, 1),
    )
    monkeypatch.setenv("SGLANG_QSA_FUSED_PREFILL_PHYSICAL_SLOTS", str(int(physical)))
    monkeypatch.setattr(indexer_module.qsa_tilelang, "is_available", lambda: True)
    result = QSAIndexer.forward_cuda(
        indexer,
        torch.ones(capacity + 1, 1),
        local_positions,
        batch,
        metadata,
        qsa_cp_context=context,
    )
    if physical:
        assert isinstance(result, QSAPrefillPhysicalSelection)
        result = result.slots
    expected = torch.tensor(expected_tokens, dtype=torch.int32).reshape(
        len(local_indices), token_topk + ratio - 1
    )
    torch.testing.assert_close(result.sort(dim=1).values, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
