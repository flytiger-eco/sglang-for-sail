"""Bounded decode scores and shared-K chain verification."""
import pytest
import torch

from sglang.kernels.ops.attention.fast_topk import fast_topk
from sglang.srt.layers.attention.qsa.mqa import tilelang_qsa_mqa_decode
from sglang.srt.utils.common import is_ppu

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_ppu(), reason="PPU required"
)


@pytest.mark.parametrize("page", [8, 16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_packed_verify_matches_independent_rows(page, dtype):
    torch.manual_seed(197)
    rows, width = 9, 4096
    q = torch.zeros(rows, 16, 128, device="cuda", dtype=torch.bfloat16)
    q[:, :4].normal_()
    keys = torch.randn(128, page, 1, 128, device="cuda", dtype=q.dtype).to(dtype)
    page_ids = torch.randint(0, 128, (3, width // page), device="cuda", dtype=torch.int32)
    table = page_ids.repeat_interleave(3, dim=0)
    lens = torch.tensor([0, 0, 0, 15, 16, 17, 511, 512, 513], device="cuda", dtype=torch.int32)
    # Invalid page entries must not be consumed, including when the final
    # query first crosses a compressed-page boundary.
    columns = torch.arange(table.shape[1], device="cuda")[None]
    table.masked_fill_(columns >= (lens[:, None] + page - 1) // page, -99999)
    independent = tilelang_qsa_mqa_decode(q, keys, table, lens, width)
    packed = tilelang_qsa_mqa_decode(q, keys, table, lens, width,
                                     mask_logits=False, verify_query_group_size=3)
    valid = torch.arange(width, device="cuda")[None] < lens[:, None]
    assert torch.equal(independent[valid], packed[valid])
    packed.masked_fill_(~valid, float("inf"))
    a = fast_topk(independent, lens, 512)
    b = fast_topk(packed, lens, 512)
    def scores(indices):
        values = independent.gather(1, indices.long().clamp_min(0))
        return torch.where(indices >= 0, values, -float("inf")).sort(-1).values
    assert torch.equal(scores(a), scores(b))


def test_packed_verify_replay_observes_new_lengths_and_page_ids():
    torch.manual_seed(199)
    q = torch.zeros(6, 16, 128, device="cuda", dtype=torch.bfloat16)
    q[:, :4].normal_()
    keys = torch.randn(256, 16, 1, 128, device="cuda", dtype=q.dtype)
    table = torch.randint(0, 256, (2, 256), device="cuda", dtype=torch.int32).repeat_interleave(3, 0)
    lens = torch.ones(6, device="cuda", dtype=torch.int32)
    for _ in range(3):
        tilelang_qsa_mqa_decode(q, keys, table, lens, 4096,
                               mask_logits=False, verify_query_group_size=3)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = tilelang_qsa_mqa_decode(q, keys, table, lens, 4096,
                                         mask_logits=False, verify_query_group_size=3)
    for length in [65, 0, 2049, 3, 4096, 16]:
        lens.fill_(length)
        table.copy_(table.reshape(2, 3, 256).flip(0).reshape(6, 256))
        graph.replay()
        expected = tilelang_qsa_mqa_decode(q, keys, table, lens, 4096)
        valid = torch.arange(4096, device="cuda")[None] < lens[:, None]
        assert torch.equal(output[valid], expected[valid])


def test_valid_page_metadata_replay_refreshes_growing_and_reused_rows():
    from sglang.srt.layers.attention.qsa.graph_metadata import (
        _qsa_graph_row_metadata_kernel,
    )

    rows, capacity, full_page, ratio = 3, 262144, 64, 4
    lengths = torch.ones(rows, device="cuda", dtype=torch.int32)
    reqs = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    request_table = torch.arange(4 * capacity, device="cuda", dtype=torch.int32).reshape(4, capacity)
    max_pages = capacity // full_page
    outputs = [torch.empty(rows, device="cuda", dtype=torch.int32) for _ in range(2)]
    pages = torch.full((rows, max_pages), -99999, device="cuda", dtype=torch.int32)
    positions = torch.empty_like(lengths)
    slots = torch.empty(rows, device="cuda", dtype=torch.int64)
    ring = torch.empty(rows * ratio, device="cuda", dtype=torch.int32)
    def launch():
        _qsa_graph_row_metadata_kernel[(rows,)](
            lengths, reqs, *outputs, pages, positions, slots, ring,
            request_table, request_table.stride(0), max_pages,
            RATIO=ratio, FULL_PAGE=full_page, PAGE_BLOCK=128, num_warps=1,
        )
    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for length in [65, capacity - 1, 1, 65536, 17]:
        lengths.copy_(torch.tensor([1, length, max(length - 3, 1)], device="cuda", dtype=torch.int32))
        reqs.copy_(reqs.roll(1))
        pages.fill_(-99999)
        graph.replay()
        used = (lengths // ratio + full_page // ratio - 1) // (full_page // ratio)
        valid = torch.arange(max_pages, device="cuda")[None] < used[:, None]
        expected = request_table[reqs.long(), ::full_page] // full_page
        assert torch.equal(pages[valid], expected[valid])
        assert bool((pages[~valid] == -99999).all())
        assert torch.equal(outputs[0], lengths // ratio)
        assert torch.equal(positions, lengths - 1)
