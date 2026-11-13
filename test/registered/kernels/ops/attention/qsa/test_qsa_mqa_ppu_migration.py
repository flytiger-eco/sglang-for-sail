"""PPU QSA MQA numerical, sub-page and graph-replay regressions."""

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.qsa.mqa import (
    tilelang_qsa_mqa_decode,
    tilelang_qsa_mqa_prefill,
    torch_qsa_mqa_decode,
    torch_qsa_mqa_prefill,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("keys", [33, 4097, 8193])
@pytest.mark.parametrize("bf16_logits", [False, True])
def test_prefill_tiles_and_logits_match_fp32_reference(keys, bf16_logits):
    torch.manual_seed(47)
    torch.backends.cuda.matmul.allow_tf32 = False
    q = torch.randn(33, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(keys, 1, 128, device="cuda", dtype=torch.bfloat16)
    starts = torch.arange(33, device="cuda", dtype=torch.int32) % 7
    ends = torch.arange(33, device="cuda", dtype=torch.int32) * (keys // 33)
    ends = ends.maximum(starts).clamp_max(keys)
    expected = torch_qsa_mqa_prefill(q, k, starts, ends)
    with envs.SGLANG_QSA_PREFILL_BF16_LOGITS.override(bf16_logits):
        actual = tilelang_qsa_mqa_prefill(q, k, starts, ends)
    assert actual.dtype == (torch.bfloat16 if bf16_logits else torch.float32)
    torch.testing.assert_close(torch.isneginf(actual), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        actual[finite].float(), expected[finite], rtol=0.02, atol=0.04
    )


@pytest.mark.parametrize("page_size", [16, 32, 64])
@pytest.mark.parametrize("storage_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_decode_persistent_grid_and_subpage_copies(page_size, storage_dtype):
    torch.manual_seed(53)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, heads, dim, max_length = 5, 8, 128, 4096
    q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(512, page_size, 1, dim, device="cuda", dtype=torch.bfloat16).to(
        storage_dtype
    )
    pages = torch.randint(
        0, 512, (rows, max_length // page_size), device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([0, 3, 17, 71, 137], device="cuda", dtype=torch.int32)
    expected = torch_qsa_mqa_decode(q, k, pages, lengths, max_length)
    for _ in range(3):
        actual = tilelang_qsa_mqa_decode(q, k, pages, lengths, max_length)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)


def test_decode_graph_replay_uses_current_lengths_and_pages():
    torch.manual_seed(59)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, page_size, max_length = 3, 16, 4096
    q = torch.randn(rows, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(512, page_size, 1, 128, device="cuda", dtype=torch.bfloat16)
    pages = torch.randint(
        0, 512, (rows, max_length // page_size), device="cuda", dtype=torch.int32
    )
    lengths = torch.tensor([0, 7, 129], device="cuda", dtype=torch.int32)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        tilelang_qsa_mqa_decode(q, k, pages, lengths, max_length)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = tilelang_qsa_mqa_decode(q, k, pages, lengths, max_length)
    lengths.copy_(torch.tensor([161, 0, 33], device="cuda", dtype=torch.int32))
    pages.copy_(torch.randint(0, 512, pages.shape, device="cuda", dtype=torch.int32))
    graph.replay()
    torch.testing.assert_close(
        output,
        torch_qsa_mqa_decode(q, k, pages, lengths, max_length),
        rtol=0.02,
        atol=0.04,
    )
