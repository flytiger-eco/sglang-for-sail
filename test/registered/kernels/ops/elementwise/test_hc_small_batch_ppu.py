"""PPU HC precision boundaries, fallback dispatch and graph workspace isolation."""

import pytest
import torch
import torch.nn.functional as F
from sglang.srt.layers.hc_small_batch_triton import (
    hc_combine_norm_small,
    hc_combine_norm_supported,
    hc_mix_down,
    hc_mix_down_supported,
)
from sglang.srt.layers.hyperconnection import GroupedGemmaRMSNorm
from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="PPU-specific HC kernels")


@pytest.mark.parametrize("rows", [2, 3, 6, 12, 24, 32])
def test_down_preserves_bf16_linear_and_activation(rows):
    torch.manual_seed(1300 + rows)
    x = torch.randn((rows, 10240), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((320, 10240), device="cuda", dtype=torch.bfloat16) * 0.01
    assert hc_mix_down_supported(x, weight)
    expected = F.silu(F.linear(x, weight) / 4)
    torch.testing.assert_close(hc_mix_down(x, weight), expected, rtol=0.016, atol=0.002)


@pytest.mark.parametrize("rows", [1, 3, 6, 12, 24, 32])
def test_combine_normalizes_the_stored_bf16_residual(rows):
    torch.manual_seed(2300 + rows)
    residual = torch.randn((rows, 10240), device="cuda", dtype=torch.bfloat16)
    block = torch.randn((rows, 2560), device="cuda", dtype=torch.bfloat16)
    normed = torch.randn_like(residual)
    weight = torch.randn((4, 10240), device="cuda", dtype=torch.bfloat16) * 0.01
    norm = GroupedGemmaRMSNorm(10240, group_size=2560).to("cuda", dtype=torch.bfloat16)
    norm.weight.data.normal_(0, 0.1)
    assert hc_combine_norm_supported(block, residual, normed, weight, norm.weight)
    updated, normalized = hc_combine_norm_small(
        block, residual, normed, weight, norm.weight, norm.variance_epsilon
    )
    gate = 2 * torch.sigmoid(F.linear(normed.float(), weight.float()) / 4)
    reference = (
        (
            residual.float().reshape(rows, 4, 2560)
            + gate.unsqueeze(-1) * block.float().unsqueeze(1)
        )
        .flatten(1)
        .to(torch.bfloat16)
    )
    torch.testing.assert_close(updated, reference, rtol=0.016, atol=0.002)
    torch.testing.assert_close(normalized, norm(updated), rtol=0.016, atol=0.002)


@pytest.mark.parametrize("rows", [0, 1, 33, 8192])
def test_down_dispatch_keeps_unsupported_rows_on_existing_path(rows):
    x = torch.empty((rows, 10240), device="cuda", dtype=torch.bfloat16)
    weight = torch.empty((320, 10240), device="cuda", dtype=torch.bfloat16)
    assert not hc_mix_down_supported(x, weight)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_down_dispatch_keeps_other_dtypes_on_existing_path(dtype):
    x = torch.empty((3, 10240), device="cuda", dtype=dtype)
    weight = torch.empty((320, 10240), device="cuda", dtype=dtype)
    assert not hc_mix_down_supported(x, weight)


def test_graph_replay_keeps_down_workspaces_independent():
    torch.manual_seed(3300)
    weight = torch.randn((320, 10240), device="cuda", dtype=torch.bfloat16) * 0.01
    x = torch.randn((3, 10240), device="cuda", dtype=torch.bfloat16)
    other = torch.randn_like(x)
    hc_mix_down(x, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        first = hc_mix_down(x, weight)
        second = hc_mix_down(other, weight)
    for scale in [0.1, 1.0, 10.0]:
        x.normal_(0, scale)
        other.normal_(0, scale)
        graph.replay()
        torch.testing.assert_close(
            first, F.silu(F.linear(x, weight) / 4), rtol=0.016, atol=0.002
        )
        torch.testing.assert_close(
            second, F.silu(F.linear(other, weight) / 4), rtol=0.016, atol=0.002
        )
