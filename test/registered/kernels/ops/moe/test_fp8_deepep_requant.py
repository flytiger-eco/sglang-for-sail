"""DeepEP requantization for 128-aligned hidden sizes, including Qwen4's 2560."""

import sys

import pytest
import torch

from sglang.kernels.ops.moe.ep_moe_kernels import (
    fp8_per_token_to_per_tensor_quant_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("hidden_size", [128, 640, 1024, 2560, 4096])
def test_requant_matches_reference_and_graph_replay(hidden_size):
    torch.manual_seed(12)
    experts, capacity = 4, 48
    x = torch.randint(-6, 7, (experts, capacity, hidden_size), device="cuda").to(
        torch.float8_e4m3fn
    )
    # DeepEP supplies scales with non-contiguous token and block dimensions.
    scales = (
        torch.randint(1, 4, (experts, hidden_size // 128, capacity), device="cuda")
        .float()
        .div_(16)
        .transpose(1, 2)
    )
    masked_m = torch.tensor([0, 1, 33, capacity], device="cuda", dtype=torch.int32)
    output_scale = torch.tensor([0.5], device="cuda")
    output = torch.full_like(x, 0.25)

    def run():
        fp8_per_token_to_per_tensor_quant_triton(
            x, scales, masked_m, output_scale, output
        )

    def check():
        expected = (x.float() * scales.repeat_interleave(128, -1) / output_scale).to(
            x.dtype
        )
        valid = torch.arange(capacity, device="cuda")[None, :] < masked_m[:, None]
        expected = torch.where(valid[:, :, None], expected.float(), 0.25)
        torch.testing.assert_close(output.float(), expected, rtol=0, atol=0)

    run()
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.fill_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    check()

    # Reuse the graph after changing which experts have valid rows, including
    # zero rows and rows beyond the kernel's 32 token programs.
    masked_m.copy_(torch.tensor([48, 32, 1, 0], device="cuda", dtype=torch.int32))
    output.fill_(0.25)
    graph.replay()
    torch.cuda.synchronize()
    check()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
