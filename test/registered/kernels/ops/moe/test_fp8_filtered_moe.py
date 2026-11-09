"""Regression: allocator contents in nonlocal EP rows must not alter local output."""

import importlib
import sys
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module", autouse=True)
def runtime():
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    from sglang.test.layer_ut_utils import init_single_process_dist

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    init_single_process_dist(master_port=29643)


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "tensor-fp8"])
@pytest.mark.parametrize(
    "poison", [float("inf"), 1e20], ids=["infinity", "large-finite"]
)
def test_filtered_rows_do_not_change_local_expert_output(fp8, poison):
    from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant

    module = importlib.import_module(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe"
    )
    torch.manual_seed(17)
    tokens, hidden, inter, experts, topk = 13, 256, 128, 2, 2
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) / 4
    w1 = (
        torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16)
        / 16
    )
    w2 = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16) / 16
    weights = torch.full((tokens, topk), 0.5, device="cuda", dtype=torch.float32)
    # This is the actual per-rank routing representation after EP remapping.
    ids = torch.tensor([[0, -1]] * tokens, device="cuda", dtype=torch.int32)
    kw = {}
    if fp8:

        def quant(w):
            pairs = [scaled_fp8_quant(e) for e in w]
            return torch.stack([q for q, _ in pairs]), torch.cat([s for _, s in pairs])

        w1, s1 = quant(w1)
        w2, s2 = quant(w2)
        kw = dict(use_fp8_w8a8=True, w1_scale=s1, w2_scale=s2)

    original_empty = torch.empty

    def run(stale_value):
        def allocate(*args, **kwargs):
            result = original_empty(*args, **kwargs)
            if (
                result.is_cuda
                and result.dtype == torch.bfloat16
                and tuple(result.shape) == (tokens * topk, inter)
            ):
                result.fill_(stale_value)
            return result

        # Simulate allocator reuse deterministically; all compute kernels remain real.
        with patch.object(torch, "empty", allocate):
            return module.fused_experts_impl(
                x.clone(), w1, w2, weights, ids, filter_expert=True, **kw
            )

    reference = run(0.0)
    actual = run(poison)
    assert torch.isfinite(reference).all()
    assert torch.isfinite(actual).all(), "unused EP rows poisoned local FP8 output"
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
