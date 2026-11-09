# SPDX-License-Identifier: Apache-2.0
"""Load-to-forward integration; DeepEP payloads are supplied without networking."""

import pytest
import torch
from test_qwen38_w4a8_sm import check_close, checkpoint

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires Hopper",
)


def loaded_layer(monkeypatch, backend="ours", bad_down=False):
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod

    monkeypatch.setenv("SGLANG_W4AFP8_MOE_BACKEND", backend)
    layer = torch.nn.Module()
    method = W4AFp8MoEMethod(W4AFp8Config())
    layer.quant_method = method
    with torch.device("cuda"):
        method.create_weights(
            layer, 4, 2560, 640, torch.bfloat16, weight_loader=lambda *a: None
        )
    gate, gs, gr = checkpoint(4, 1280, 2560)
    down, ds, dr = checkpoint(4, 2560, 640)
    if bad_down:
        down[0, 0, 0] = 8
    with torch.no_grad():
        layer.w13_weight.copy_(gate)
        layer.w2_weight.copy_(down)
        layer.w13_weight_scale_inv.copy_(gs)
        layer.w2_weight_scale_inv.copy_(ds)
        layer.w13_input_scale.fill_(0.5)
        layer.w2_input_scale.fill_(0.03125)
    method.create_moe_runner(
        layer, MoeRunnerConfig(num_experts=4, routed_scaling_factor=1.25)
    )
    method.process_weights_after_loading(layer)
    ptrs = (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
    method.process_weights_after_loading(layer)
    assert ptrs == (layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr())
    return layer, gr, dr, gate, down


def reference_expert(x, gate, down):
    x = (x.float() / 0.5).clamp(-448, 448).to(torch.float8_e4m3fn).float() * 0.5
    z = x @ gate.T
    z = torch.nn.functional.silu(z[..., :640]) * z[..., 640:]
    z = (z / 0.03125).clamp(-448, 448).to(torch.float8_e4m3fn).float() * 0.03125
    return (z @ down.T).bfloat16()


def test_checkpoint_fallback_is_atomic(monkeypatch):
    layer, _, _, gate, down = loaded_layer(monkeypatch, backend="auto", bad_down=True)
    assert not layer.quant_method.use_sm
    assert torch.equal(layer.w13_weight, gate)
    assert torch.equal(layer.w2_weight, down)
    with pytest.raises(ValueError, match="contains -8"):
        loaded_layer(monkeypatch, backend="ours", bad_down=True)


@pytest.mark.parametrize("mode", ["standard", "normal", "low_latency"])
def test_loader_to_forward_and_graph(monkeypatch, mode):
    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.runtime_context import get_parallel

    torch.backends.cuda.matmul.allow_tf32 = False
    layer, gate, down, _, _ = loaded_layer(monkeypatch)
    assert layer.quant_method.use_sm
    counts = [0, 1, 33, 65]
    if mode == "low_latency":
        a = (torch.randn((4, 65, 2560), device="cuda") * 0.5).to(torch.float8_e4m3fn)
        scales = torch.full((4, 65, 20), 0.5, device="cuda")
        scales[..., ::3] = 0.25
        masked = torch.tensor(counts, device="cuda", dtype=torch.int32)
        ids = torch.zeros((32, 10), device="cuda", dtype=torch.int64)
        dispatch = DeepEPLLDispatchOutput(
            a, scales, ids, torch.ones_like(ids, dtype=torch.float32), masked, 1
        )

        def run():
            return layer.quant_method.apply_deepep_ll(layer, dispatch)

        def verify(out):
            for expert, count in enumerate(counts):
                if count:
                    x = (
                        a[expert, :count].float().reshape(count, 20, 128)
                        * scales[expert, :count, :, None]
                    ).reshape(count, 2560)
                    check_close(
                        out[expert, :count],
                        reference_expert(x, gate[expert], down[expert]),
                    )
    else:
        a = (torch.randn((32, 2560), device="cuda") * 0.25).bfloat16()
        ids = torch.tensor(
            [[1, 3], [3, 2], [3, -1], [-1, -1]] * 8, device="cuda", dtype=torch.int64
        )
        weights = torch.rand((32, 2), device="cuda")
        if mode == "standard":
            dispatch = StandardDispatchOutput(
                a, None, StandardTopKOutput(weights, ids, torch.empty(0, device="cuda"))
            )

            def run():
                return layer.quant_method.apply(layer, dispatch).hidden_states
        else:
            dispatch = DeepEPNormalDispatchOutput(a, None, ids, weights, [0, 8, 8, 24])

            def run():
                return layer.quant_method.apply_deepep_normal(layer, dispatch)

        def verify(out):
            reference = torch.zeros_like(a, dtype=torch.float32)
            for expert in range(4):
                rows, slots = torch.where(ids == expert)
                if rows.numel():
                    z = reference_expert(a[rows], gate[expert], down[expert]).float()
                    reference.index_add_(0, rows, z * weights[rows, slots, None])
            if mode == "standard":
                reference *= 1.25
            check_close(out, reference.bfloat16())

    # Force only topology accessors; execute the actual loader and CUDA/Triton ops.
    with get_parallel().override(moe_ep_size=4, moe_ep_rank=0):
        for _ in range(3):
            out = run()
        torch.cuda.synchronize()
        verify(out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        verify(out)
        if mode == "low_latency":
            counts = [33, 0, 65, 1]
            masked.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        else:
            ids.copy_(ids.roll(1, dims=0))
        graph.replay()
        torch.cuda.synchronize()
        verify(out)
        if mode != "low_latency":
            ids.fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.count_nonzero(out).item() == 0
