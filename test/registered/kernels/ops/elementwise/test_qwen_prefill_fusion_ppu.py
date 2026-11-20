"""Prefill fusion must retain BF16 rounding, padding, history and output ownership."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from sglang.kernels.ops.mamba.qwen4_ple_prefill import (
    apply_ple_prefill_gate,
    can_pack_ple_prefill,
    pack_ple_prefill,
)
from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import (
    sigmoid_gate_mul_broadcast_bf16,
)
from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_ppu(), reason="PPU required"
)


def packing_inputs(dtype=torch.bfloat16):
    lengths = torch.tensor([5, 0, 2], device="cuda", dtype=torch.long)
    starts = torch.tensor([0, 5, 5, 7], device="cuda", dtype=torch.long)
    slots = torch.tensor([2, 0, 1], device="cuda", dtype=torch.long)
    x = torch.randn((7, 96), device="cuda", dtype=torch.bfloat16)
    # A strided state pool tests all three passed strides.
    storage = torch.randn((10, 96, 6), device="cuda", dtype=dtype)
    return x, storage[:, :, ::2], slots, lengths, starts


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_history_padding_and_strides(dtype):
    x, state, slots, lengths, starts = packing_inputs(dtype)
    result = pack_ple_prefill(x, state, slots, lengths, starts, 5)
    for row, length in enumerate([5, 0, 2]):
        history = state[slots[row]].to(x.dtype)
        tail = torch.zeros((96, 5), device="cuda", dtype=x.dtype)
        tail[:, :length] = x[starts[row] : starts[row + 1]].t()
        torch.testing.assert_close(
            result[row], torch.cat([history, tail], 1), rtol=0, atol=0
        )


@pytest.mark.parametrize("bad", ["missing_starts", "fp16_state", "cpu_indices"])
def test_unsupported_packing_metadata(bad):
    x, state, slots, lengths, starts = packing_inputs()
    if bad == "missing_starts":
        starts = None
    elif bad == "fp16_state":
        state = state.half()
    else:
        slots = slots.cpu()
    assert not can_pack_ple_prefill(x, state, slots, lengths, starts, 5)
    with pytest.raises(ValueError):
        pack_ple_prefill(x, state, slots, lengths, starts, 5)


@pytest.mark.parametrize("rows", [0, 1, 13, 512])
@torch.inference_mode()
def test_ple_broadcast_preserves_nonfinite_values(rows):
    gate = torch.randn((rows, 4, 1), device="cuda", dtype=torch.bfloat16)
    value = torch.randn((rows, 2560), device="cuda", dtype=torch.bfloat16)
    if rows:
        gate[0, :, 0] = torch.tensor([0, 1, float("nan"), -0.0], device="cuda")
        value[0, :3] = torch.tensor(
            [float("inf"), -float("inf"), float("nan")], device="cuda"
        )
    torch.testing.assert_close(
        apply_ple_prefill_gate(gate, value),
        gate * value.unsqueeze(1),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 13, 512])
@torch.inference_mode()
def test_shared_sigmoid_rounding_and_fallback(dtype, rows):
    x = torch.randn((rows, 2560), device="cuda", dtype=dtype)
    gate = torch.randn((rows, 1), device="cuda", dtype=dtype)
    torch.testing.assert_close(
        sigmoid_gate_mul_broadcast_bf16(x, gate),
        torch.sigmoid(gate) * x,
        rtol=0,
        atol=0,
    )


@torch.inference_mode()
def test_convolution_live_and_radix_states(monkeypatch):
    import sglang.srt.models.qwen4_exp as model
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    x, state, slots, lengths, starts = packing_inputs(torch.float32)
    original = state.clone()
    layer = model.Qwen4ExpPLELayer.__new__(model.Qwen4ExpPLELayer)
    torch.nn.Module.__init__(layer)
    layer.layer_id = 0
    layer.conv_channels = 96
    layer.short_conv_state_len = 3
    layer.short_conv_dilation = 1
    layer.ple_embedding = SimpleNamespace(enable_ple_fusion=True)
    layer.conv1d = torch.nn.Conv1d(
        96, 96, 4, groups=96, bias=False, device="cuda", dtype=x.dtype
    )
    pool = SimpleNamespace(short_conv_layer_cache=lambda _: state)
    monkeypatch.setattr(model, "get_req_to_token_pool", lambda: pool)
    track_slots = torch.tensor([7, 8, 9], device="cuda", dtype=torch.long)
    track_offsets = torch.tensor([3, 0, 1], device="cuda", dtype=torch.long)
    monkeypatch.setattr(
        model, "_ple_track_targets", lambda *_: (track_slots, track_offsets)
    )
    req = torch.tensor([0, 0, 0, 0, 0, 2, 2], device="cuda", dtype=torch.long)
    offsets = torch.tensor([0, 1, 2, 3, 4, 0, 1], device="cuda", dtype=torch.long)
    batch = SimpleNamespace(
        use_decode_fast_path=False,
        mode=ForwardMode.EXTEND,
        state_indices=slots,
        lengths=lengths,
        query_start_loc=starts,
        row_width=5,
        req_indices=req,
        token_offsets=offsets,
    )
    result = layer._short_conv(x, None, batch)
    expected = original.clone()
    outputs = []
    for row, length in enumerate([5, 0, 2]):
        segment = x[starts[row] : starts[row + 1]].t()
        full = torch.cat([original[slots[row]].to(x.dtype), segment], 1)
        expected[slots[row]] = full[:, length : length + 3].to(expected.dtype)
        offset = [3, 0, 1][row]
        expected[track_slots[row]] = full[:, offset : offset + 3].to(expected.dtype)
        if length:
            outputs.append(
                F.silu(F.conv1d(full.unsqueeze(0), layer.conv1d.weight, groups=96))
                .squeeze(0)
                .t()
            )
    torch.testing.assert_close(result, torch.cat(outputs), rtol=0, atol=0)
    torch.testing.assert_close(state, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_graph_output_ownership():
    x = torch.randn((13, 2560), device="cuda", dtype=torch.bfloat16)
    gate = torch.randn((13, 1), device="cuda", dtype=torch.bfloat16)
    sigmoid_gate_mul_broadcast_bf16(x, gate)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        a = sigmoid_gate_mul_broadcast_bf16(x, gate)
        b = sigmoid_gate_mul_broadcast_bf16(x * 2, gate)
    graph.replay()
    torch.cuda.synchronize()
    assert a.data_ptr() != b.data_ptr()
    torch.testing.assert_close(a, torch.sigmoid(gate) * x, rtol=0, atol=0)
