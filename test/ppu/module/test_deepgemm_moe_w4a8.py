"""Run on M890 after installing DeepGEMM-for-sail PR #36."""

import json
import os
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.ppu_deepgemm_moe import deep_moe_impl_fused
from sglang.srt.utils import get_device_sm, is_ppu

pytestmark = pytest.mark.skipif(
    not is_ppu() or get_device_sm() != 89 or os.getenv("SGLANG_TEST_PPU_W4A8") != "1",
    reason="Requires M890, W4A8 DeepGEMM and SGLANG_TEST_PPU_W4A8=1",
)


def _unpack_mxfp4(weight, scale):
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weight.device, dtype=torch.float32
    )
    return (
        values[codes & 7]
        * torch.where(codes < 8, 1.0, -1.0)
        * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=-1)
    )


def _case(m, e=4, hidden=256, intermediate=128, top_k=2, scale_shift=0):
    import deep_gemm

    torch.manual_seed(42)
    device = "cuda"
    x = torch.randn((m, hidden), device=device, dtype=torch.bfloat16)
    w13 = torch.randint(
        0, 256, (e, 2 * intermediate, hidden // 2), device=device, dtype=torch.uint8
    )
    w2 = torch.randint(
        0, 256, (e, hidden, intermediate // 2), device=device, dtype=torch.uint8
    )
    s13 = torch.randint(
        124 + scale_shift,
        131 + scale_shift,
        (e, 2 * intermediate, hidden // 32),
        device=device,
        dtype=torch.uint8,
    )
    s2 = torch.randint(
        124 + scale_shift,
        131 + scale_shift,
        (e, hidden, intermediate // 32),
        device=device,
        dtype=torch.uint8,
    )
    w13_pre, s13_pre, inv13 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, w13, s13
    )
    down_k_pad = (-intermediate) % 128
    w2_pre, s2_pre, inv2 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA,
        False,
        torch.nn.functional.pad(w2, (0, down_k_pad // 2)),
        torch.nn.functional.pad(s2, (0, down_k_pad // 32), value=127),
    )
    ids = torch.randint(0, e, (m, top_k), device=device, dtype=torch.int32)
    weights = torch.rand((m, top_k), device=device, dtype=torch.float32)
    weights /= weights.sum(-1, keepdim=True)
    return (
        x,
        (w13, s13, w13_pre, s13_pre, inv13),
        (w2, s2, w2_pre, s2_pre, inv2),
        ids,
        weights,
    )


def _prepare(case, precision):
    import deep_gemm

    _, (w13, s13, w13_pre, s13_pre, inv13), (w2, s2, w2_pre, s2_pre, inv2), _, _ = case
    if precision == "w4a8":
        return (
            w13_pre,
            w2_pre,
            s13_pre,
            s2_pre,
            dict(use_mxfp4_w4a8=True, w1_inv_scale=inv13, w2_inv_scale=inv2),
        )
    if precision == "w4a4":
        return (
            w13,
            w2,
            deep_gemm.preprocess_mxfp4_scales(s13),
            deep_gemm.preprocess_mxfp4_scales(s2),
            dict(use_mxfp4=True, block_shape=[0, 16]),
        )
    w13_pre, s13_pre = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False, w13, s13
    )
    w2_pre, s2_pre = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A16_MXFP4_MMA, False, w2, s2
    )
    return w13_pre, w2_pre, s13_pre, s2_pre, dict(use_mxfp4_w4a16=True)


def _run(case, fused, precision="w4a8", prepared=None, activation="situ"):
    x, _, _, ids, weights = case
    w13, w2, s13, s2, quant_args = prepared or _prepare(case, precision)
    envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.set(fused)
    out = torch.empty_like(x)
    deep_moe_impl_fused(
        x,
        w13,
        w2,
        s13,
        s2,
        weights,
        ids,
        activation=activation,
        gemm1_alpha=4.0 if activation == "situ" else None,
        gemm1_limit=25.0 if activation == "situ" else None,
        out_hidden_states=out,
        **quant_args,
    )
    return out


def _reference(case, activation="situ"):
    x, (w13, s13, *_), (w2, s2, *_), ids, weights = case
    w13 = _unpack_mxfp4(w13, s13)
    w2 = _unpack_mxfp4(w2, s2)
    result = torch.zeros_like(x, dtype=torch.float32)
    for row in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            gate, up = (x[row].float() @ w13[expert].T).chunk(2)
            if activation == "situ":
                activated = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
                activated *= 25.0 * torch.tanh(up / 25.0)
            else:
                activated = torch.nn.functional.silu(gate) * up
            result[row] += (activated @ w2[expert].T) * weights[row, slot]
    return result.bfloat16()


def _relative_l2(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.parametrize("m,hidden", [(1, 256), (8, 3584), (2048, 1076), (4096, 3584)])
def test_w4a8_fp8_quant_folds_inverse_without_changing_codes(m, hidden):
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

    x = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16)
    values, scales = sglang_per_token_quant_fp8(x)
    fused_values, fused_scales = sglang_per_token_quant_fp8(x, scale_multiplier=0.0625)
    torch.testing.assert_close(fused_values.float(), values.float(), rtol=0, atol=0)
    torch.testing.assert_close(fused_scales, scales * 0.0625, rtol=1e-5, atol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_w4a8_silu_accuracy(fused):
    case = _case(8)
    try:
        reference = _reference(case, activation="silu")
        errors = {}
        for precision in ("w4a8", "w4a4", "w4a16"):
            result = _run(case, fused, precision, activation="silu")
            assert torch.isfinite(result).all()
            errors[precision] = _relative_l2(result, reference)
        assert errors["w4a16"] < 0.02
        assert errors["w4a8"] < errors["w4a4"]
    finally:
        envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.set(False)


def _masked_case(e, hidden, intermediate, tokens=16):
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
    from sglang.srt.layers.quantization.ppu_mxfp4_utils import downcast_to_mxfp4

    case = _case(1, e=e, hidden=hidden, intermediate=intermediate)
    x = torch.randn((e, tokens, hidden), device="cuda", dtype=torch.bfloat16)
    masked_m = torch.randint(0, 3, (e,), device="cuda", dtype=torch.int32)
    masked_m[0] = 4
    fp8, fp8_scale = sglang_per_token_quant_fp8(x.reshape(-1, hidden))
    fp4, fp4_scale = downcast_to_mxfp4(x, axis=-1)
    inputs = {
        "w4a8": (fp8.reshape(e, tokens, hidden), fp8_scale.reshape(e, tokens, 1)),
        "w4a4": (fp4, fp4_scale),
        "w4a16": (x, None),
    }
    runner = object.__new__(DeepGemmRunnerCore)
    runner.config = SimpleNamespace(
        activation="situ", gemm1_alpha=4.0, gemm1_clamp_limit=25.0, top_k=16
    )
    runner.swiglu_limit = None
    methods = {
        "w4a8": runner._run_w4a8_masked_gemm,
        "w4a4": runner._run_masked_fp4_gemm,
        "w4a16": runner._run_masked_int4_gemm,
    }
    calls = {}
    for precision in methods:
        w13, w2, s13, s2, _ = _prepare(case, precision)
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            w13_scale=s13,
            w2_scale=s2,
            use_mxfp4_w4a8=precision == "w4a8",
            use_mxfp4=precision == "w4a4",
            use_mxfp4_w4a16=precision == "w4a16",
            w13_inv_scale=case[1][-1],
            w2_inv_scale=case[2][-1],
        )
        hidden_states, scale = inputs[precision]
        runner_input = DeepGemmRunnerInput(
            hidden_states=hidden_states,
            hidden_states_scale=scale,
            use_masked_gemm=True,
            masked_m=masked_m,
            expected_m=2,
        )
        calls[precision] = (
            lambda method=methods[precision], ri=runner_input, qi=quant_info: method(
                ri, qi, {"hidden_states_device": x.device}
            )
        )
    valid = torch.arange(tokens, device="cuda")[None, :] < masked_m[:, None]
    return calls, valid


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("m,intermediate", [(1, 128), (8, 128), (64, 128), (8, 192)])
@pytest.mark.parametrize("scale_shift", [0, 13])
def test_kimi_k3_w4a8_accuracy(m, intermediate, fused, scale_shift):
    case = _case(m, intermediate=intermediate, scale_shift=scale_shift)
    if scale_shift:
        assert case[1][-1] != 1.0 and case[2][-1] != 1.0
    try:
        reference = _reference(case)
        errors = {}
        for precision in ("w4a8", "w4a4", "w4a16"):
            result = _run(case, fused, precision)
            assert torch.isfinite(result).all()
            errors[precision] = _relative_l2(result, reference)
        print(
            f"M={m} intermediate={intermediate} fused={fused} shift={scale_shift}: {errors}"
        )
        assert errors["w4a16"] < 0.02
        assert errors["w4a8"] < errors["w4a4"]
        assert errors["w4a8"] < (0.35 if scale_shift else 0.15)
    finally:
        envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.set(False)


@pytest.mark.parametrize("activation", ["situ", "silu"])
def test_kimi_k3_w4a8_masked_accuracy(activation):
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

    case = _case(16)
    x, (w13, s13, w13_pre, s13_pre, inv13), (w2, s2, w2_pre, s2_pre, inv2), _, _ = case
    e, t = 4, 4
    q, scale = sglang_per_token_quant_fp8(x)
    masked_m = torch.tensor([4, 2, 1, 0], device=x.device, dtype=torch.int32)
    runner_input = DeepGemmRunnerInput(
        hidden_states=q.reshape(e, t, -1),
        hidden_states_scale=scale.reshape(e, t, 1),
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=3,
    )
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13_pre,
        w2_weight=w2_pre,
        w13_scale=s13_pre,
        w2_scale=s2_pre,
        use_mxfp4_w4a8=True,
        w13_inv_scale=inv13,
        w2_inv_scale=inv2,
    )
    runner = object.__new__(DeepGemmRunnerCore)
    runner.config = type(
        "Config",
        (),
        {"activation": activation, "gemm1_alpha": 4.0, "gemm1_clamp_limit": 25.0},
    )()
    runner.swiglu_limit = None
    out = runner._run_w4a8_masked_gemm(runner_input, quant_info, {})
    w13_ref = _unpack_mxfp4(w13, s13)
    w2_ref = _unpack_mxfp4(w2, s2)
    actual_rows = []
    expected_rows = []
    for expert, count in enumerate(masked_m.tolist()):
        for row in range(count):
            gate, up = (x[expert * t + row].float() @ w13_ref[expert].T).chunk(2)
            if activation == "situ":
                activated = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
                activated *= 25.0 * torch.tanh(up / 25.0)
            else:
                activated = torch.nn.functional.silu(gate) * up
            reference = (activated @ w2_ref[expert].T).bfloat16()
            actual_rows.append(out[expert, row])
            expected_rows.append(reference)
    assert _relative_l2(torch.stack(actual_rows), torch.stack(expected_rows)) < 0.2


@pytest.mark.skipif(
    os.getenv("SGLANG_TEST_PPU_W4A8_REAL_SHAPE") != "1",
    reason="Set SGLANG_TEST_PPU_W4A8_REAL_SHAPE=1 for K3 TP16 dimensions",
)
def test_kimi_k3_w4a8_tp16_shape():
    case = _case(1, e=896, hidden=3584, intermediate=192, top_k=16)
    try:
        w4a8 = _run(case, False)
        w4a4 = _run(case, False, "w4a4")
        w4a16 = _run(case, False, "w4a16")
        assert _relative_l2(w4a8, w4a16) < _relative_l2(w4a4, w4a16)
        assert _relative_l2(w4a8, w4a16) < 0.2
        torch.testing.assert_close(_run(case, True), w4a8, rtol=0, atol=0)
    finally:
        envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.set(False)


@pytest.mark.skipif(
    not os.getenv("SGLANG_TEST_PPU_W4A8_CHECKPOINT"),
    reason="Set SGLANG_TEST_PPU_W4A8_CHECKPOINT to the K3 checkpoint directory",
)
def test_kimi_k3_w4a8_checkpoint_experts():
    import deep_gemm
    from safetensors import safe_open

    root = Path(os.environ["SGLANG_TEST_PPU_W4A8_CHECKPOINT"])
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = "language_model.model.layers.12.block_sparse_moe.experts."
    shard = root / weight_map[prefix + "0.w1.weight_packed"]
    gate_up, gate_up_scale, down, down_scale = [], [], [], []
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        for expert in range(8):
            name = prefix + str(expert) + "."
            gate_up.append(
                torch.cat(
                    (
                        checkpoint.get_tensor(name + "w1.weight_packed")[:192],
                        checkpoint.get_tensor(name + "w3.weight_packed")[:192],
                    )
                )
            )
            gate_up_scale.append(
                torch.cat(
                    (
                        checkpoint.get_tensor(name + "w1.weight_scale")[:192],
                        checkpoint.get_tensor(name + "w3.weight_scale")[:192],
                    )
                )
            )
            down.append(checkpoint.get_tensor(name + "w2.weight_packed")[:, :96])
            down_scale.append(checkpoint.get_tensor(name + "w2.weight_scale")[:, :6])

    w13, s13 = torch.stack(gate_up).cuda(), torch.stack(gate_up_scale).cuda()
    w2, s2 = torch.stack(down).cuda(), torch.stack(down_scale).cuda()
    w13_pre, s13_pre, inv13 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA, False, w13, s13
    )
    w2_pre, s2_pre, inv2 = deep_gemm.preprocess_weights_and_scales(
        deep_gemm.PreprocessMode.W4A8_MXFP4_MMA,
        False,
        torch.nn.functional.pad(w2, (0, 32)),
        torch.nn.functional.pad(s2, (0, 2), value=127),
    )
    torch.manual_seed(123)
    x = torch.randn((8, 3584), device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 8, (8, 4), device="cuda", dtype=torch.int32)
    scores = torch.rand((8, 4), device="cuda")
    scores /= scores.sum(-1, keepdim=True)
    case = (
        x,
        (w13, s13, w13_pre, s13_pre, inv13),
        (w2, s2, w2_pre, s2_pre, inv2),
        ids,
        scores,
    )
    try:
        reference = _reference(case)
        errors = {
            precision: _relative_l2(_run(case, False, precision), reference)
            for precision in ("w4a8", "w4a4", "w4a16")
        }
        print(f"K3 layer 12 checkpoint experts 0..7 relative L2: {errors}")
        assert errors["w4a16"] < 0.02
        assert errors["w4a8"] < 0.1
        assert errors["w4a8"] < errors["w4a4"]
    finally:
        envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.set(False)


@pytest.mark.parametrize("e,intermediate", [(8, 512), (112, 3072)])
def test_kimi_k3_w4a8_masked_latency(e, intermediate):
    calls, valid = _masked_case(e, 3584, intermediate)
    timings = {}
    with patch("sglang.srt.layers.moe.moe_runner.deep_gemm.dispose_tensor"):
        outputs = {precision: call() for precision, call in calls.items()}
        reference = outputs["w4a16"][valid]
        errors = {
            precision: _relative_l2(output[valid], reference)
            for precision, output in outputs.items()
        }
        assert errors["w4a8"] < errors["w4a4"]
        for precision, call in calls.items():
            for _ in range(5):
                call()
            torch.cuda.synchronize()
            samples = []
            for _ in range(20):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                call()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            timings[precision] = statistics.median(samples)
    print(
        f"masked E={e} H=3584 I={intermediate} valid={int(valid.sum())}: ms={timings} rel_to_w4a16={errors}"
    )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("m", [1, 32, 256])
def test_kimi_k3_w4a8_latency(m, fused):
    case = _case(
        m,
        e=int(os.getenv("SGLANG_TEST_PPU_W4A8_EXPERTS", "32")),
        hidden=int(os.getenv("SGLANG_TEST_PPU_W4A8_HIDDEN", "256")),
        intermediate=int(os.getenv("SGLANG_TEST_PPU_W4A8_INTERMEDIATE", "128")),
        top_k=int(os.getenv("SGLANG_TEST_PPU_W4A8_TOPK", "2")),
    )
    try:
        timings = {}
        for precision in ("w4a8", "w4a4", "w4a16"):
            prepared = _prepare(case, precision)
            for _ in range(5):
                _run(case, fused, precision, prepared)
            torch.cuda.synchronize()
            samples = []
            for _ in range(20):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _run(case, fused, precision, prepared)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            timings[precision] = statistics.median(samples)
        print(f"M={m} E={case[1][0].shape[0]} fused={fused} median_ms={timings}")
    finally:
        envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED.set(False)
