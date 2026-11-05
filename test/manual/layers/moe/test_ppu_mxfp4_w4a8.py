"""One-GPU W4A8 integration checks; requires the matching PPU DeepGEMM build.

Uses real weight preprocessing, activation quantization and both GEMMs.
This does not replace checkpoint accuracy or multi-node DeepEP validation.
"""

import tempfile

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.runtime_context import get_flags, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="requires a PPU")


@pytest.fixture(scope="module", autouse=True)
def runtime():
    publish(
        ServerArgs(model_path="dummy", device="cuda", moe_runner_backend="deep_gemm"),
        role="test",
    )
    get_flags().moe.runner_backend = MoeRunnerBackend.DEEP_GEMM
    get_flags().moe.a2a_backend = MoeA2ABackend.DEEPEP
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    with tempfile.TemporaryDirectory(prefix="w4a8-single-rank-") as directory:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{directory}/rendezvous",
            backend="nccl",
        )
        initialize_model_parallel()
        try:
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
            reset_context()


def decode_weights(packed, scales):
    lookup = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    return lookup[codes] * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def quantize_activation_reference(x):
    groups = x.float().reshape(*x.shape[:-1], 1, x.shape[-1])
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
    q = (groups / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (q.float() * scale).reshape(x.shape)


def make_layer(intermediate=256):
    e, h, topk = 4, 512, 2
    config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        is_fp4_experts=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    runner_config = MoeRunnerConfig(
        num_experts=e,
        num_local_experts=e,
        hidden_size=h,
        intermediate_size_per_partition=intermediate,
        top_k=topk,
        params_dtype=torch.bfloat16,
        inplace=False,
    )
    method = Fp8MoEMethod(config, use_ppu_w4a8=True)
    layer = torch.nn.Module()
    layer.moe_runner_config = runner_config
    with torch.device("cuda"):
        method.create_weights(layer, e, h, intermediate, torch.bfloat16)
    reference_weights = []
    for name in ("w13", "w2"):
        weight = getattr(layer, f"{name}_weight")
        scale = getattr(layer, f"{name}_weight_scale_inv")
        packed = torch.randint(256, weight.shape, device="cuda", dtype=torch.uint8)
        raw = torch.randint(119, 124, scale.shape, device="cuda", dtype=torch.uint8)
        weight.data.copy_(packed.view(torch.int8))
        scale.data.copy_(raw)
        reference_weights.append(decode_weights(packed, raw))
    method.process_weights_after_loading(layer)
    method.create_moe_runner(layer, runner_config)
    assert layer.w13_weight.dtype == torch.uint8
    assert layer.w2_weight.dtype == torch.uint8
    assert layer.w13_weight.shape[-1] == h // 2
    assert layer.w2_weight.shape[-1] == intermediate // 2
    return layer, method, reference_weights


def reference_moe(x, ids, route_weights, weights):
    gate_up, down = weights
    h = quantize_activation_reference(x)
    result = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for expert in range(gate_up.shape[0]):
        tokens, slots = torch.where(ids == expert)
        if tokens.numel() == 0:
            continue
        gu = (h[tokens] @ gate_up[expert].T).bfloat16()
        gate, up = gu.float().chunk(2, dim=-1)
        activated = (F.silu(gate) * up).bfloat16()
        quantized = quantize_activation_reference(activated)
        out = (quantized @ down[expert].T).bfloat16()
        result.index_add_(0, tokens, out.float() * route_weights[tokens, slots, None])
    return result.bfloat16()


@torch.inference_mode()
def test_contiguous_moe():
    torch.manual_seed(20261104)
    layer, method, weights = make_layer()
    counts = [0, 1, 17, 65]
    rows = torch.tensor(counts, device="cuda", dtype=torch.int32)
    indices = torch.repeat_interleave(
        torch.arange(4, device="cuda", dtype=torch.int32), rows.long()
    )
    x = torch.randn(sum(counts), 512, device="cuda", dtype=torch.bfloat16)
    expected = reference_moe(
        x, indices[:, None], torch.ones(sum(counts), 1, device="cuda"), weights
    )
    q, scales = sglang_per_token_group_quant_fp8(
        x,
        x.shape[-1],
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    )
    quant = DeepGemmMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_scale=layer.w13_weight_scale_inv,
        w2_scale=layer.w2_weight_scale_inv,
        w13_w4a8_inv_scale=layer.w13_w4a8_inv_scale,
        w2_w4a8_inv_scale=layer.w2_w4a8_inv_scale,
        use_fp8=True,
        per_channel_quant=True,
        is_fp4_experts=True,
        use_mxfp4_w4a8=True,
        block_shape=None,
    )
    actual = method.runner.runner_core.run(
        DeepGemmRunnerInput(q, scales, use_masked_gemm=False, m_indices=indices),
        quant,
        {
            "all_tokens": x.shape[0],
            "hidden_states_device": x.device,
            "hidden_states_dtype": x.dtype,
            "hidden_states_shape": x.shape,
            "m_rows": rows,
        },
    ).hidden_states
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    delta = (actual.float() - expected.float()).square().sum()
    energy = actual.float().square().sum() + expected.float().square().sum()
    difference = (delta / energy.clamp_min(1e-20)).item()
    print(f"contiguous normalized_squared_error={difference:.8g}", flush=True)
    assert difference < 0.0015


@pytest.mark.parametrize("intermediate", [1536, 2048])
@torch.inference_mode()
def test_masked_moe_graph_replay(intermediate):
    torch.manual_seed(20261104)
    layer, method, weights = make_layer(intermediate)
    x = torch.randn(4, 128, 512, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([0, 1, 17, 65], device="cuda", dtype=torch.int32)
    quant = DeepGemmMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_scale=layer.w13_weight_scale_inv,
        w2_scale=layer.w2_weight_scale_inv,
        w13_w4a8_inv_scale=layer.w13_w4a8_inv_scale,
        w2_w4a8_inv_scale=layer.w2_w4a8_inv_scale,
        use_fp8=True,
        per_channel_quant=True,
        is_fp4_experts=True,
        use_mxfp4_w4a8=True,
        block_shape=None,
    )

    def forward():
        q, scales = sglang_per_token_group_quant_fp8(
            x.view(-1, x.shape[-1]), x.shape[-1]
        )
        q = q.view(x.shape)
        scales = scales.view(*x.shape[:-1], 1)
        return method.runner.runner_core.run(
            DeepGemmRunnerInput(
                q, scales, use_masked_gemm=True, masked_m=counts, expected_m=32
            ),
            quant,
            {"hidden_states_device": x.device},
        ).hidden_states

    def check_reference(actual):
        for expert, count in enumerate(counts.tolist()):
            if not count:
                continue
            ids = torch.full((count, 1), expert, device="cuda", dtype=torch.int32)
            expected = reference_moe(
                x[expert, :count], ids, torch.ones(count, 1, device="cuda"), weights
            )
            observed = actual[expert, :count]
            assert torch.isfinite(observed).all()
            delta = (observed.float() - expected.float()).square().sum()
            energy = observed.float().square().sum() + expected.float().square().sum()
            assert (delta / energy.clamp_min(1e-20)).item() < 0.0015

    capture_flags = get_flags().capture
    previous_dispose = capture_flags.disable_dispose_tensor
    capture_flags.disable_dispose_tensor = True
    try:
        for _ in range(3):
            eager = forward()
        torch.cuda.synchronize()
        check_reference(eager)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward()
        for factor, occupancy in [(0.25, [65, 0, 1, 17]), (2.0, [17, 65, 0, 1])]:
            x.copy_(torch.randn_like(x) * factor)
            counts.copy_(torch.tensor(occupancy, device="cuda", dtype=torch.int32))
            graph.replay()
            eager = forward()
            torch.cuda.synchronize()
            for expert, count in enumerate(occupancy):
                torch.testing.assert_close(
                    captured[expert, :count], eager[expert, :count], rtol=0, atol=0
                )
            check_reference(captured)
    finally:
        capture_flags.disable_dispose_tensor = previous_dispose
