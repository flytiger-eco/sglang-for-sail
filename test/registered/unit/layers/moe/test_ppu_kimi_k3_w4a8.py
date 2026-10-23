"""CPU contract checks for the PPU Kimi-K3 W4A8 MoE path."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.kernels.ops.quantization import fp8_kernel as fp8_module
from sglang.srt.layers.deep_gemm_wrapper import compile_utils, entrypoint
from sglang.srt.layers.moe import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_layer
from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module
from sglang.srt.layers.moe.moe_runner import ppu_deepgemm_moe as ppu_module
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.moe.token_dispatcher import deepep as deepep_module
from sglang.srt.layers.quantization import mxfp4


def _quant_info():
    return DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((2, 256, 128), dtype=torch.uint8),
        w2_weight=torch.empty((2, 256, 64), dtype=torch.uint8),
        w13_scale=torch.empty((2, 4, 512), dtype=torch.uint8),
        w2_scale=torch.empty((2, 4, 256), dtype=torch.uint8),
        use_mxfp4_w4a8=True,
        w13_inv_scale=2.0,
        w2_inv_scale=4.0,
    )


def _runner(activation="situ"):
    runner = object.__new__(DeepGemmRunnerCore)
    runner.config = SimpleNamespace(
        activation=activation, gemm1_alpha=4.0, gemm1_clamp_limit=25.0
    )
    runner.swiglu_limit = None
    return runner


def test_w4a8_fp8_quant_selects_fused_scale_writer():
    x = torch.empty((2, 256), dtype=torch.bfloat16)
    with patch.object(fp8_module, "sgl_per_token_quant_fp8") as aot:
        fp8_module.sglang_per_token_quant_fp8(x)
        fp8_module.sglang_per_token_quant_fp8(x, scale_multiplier=2.0)
        assert [call.args[-1] for call in aot.call_args_list] == [1.0, 2.0]


def test_w4a8_fp8_quant_wrapper_forwards_scale():
    from sglang.kernels.ops import quantization
    from sglang.kernels.spec import KernelBackend

    x = torch.empty((2, 256), dtype=torch.bfloat16)
    q = torch.empty((2, 256), dtype=torch.float8_e4m3fn)
    scale = torch.empty((2, 1), dtype=torch.float32)
    aot = Mock()
    with (
        patch.dict(
            sys.modules,
            {"sgl_kernel": SimpleNamespace(sgl_per_token_quant_fp8=aot)},
        ),
        patch.object(quantization, "get_kernel") as resolve,
    ):
        quantization.sgl_per_token_quant_fp8(x, q, scale, 2.0)
        aot.assert_called_once_with(x, q, scale, 2.0)
        resolve.assert_not_called()

        quantization.sgl_per_token_quant_fp8(x, q, scale)
        resolve.assert_called_once_with(
            "quantization.sgl_per_token_quant_fp8", KernelBackend.JIT
        )
        resolve.return_value.assert_called_once_with(x, q, scale)


def test_w4a8_normal_dispatch_quant_folds_scale():
    dispatcher = object.__new__(deepep_module._DeepEPDispatcherImplNormal)
    dispatcher.use_fp8 = True
    dispatcher.async_finish = False
    dispatcher.quant_config = {"is_channel_quant": True, "w4a8_scale_multiplier": 2.0}
    x = torch.empty((1, 256), dtype=torch.bfloat16)
    topk = SimpleNamespace(
        topk_weights=torch.ones((1, 1)),
        topk_ids=torch.zeros((1, 1), dtype=torch.int64),
    )
    with (
        patch.object(deepep_module.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        patch.object(deepep_module, "sglang_per_token_group_quant_fp8") as quantize,
    ):
        dispatcher.dispatch_a(x, topk)
    assert quantize.call_args.kwargs["scale_multiplier"] == 2.0
    assert quantize.call_args.args[1] == 256


def test_w4a8_standard_preprocess_folds_scale():
    from sglang.kernels.ops.moe import ep_moe_kernels

    x = torch.empty((1, 256), dtype=torch.bfloat16)
    topk = (torch.ones((1, 1)), torch.zeros((1, 1), dtype=torch.int32), None)
    output = (
        torch.zeros(2, dtype=torch.int32),
        None,
        torch.zeros((1, 1), dtype=torch.int32),
        torch.empty((2, 1, 256), dtype=torch.float8_e4m3fn),
        torch.ones((2, 1, 1), dtype=torch.float32),
    )
    config = SimpleNamespace(num_experts=4, num_local_experts=2, top_k=1, inplace=False)
    with (
        patch.object(runner_module, "_is_ppu", True),
        patch.object(
            runner_module.deep_gemm_sm120,
            "allows_masked_standard_layout",
            return_value=False,
        ),
        patch.object(
            ep_moe_kernels, "moe_ep_deepgemm_preprocess", return_value=output
        ) as preprocess,
    ):
        result = runner_module.pre_permute_standard_to_deep_gemm(
            SimpleNamespace(hidden_states=x, topk_output=topk),
            _quant_info(),
            config,
            {},
        )
    assert preprocess.call_args.kwargs["scale_multiplier"] == 2.0
    assert result.w13_scale_compensated


@pytest.mark.parametrize("a2a", [MoeA2ABackend.NONE, MoeA2ABackend.DEEPEP])
def test_w4a8_selector_and_dispatch(a2a):
    with (
        patch.object(mxfp4, "_is_ppu", True),
        patch.object(mxfp4, "get_device_sm", return_value=89),
        patch.object(
            mxfp4, "get_moe_runner_backend", return_value=MoeRunnerBackend.DEEP_GEMM
        ),
        patch.object(mxfp4, "get_moe_a2a_backend", return_value=a2a),
        patch.object(
            mxfp4,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
            ),
        ),
        patch.object(mxfp4.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", True),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A8, "get", return_value=True
        ),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16, "get", return_value=False
        ),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA, "get", return_value=False
        ),
        patch.object(fused_layer, "_is_ppu", True),
    ):
        method = mxfp4.Mxfp4MoEMethod("model.layers.0.experts")
        layer = SimpleNamespace(quant_method=method, quant_config=None)
        config = fused_layer.FusedMoE._build_dispatcher_quant_config(layer)

    assert method.use_deepgemm_mxfp4_w4a8
    assert not method.use_deepgemm_mxfp4_w4a16_mma
    assert config == {"dispatcher_output_dtype": "fp8", "is_channel_quant": True}


def test_w4a8_opt_in_does_not_change_other_platforms():
    with (
        patch.object(mxfp4, "_is_ppu", False),
        patch.object(
            mxfp4, "get_moe_runner_backend", return_value=MoeRunnerBackend.DEEP_GEMM
        ),
        patch.object(
            mxfp4,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(flashinfer_mxfp4_moe_precision="default")
            ),
        ),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A8, "get", return_value=True
        ),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16, "get", return_value=False
        ),
        patch.object(
            mxfp4.envs.SGLANG_SAIL_DEEPGEMM_MXFP4_W4A16_MMA, "get", return_value=False
        ),
    ):
        method = mxfp4.Mxfp4MoEMethod("model.layers.0.experts")

    assert not method.use_deepgemm_mxfp4_w4a8


@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_w4a8_weight_creation_accepts_activation(activation):
    method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
    method.use_marlin = False
    method.use_deep_gemm = True
    method.use_deepgemm_mxfp4_w4a8 = True
    method.use_deepgemm_mxfp4_w4a16_valu = False
    method.use_deepgemm_mxfp4_w4a16_mma = False
    layer = torch.nn.Module()
    layer.num_local_experts = 2
    layer.hidden_size = 256
    layer.intermediate_size_per_partition = 128
    layer.moe_runner_config = SimpleNamespace(activation=activation)
    with patch.object(mxfp4, "_is_ppu", True):
        method.create_weights(layer, 2, 256, 128, torch.bfloat16)
    assert layer.w13_weight.shape == (2, 256, 128)


@pytest.mark.parametrize("a2a", [MoeA2ABackend.NONE, MoeA2ABackend.DEEPEP])
def test_w4a8_preprocess_keeps_independent_inverse_scales(a2a):
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.zeros((2, 256, 128), dtype=torch.uint8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.full((2, 256, 8), 127, dtype=torch.uint8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.zeros((2, 256, 64), dtype=torch.uint8), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.full((2, 256, 4), 127, dtype=torch.uint8), requires_grad=False
        ),
        dispatcher=SimpleNamespace(quant_config={"dispatcher_output_dtype": "fp8"}),
    )
    method = SimpleNamespace(use_deepgemm_mxfp4_w4a8=True)
    calls = []

    def preprocess(mode, fuse_activation, weight, scale):
        calls.append((mode, fuse_activation, weight.shape, scale.shape))
        return (
            weight,
            torch.full(
                (scale.shape[0], scale.shape[1] // 64, weight.shape[-1] * 4),
                len(calls),
                dtype=torch.uint8,
            ),
            float(len(calls) + 1),
        )

    fake_dg = SimpleNamespace(
        PreprocessMode=SimpleNamespace(W4A8_MXFP4_MMA=sentinel.w4a8),
        preprocess_weights_and_scales=preprocess,
    )
    with (
        patch.dict(sys.modules, {"deep_gemm": fake_dg}),
        patch.object(torch.cuda, "empty_cache"),
        patch.object(mxfp4, "get_moe_a2a_backend", return_value=a2a),
    ):
        mxfp4.Mxfp4MoEMethod.process_weights_after_loading(method, layer)

    assert [call[:2] for call in calls] == [(sentinel.w4a8, False)] * 2
    assert (layer.w13_weight_inv_scale, layer.w2_weight_inv_scale) == (2.0, 3.0)
    assert layer.w13_weight_scale.shape == (2, 4, 512)
    assert layer.w2_weight_scale.shape == (2, 4, 256)
    assert layer.dispatcher.quant_config.get("w4a8_scale_multiplier") == (
        2.0 if a2a.is_deepep() else None
    )


def test_w4a8_preprocess_pads_tp_down_k_after_loading():
    layer = SimpleNamespace(
        w13_weight=torch.nn.Parameter(
            torch.zeros((2, 384, 128), dtype=torch.uint8), requires_grad=False
        ),
        w13_weight_scale=torch.nn.Parameter(
            torch.full((2, 384, 8), 127, dtype=torch.uint8), requires_grad=False
        ),
        w2_weight=torch.nn.Parameter(
            torch.ones((2, 256, 96), dtype=torch.uint8), requires_grad=False
        ),
        w2_weight_scale=torch.nn.Parameter(
            torch.full((2, 256, 6), 125, dtype=torch.uint8), requires_grad=False
        ),
    )
    method = SimpleNamespace(use_deepgemm_mxfp4_w4a8=True)
    inputs = []

    def preprocess(mode, fuse_activation, weight, scale):
        inputs.append((weight.clone(), scale.clone()))
        return weight, scale, 1.0

    fake_dg = SimpleNamespace(
        PreprocessMode=SimpleNamespace(W4A8_MXFP4_MMA=sentinel.w4a8),
        preprocess_weights_and_scales=preprocess,
    )
    with (
        patch.dict(sys.modules, {"deep_gemm": fake_dg}),
        patch.object(torch.cuda, "empty_cache"),
        patch.object(mxfp4, "get_moe_a2a_backend", return_value=MoeA2ABackend.NONE),
    ):
        mxfp4.Mxfp4MoEMethod.process_weights_after_loading(method, layer)

    assert inputs[0][0].shape == (2, 384, 128)
    assert inputs[1][0].shape == (2, 256, 128)
    assert inputs[1][1].shape == (2, 256, 8)
    assert torch.count_nonzero(inputs[1][0][..., 96:]) == 0
    assert torch.all(inputs[1][1][..., 6:] == 127)


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("activation", ["silu", "situ"])
@pytest.mark.parametrize("compensated", [False, True])
def test_w4a8_runner_scales_both_gemms(masked, activation, compensated):
    quant_info = _quant_info()
    shape = (2, 3, 256) if masked else (3, 256)
    quantized_input = torch.empty(shape, dtype=torch.float8_e4m3fn)
    scale = torch.full((*shape[:-1], 1), 3.0)
    masked_m = torch.tensor([1, 2], dtype=torch.int32) if masked else None
    runner_input = DeepGemmRunnerInput(
        hidden_states=quantized_input,
        hidden_states_scale=scale,
        use_masked_gemm=masked,
        masked_m=masked_m,
        expected_m=2 if masked else None,
        m_indices=torch.zeros((3,), dtype=torch.int32) if not masked else None,
        w13_scale_compensated=compensated,
    )
    down_input = torch.empty((*shape[:-1], 128), dtype=torch.float8_e4m3fn)
    down_scale = torch.full((*shape[:-1], 1), 7.0)
    fake_dg = SimpleNamespace(
        ActivationQuantMode=SimpleNamespace(FP8_CHANNEL=sentinel.fp8_channel),
        situ_and_mul=lambda *args, **kwargs: (down_input, down_scale),
        silu_and_mul=lambda *args, **kwargs: (down_input, down_scale),
    )
    fn = (
        "grouped_gemm_nt_f8f4bf16_masked"
        if masked
        else "grouped_gemm_nt_f8f4bf16_nopad"
    )
    with (
        patch.dict(sys.modules, {"deep_gemm": fake_dg}),
        patch.object(runner_module.deep_gemm_wrapper, fn) as gemm,
        patch.object(
            fake_dg,
            f"{activation}_and_mul",
            wraps=getattr(fake_dg, f"{activation}_and_mul"),
        ) as act,
    ):
        output = (
            _runner(activation)._run_w4a8_masked_gemm(runner_input, quant_info, {})
            if masked
            else _runner(activation)._run_w4a8_contiguous_gemm(
                runner_input, quant_info, {"all_tokens": 3}
            )
        )

    assert output.shape == shape
    assert gemm.call_count == 2
    torch.testing.assert_close(
        gemm.call_args_list[0].args[0][1], scale if compensated else scale * 2
    )
    assert gemm.call_args_list[1].args[0][0] is down_input
    assert gemm.call_args_list[1].args[0][1] is down_scale
    assert act.call_args.args[:2] == (
        sentinel.fp8_channel,
        gemm.call_args_list[0].args[2],
    )
    assert act.call_args.kwargs["weight_inv_scale"] == 4.0
    if masked:
        assert act.call_args.kwargs["masked_m"] is masked_m
        assert gemm.call_args_list[1].args[-2:] == (masked_m, 2)


@pytest.mark.parametrize("layout", ["nopad", "masked", "fused"])
def test_w4a8_wrapper_uses_exported_grouped_gemm_name(layout):
    lhs_shape = (2, 8, 256) if layout == "masked" else (8, 256)
    lhs = (torch.empty(lhs_shape), torch.empty(lhs_shape[:-1] + (1,)))
    rhs = (torch.empty((2, 256, 128)), torch.empty((2, 256, 8)))
    out = torch.empty(lhs_shape[:-1] + (256,))
    extra_args = {
        "nopad": (sentinel.indices, sentinel.rows),
        "masked": (sentinel.masked_m, 8),
        "fused": (
            sentinel.rows,
            sentinel.experts,
            sentinel.tokens,
            sentinel.aligned,
            sentinel.config,
        ),
    }[layout]
    api_name = f"m_grouped_gemm_fp8_fp4_bf16_nt_{layout}"
    fake_dg = SimpleNamespace(**{api_name: Mock()})
    with (
        patch.object(entrypoint, "_is_ppu", True),
        patch.object(entrypoint, "deep_gemm", fake_dg, create=True),
        patch.object(
            compile_utils, "deep_gemm_execution_hook", return_value=nullcontext()
        ),
    ):
        getattr(entrypoint, f"grouped_gemm_nt_f8f4bf16_{layout}")(
            lhs, rhs, out, *extra_args
        )
    getattr(fake_dg, api_name).assert_called_once_with(lhs, rhs, out, *extra_args)


@pytest.mark.parametrize(
    "kernel_type,executor_name",
    [
        (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F4BF16_NOPAD,
            "_GroupedNopadWarmupExecutor_w4a8",
        ),
        (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F4BF16_MASKED,
            "_GroupedMaskedWarmupExecutor_w4a8",
        ),
        (
            compile_utils.DeepGemmKernelType.GROUPED_GEMM_NT_F8F4BF16_FUSED,
            "_GroupedFusedWarmupExecutor_w4a8",
        ),
    ],
)
def test_w4a8_precompile_registers_each_layout(kernel_type, executor_name):
    with patch.object(
        compile_utils, executor_name, return_value=sentinel.executor
    ) as executor:
        result = compile_utils._BaseWarmupExecutor.create(
            kernel_type, max_m=8, n=256, k=256, num_groups=4
        )
    assert result is sentinel.executor
    executor.assert_called_once_with(max_m=8, n=256, k=256, num_groups=4)
    assert (
        compile_utils._BaseWarmupExecutor.get_memory_requirement(
            kernel_type, max_m=8, n=256, k=256, num_groups=4
        )
        > 0
    )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("activation", ["silu", "situ"])
def test_w4a8_none_uses_fused_or_nopad_without_changing_combine(fused, activation):
    hidden_states = torch.empty((2, 256), dtype=torch.bfloat16)
    quantized = torch.empty((2, 256), dtype=torch.float8_e4m3fn)
    row_scale = torch.full((2, 1), 3.0)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    rows = torch.tensor([1, 1], dtype=torch.int32)
    indices = torch.tensor([0, 1], dtype=torch.int32)
    down_input = torch.empty((2, 128), dtype=torch.float8_e4m3fn)
    down_scale = torch.ones((2, 1), dtype=torch.float32)
    fake_dg = SimpleNamespace(
        ActivationQuantMode=SimpleNamespace(FP8_CHANNEL=sentinel.fp8_channel),
        situ_and_mul=lambda *args, **kwargs: (down_input, down_scale),
        silu_and_mul=lambda *args, **kwargs: (down_input, down_scale),
    )
    with (
        patch.dict(sys.modules, {"deep_gemm": fake_dg}),
        patch.object(
            ppu_module,
            "sglang_per_token_quant_fp8",
            side_effect=lambda x, scale_multiplier: (
                quantized,
                row_scale * scale_multiplier,
            ),
            create=True,
        ) as quantize,
        patch.object(
            ppu_module.envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED, "get", return_value=fused
        ),
        patch.object(
            ppu_module,
            "grouped_gemm_nt_f8f4bf16_fused",
            return_value=(rows, indices, indices),
        ) as fused_gemm,
        patch.object(
            ppu_module.deep_gemm_wrapper, "grouped_gemm_nt_f8f4bf16_nopad"
        ) as nopad_gemm,
        patch.object(
            ppu_module,
            "deepgemm_moe_permute",
            side_effect=lambda **kwargs: (
                quantized,
                kwargs["aq_scale"],
                indices,
                indices,
                rows,
            ),
        ) as permute,
        patch.object(
            fake_dg,
            f"{activation}_and_mul",
            wraps=getattr(fake_dg, f"{activation}_and_mul"),
        ) as act,
        patch.object(ppu_module, "ep_gather") as gather,
        patch.object(ppu_module, "SGLANG_PROFILE_NVTX", False),
    ):
        ppu_module.deep_moe_impl_fused(
            hidden_states,
            torch.empty((2, 256, 128), dtype=torch.uint8),
            torch.empty((2, 256, 64), dtype=torch.uint8),
            torch.empty((2, 4, 512), dtype=torch.uint8),
            torch.empty((2, 4, 256), dtype=torch.uint8),
            topk_weights,
            topk_ids,
            use_mxfp4_w4a8=True,
            w1_inv_scale=2.0,
            w2_inv_scale=4.0,
            activation=activation,
            gemm1_alpha=4.0 if activation == "situ" else None,
            gemm1_limit=25.0 if activation == "situ" else None,
            out_hidden_states=torch.empty_like(hidden_states),
        )

    assert fused_gemm.call_count == int(fused)
    assert permute.call_count == int(not fused)
    assert nopad_gemm.call_count == (1 if fused else 2)
    first = fused_gemm.call_args if fused else nopad_gemm.call_args_list[0]
    first_scale = first.args[1] if fused else first.args[0][1]
    torch.testing.assert_close(first_scale, torch.full((2, 1), 6.0))
    assert nopad_gemm.call_args.args[0][0] is down_input
    assert nopad_gemm.call_args.args[0][1] is down_scale
    assert quantize.call_args.kwargs["scale_multiplier"] == 2.0
    act.assert_called_once()
    gather.assert_called_once()
