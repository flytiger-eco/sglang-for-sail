"""CPU checks for PPU Kimi-K3 MoE compatibility paths."""

import sys
from types import SimpleNamespace
from unittest.mock import ANY, patch, sentinel

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers.moe.moe_runner import acext as acext_module
from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_module
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)


def _w4a16_runner():
    runner = object.__new__(DeepGemmRunnerCore)
    runner.config = SimpleNamespace(
        activation="situ",
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
        top_k=8,
    )
    runner.swiglu_limit = None
    return runner


def _mxfp4_w4a16_quant_info():
    return DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((2, 1, 1), dtype=torch.uint8),
        w2_weight=torch.empty((2, 1, 1), dtype=torch.uint8),
        w13_scale=torch.empty((2, 1, 1), dtype=torch.uint8),
        w2_scale=torch.empty((2, 1, 1), dtype=torch.uint8),
        use_mxfp4_w4a16=True,
    )


def test_acext_situ_falls_back_before_importing_acext():
    sentinel = object()
    runner_config = SimpleNamespace(activation="situ")
    with (
        patch.object(acext_module, "is_ppu", return_value=True),
        patch.object(acext_module.logger, "info_once", create=True),
        patch.object(
            acext_module,
            "fused_experts_none_to_triton",
            return_value=sentinel,
        ) as fallback,
    ):
        result = acext_module.fused_experts_none_to_acext(
            "dispatch", "quant", runner_config
        )

    assert result is sentinel
    fallback.assert_called_once_with("dispatch", "quant", runner_config)


def test_mxfp4_w4a16_reuses_deepep_w4a16_runner_paths():
    runner = _w4a16_runner()
    quant_info = _mxfp4_w4a16_quant_info()
    contiguous_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((1, 4), dtype=torch.bfloat16),
        hidden_states_scale=None,
        use_masked_gemm=False,
    )
    masked_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((2, 1, 4), dtype=torch.bfloat16),
        hidden_states_scale=None,
        use_masked_gemm=True,
    )

    with (
        patch.object(deep_gemm_module, "SGLANG_PROFILE_NVTX", False),
        patch.object(
            runner,
            "_run_int4_contiguous_gemm",
            return_value=sentinel.contiguous_output,
        ) as contiguous,
        patch.object(
            runner,
            "_run_masked_int4_gemm",
            return_value=sentinel.masked_output,
        ) as masked,
    ):
        contiguous_output = runner.run(contiguous_input, quant_info, {})
        masked_output = runner.run(masked_input, quant_info, {})

    assert contiguous_output.hidden_states is sentinel.contiguous_output
    assert masked_output.hidden_states is sentinel.masked_output
    contiguous.assert_called_once_with(contiguous_input, quant_info, {})
    masked.assert_called_once_with(masked_input, quant_info, {})


def test_w4a16_deepep_uses_k3_situ():
    runner = _w4a16_runner()
    with patch("sglang.kernels.ops.kimi_k3.situ_and_mul") as situ_and_mul:
        runner._apply_situ_and_mul(
            sentinel.gateup_output,
            sentinel.down_input,
        )

    situ_and_mul.assert_called_once_with(
        sentinel.gateup_output,
        sentinel.down_input,
        4.0,
        25.0,
    )


def test_w4a16_deepep_ll_uses_masked_k3_situ():
    runner = _w4a16_runner()
    with patch("sglang.kernels.ops.kimi_k3.situ_and_mul_masked") as situ_and_mul:
        runner._apply_situ_and_mul(
            sentinel.gateup_output,
            sentinel.down_input,
            sentinel.masked_m,
            expected_m=3,
        )

    situ_and_mul.assert_called_once_with(
        sentinel.gateup_output,
        sentinel.down_input,
        sentinel.masked_m,
        4.0,
        25.0,
        8,
        3,
    )


def test_w4a16_deepep_ll_passes_mask_to_activation():
    runner = _w4a16_runner()
    masked_m = torch.tensor([1, 2], dtype=torch.int32)
    runner_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((2, 3, 4), dtype=torch.bfloat16),
        hidden_states_scale=None,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=2,
    )

    with (
        patch.object(
            deep_gemm_module.deep_gemm_wrapper,
            "grouped_gemm_nt_bf16i4bf16_masked",
        ) as grouped_gemm,
        patch.object(deep_gemm_module, "dispose_tensor"),
        patch.object(runner, "_apply_situ_and_mul") as apply_situ,
    ):
        output = runner._run_masked_int4_gemm(
            runner_input,
            _mxfp4_w4a16_quant_info(),
            {"hidden_states_device": torch.device("cpu")},
        )

    assert output.shape == (2, 3, 4)
    assert grouped_gemm.call_count == 2
    assert apply_situ.call_args.args[2] is masked_m
    assert apply_situ.call_args.kwargs["expected_m"] == 2


def test_w4a4_deepep_uses_k3_situ_mxfp4_post_quant():
    runner = _w4a16_runner()
    runner_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((2, 32), dtype=torch.uint8),
        hidden_states_scale=torch.empty((2, 2), dtype=torch.uint16),
        use_masked_gemm=False,
        m_indices=torch.zeros((2,), dtype=torch.int32),
    )
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((2, 128, 32), dtype=torch.uint8),
        w2_weight=torch.empty((2, 64, 32), dtype=torch.uint8),
        w13_scale=torch.empty((2, 2, 128), dtype=torch.uint8),
        w2_scale=torch.empty((2, 1, 128), dtype=torch.uint8),
        use_mxfp4=True,
    )
    with (
        patch.dict(
            sys.modules,
            {"deep_gemm": SimpleNamespace(preprocess_mxfp4_scales=lambda x: x)},
        ),
        patch.object(
            deep_gemm_module.deep_gemm_wrapper,
            "grouped_gemm_nt_f4f4bf16_nopad",
        ) as grouped_gemm,
        patch.object(deep_gemm_module, "dispose_tensor"),
        patch(
            "sglang.kernels.ops.kimi_k3.situ_and_mul_post_quant_mxfp4",
        ) as activation,
    ):
        runner._run_fp4_contiguous_gemm(
            runner_input,
            quant_info,
            {
                "all_tokens": 2,
                "hidden_states_device": torch.device("cpu"),
                "hidden_states_dtype": torch.uint8,
                "hidden_states_shape": (2, 64),
            },
        )

    activation.assert_called_once_with(ANY, ANY, ANY, 4.0, 25.0)
    assert activation.call_args.args[1].shape == (2, 32)
    assert activation.call_args.args[2].shape == (1, 2)
    assert grouped_gemm.call_args_list[1].args[0][1].shape == (2, 1)


def test_w4a4_deepep_ll_uses_masked_k3_situ_mxfp4_post_quant():
    runner = _w4a16_runner()
    masked_m = torch.tensor([1, 2], dtype=torch.int32)
    runner_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((2, 3, 32), dtype=torch.uint8),
        hidden_states_scale=torch.empty((2, 1, 3), dtype=torch.uint16),
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=2,
    )
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((2, 128, 32), dtype=torch.uint8),
        w2_weight=torch.empty((2, 64, 32), dtype=torch.uint8),
        w13_scale=torch.empty((2, 2, 128), dtype=torch.uint8),
        w2_scale=torch.empty((2, 1, 128), dtype=torch.uint8),
        use_mxfp4=True,
    )
    with (
        patch.object(
            deep_gemm_module.deep_gemm_wrapper,
            "grouped_gemm_nt_f4f4bf16_masked",
        ) as grouped_gemm,
        patch.object(deep_gemm_module, "dispose_tensor"),
        patch(
            "sglang.kernels.ops.kimi_k3.situ_and_mul_masked_post_quant_mxfp4",
        ) as activation,
    ):
        runner._run_masked_fp4_gemm(
            runner_input,
            quant_info,
            {"hidden_states_device": torch.device("cpu")},
        )

    activation.assert_called_once_with(ANY, ANY, ANY, masked_m, 4.0, 25.0, 8, 2)
    assert activation.call_args.args[1].shape == (2, 3, 32)
    assert activation.call_args.args[2].shape == (2, 1, 3)
    assert grouped_gemm.call_args_list[1].args[0][1].shape == (2, 3, 1)
