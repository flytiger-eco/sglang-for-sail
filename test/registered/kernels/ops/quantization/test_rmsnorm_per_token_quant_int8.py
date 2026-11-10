"""Precision tests for the fused RMSNorm + per-token int8 quant producer.

The ground truth is computed on the **CPU** in float64 from the same inputs,
so the kernel is compared with a result no accelerator kernel produced. The
int8 contract it has to keep is per_token_quant_int8's::

    scale = max(max|y|, 1e-10) / 127
    q     = round_half_even(y * 127 / max(absmax, 1e-10))

with the normalized row rounded to bf16 first. Against that reference the
fused launch must keep every element within one bf16 ulp of the correctly
rounded row, match the scale to fp32 precision, and stay within one int8 step
of the reference payload.

The producer is also wired into ``W8A8Int8LinearMethod`` through a
pre-quantized ``(int8, scale, out_dtype)`` tuple, so the GEMM must consume that
pair exactly like the path that quantizes inside the call.
"""

import sys
from contextlib import nullcontext

import pytest
import torch

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.kernels.ops.quantization.per_token_quant_fp8 import (
    fused_rmsnorm_per_token_quant_int8,
)
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8LinearMethod
from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

EPS = 1e-6
INT8_MAX = 127
ABSMAX_FLOOR = 1e-10
# bf16 keeps 8 significand bits, so one ulp is at most 2**-7 of the value (a
# value just above a power of two has the coarser spacing).
BF16_REL_ULP = 2.0**-7


def _cpu_reference(x, weight, eps=EPS):
    """Ground truth: float64 math + the int8 contract, all on the CPU."""
    xc = x.detach().to(device="cpu", dtype=torch.float64)
    wc = weight.detach().to(device="cpu", dtype=torch.float64)
    var = (xc * xc).mean(dim=-1, keepdim=True)
    y_ref = (xc * torch.rsqrt(var + eps) * wc).to(torch.bfloat16)

    y64 = y_ref.double()
    absmax = y64.abs().amax(dim=-1, keepdim=True).clamp_min(ABSMAX_FLOOR)
    scale_ref = (absmax / INT8_MAX).to(torch.float32)
    q_ref = torch.round(y64 * (INT8_MAX / absmax)).clamp(-128, INT8_MAX)
    return y_ref, q_ref.to(torch.int8), scale_ref


def _make_input(num_tokens, hidden, *, dtype=torch.bfloat16, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = (
        torch.randn(
            (num_tokens, hidden),
            generator=generator,
            dtype=torch.float32,
            device="cuda",
        )
        * 3.0
    )
    weight = (
        torch.randn((hidden,), generator=generator, dtype=torch.float32, device="cuda")
        * 0.1
        + 1.0
    )
    return x.to(dtype), weight.to(dtype)


def _assert_within_ulp(actual, expected, *, max_ulps=1.0, max_differing=1e-4):
    """Bound the element error against a bf16 ground truth, and its footprint."""
    exp = expected.float()
    diff = (actual.float() - exp).abs()
    row_scale = exp.abs().amax(dim=-1, keepdim=True)
    limit = row_scale * BF16_REL_ULP * max_ulps
    assert torch.all(diff <= limit), (
        f"element outside {max_ulps} bf16 ulp of the CPU reference: "
        f"max diff {diff.max().item():.3e}"
    )
    differing = (
        (actual.view(torch.int16) != expected.view(torch.int16)).float().mean().item()
    )
    assert differing <= max_differing, (
        f"{differing:.3e} of the elements differ from the correctly rounded "
        f"value (> {max_differing:.1e}); the fp32 variance only perturbs the "
        "last bit, so only rounding-boundary elements may flip"
    )


def _assert_scale_matches_row(y, scale):
    expected = (
        y.float().abs().amax(dim=-1, keepdim=True).clamp_min(ABSMAX_FLOOR) / INT8_MAX
    )
    torch.testing.assert_close(scale, expected, rtol=1e-6, atol=0)


def _assert_payload_close(actual, expected, *, max_fraction=2e-2):
    """Every element within one int8 step of the CPU quantization.

    The kernel scales by an fp32 reciprocal where the reference divides in
    float64, so boundary elements may round the other way; a wrong scale moves
    whole rows by far more than one step.
    """
    beyond = (actual.float() - expected.float()).abs() > 1.0
    assert not beyond.any(), (
        f"{int(beyond.sum())} elements are more than one int8 step from the CPU "
        "quantization"
    )
    mismatched = (actual != expected).float().mean().item()
    assert mismatched <= max_fraction, (
        f"{mismatched:.3e} of the payload bits differ from the CPU quantization "
        f"(> {max_fraction:.1e})"
    )


@pytest.mark.parametrize("num_tokens", [1, 2, 8, 9, 257, 4097], ids=lambda n: f"m{n}")
@pytest.mark.parametrize("hidden", [1280, 512])
def test_matches_cpu_reference(num_tokens, hidden):
    """Row, scale and payload all follow the int8 contract on the CPU."""
    x, weight = _make_input(num_tokens, hidden)
    y_ref, q_ref, scale_ref = _cpu_reference(x, weight)
    y, q, scale = fused_rmsnorm_per_token_quant_int8(x, weight, EPS)

    assert y.shape == x.shape and q.shape == x.shape
    assert q.dtype == torch.int8
    assert scale.shape == (num_tokens, 1) and scale.dtype == torch.float32

    _assert_within_ulp(y.cpu(), y_ref)
    _assert_scale_matches_row(y, scale)
    torch.testing.assert_close(scale.cpu(), scale_ref, rtol=1e-6, atol=0)
    _assert_payload_close(q.cpu(), q_ref)


@pytest.mark.parametrize("hidden", [1280, 512])
def test_strided_rows_read_the_row_stride(hidden):
    """q_lora is the leading slice of the fused wqkv_a row, stride(0) > hidden."""
    num_tokens = 4097
    fused_width = hidden + 512
    x, _ = _make_input(num_tokens, fused_width)
    _, weight = _make_input(1, hidden)
    # Poison the trailing columns so a kernel that ignores stride(0) reads them
    # as part of the row instead of silently matching.
    x[:, hidden:] = 1024.0

    y_strided, q_strided, scale_strided = fused_rmsnorm_per_token_quant_int8(
        x[:, :hidden], weight, EPS
    )
    y_dense, q_dense, scale_dense = fused_rmsnorm_per_token_quant_int8(
        x[:, :hidden].contiguous(), weight, EPS
    )

    assert torch.equal(y_strided, y_dense), "strided rows changed the normalized output"
    assert torch.equal(q_strided, q_dense)
    assert torch.equal(scale_strided, scale_dense)


@pytest.mark.parametrize("num_tokens", [1, 8192])
def test_all_zero_rows_keep_the_absmax_floor(num_tokens):
    """A zero row keeps the 1e-10 absmax floor instead of dividing by zero."""
    hidden = 1280
    x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((hidden,), dtype=torch.bfloat16, device="cuda")

    y, q, scale = fused_rmsnorm_per_token_quant_int8(x, weight, EPS)

    assert torch.isfinite(scale).all()
    assert torch.equal(y, torch.zeros_like(y))
    torch.testing.assert_close(
        scale, torch.full_like(scale, ABSMAX_FLOOR / INT8_MAX), rtol=1e-6, atol=0
    )
    assert torch.equal(q, torch.zeros_like(q))


def test_uses_full_weight_row():
    """The kernel multiplies by the full weight row, not by weight[0]."""
    hidden = 1280
    x, _ = _make_input(64, hidden)
    weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.bfloat16, device="cuda")
    y_ref, q_ref, _ = _cpu_reference(x, weight)
    y, q, _ = fused_rmsnorm_per_token_quant_int8(x, weight, EPS)

    _assert_within_ulp(y.cpu(), y_ref)
    _assert_payload_close(q.cpu(), q_ref)


def test_rejects_hidden_dim_not_multiple_of_16():
    x = torch.randn((4, 500), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((500,), dtype=torch.bfloat16, device="cuda")
    # tvm-ffi surfaces a host CHECK through the generated module, so match the
    # message rather than a specific exception class.
    with pytest.raises(Exception, match="multiple of 16"):
        fused_rmsnorm_per_token_quant_int8(x, weight, EPS)


def _int8_gemm_available() -> bool:
    """Whether any int8 GEMM branch of the linear can run on this build."""
    if is_sail():
        # Exercised through SGLANG_SAIL_DEEPGEMM_DENSE below.
        return True
    return hasattr(torch.ops.sgl_kernel, "int8_scaled_mm")


class _StandInLinear(torch.nn.Module):
    """Holds the two tensors W8A8Int8LinearMethod reads, nothing else."""

    def __init__(self, weight, weight_scale):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)


@pytest.mark.skipif(not _int8_gemm_available(), reason="no int8 GEMM path here")
@pytest.mark.parametrize("num_tokens", [1, 7, 4096])
def test_apply_consumes_the_prequantized_pair(num_tokens):
    """Handing the GEMM (int8, scale) must match letting it quantize the bf16 row."""
    hidden, out_features = 1280, 256
    generator = torch.Generator(device="cuda").manual_seed(7)
    # Mirror layer.weight: a [K, N] view of a contiguous [N, K] buffer.
    weight = torch.randint(
        -127,
        128,
        (out_features, hidden),
        generator=generator,
        device="cuda",
        dtype=torch.int8,
    ).t()
    weight_scale = (
        torch.rand(out_features, 1, generator=generator, device="cuda") * 0.02 + 0.005
    )
    layer = _StandInLinear(weight, weight_scale)
    method = W8A8Int8LinearMethod.__new__(W8A8Int8LinearMethod)
    x, _ = _make_input(num_tokens, hidden)

    # The default int8 GEMM op is not in every wheel; on SAIL the deepgemm
    # branch is the one that consumes a per-token pair, so pin it there.
    branch = (
        envs.SGLANG_SAIL_DEEPGEMM_DENSE.override(True) if is_sail() else nullcontext()
    )
    with branch:
        expected = method.apply(layer, x)
        quant, scale = per_token_quant_int8(x)
        actual = method.apply(layer, (quant, scale, torch.bfloat16))

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
        "pre-quantized operand changed the GEMM result"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
