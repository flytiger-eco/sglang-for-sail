"""Precision tests for the fused RMSNorm + per-token FP8 quant producer.

The ground truth is computed on the **CPU** in float64 from the same inputs:
the kernel under test runs in fp32 on the accelerator, so a GPU-vs-GPU
comparison could only show that two kernels agree, not which one is right.
Against that reference the fused launch has to keep three contracts:

* every element of the normalized row lands within one bf16 ulp of the
  correctly rounded value, and only a handful of rows sit on a rounding
  boundary at all (the fp32 variance differs from exact math in its last bit),
* the per-token scale equals max|y|/448 of the row this launch produced, and
  matches the CPU scale to fp32 precision,
* the FP8 payload matches the CPU payload everywhere except elements parked on
  an fp8 bin boundary.

The producer is also wired into ``apply_fp8_linear`` through a pre-quantized
``(fp8, scale, out_dtype)`` tuple, so the GEMM must consume that pair exactly
like the bf16 path that quantizes inside the call.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8
from sglang.kernels.ops.quantization.per_token_quant_fp8 import (
    fused_rmsnorm_per_token_quant_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

EPS = 1e-6
# bf16 keeps 8 significand bits, so one ulp is at most 2**-7 of the value (a
# value just above a power of two has the coarser spacing).
BF16_REL_ULP = 2.0**-7
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _fp8_gemm_supported() -> bool:
    """The per-token FP8 GEMM needs sm >= 89 on this backend."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


def _cpu_reference(x, weight, eps=EPS):
    """Ground truth: float64 math + the same fp8 rounding, all on the CPU."""
    xc = x.detach().to(device="cpu", dtype=torch.float64)
    wc = weight.detach().to(device="cpu", dtype=torch.float64)
    var = (xc * xc).mean(dim=-1, keepdim=True)
    y_ref = (xc * torch.rsqrt(var + eps) * wc).to(torch.bfloat16)

    y64 = y_ref.double()
    scale_ref = (y64.abs().amax(dim=-1, keepdim=True) / FP8_MAX).to(torch.float32)
    q_ref = (y64 / scale_ref.double().clamp_min(1e-300)).clamp(-FP8_MAX, FP8_MAX)
    return y_ref, q_ref.to(torch.float8_e4m3fn), scale_ref


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


def _assert_scale_matches_row(y, scale, hidden):
    expected = y.float().abs().amax(dim=-1, keepdim=True) / FP8_MAX
    torch.testing.assert_close(
        scale,
        expected,
        rtol=1e-6,
        atol=0,
        msg=lambda m: (
            "per-token scale is not max|y|/448 of the row this launch produced "
            f"(hidden={hidden}): {m}"
        ),
    )


def _assert_fp8_matches_cpu(actual, expected, *, max_fraction=1e-2):
    """Every element within one fp8 step of the CPU quantization.

    The kernel scales by the fp32 reciprocal while the reference divides in
    float64, so elements on a bin boundary may round the other way; a wrong
    scale or a broken store moves whole rows, not single steps.
    """
    a, b = actual.float(), expected.float()
    # E4M3 spacing is 2**-3 relative in the upper part of a binade and a fixed
    # 2**-9 below the subnormal threshold, so bound by the larger of the two.
    step = torch.maximum(b.abs() * 2.0**-3, torch.full_like(b, 2.0**-9))
    beyond = (a - b).abs() > step
    assert not beyond.any(), (
        f"{int(beyond.sum())} elements are more than one fp8 step from the CPU "
        "quantization"
    )
    mismatched = (
        (actual.view(torch.uint8) != expected.view(torch.uint8)).float().mean().item()
    )
    # A smoke alarm only: the one-step bound above is the real contract, and on
    # small batches a couple of boundary flips already move the fraction.
    assert mismatched <= max_fraction, (
        f"{mismatched:.3e} of the payload bits differ from the CPU quantization "
        f"(> {max_fraction:.1e})"
    )


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


@pytest.mark.parametrize("num_tokens", [1, 2, 8, 9, 257, 4097], ids=lambda n: f"m{n}")
@pytest.mark.parametrize("hidden", [1280, 512])
def test_matches_cpu_reference(num_tokens, hidden):
    """Precision floor: one bf16 ulp of the CPU float64 result, nothing wider."""
    x, weight = _make_input(num_tokens, hidden)
    y_ref, q_ref, scale_ref = _cpu_reference(x, weight)
    y, q, scale = fused_rmsnorm_per_token_quant_fp8(x, weight, EPS)

    assert y.shape == x.shape and q.shape == x.shape
    assert q.dtype == torch.float8_e4m3fn
    assert scale.shape == (num_tokens, 1) and scale.dtype == torch.float32

    _assert_within_ulp(y.cpu(), y_ref)
    _assert_scale_matches_row(y, scale, hidden)
    torch.testing.assert_close(scale.cpu(), scale_ref, rtol=1e-6, atol=0)
    _assert_fp8_matches_cpu(q.cpu(), q_ref)


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

    y_strided, q_strided, scale_strided = fused_rmsnorm_per_token_quant_fp8(
        x[:, :hidden], weight, EPS
    )
    y_dense, q_dense, scale_dense = fused_rmsnorm_per_token_quant_fp8(
        x[:, :hidden].contiguous(), weight, EPS
    )

    assert torch.equal(y_strided, y_dense), "strided rows changed the normalized output"
    assert torch.equal(q_strided.view(torch.uint8), q_dense.view(torch.uint8))
    assert torch.equal(scale_strided, scale_dense)


@pytest.mark.parametrize("num_tokens", [1, 8192])
def test_all_zero_rows_produce_zero_payload(num_tokens):
    """A zero row keeps a finite norm and a zero scale instead of NaN."""
    hidden = 1280
    x = torch.zeros((num_tokens, hidden), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((hidden,), dtype=torch.bfloat16, device="cuda")

    y, q, scale = fused_rmsnorm_per_token_quant_fp8(x, weight, EPS)

    assert torch.isfinite(y).all() and torch.isfinite(scale).all()
    assert torch.equal(y, torch.zeros_like(y))
    assert torch.equal(scale, torch.zeros_like(scale))
    assert torch.equal(q.float(), torch.zeros_like(q.float()))


def test_uses_full_weight_row():
    """The kernel multiplies by the full weight row, not by weight[0]."""
    hidden = 1280
    x, _ = _make_input(64, hidden)
    weight = torch.linspace(0.5, 1.5, hidden, dtype=torch.bfloat16, device="cuda")
    y_ref, q_ref, _ = _cpu_reference(x, weight)
    y, q, _ = fused_rmsnorm_per_token_quant_fp8(x, weight, EPS)

    _assert_within_ulp(y.cpu(), y_ref)
    _assert_fp8_matches_cpu(q.cpu(), q_ref)


def test_rejects_hidden_dim_not_multiple_of_16():
    x = torch.randn((4, 500), dtype=torch.bfloat16, device="cuda")
    weight = torch.ones((500,), dtype=torch.bfloat16, device="cuda")
    # tvm-ffi surfaces a host CHECK through the generated module, so match the
    # message rather than a specific exception class.
    with pytest.raises(Exception, match="multiple of 16"):
        fused_rmsnorm_per_token_quant_fp8(x, weight, EPS)


@pytest.mark.skipif(
    not _fp8_gemm_supported(), reason="per-token FP8 GEMM needs sm >= 89"
)
@pytest.mark.parametrize("num_tokens", [1, 7, 4096])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, None])
def test_apply_fp8_linear_consumes_the_prequantized_pair(num_tokens, output_dtype):
    """Handing the GEMM (fp8, scale) must match letting it quantize the bf16 row."""
    hidden, out_features = 1280, 256
    generator = torch.Generator(device="cuda").manual_seed(7)
    # Mirror layer.weight: a [K, N] view of a contiguous [N, K] buffer, so the
    # weight.t() the NT GEMM receives is contiguous.
    weight = (
        torch.randn((out_features, hidden), generator=generator, device="cuda")
        .to(torch.float8_e4m3fn)
        .t()
    )
    weight_scale = (
        torch.rand(out_features, 1, generator=generator, device="cuda") * 0.02 + 0.005
    )
    x, _ = _make_input(num_tokens, hidden)

    expected = apply_fp8_linear(x, weight, weight_scale, cutlass_fp8_supported=False)
    quant, scale = sglang_per_token_quant_fp8(x)
    actual = apply_fp8_linear(
        quant,
        weight,
        weight_scale,
        input_scale=scale,
        cutlass_fp8_supported=False,
        pre_quant_output_dtype=output_dtype,
    )

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (
        "pre-quantized operand changed the GEMM result"
    )


@pytest.mark.skipif(
    not _fp8_gemm_supported(), reason="per-token FP8 GEMM needs sm >= 89"
)
def test_apply_fp8_linear_prequantized_from_fused_producer():
    """The tuple the model builds round-trips through the GEMM unchanged."""
    num_tokens, hidden, out_features = 257, 1280, 128
    generator = torch.Generator(device="cuda").manual_seed(11)
    weight = (
        torch.randn((out_features, hidden), generator=generator, device="cuda")
        .to(torch.float8_e4m3fn)
        .t()
    )
    weight_scale = (
        torch.rand(out_features, 1, generator=generator, device="cuda") * 0.02 + 0.005
    )
    x, norm_weight = _make_input(num_tokens, hidden)

    y, quant, scale = fused_rmsnorm_per_token_quant_fp8(x, norm_weight, EPS)
    expected = apply_fp8_linear(y, weight, weight_scale, cutlass_fp8_supported=False)
    actual = apply_fp8_linear(
        quant,
        weight,
        weight_scale,
        input_scale=scale,
        cutlass_fp8_supported=False,
        pre_quant_output_dtype=torch.bfloat16,
    )

    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
