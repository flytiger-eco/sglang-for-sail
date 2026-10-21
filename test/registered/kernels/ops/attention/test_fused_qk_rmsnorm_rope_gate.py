"""Correctness tests for fused Q/K GemmaRMSNorm + NeoX RoPE + gate copy."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available() or torch.version.cuda is None:
    pytest.skip("Requires a CUDA-compatible GPU.", allow_module_level=True)

from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (  # noqa: E402
    _fused_qk_rmsnorm_rope_gate_kernel,
    fused_qk_gemma_rmsnorm_rope_gate,
)

DEVICE = "cuda"
EPS = 1e-6
MAX_POSITION = 512


def _gemma_rmsnorm(x, weight, eps):
    x_fp32 = x.float()
    variance = x_fp32.square().mean(dim=-1, keepdim=True)
    # RoPE consumes the rounded RMSNorm result, not the unrounded FP32 value.
    return (x_fp32 * torch.rsqrt(variance + eps) * (weight.float() + 1)).to(x.dtype)


def _neox_rope(x, cache, positions, rotary_dim):
    cos, sin = cache.index_select(0, positions.long()).float().chunk(2, dim=-1)
    cos, sin = cos[:, None, :], sin[:, None, :]
    half = rotary_dim // 2
    first = x[..., :half].float()
    second = x[..., half:rotary_dim].float()
    return torch.cat(
        (
            first * cos - second * sin,
            second * cos + first * sin,
            x[..., rotary_dim:].float(),
        ),
        dim=-1,
    ).to(x.dtype)


def _reference(inputs, num_q_heads, num_kv_heads, head_dim, rotary_dim, has_gate, eps):
    q_gate, k, q_weight, k_weight, cache, positions = inputs
    tokens = q_gate.shape[0]
    q_gate = q_gate.reshape(tokens, num_q_heads, (1 + int(has_gate)) * head_dim)
    q = q_gate[..., :head_dim]
    k = k.reshape(tokens, num_kv_heads, head_dim)
    q = _neox_rope(_gemma_rmsnorm(q, q_weight, eps), cache, positions, rotary_dim)
    k = _neox_rope(_gemma_rmsnorm(k, k_weight, eps), cache, positions, rotary_dim)
    gate = q_gate[..., head_dim:].clone() if has_gate else None
    return q.reshape(tokens, -1), k.reshape(tokens, -1), gate


def _make_inputs(
    num_tokens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    dtype,
    has_gate,
    contiguous=False,
    positions_dtype=torch.int64,
    cache_dtype=torch.float32,
    padded_cache=False,
):
    torch.manual_seed(0)
    q_size = num_q_heads * (1 + int(has_gate)) * head_dim
    k_size = num_kv_heads * head_dim
    qkv = torch.randn(num_tokens, q_size + 2 * k_size, dtype=dtype, device=DEVICE)
    q_gate, k, _ = qkv.split([q_size, k_size, k_size], dim=-1)
    if contiguous:
        q_gate, k = q_gate.contiguous(), k.contiguous()
    elif num_tokens > 1:
        assert not q_gate.is_contiguous()
        assert not k.is_contiguous()
    q_weight = torch.randn(head_dim, dtype=dtype, device=DEVICE)
    k_weight = torch.randn(head_dim, dtype=dtype, device=DEVICE)

    inv_freq = 10000.0 ** (
        -torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEVICE) / rotary_dim
    )
    phase = torch.outer(
        torch.arange(MAX_POSITION, dtype=torch.float32, device=DEVICE), inv_freq
    )
    cache = torch.cat((phase.cos(), phase.sin()), dim=-1).to(cache_dtype)
    if padded_cache:
        storage = torch.full(
            (MAX_POSITION, rotary_dim + 13),
            float("nan"),
            dtype=cache_dtype,
            device=DEVICE,
        )
        storage[:, :rotary_dim].copy_(cache)
        cache = storage[:, :rotary_dim]
        assert cache.stride(0) > rotary_dim and cache.stride(1) == 1

    # Nonmonotonic, repeated positions include the first and last cache rows.
    position_values = [MAX_POSITION - 1, 0, 137, 3, MAX_POSITION - 1, 0, 137]
    positions = torch.tensor(
        (position_values * ((num_tokens + 6) // 7))[:num_tokens],
        dtype=positions_dtype,
        device=DEVICE,
    )
    return (q_gate, k, q_weight, k_weight, cache, positions), qkv


def _assert_bitwise_equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


def _check_outputs(
    inputs,
    qkv,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    has_gate,
    eps=EPS,
):
    before = [tensor.clone() for tensor in (*inputs, qkv)]
    expected = _reference(
        inputs, num_q_heads, num_kv_heads, head_dim, rotary_dim, has_gate, eps
    )
    actual = fused_qk_gemma_rmsnorm_rope_gate(
        *inputs,
        eps,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        has_gate=has_gate,
    )
    for got, ref in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
        assert got.is_contiguous()
    if has_gate:
        assert actual[2].shape == (inputs[0].shape[0], num_q_heads, head_dim)
        assert actual[2].is_contiguous()
        _assert_bitwise_equal(actual[2], expected[2])
    else:
        assert actual[2] is None
    for tensor, snapshot in zip((*inputs, qkv), before):
        _assert_bitwise_equal(tensor, snapshot)
    return actual


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_gate", [True, False])
@pytest.mark.parametrize(
    "num_tokens,num_q_heads,num_kv_heads,head_dim,rotary_dim,contiguous",
    [
        (1, 64, 4, 256, 64, False),  # Qwen3.8-Max TP1
        (7, 32, 2, 256, 64, False),  # TP2
        (128, 16, 1, 256, 64, False),  # TP4
        (7, 8, 1, 256, 64, False),  # TP8
        (17, 8, 2, 128, 128, False),  # Full RoPE
        (33, 6, 2, 96, 60, False),  # Both head and half-RoPE masks have tails
        (7, 4, 1, 96, 96, True),  # Full RoPE with non-power-of-two dimensions
        (7, 8, 1, 256, 64, True),  # Contiguous Q/K
    ],
)
@torch.inference_mode()
def test_fused_qk_rmsnorm_rope_gate_matches_reference(
    dtype,
    has_gate,
    num_tokens,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    contiguous,
):
    inputs, qkv = _make_inputs(
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        dtype,
        has_gate,
        contiguous=contiguous,
    )
    _check_outputs(
        inputs, qkv, num_q_heads, num_kv_heads, head_dim, rotary_dim, has_gate
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("positions_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("cache_in_input_dtype", [False, True])
@torch.inference_mode()
def test_positions_and_padded_cache(dtype, positions_dtype, cache_in_input_dtype):
    inputs, qkv = _make_inputs(
        7,
        3,
        2,
        96,
        60,
        dtype,
        True,
        positions_dtype=positions_dtype,
        cache_dtype=dtype if cache_in_input_dtype else torch.float32,
        padded_cache=True,
    )
    # Norm weights can also be stored in FP32 independently of the input dtype.
    inputs = (*inputs[:2], inputs[2].float(), inputs[3].float(), *inputs[4:])
    _check_outputs(inputs, qkv, 3, 2, 96, 60, True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("eps", [1e-6, 1e-2])
@torch.inference_mode()
def test_zero_and_small_inputs(dtype, eps):
    inputs, qkv = _make_inputs(7, 3, 2, 96, 60, dtype, True)
    q_gate, k, q_weight, k_weight, _, _ = inputs
    q = q_gate.reshape(7, 3, 192)[..., :96]
    q.mul_(1e-4)
    k.mul_(1e-4)
    q[0].zero_()
    k[0].zero_()
    q_weight.zero_()
    k_weight.zero_()
    q_weight[::3] = -1
    k_weight[::3] = -1
    actual = _check_outputs(inputs, qkv, 3, 2, 96, 60, True, eps)
    assert torch.count_nonzero(actual[0][0]).item() == 0
    assert torch.count_nonzero(actual[1][0]).item() == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_gate_preserves_special_value_bits(dtype):
    inputs, qkv = _make_inputs(7, 4, 2, 128, 64, dtype, True)
    bits = (
        [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC1, 0x0001, 0x007F]
        if dtype == torch.bfloat16
        else [0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E01, 0x0001, 0x03FF]
    )
    # Include signed zero, infinities, a NaN payload, and subnormal values.
    pattern = torch.tensor(bits, dtype=torch.uint16).view(dtype).to(DEVICE)
    gate = inputs[0].reshape(7, 4, 256)[..., 128:]
    gate.copy_(
        pattern.repeat((gate.numel() + len(bits) - 1) // len(bits))[
            : gate.numel()
        ].reshape_as(gate)
    )
    _check_outputs(inputs, qkv, 4, 2, 128, 64, True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_rmsnorm_rounds_before_rope(dtype):
    inputs, qkv = _make_inputs(7, 2, 1, 64, 32, dtype, False)
    q, k, q_weight, k_weight, cache, positions = inputs
    q.fill_(1)
    k.fill_(1)
    weight = 2**-9 if dtype == torch.bfloat16 else 2**-12
    angle = 0.003 if dtype == torch.bfloat16 else 0.0004
    q_weight.fill_(weight)
    k_weight.fill_(weight)
    cache[:, :16].fill_(torch.cos(torch.tensor(angle)).item())
    cache[:, 16:].fill_(torch.sin(torch.tensor(angle)).item())
    actual = _check_outputs(inputs, qkv, 2, 1, 64, 32, False)
    expected = _reference(inputs, 2, 1, 64, 32, False, EPS)
    for got, ref in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)

    # Prove this fixture distinguishes a norm that omits the intermediate cast.
    raw_norm = torch.ones(7, 2, 64, device=DEVICE) * (
        torch.rsqrt(torch.tensor(1 + EPS, device=DEVICE)) * (1 + weight)
    )
    unrounded = _neox_rope(raw_norm, cache, positions, 32).to(dtype).reshape(7, -1)
    assert not torch.equal(unrounded, expected[0])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_k_programs_do_not_write_gate(dtype):
    # Launch only K programs, so a Q store cannot hide an erroneous K gate store.
    inputs, _ = _make_inputs(7, 2, 2, 96, 60, dtype, True)
    q_gate, k, q_weight, k_weight, cache, positions = inputs
    q_out = torch.empty(7, 192, dtype=dtype, device=DEVICE)
    k_out = torch.empty_like(q_out)
    gate = torch.full((7, 2, 96), 42, dtype=dtype, device=DEVICE)
    before = gate.clone()
    _fused_qk_rmsnorm_rope_gate_kernel[(7, 2)](
        q_gate,
        k,
        q_out,
        k_out,
        gate,
        q_weight,
        k_weight,
        cache,
        positions,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate.stride(0),
        cache.stride(0),
        NUM_Q_HEADS=0,
        NUM_KV_HEADS=2,
        HEAD_DIM=96,
        ROTARY_DIM=60,
        HALF_ROTARY=30,
        HEAD_BLOCK=128,
        ROT_HALF_BLOCK=32,
        EPS=EPS,
        FP16=dtype == torch.float16,
        HAS_PASS=True,
        HAS_GATE=True,
        ENABLE_PDL=False,
    )
    _assert_bitwise_equal(gate, before)
    _, expected_k, _ = _reference(inputs, 2, 2, 96, 60, True, EPS)
    torch.testing.assert_close(k_out, expected_k, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
