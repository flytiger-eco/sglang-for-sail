"""Sail GDN VERIFY projection and convolution helpers."""

import torch
import triton
import triton.language as tl


def prepare_verify_conv(layer, qkvz, ba, conv_states, draft, retrieve_next_token):
    qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
    rows = qkvz.shape[0]
    weight = layer.conv_weights
    if (
        draft <= 0
        or rows == 0
        or rows % draft
        or retrieve_next_token is not None
        or layer.num_v_heads != 48
        or layer.head_v_dim != 128
        or qkv_dim != 10240
        or qkvz.shape != (rows, qkv_dim + layer.v_dim)
        or ba.shape != (rows, 2 * layer.num_v_heads)
        or qkvz.dtype != torch.bfloat16
        or not qkvz.is_cuda
        or weight.dtype != qkvz.dtype
        or ba.dtype != qkvz.dtype
        or conv_states.dtype != qkvz.dtype
        or weight.shape != (qkv_dim, 4)
        or qkvz.stride(-1) != 1
        or ba.stride(-1) != 1
        or torch.compiler.is_compiling()
    ):
        return None

    if weight.stride(0) == 1:
        packed = weight
    else:
        version = None if weight.is_inference() else weight._version
        cached = getattr(layer, "_verify_packed_conv_weight", None)
        if cached is not None and cached[0] is weight and cached[1] == version:
            packed = cached[2]
        elif torch.cuda.is_current_stream_capturing():
            return None
        else:
            packed = weight.t().contiguous().t()
            layer._verify_packed_conv_weight = (weight, version, packed)

    z = qkvz[:, qkv_dim:].view(rows, layer.num_v_heads, layer.head_v_dim)
    b = ba.new_empty((rows, layer.num_v_heads))
    a = torch.empty_like(b)
    return qkvz[:, :qkv_dim], z, b, a, packed


@triton.jit
def _exact_fp8_quant(X, Q, S, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    val = tl.load(X + row * H + col, col < H, 0).to(tl.float32)
    scale = tl.div_rn(tl.max(tl.abs(val)), 448.0)
    inv = tl.div_rn(1.0, tl.where(scale == 0.0, 1.0, scale))
    q = tl.clamp(val * tl.where(scale == 0.0, 0.0, inv), -448.0, 448.0)
    tl.store(Q + row * H + col, q.to(Q.dtype.element_ty), col < H)
    tl.store(S + row, scale)


def _per_channel_fp8_projection(linear) -> bool:
    """Whether ``linear`` is a per-channel FP8 projection the PPU DeepGEMM
    channelwise path can consume.

    Two quantizers reach this model: Quark's ``QuarkW8A8Fp8`` per-token scheme
    and the ``W8A8Fp8LinearMethod`` that serves the dense FP8 layers of a
    mixed-precision checkpoint. Only the first one publishes a ``scheme``, so
    the layout and scale contract is checked directly instead of the class name.
    """
    scheme = getattr(linear, "scheme", None)
    if scheme is not None and not (
        getattr(scheme, "per_token", False)
        and getattr(scheme, "weight_qscheme", None) == "per_channel"
    ):
        return False

    weight = getattr(linear, "weight", None)
    weight_scale = getattr(linear, "weight_scale", None)
    return (
        weight is not None
        and weight_scale is not None
        and weight.dtype == torch.float8_e4m3fn
        # Both quantizers publish the transposed (in, out) layout with the
        # output dim contiguous, which is what the GEMM consumes.
        and weight.shape == (2560, 16384)
        and weight.stride(0) == 1
        and weight_scale.numel() == weight.shape[1]
        and weight_scale.is_contiguous()
        and getattr(linear, "input_scale", None) is None
        and getattr(linear, "bias", None) is None
        and not getattr(linear, "gather_output", False)
    )


def verify_qkvz_projection(linear, x):
    from sglang.srt.layers import deep_gemm_wrapper
    from sglang.srt.layers.quantization.fp8_utils import gemm_nt_f8f8bf16
    from sglang.srt.utils import is_ppu

    if (
        not is_ppu()
        or not deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        or not _per_channel_fp8_projection(linear)
        or not isinstance(x, torch.Tensor)
        or x.ndim != 2
        or x.shape[1] != 2560
        or not 0 < x.shape[0] <= 192
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or torch.compiler.is_compiling()
    ):
        return None

    rows, hidden = x.shape
    weight = linear.weight
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    _exact_fp8_quant[(rows,)](
        x, q, scale, hidden, triton.next_power_of_2(hidden), num_warps=8
    )
    out = x.new_empty((rows, weight.shape[1]))
    # Same PPU call as ``apply_fp8_linear``'s channelwise branch: the 5-argument
    # custom op wrapping ``deep_gemm_wrapper.gemm_nt_f8f8bf16``.
    gemm_nt_f8f8bf16(
        q, scale, weight.t(), linear.weight_scale.reshape(-1, 1).contiguous(), out
    )
    return out
