"""Contiguous row-major channel bias addition for PPU FP8 linear outputs."""
import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _channel_bias_kernel(
    output, bias, numel: tl.constexpr, width: tl.constexpr,
    bias_stride: tl.constexpr, BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(output + offsets, offsets < numel, other=0).to(tl.float32)
    bias_values = tl.load(bias + (offsets % width) * bias_stride).to(tl.float32)
    tl.store(output + offsets, values + bias_values, offsets < numel)


def _channel_bias_fake(output: torch.Tensor, bias: torch.Tensor) -> None:
    return None


@register_custom_op(mutates_args=["output"], fake_impl=_channel_bias_fake)
def add_channel_bias_(output: torch.Tensor, bias: torch.Tensor) -> None:
    if output.numel() == 0:
        return
    assert output.ndim == 2 and output.is_contiguous()
    assert bias.ndim == 1 and bias.numel() == output.shape[1]
    _channel_bias_kernel[(triton.cdiv(output.numel(), 2048),)](
        output, bias, output.numel(), output.shape[1], bias.stride(0),
        BLOCK=2048, num_warps=4,
    )
