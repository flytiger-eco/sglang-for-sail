# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache
from typing import Callable, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUW8A8Int8DynamicLinearMethod,
)
from sglang.srt.layers.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.utils import requantize_with_max_scale
from sglang.srt.utils import is_cuda, is_ppu

__all__ = ["CompressedTensorsW8A8Int8", "NPUCompressedTensorsW8A8Int8"]

_is_cuda = is_cuda()
_is_ppu = is_ppu()
if _is_cuda:
    from sgl_kernel import int8_scaled_mm


@lru_cache(maxsize=1)
def _get_deepgemm_int8_gemm():
    """Resolve the PPU DeepGEMM int8 NT GEMM lazily.

    PPU wheels never register ``sgl_kernel.int8_scaled_mm`` (it is a CUTLASS
    kernel), so dense int8 checkpoints need the PPU DeepGEMM path. The import is
    kept out of module scope so non-INT8 models never load ``deep_gemm``.
    """
    from sglang.srt.layers.quantization.w8a8_int8 import gemm_nt_i8i8bf16

    return gemm_nt_i8i8bf16


def _ppu_deepgemm_int8_enabled() -> bool:
    """Dense int8 on PPU is routed to DeepGEMM behind an env switch, default off.

    The CUDA path stays untouched: when the switch is off this scheme behaves
    exactly as before and only ``int8_scaled_mm`` is used.
    """
    return _is_ppu and envs.SGLANG_SAIL_DEEPGEMM_DENSE.get()


class CompressedTensorsW8A8Int8(CompressedTensorsLinearScheme):
    def __init__(
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool
    ):
        self.strategy = strategy
        self.is_static_input_scheme = is_static_input_scheme
        self.input_symmetric = input_symmetric
        # DeepGEMM NT consumes row-major [N, K] weights, while the CUDA cutlass
        # kernel wants the transposed layout; the choice is latched at load time.
        self.use_deepgemm_int8 = _ppu_deepgemm_int8_enabled()
        self.deepgemm_int8_ready = False

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=torch.int8
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )

        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1), dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader,
            )
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader,
            )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(1, dtype=torch.float32), weight_loader=weight_loader
            )
            layer.register_parameter("input_scale", input_scale)

            if not self.input_symmetric:
                # Note: compressed-tensors stores the zp using the same dtype
                # as the weights
                # AZP loaded as int8 but used as int32
                input_zero_point = PerTensorScaleParameter(
                    data=torch.empty(1, dtype=torch.int8), weight_loader=weight_loader
                )
                layer.register_parameter("input_zero_point", input_zero_point)

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def process_weights_after_loading(self, layer) -> None:
        if self.use_deepgemm_int8 and not self.input_symmetric:
            raise NotImplementedError(
                "The PPU DeepGEMM int8 dense path does not support asymmetric "
                "activation quantization (azp); unset SGLANG_SAIL_DEEPGEMM_DENSE to "
                "use the default int8 path for this checkpoint."
            )

        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per channel
        if self.strategy == QuantizationStrategy.TENSOR:
            max_w_scale, weight = requantize_with_max_scale(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                logical_widths=layer.logical_widths,
            )

            if self.use_deepgemm_int8:
                # per-tensor scale arrives as a 0-dim scalar; DeepGEMM wants one
                # scale per output channel.
                max_w_scale = (
                    max_w_scale.reshape(1, 1).expand(weight.shape[0], 1).contiguous()
                )
            # Keep [N, K]: DeepGEMM NT consumes the row-major checkpoint layout.
            # ``.t().t()`` is the identity view but returns a plain Tensor, which
            # Parameter() requires (ModelWeightParameter cannot be re-wrapped).
            layer.weight = Parameter(
                weight.t().t() if self.use_deepgemm_int8 else weight.t(),
                requires_grad=False,
            )
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # If channelwise, scales are already lined up, so just transpose.
        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight = layer.weight
            weight_scale = layer.weight_scale.data

            layer.weight = Parameter(
                weight.t().t() if self.use_deepgemm_int8 else weight.t(),
                requires_grad=False,
            )
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(f"Unknown quantization strategy {self.strategy}")

        if self.use_deepgemm_int8:
            # Fail at load time instead of at the first prefill: DeepGEMM NT needs
            # a 2-D int8 [N, K] weight and one float scale per output channel.
            weight = layer.weight
            weight_scale = layer.weight_scale
            if weight.dim() != 2 or weight.dtype != torch.int8:
                raise RuntimeError(
                    "DeepGEMM int8 dense path expects a 2-D int8 [N, K] weight, got "
                    f"shape {tuple(weight.shape)} dtype {weight.dtype}."
                )
            if weight_scale.numel() != weight.shape[0]:
                raise RuntimeError(
                    "DeepGEMM int8 dense path expects one weight scale per output "
                    f"channel: got {weight_scale.numel()} scales for "
                    f"{weight.shape[0]} channels."
                )
            self.deepgemm_int8_ready = True

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, "input_scale"):
            if self.input_symmetric:
                layer.input_scale = Parameter(
                    layer.input_scale.max(), requires_grad=False
                )
            else:
                input_scale = layer.input_scale
                input_zero_point = layer.input_zero_point

                # reconstruct the ranges
                int8_traits = torch.iinfo(torch.int8)
                azps = input_zero_point.to(dtype=torch.int32)
                range_max = (input_scale * (int8_traits.max - azps)).max()
                range_min = (input_scale * (int8_traits.min - azps)).min()

                scale = (range_max - range_min) / (int8_traits.max - int8_traits.min)

                # AZP loaded as int8 but used as int32
                azp = (int8_traits.min - range_min / scale).to(dtype=torch.int32)

                layer.input_scale = Parameter(scale, requires_grad=False)
                layer.input_zero_point = Parameter(azp, requires_grad=False)
        else:
            layer.input_scale = None
            layer.input_zero_point = None

        # azp_adj is the AZP adjustment term, used to account for weights.
        # It does not depend on scales or azp, so it is the same for
        # static and dynamic quantization.
        # For more details, see csrc/quantization/cutlass_w8a8/Epilogues.md
        # https://github.com/vllm-project/vllm/blob/8d59dbb00044a588cab96bcdc028006ed922eb06/csrc/quantization/cutlass_w8a8/Epilogues.md
        if not self.input_symmetric:
            weight = layer.weight
            azp_adj = weight.sum(dim=0, keepdim=True, dtype=torch.int32)
            if self.is_static_input_scheme:
                # cutlass_w8a8 requires azp to be folded into azp_adj
                # in the per-tensor case
                azp_adj = layer.input_zero_point * azp_adj
            layer.azp_adj = Parameter(azp_adj, requires_grad=False)
        else:
            layer.azp_adj = None

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # TODO: add cutlass_scaled_mm_azp support
        x_q, x_scale = per_token_quant_int8(x)

        if self.use_deepgemm_int8:
            if bias is not None:
                raise NotImplementedError(
                    "The PPU DeepGEMM int8 dense path has no bias support; unset "
                    "SGLANG_SAIL_DEEPGEMM_DENSE to use the default int8 path."
                )
            if not self.deepgemm_int8_ready:
                raise RuntimeError(
                    "DeepGEMM int8 dispatch was not validated at weight-load time "
                    "(process_weights_after_loading); refusing to run."
                )
            try:
                gemm = _get_deepgemm_int8_gemm()
            except ImportError as exc:
                raise RuntimeError(
                    "SGLANG_SAIL_DEEPGEMM_DENSE is enabled but the PPU DeepGEMM "
                    "wrapper is unavailable in this image; install a PPU image that "
                    "ships deep_gemm, or unset the switch."
                ) from exc
            x_q_2d = x_q.view(-1, x_q.shape[-1])
            x_scale_2d = x_scale.view(-1, 1)
            out = torch.empty(
                (x_q_2d.shape[0], layer.weight.shape[0]),
                dtype=torch.bfloat16,
                device=x_q.device,
            )
            gemm(x_q_2d, x_scale_2d, layer.weight, layer.weight_scale, out)
            return out.to(x.dtype).view(*x_q.shape[:-1], layer.weight.shape[0])

        return int8_scaled_mm(
            x_q, layer.weight, x_scale, layer.weight_scale, out_dtype=x.dtype, bias=bias
        )


class NPUCompressedTensorsW8A8Int8(CompressedTensorsW8A8Int8):
    def __init__(
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool
    ):
        super().__init__(strategy, is_static_input_scheme, input_symmetric)
        # TODO: Currently, NPU kernel for static quant requires quant_bias field,
        # which can't be replicated in compressed-tensors.
        if self.is_static_input_scheme:
            raise NotImplementedError(
                "Static compressed-tensors scheme is not yet supported on NPU."
            )
        self.kernel = NPUW8A8Int8DynamicLinearMethod()

    @classmethod
    def get_min_capability(cls) -> int:
        return NotImplementedError

    def process_weights_after_loading(self, layer):
        return self.kernel.process_weights_after_loading(layer)

    def apply_weights(self, layer, x, bias):
        return self.kernel.apply(layer, x, bias)
