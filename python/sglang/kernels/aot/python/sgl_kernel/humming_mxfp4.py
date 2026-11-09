# SPDX-License-Identifier: Apache-2.0
"""Optional Humming MXFP4/E4M3 grouped GEMM entry points.

Humming owns code generation and heuristic selection. This adapter consumes
native, untransformed Humming MXFP4 checkpoint tensors; it never requantizes
an INT4 checkpoint or runs a tile search. Serving uses SGLang's Humming runner.
"""

import json

import torch


@torch.no_grad()
def prepare_humming_mxfp4_moe(
    weight: torch.Tensor, weight_scale: torch.Tensor, *, shape_n: int, shape_k: int
):
    """Prepare packed E2M1 weights and group-32 E8M0 scales once at loading.

    Tensor shapes/dtypes must match Humming's native checkpoint schema. The
    returned HummingLayer owns the transformed tensors and heuristic configs.
    """
    try:
        from humming.layer import HummingLayer, HummingMethod
    except ImportError as exc:
        raise ImportError(
            "Install humming-kernels for the optional MXFP4 backend"
        ) from exc
    if not weight.is_cuda or weight_scale.device != weight.device or weight.ndim != 3:
        raise ValueError("MXFP4 weights/scales must be CUDA tensors on the same device")
    with torch.cuda.device(weight.device):
        layer = HummingLayer(
            shape_n=shape_n,
            shape_k=shape_k,
            num_experts=weight.shape[0],
            weight_config={
                "dtype": "float4e2m1",
                "scale_dtype": "float8e8m0",
                "group_size": 32,
            },
            input_config={"dtype": "float8e4m3", "group_size": 0},
            torch_dtype=torch.bfloat16,
        ).to(weight.device)
        for name, tensor in (("weight", weight), ("weight_scale", weight_scale)):
            target = getattr(layer, name)
            if tensor.shape != target.shape or tensor.dtype != target.dtype:
                raise ValueError(
                    f"{name} must have native Humming shape {tuple(target.shape)} and dtype {target.dtype}"
                )
            target.copy_(tensor)
        layer.transform()
        layer._sgl_grouped_configs = {}
        for mode in ("grouped_contiguous", "grouped_masked"):
            config = HummingMethod.get_default_tuning_configs(layer, gemm_type=mode)
            layer._sgl_grouped_configs[mode] = (
                json.dumps(
                    {
                        "gemm_type": mode,
                        "use_f16_accum": False,
                        "use_m_major_input_scale": False,
                    }
                ),
                json.dumps(config),
            )
        return layer


def humming_mxfp4_moe_mm(
    layer,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    input_scale: torch.Tensor,
    expert_layout: torch.Tensor,
    *,
    masked: bool = False,
    valid_shape_m: int = 0,
) -> None:
    """FP8 input -> BF16 output with Humming's default heuristic.

    Contiguous layout: inputs [rows,K], expert_layout int64 [E+1] offsets.
    Masked layout: inputs [E,capacity,K], expert_layout int32 [E] counts.
    Capacity must be padded to a multiple of 4, as in DeepEP masked buffers.
    Input scales use Humming's per-row shape (group_size=0). Masked outputs
    outside each expert's count are unspecified. Warm up before graph capture.
    """
    mode = "grouped_masked" if masked else "grouped_contiguous"
    compute, tuning = layer._sgl_grouped_configs[mode]
    if masked:
        # Humming uses flattened storage with a fixed per-expert capacity.
        # These views preserve DeepEP's expert-major layout without a copy.
        if inputs.ndim != 3 or outputs.ndim != 3:
            raise ValueError("masked inputs/outputs must be [E,capacity,K/N]")
        if inputs.shape[1] % 4:
            raise ValueError(
                "Humming masked capacity must be padded to a multiple of 4"
            )
        inputs = inputs.view(-1, inputs.shape[-1])
        outputs = outputs.view(-1, outputs.shape[-1])
        input_scale = input_scale.view(-1, 1)
    layer(
        inputs=inputs,
        outputs=outputs,
        input_scale=input_scale,
        expert_layout=expert_layout,
        valid_shape_m=valid_shape_m,
        compute_config=compute,
        tuning_config=tuning,
    )
