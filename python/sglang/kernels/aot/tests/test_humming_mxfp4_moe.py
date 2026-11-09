# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from sgl_kernel.humming_mxfp4 import humming_mxfp4_moe_mm, prepare_humming_mxfp4_moe

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="Qwen38 validation requires Hopper",
)


@pytest.mark.parametrize("n,k", [(1280, 2560), (2560, 640)])
@pytest.mark.parametrize("masked", [False, True])
def test_native_mxfp4_grouped_checkpoint(n, k, masked):
    from humming.layer import HummingLayer
    from humming.schema import HummingWeightSchema

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(125)
    e, capacity = 4, 132
    source = HummingLayer(
        shape_n=n,
        shape_k=k,
        num_experts=e,
        weight_config={
            "dtype": "float4e2m1",
            "scale_dtype": "float8e8m0",
            "group_size": 32,
        },
        input_config={"dtype": "float8e4m3", "group_size": 0},
        torch_dtype=torch.bfloat16,
    ).cuda()
    tensors = HummingWeightSchema.quant_tensor(
        (torch.randn((e, n, k), device="cuda") * 0.02).bfloat16(),
        source.weight_schema,
        torch.bfloat16,
        allow_negative_scale=False,
    )
    weight = source.weight_schema.dequant_tensors(tensors).float()
    layer = prepare_humming_mxfp4_moe(
        tensors["weight"], tensors["weight_scale"], shape_n=n, shape_k=k
    )
    counts = [0, 1, 33, 129]
    layout = torch.tensor(
        counts if masked else [0, 0, 1, 34, 163],
        device="cuda",
        dtype=torch.int32 if masked else torch.int64,
    )
    shape = (e, capacity, k) if masked else (sum(counts), k)
    a = (torch.randn(shape, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    scales = torch.full((*shape[:-1], 1), 0.5, device="cuda")
    out = torch.empty((*shape[:-1], n), device="cuda", dtype=torch.bfloat16)

    def run():
        humming_mxfp4_moe_mm(
            layer, a, out, scales, layout, masked=masked, valid_shape_m=sum(counts)
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for phase in range(2):
        if phase:
            counts = [33, 0, 129, 1]
            layout.copy_(
                torch.tensor(
                    counts if masked else [0, 33, 33, 162, 163],
                    device="cuda",
                    dtype=layout.dtype,
                )
            )
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        start = 0
        for expert, count in enumerate(counts):
            if count:
                x = a[expert, :count] if masked else a[start : start + count]
                actual = out[expert, :count] if masked else out[start : start + count]
                reference = ((x.float() * 0.5) @ weight[expert].T).bfloat16().float()
                assert torch.isfinite(actual).all()
                error = (actual.float() - reference).abs().sum() / reference.abs().sum()
                assert error < 0.035
                assert (
                    torch.nn.functional.cosine_similarity(
                        actual.float().flatten(), reference.flatten(), dim=0
                    )
                    > 0.995
                )
            start += count
