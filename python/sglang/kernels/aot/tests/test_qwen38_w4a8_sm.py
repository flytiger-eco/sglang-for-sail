# SPDX-License-Identifier: Apache-2.0
"""Hopper correctness tests, including checkpoint layout and graph routing updates.

Run on an idle H800: CUDA_VISIBLE_DEVICES=<uuid> pytest -q this_file.py
"""

import pytest
import torch
from sgl_kernel.w4a8_sm import (
    cutlass_w4a8_sm_moe_mm,
    cutlass_w4a8_sm_moe_mm_swiglu,
    prepare_w4a8_sm_weight,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="requires Hopper",
)


def checkpoint(e, n, k, seed=12):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(-7, 8, (e, n, k), generator=gen, device="cuda", dtype=torch.int8)
    scales = (
        torch.rand((e, n, k // 128), generator=gen, device="cuda") * 0.015 + 0.002
    ).bfloat16()
    unsigned = q.to(torch.int16) & 15
    packed = (unsigned[..., ::2] | (unsigned[..., 1::2] << 4)).to(torch.int8)
    reference = (
        q.float().reshape(e, n, k // 128, 128) * scales.float().unsqueeze(-1)
    ).reshape(e, n, k)
    return packed, scales, reference


def check_close(actual, expected):
    x, y = actual.float().flatten(), expected.float().flatten()
    assert torch.isfinite(x).all()
    relative_l1 = (x - y).abs().sum() / y.abs().sum().clamp_min(1e-8)
    cosine = torch.nn.functional.cosine_similarity(x, y, dim=0)
    assert relative_l1 < 0.035, relative_l1.item()
    assert cosine > 0.995, cosine.item()


def test_load_time_repacking_and_negative_eight():
    for gate in (False, True):
        n, k = (1280, 2560) if gate else (2560, 640)
        original, scales, _ = checkpoint(1, n, k)
        prepared = prepare_w4a8_sm_weight(original, scales, fused_gate_up=gate)
        # Decode the kernel format independently and recover every original nibble.
        u = prepared.weight.to(torch.uint8)
        sm = torch.stack((u & 15, u >> 4), -1).reshape(1, n, k)
        sm = sm.reshape(1, n // 16, 2, 8, k // 4, 4).permute(0, 1, 3, 4, 2, 5)
        sm = sm.contiguous().reshape(1, n // 16, 8, k // 4, 8)[
            ..., [0, 2, 4, 6, 1, 3, 5, 7]
        ]
        sm = sm.reshape(1, n // 16, 8, k // 4, 2, 4).permute(0, 1, 4, 2, 3, 5)
        sm = sm.contiguous().reshape(1, n, k)
        q = torch.where(sm >= 8, 8 - sm.to(torch.int16), sm.to(torch.int16))
        if gate:
            split = torch.empty_like(q)
            for block in range(n // 16):
                split[:, block * 8 : block * 8 + 8] = q[:, block * 16 : block * 16 + 8]
                split[:, n // 2 + block * 8 : n // 2 + block * 8 + 8] = q[
                    :, block * 16 + 8 : block * 16 + 16
                ]
            q = split
        repacked = ((q[..., ::2] & 15) | ((q[..., 1::2] & 15) << 4)).to(torch.int8)
        assert torch.equal(repacked, original)
        bad = original.clone()
        bad[0, 0, 0] = 8
        with pytest.raises(ValueError, match="contains -8"):
            prepare_w4a8_sm_weight(bad, scales, fused_gate_up=gate)
        assert bad[0, 0, 0].item() == 8


@pytest.mark.parametrize(
    "gate,config_id", [(True, x) for x in range(7)] + [(False, x) for x in range(5)]
)
@pytest.mark.parametrize("masked", [False, True])
def test_all_tiles_with_empty_experts_and_graph(gate, config_id, masked):
    torch.backends.cuda.matmul.allow_tf32 = False
    e = 4
    n, k = (1280, 2560) if gate else (2560, 640)
    out_n = 640 if gate else n
    weight, scales, reference_w = checkpoint(e, n, k)
    prepared = prepare_w4a8_sm_weight(weight, scales, fused_gate_up=gate)
    a_scale = torch.tensor([0.5], device="cuda")
    out_scale = torch.tensor([0.03125], device="cuda")
    capacity = 161
    counts_list = [0, 1, 33, 161]
    counts = torch.tensor(counts_list, device="cuda", dtype=torch.int32)
    offsets = torch.cat(
        (torch.zeros(1, device="cuda", dtype=torch.int32), counts.cumsum(0).int())
    )[:-1].contiguous()
    problems = torch.stack(
        (torch.full_like(counts, n), counts, torch.full_like(counts, k)), 1
    )
    stride_k = torch.full((e, 3), k, device="cuda", dtype=torch.int64)
    stride_n = torch.full((e, 3), n, device="cuda", dtype=torch.int64)
    stride_d = torch.full((e, 3), out_n, device="cuda", dtype=torch.int64)
    shape = (e, capacity, k) if masked else (sum(counts_list), k)
    a = (torch.randn(shape, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    output = torch.empty(
        (*shape[:-1], out_n),
        device="cuda",
        dtype=torch.float8_e4m3fn if gate else torch.bfloat16,
    )
    packed_scales = prepared.scales_k512 if gate and config_id == 6 else prepared.scales

    def run():
        if gate:
            cutlass_w4a8_sm_moe_mm_swiglu(
                output,
                a,
                prepared.weight,
                a_scale,
                packed_scales,
                out_scale,
                offsets,
                problems,
                stride_k,
                stride_k,
                stride_d,
                stride_n,
                config_id=config_id,
            )
        else:
            cutlass_w4a8_sm_moe_mm(
                output,
                a,
                prepared.weight,
                a_scale,
                packed_scales,
                offsets,
                problems,
                stride_k,
                stride_k,
                stride_d,
                stride_n,
                config_id=config_id,
            )

    def verify(cs):
        start = 0
        for expert, count in enumerate(cs):
            if count:
                x = a[expert, :count] if masked else a[start : start + count]
                actual = (
                    output[expert, :count] if masked else output[start : start + count]
                )
                ref = (x.float() * a_scale) @ reference_w[expert].T
                if gate:
                    ref = (
                        (
                            (torch.nn.functional.silu(ref[:, :640]) * ref[:, 640:])
                            / out_scale
                        )
                        .clamp(-448, 448)
                        .to(torch.float8_e4m3fn)
                    )
                else:
                    ref = ref.bfloat16()
                check_close(actual, ref)
            start += count

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    verify(counts_list)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    verify(counts_list)
    # Same captured graph, different device routing (including a newly active expert).
    counts_list = [33, 0, 161, 1]
    counts.copy_(torch.tensor(counts_list, device="cuda", dtype=torch.int32))
    problems[:, 1].copy_(counts)
    offsets.copy_(
        torch.cat(
            (torch.zeros(1, device="cuda", dtype=torch.int32), counts.cumsum(0).int())
        )[:-1]
    )
    a.copy_((torch.randn(shape, device="cuda") * 0.5).to(a.dtype))
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    verify(counts_list)
