"""Check live state, radix boundary snapshots, padding, and cache restores."""

from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.srt.layers.attention.linear.kernels.gdn_pla import (
    PlaGDNKernel,
    build_pla_prefill_layout,
)


def close(actual, expected):
    actual, expected = actual.float().flatten(), expected.float().flatten()
    cos = F.cosine_similarity(actual[None], expected[None]).item()
    relative = ((actual - expected).abs().max() / (expected.abs().max() + 1e-8)).item()
    assert torch.isfinite(actual).all() and cos > 0.999 and relative < 0.05, (
        cos,
        relative,
    )


@pytest.mark.parametrize(
    "lengths,chunks",
    [
        ([2048], [-1]),
        ([2092], [32]),
        ([31], [0]),
        ([654, 1438], [10, 22]),
        ([31, 2048, 77], [0, -1, 1]),
        ([0, 129], [-1, 2]),
        ([4097, 8193], [64, 128]),
        ([32749], [511]),
    ],
)
def test_layout_covers_every_valid_token(lengths, chunks):
    plan = build_pla_prefill_layout(lengths, chunks, 64)
    positions = []
    for row, (length, prefix) in enumerate(zip(plan.lengths, plan.prefix_lengths)):
        start = plan.starts[row]
        positions.extend(range(start, start + prefix))
        positions.extend(range(start + prefix, start + length))
    assert sorted(positions) == list(range(sum(lengths)))
    assert plan.snapshot_rows == tuple(i for i, x in enumerate(chunks) if x >= 0)


@pytest.mark.parametrize("lengths,chunks", [([63], [2]), ([-1], [-1]), ([64], [])])
def test_bad_metadata_is_rejected(lengths, chunks):
    with pytest.raises(ValueError):
        build_pla_prefill_layout(lengths, chunks, 64)


@pytest.fixture(scope="module")
def kernel():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires PPU1.5 and the compiled PLA prefill extension")
    return PlaGDNKernel()


@pytest.mark.parametrize(
    "lengths,chunks,padding",
    [
        ([2048], [-1], 0),
        ([2092], [32], 0),
        ([31], [0], 7),
        ([654, 1438], [10, 22], 0),
        ([31, 2048, 77], [0, -1, 1], 19),
        ([0, 129], [-1, 2], 0),
        ([4097, 8193], [64, 128], 31),
        ([32749], [511], 19),
    ],
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_packed_state_and_radix_snapshot(kernel, lengths, chunks, padding, index_dtype):
    torch.manual_seed(20261119)
    h, hk, d = 24, 8, 128
    total = sum(lengths)
    width = (2 * hk + h) * d
    x = torch.randn(total + padding, width, device="cuda", dtype=torch.bfloat16) * 0.1
    gates = torch.randn(total + padding, 2 * h, device="cuda", dtype=torch.bfloat16)
    a, b = gates[:, :h], gates[:, h:]
    alog = torch.linspace(-2, -1, h, device="cuda")
    bias = torch.zeros(h, device="cuda")
    indices = torch.tensor([5, 2, 6][: len(lengths)], device="cuda", dtype=index_dtype)
    initial = torch.randn(8, h, d, d, device="cuda") * 0.01
    reference_pool = initial.clone()
    pool = initial.clone()
    # The reference chunk-index helper does not accept empty rows. Empty rows
    # carry no tokens, so excluding only those rows preserves the golden work.
    active = [i for i, n in enumerate(lengths) if n]
    raw = tuple(accumulate((lengths[i] for i in active), initial=0))
    cu = torch.tensor(raw, device="cuda", dtype=torch.int32)
    ref_indices = indices[torch.tensor(active, device="cuda")]
    q, k, v = fused_qkv_split_gdn_prefill(x[:total], hk, hk, h, d, d, d)
    g, beta = fused_gdn_gating(alog, a[:total], b[:total], bias)
    reference, _, history = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=reference_pool,
        initial_state_indices=ref_indices,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
    )
    plan = kernel.prepare_plan(lengths, chunks, 64, x.device)
    assert kernel.can_use_packed_prefill(x, pool, hk, h, d, plan)
    output, snapshot = kernel.packed_extend(
        x,
        a,
        b,
        A_log=alog,
        dt_bias=bias,
        ssm_states=pool,
        cache_indices=indices,
        num_qk_heads=hk,
        num_v_heads=h,
        plan=plan,
    )
    torch.cuda.synchronize()
    close(output[:, :total], reference)
    close(pool[indices], reference_pool[indices])
    assert torch.count_nonzero(output[:, total:]) == 0
    unused = [i for i in range(8) if i not in [5, 2, 6][: len(lengths)]]
    assert torch.equal(pool[unused], initial[unused])
    if plan.layout.split:
        offsets = tuple(accumulate(((n + 63) // 64 for n in lengths), initial=0))
        for row in plan.layout.snapshot_rows:
            expected = history[0, offsets[row] + chunks[row]].float()
            close(snapshot[row], expected)
            # A later request resumes from the cached boundary, not the live
            # row's final state; exercise the actual FP32 pool restore layout.
            t = 64
            x2 = torch.randn(t, width, device="cuda", dtype=x.dtype) * 0.1
            q2, k2, v2 = fused_qkv_split_gdn_prefill(x2, hk, hk, h, d, d, d)
            g2, beta2 = fused_gdn_gating(
                alog,
                a[:t] if total >= t else gates.new_zeros(t, h),
                b[:t] if total >= t else gates.new_zeros(t, h),
                bias,
            )
            restored = expected[None].clone()
            restored_pla = snapshot[row : row + 1].clone()
            ix = torch.zeros(1, device="cuda", dtype=torch.int32)
            out_ref = chunk_gated_delta_rule(
                q2,
                k2,
                v2,
                g2,
                beta2,
                initial_state=restored,
                initial_state_indices=ix,
                use_qk_l2norm_in_kernel=True,
            )[0]
            out_new = chunk_gated_delta_rule(
                q2,
                k2,
                v2,
                g2,
                beta2,
                initial_state=restored_pla,
                initial_state_indices=ix,
                use_qk_l2norm_in_kernel=True,
            )[0]
            close(out_new, out_ref)
    else:
        assert snapshot is None
    assert not kernel.can_use_packed_prefill(x, pool.bfloat16(), hk, h, d, plan)
