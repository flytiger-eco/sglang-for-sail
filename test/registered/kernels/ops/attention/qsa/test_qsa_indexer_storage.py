"""Fused Qwen4 indexer storage dtype and reserved-slot contracts."""

import pytest
import torch

from sglang.kernels.ops.attention.qsa_indexer import (
    qsa_index_k_compress_store,
    qsa_index_q_norm_rope_store,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def rope_inputs():
    cos_sin = torch.cat(
        (torch.ones(64, 64, device="cuda"), torch.zeros(64, 64, device="cuda")), dim=1
    )
    return cos_sin, torch.zeros(64, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("storage_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_q_prep_writes_native_storage_dtype_and_zero_head_padding(storage_dtype):
    rows, dim, heads = 4, 128, 8
    qk = torch.ones(rows, (heads + 1) * dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(rows, device="cuda", dtype=torch.long)
    cos_sin, axes = rope_inputs()
    weight = torch.zeros(dim, device="cuda", dtype=torch.bfloat16)
    locations = torch.arange(1, rows + 1, device="cuda", dtype=torch.long)
    raw = torch.zeros(8, dim, device="cuda", dtype=torch.bfloat16)
    rope_state = torch.zeros(8, 3, device="cuda", dtype=torch.long)
    output = qsa_index_q_norm_rope_store(
        qk,
        positions,
        cos_sin,
        axes,
        weight,
        locations,
        raw,
        rope_state,
        heads,
        dim,
        1e-6,
        True,
        q_heads_padded=16,
        out_dtype=storage_dtype,
    )
    assert output.dtype == storage_dtype
    torch.testing.assert_close(
        output[:, :heads].float(),
        torch.ones_like(output[:, :heads].float()),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output[:, heads:].float(),
        torch.zeros_like(output[:, heads:].float()),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        raw[locations], torch.ones(rows, dim, device="cuda", dtype=torch.bfloat16)
    )
    torch.testing.assert_close(rope_state[locations], positions[:, None].expand(-1, 3))


@pytest.mark.parametrize("storage_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_compression_skips_padding_before_reading_invalid_source_rows(storage_dtype):
    dim = 128
    raw = torch.ones(8, dim, device="cuda", dtype=torch.bfloat16)
    groups = torch.tensor(
        [[1, 2, 3, 4], [-123, -123, -123, -123]], device="cuda", dtype=torch.int32
    )
    positions = torch.zeros(8, 3, device="cuda", dtype=torch.long)
    cos_sin, axes = rope_inputs()
    weight = torch.zeros(dim, device="cuda", dtype=torch.bfloat16)
    writes = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    output = torch.full((4, dim), -7, device="cuda", dtype=torch.bfloat16).to(
        storage_dtype
    )
    qsa_index_k_compress_store(
        raw,
        groups,
        positions,
        cos_sin,
        axes,
        weight,
        writes,
        output,
        4,
        dim,
        1e-6,
        True,
    )
    torch.testing.assert_close(
        output[1].float(), torch.ones(dim, device="cuda"), rtol=0, atol=0
    )
    torch.testing.assert_close(
        output[0].float(), torch.full((dim,), -7.0, device="cuda"), rtol=0, atol=0
    )
