import pytest
import torch

from sglang.srt.layers.attention.qsa.fa3 import fa3_sparse_attention, resolve_fa3_qsa


@pytest.mark.parametrize("storage_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("heads,kv_heads", [(12, 2), (24, 3)])
def test_premapped_physical_slots_match_logical_adapter(storage_dtype, heads, kv_heads):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    resolve_fa3_qsa()
    torch.manual_seed(42)
    rows, pool, width = 6, 192, 19
    q = torch.randn(rows, heads, 256, device="cuda", dtype=torch.bfloat16)
    k = (
        torch.randn(pool, kv_heads, 256, device="cuda", dtype=torch.bfloat16) * 0.5
    ).to(storage_dtype)
    v = (
        torch.randn(pool, kv_heads, 256, device="cuda", dtype=torch.bfloat16) * 0.5
    ).to(storage_dtype)
    indices = torch.randint(0, pool, (rows, width), device="cuda", dtype=torch.int32)
    indices[:, ::4] = -1
    indices[:, 1] = pool + 31
    indices[0].fill_(-1)
    owners = torch.zeros(rows, device="cuda", dtype=torch.int32)
    slots = torch.arange(pool, device="cuda", dtype=torch.int32)[None, :]
    positions = torch.full((rows,), pool - 1, device="cuda", dtype=torch.int32)
    kwargs = {"scale": 0.0625, "k_scale": 0.75, "v_scale": 1.25}
    expected = fa3_sparse_attention(
        q, k, v, indices, owners, slots, positions, **kwargs
    )
    actual = fa3_sparse_attention(
        q, k, v, indices, None, None, None, indices_are_physical=True, **kwargs
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], torch.zeros_like(actual[0]), rtol=0, atol=0)
