"""Precompile PPU CP kernels using scratch, without touching live KV states."""

from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def warmup_qsa_cp_prefill(device, kv_heads, kv_head_dim, warm_fp8_store, hidden_size):
    from sglang.kernels.ops.attention.qsa_cp_q_prep import qprep

    x = torch.zeros((2, 640), device=device, dtype=torch.bfloat16)
    weight = torch.zeros(128, device=device, dtype=x.dtype)
    cos_sin = torch.ones((2, 64), device=device, dtype=torch.float32)
    axis = torch.zeros(32, device=device, dtype=torch.int32)
    positions = torch.arange(2, device=device, dtype=torch.int64)
    for pos in (positions, positions.repeat(3, 1)):
        qprep(x, pos, cos_sin, axis, weight)
    if warm_fp8_store:
        from sglang.kernels.ops.attention.qsa_cp_fp8_store import store_cp_kv

        packed = torch.zeros(
            (2, kv_heads, 2 * kv_head_dim), device=device, dtype=x.dtype
        )
        k, v = packed.split(kv_head_dim, dim=-1)
        keys = torch.empty(
            (2, kv_heads, kv_head_dim), device=device, dtype=torch.float8_e4m3fn
        )
        values = torch.empty_like(keys)
        for dtype in (torch.int32, torch.int64):
            loc = torch.arange(2, device=device, dtype=dtype)
            store_cp_kv(
                k, v, keys, values, loc,
                total_rows=2, cp_size=2, local_capacity=1,
            )

    from sglang.kernels.ops.moe.triton_sigmoid_gate_mul import sigmoid_gate_mul_add_bf16

    # Scalar M is not specialized by the flat kernel, so one scratch shape
    # also precompiles the long-context calls without allocating long buffers.
    main = torch.zeros((2, hidden_size), device=device, dtype=x.dtype)
    shared = torch.zeros_like(main)
    gate = torch.zeros((2, 1), device=device, dtype=x.dtype)
    sigmoid_gate_mul_add_bf16(main, shared, gate)
