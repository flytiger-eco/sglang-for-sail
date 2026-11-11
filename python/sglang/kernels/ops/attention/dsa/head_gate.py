import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_head_gate_q_scale_kernel(
    gate_ptr,
    q_scale_ptr,
    out_ptr,
    gate_stride,
    n_heads,
    gate_scale,
    softmax_scale,
    BLOCK: tl.constexpr,
):
    # out = float(bf16(float(gate) * gate_scale)) * q_scale * softmax_scale, i.e.
    # bf16 `gate * n_heads**-0.5` then the fp32 q-scale fold, in one launch
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_heads
    g = tl.load(gate_ptr + row * gate_stride + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    g = (g * gate_scale).to(tl.bfloat16).to(tl.float32)
    qs = tl.load(q_scale_ptr + row * n_heads + offs, mask=mask, other=0.0)
    tl.store(out_ptr + row * n_heads + offs, g * qs * softmax_scale, mask=mask)


def bf16_head_gate_q_scale(
    gate: torch.Tensor, q_scale: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """DSA indexer head weights [tokens, n_heads, 1] fp32 from the bf16 head
    gate [tokens, n_heads] (row stride free) and the per-head q scales."""
    num_tokens, n_heads = gate.shape
    out = torch.empty((num_tokens, n_heads, 1), dtype=torch.float32, device=gate.device)
    if num_tokens > 0:
        _bf16_head_gate_q_scale_kernel[(num_tokens,)](
            gate,
            q_scale.contiguous(),
            out,
            gate.stride(0),
            n_heads,
            n_heads**-0.5,
            softmax_scale,
            BLOCK=triton.next_power_of_2(n_heads),
        )
    return out


@triton.jit
def _indexer_q_quant_head_gate_kernel(
    Q,
    G,
    Q8,
    W,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    GATE_STRIDE: tl.constexpr,
    GATE_SCALE: tl.constexpr,
    SOFTMAX_SCALE: tl.constexpr,
    ROUND_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Keep act_quant's arithmetic and the BF16 gate rounding point unchanged.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, 128)
    q = tl.load(
        Q + rows[:, None] * 128 + cols[None, :], rows[:, None] < ROWS, other=0
    ).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(q), 1), 1e-4)
    scale = amax * (1.0 / 448.0)
    if ROUND_SCALE:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    quant = tl.minimum(tl.maximum(q / scale[:, None], -448.0), 448.0)
    tl.store(Q8 + rows[:, None] * 128 + cols[None, :], quant, rows[:, None] < ROWS)
    gate = tl.load(
        G + rows // HEADS * GATE_STRIDE + rows % HEADS, rows < ROWS, other=0
    ).to(tl.float32)
    gate = (gate * GATE_SCALE).to(tl.bfloat16).to(tl.float32)
    tl.store(W + rows, gate * scale * SOFTMAX_SCALE, rows < ROWS)


def indexer_q_quant_head_gate(
    query: torch.Tensor,
    gate: torch.Tensor,
    softmax_scale: float,
    scale_fmt: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kimi's no-RoPE indexer Q quantization and BF16 head-gate scaling.

    Equivalent to act_quant(query, 128, scale_fmt), followed by
    float(bf16(gate * heads**-0.5)) * q_scale * softmax_scale.
    Query is contiguous [tokens, heads, 128]; gate may have a padded row stride.
    """
    assert query.is_cuda and query.dtype == torch.bfloat16
    assert query.ndim == 3 and query.shape[-1] == 128 and query.is_contiguous()
    tokens, heads, _ = query.shape
    assert gate.shape == (tokens, heads) and gate.stride(-1) == 1
    assert gate.device == query.device and gate.dtype == torch.bfloat16
    q8 = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    weights = torch.empty((tokens, heads, 1), device=query.device, dtype=torch.float32)
    if tokens:
        _indexer_q_quant_head_gate_kernel[(triton.cdiv(tokens * heads, 32),)](
            query,
            gate,
            q8,
            weights,
            tokens * heads,
            heads,
            gate.stride(0),
            heads**-0.5,
            softmax_scale,
            scale_fmt is not None,
            32,
            num_stages=0 if scale_fmt is not None else 2,
        )
    return q8, weights
