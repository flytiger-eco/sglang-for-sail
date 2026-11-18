"""Pack PLE Prefill convolution input without padded token intermediates.

The starts/lengths/slot indices come from _prepare_ple_batch. Every output
position is written, including zero padding and the cached-history prefix.
Convolution arithmetic and live/radix-boundary state updates remain unchanged.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_ple_prefill_kernel(
    X,
    S,
    SLOTS,
    L,
    START,
    OUT,
    C: tl.constexpr,
    T,
    SZ,
    SS0,
    SS1,
    SS2,
    BC: tl.constexpr,
    BT: tl.constexpr,
):
    seq = tl.program_id(0)
    channels = tl.program_id(1) * BC + tl.arange(0, BC)
    times = tl.program_id(2) * BT + tl.arange(0, BT)
    length = tl.load(L + seq)
    start = tl.load(START + seq)
    slot = tl.load(SLOTS + seq)
    prefix = tl.load(
        S + slot.to(tl.int64) * SS0 + channels[:, None] * SS1 + times[None, :] * SS2,
        (channels[:, None] < C) & (times[None, :] < SZ),
        0,
    )
    position = times - SZ
    value = tl.load(
        X + (start.to(tl.int64) + position[None, :]) * C + channels[:, None],
        (channels[:, None] < C)
        & (position[None, :] >= 0)
        & (position[None, :] < length),
        0,
    )
    output = tl.where(times[None, :] < SZ, prefix, value)
    tl.store(
        OUT + (seq * C + channels[:, None].to(tl.int64)) * T + times[None, :],
        output,
        (channels[:, None] < C) & (times[None, :] < T),
    )


def can_pack_ple_prefill(x, state, state_indices, lengths, starts, row_width):
    if starts is None or x.ndim != 2 or state.ndim != 3 or lengths.ndim != 1:
        return False
    rows = lengths.shape[0]
    return (
        row_width >= 0
        and x.shape[1] == state.shape[1]
        and x.dtype == torch.bfloat16
        and state.dtype in (torch.bfloat16, torch.float32)
        and x.is_contiguous()
        and state_indices.shape == (rows,)
        and starts.shape == (rows + 1,)
        and all(
            t.dtype == torch.int64 and t.is_contiguous()
            for t in (state_indices, lengths, starts)
        )
        and all(
            t.is_cuda and t.device == x.device
            for t in (x, state, state_indices, lengths, starts)
        )
    )


def pack_ple_prefill(x, state, state_indices, lengths, starts, row_width):
    if not can_pack_ple_prefill(x, state, state_indices, lengths, starts, row_width):
        raise ValueError("unsupported PLE Prefill packing layout or dtype")
    count, channels = lengths.shape[0], x.shape[1]
    width = row_width + state.shape[2]
    out = torch.empty((count, channels, width), device=x.device, dtype=x.dtype)
    if out.numel():
        # Measured PPU large-row layout; small widths use a square tile.
        block_t = 32 if row_width <= 128 else 64
        _pack_ple_prefill_kernel[
            (count, triton.cdiv(channels, 32), triton.cdiv(width, block_t))
        ](
            x,
            state,
            state_indices,
            lengths,
            starts,
            out,
            channels,
            width,
            state.shape[2],
            *state.stride(),
            32,
            block_t,
            num_warps=4,
        )
    return out


@triton.jit
def _apply_ple_prefill_gate_kernel(G, V, OUT, GROUPS, WIDTH: tl.constexpr):
    groups = tl.program_id(0) * 4 + tl.arange(0, 4)
    channels = tl.program_id(1) * 256 + tl.arange(0, 256)
    gate = tl.load(G + groups, groups < GROUPS, 0).to(tl.float32)
    mask = (groups[:, None] < GROUPS) & (channels[None, :] < WIDTH)
    value = tl.load(V + (groups[:, None] // 4) * WIDTH + channels[None, :], mask, 0).to(
        tl.float32
    )
    tl.store(
        OUT + groups[:, None] * WIDTH + channels[None, :], gate[:, None] * value, mask
    )


def apply_ple_prefill_gate(gate, value):
    """Apply the already-computed BF16 gate without changing gate arithmetic."""
    supported = (
        gate.ndim == 3
        and gate.shape[1:] == (4, 1)
        and value.shape == (gate.shape[0], 2560)
        and gate.dtype == value.dtype == torch.bfloat16
        and gate.is_cuda
        and gate.device == value.device
        and gate.is_contiguous()
        and value.is_contiguous()
    )
    if not supported:
        return gate * value.unsqueeze(-2)
    out = torch.empty((gate.shape[0], 4, 2560), device=value.device, dtype=value.dtype)
    if out.numel():
        _apply_ple_prefill_gate_kernel[(gate.shape[0], 10)](
            gate,
            value,
            out,
            gate.shape[0] * 4,
            2560,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
