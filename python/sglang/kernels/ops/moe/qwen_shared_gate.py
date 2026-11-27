"""Sail Qwen shared gate for the MegaMoE overlap path."""

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_gate_rows(x, w, shared, out, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offset = row * H + col
    h = tl.load(x + offset, col < H, 0).to(tl.float32)
    weight = tl.load(w + col, col < H, 0).to(tl.float32)
    logits = tl.sum(h * weight, 0).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(logits).to(tl.bfloat16).to(tl.float32)
    value = tl.load(shared + offset, col < H, 0).to(tl.float32)
    tl.store(out + offset, gate * value, col < H)


def qwen_shared_gate(x: torch.Tensor, weight: torch.Tensor, shared: torch.Tensor):
    out = torch.empty_like(shared)
    rows, hidden = x.shape
    _shared_gate_rows[(rows,)](
        x, weight, shared, out, hidden, triton.next_power_of_2(hidden), num_warps=8
    )
    return out
