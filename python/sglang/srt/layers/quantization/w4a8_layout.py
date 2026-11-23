"""Packed INT4 expert weight conversion for the W4A8-Int8 MoE DeepGEMM path.

Checkpoint format (``mixed_precision_w4``, verified on the 810E artefact
against its FP8 source weights):

* per expert a ``uint8 [N, K // 2]`` buffer, packed along K;
* byte ``i`` holds logical ``k = 2 * i`` in the low nibble and ``k = 2 * i + 1``
  in the high nibble, both two's-complement signed INT4;
* inside every 32-element K group (16 bytes) the four-element tiles are stored
  in the order ``0, 4, 1, 5, 2, 6, 3, 7``, i.e. the three-bit tile index is
  rotated left by one.

That byte order is exactly what DeepGEMM's A8W4 weight preprocess produces in
its first step, so the checkpoint is *already past* that step.

``preprocess_a8w4_weight`` (PPU DeepGEMM ``674cfa5``,
``csrc/jit_kernels/heuristics/common_preprocess.hpp``,
``PreprocessMode::W4A8_INT8``) expects the portable order and applies three
steps:

1. transpose the 2-byte units inside every 16-byte K32 group
   (``view([..., K / 32, 2, 4, 2]).transpose(-3, -2)``);
2. repack each four-byte converter group ``{a, b, c, d, e, f, g, h}`` into
   ``{a, e}, {b, f}, {c, g}, {d, h}``;
3. flip every nibble's sign bit (``^ 0x88``) to turn two's-complement INT4 into
   the unsigned zero-point-8 encoding the PPU register converter wants.

Reusing that entry point here would run step 1 a second time. The rotation has
order three, so a second application is *not* the identity and would silently
scramble every K32 group; :func:`repack_a8w4_ckpt_to_deepgemm` therefore applies
steps 2 and 3 only.

The module depends on torch only, so the same code runs on a checkpoint shard
outside the serving stack (see ``--verify`` below).
"""

from __future__ import annotations

import torch

# Stored tile order of one 32-element K group relative to the logical order.
A8W4_TILE_PERMUTATION_STORED_TO_LOGICAL = (0, 4, 1, 5, 2, 6, 3, 7)

# Working-set budget for the chunked conversion, in bytes.
A8W4_REPACK_CHUNK_BYTES = 64 << 20


def unpack_a8w4_pairs(packed: torch.Tensor, low_nibble_even: bool = True) -> torch.Tensor:
    """Unpack uint8-stored signed int4 pairs along the last dim to int16."""
    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)
    pair = torch.stack((low, high) if low_nibble_even else (high, low), dim=-1)
    return pair.reshape(*packed.shape[:-1], -1)


def swizzle_a8w4_k32(weight: torch.Tensor) -> torch.Tensor:
    """Step 1 of DeepGEMM's A8W4 preprocess: transpose 2-byte units per K32.

    Kept so the conversion can be cross-checked against
    ``preprocess_a8w4_weight`` (three applications are the identity).
    """
    num_k_bytes = weight.shape[-1]
    if num_k_bytes % 16:
        raise ValueError(f"K must be a multiple of 32, got {2 * num_k_bytes}")
    grouped = weight.reshape(*weight.shape[:-1], num_k_bytes // 16, 2, 4, 2)
    return grouped.transpose(-3, -2).contiguous().flatten(-4)


def _repack_converter_groups(blocks: torch.Tensor) -> torch.Tensor:
    """Steps 2 and 3 of DeepGEMM's A8W4 preprocess on ``[..., 4]`` byte groups."""
    low = blocks & 0x0F  # per byte: a, c, e, g
    high = (blocks >> 4) & 0x0F  # per byte: b, d, f, h
    out = torch.empty_like(blocks)
    # {a,b,c,d,e,f,g,h} -> bytes {a,e}, {b,f}, {c,g}, {d,h}
    out[..., 0] = low[..., 0] | (low[..., 2] << 4)  # {a, e}
    out[..., 1] = high[..., 0] | (high[..., 2] << 4)  # {b, f}
    out[..., 2] = low[..., 1] | (low[..., 3] << 4)  # {c, g}
    out[..., 3] = high[..., 1] | (high[..., 3] << 4)  # {d, h}
    # Two's-complement INT4 -> unsigned INT4 with zero point 8.
    return out ^ 0x88


def repack_a8w4_ckpt_to_deepgemm(packed: torch.Tensor) -> torch.Tensor:
    """Convert a checkpoint ``[E, N, K // 2]`` buffer to the DeepGEMM rhs layout.

    The checkpoint is already in the post-step-1 byte order, so this applies the
    converter-group repack and the sign-bit flip only. The input is never
    modified; a new ``uint8`` tensor is returned.
    """
    if packed.dim() != 3:
        raise ValueError(f"expected [E, N, K // 2], got {tuple(packed.shape)}")
    if packed.dtype == torch.int8:
        packed = packed.view(torch.uint8)
    elif packed.dtype != torch.uint8:
        raise ValueError(f"expected an int8/uint8 packed buffer, got {packed.dtype}")

    num_experts, n, num_k_bytes = packed.shape
    if n % 8:
        raise ValueError(f"DeepGEMM i8i4 requires N % 8 == 0, got N={n}")
    if (2 * num_k_bytes) % 32 or num_k_bytes % 4:
        raise ValueError(f"DeepGEMM i8i4 requires K % 32 == 0, got K={2 * num_k_bytes}")

    out = torch.empty_like(packed)
    rows = packed.numel() // (num_experts * num_k_bytes)
    bytes_per_expert = max(1, rows * num_k_bytes)
    chunk = max(1, A8W4_REPACK_CHUNK_BYTES // bytes_per_expert)
    groups = num_k_bytes // 4
    for begin in range(0, num_experts, chunk):
        end = min(begin + chunk, num_experts)
        src = packed[begin:end].reshape(end - begin, rows, groups, 4)
        out[begin:end] = _repack_converter_groups(src).reshape_as(out[begin:end])
    return out


def _verify_against_deep_gemm(weight: torch.Tensor) -> dict:
    """Cross-check the conversion against ``preprocess_a8w4_weight``.

    ``swizzle`` has order three, so ``preprocess(swizzle(swizzle(w)))`` is
    DeepGEMM's steps 2+3 applied to ``w``: it must match
    :func:`repack_a8w4_ckpt_to_deepgemm` byte for byte. ``preprocess(w)``
    applies step 1 once and must *not* match.
    """
    from deep_gemm import preprocess_a8w4_weight

    if weight.dtype == torch.int8:
        weight = weight.view(torch.uint8)
    ours = repack_a8w4_ckpt_to_deepgemm(weight)
    twice = swizzle_a8w4_k32(swizzle_a8w4_k32(weight))
    return {
        "steps_2_3_bit_exact": torch.equal(ours, preprocess_a8w4_weight(twice)),
        "matches_full_preprocess": torch.equal(ours, preprocess_a8w4_weight(weight)),
    }


if __name__ == "__main__":
    import argparse
    import json

    from safetensors import safe_open

    parser = argparse.ArgumentParser(
        description="Check the W4A8 packed-int4 conversion against DeepGEMM."
    )
    parser.add_argument("--shard", required=True, help="safetensors shard path")
    parser.add_argument("--weight-key", required=True)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="cross-check against deep_gemm.preprocess_a8w4_weight",
    )
    args = parser.parse_args()

    with safe_open(args.shard, framework="pt") as handle:
        tensor_slice = handle.get_slice(args.weight_key)
        weight_shape = tensor_slice.get_shape()
        num_experts = weight_shape[0]
        step = max(1, num_experts // args.experts)
        ids = list(range(0, num_experts, step))[: args.experts]
        packed = torch.stack([tensor_slice[i] for i in ids]).contiguous()

    report = {
        "weight_key": args.weight_key,
        "weight_shape": list(weight_shape),
        "experts_sampled": ids,
        "converted_shape": list(repack_a8w4_ckpt_to_deepgemm(packed).shape),
    }
    if args.verify:
        report["vs_deep_gemm"] = _verify_against_deep_gemm(packed)
    print(json.dumps(report, indent=2))
