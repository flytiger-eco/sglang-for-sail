"""Unit tests for the `dequantize_and_gather_k_cuda` JIT kernel.

Compares the CUDA JIT kernel implemented in
`python/sglang/jit_kernel/csrc/deepseek_v4/dequantize_gather.cuh`
against a pure-PyTorch reference that mirrors the per-token byte layout of
the DeepseekV4 paged FP8 K cache.

Layout (per token, total 584 bytes split across the two intra-page regions):
    [   448 B FP8 nope    | 128 B BF16 rope ]   data
    [   7 UE8M0 scales    | 1 B pad         ]   scales
Per page (ordered):
    page_size * 576 bytes of token data, then page_size * 8 bytes of scales.
Per-token output (BF16): 448 dequantized FP8 + 64 directly-copied BF16 = 512.
"""

import sys

import pytest
import torch

from sglang.jit_kernel.deepseek_v4 import dequantize_and_gather_k_cuda

try:
    from sglang.test.ci.ci_register import register_cuda_ci

    register_cuda_ci(est_time=10, suite="nightly-1-gpu", nightly=True)
except Exception:  # pragma: no cover - optional CI hook
    pass


# ─── Layout constants (match dequantize_gather.cuh) ─────────────────────────
FP8_DIM = 448
BF16_DIM = 64
SCALE_DIM = 8  # 7 active scales + 1 pad
QUANT_BLOCK = 64
N_QUANT_BLOCKS = 7  # FP8_DIM // QUANT_BLOCK
TOKEN_DATA_SIZE = 576  # FP8_DIM + BF16_DIM*2
OUTPUT_DIM = 512  # FP8_DIM + BF16_DIM

DEVICE = "cuda"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for the JIT kernel"
)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_random_k_cache(num_pages: int, block_size: int, seed: int = 0) -> torch.Tensor:
    """Build a paged uint8 K cache populated with valid FP8/BF16/UE8M0 bytes.

    Layout per page: `block_size * TOKEN_DATA_SIZE` data bytes followed by
    `block_size * SCALE_DIM` scale bytes (= `block_size * 584` bytes total).
    """
    page_bytes = block_size * (TOKEN_DATA_SIZE + SCALE_DIM)
    g = torch.Generator(device="cpu").manual_seed(seed)
    cache = torch.zeros(num_pages, page_bytes, dtype=torch.uint8)

    # FP8 bytes (e4m3fn): the bit patterns 0x7F and 0xFF encode NaN in this
    # format. Remap them to 0 so the reference and kernel both produce
    # well-defined finite values that can be compared directly.
    fp8 = torch.randint(
        0, 256, (num_pages, block_size, FP8_DIM), generator=g, dtype=torch.int32
    )
    fp8[(fp8 == 0x7F) | (fp8 == 0xFF)] = 0

    # BF16 rope bytes: small-magnitude random bf16 reinterpreted as uint8.
    bf16 = (torch.randn((num_pages, block_size, BF16_DIM), generator=g) * 2.0).to(
        torch.bfloat16
    )
    bf16_bytes = (
        bf16.contiguous().view(torch.uint8).view(num_pages, block_size, BF16_DIM * 2)
    )

    token_data = torch.empty(num_pages, block_size, TOKEN_DATA_SIZE, dtype=torch.uint8)
    token_data[..., :FP8_DIM] = fp8.to(torch.uint8)
    token_data[..., FP8_DIM:TOKEN_DATA_SIZE] = bf16_bytes
    cache[:, : block_size * TOKEN_DATA_SIZE] = token_data.reshape(num_pages, -1)

    # UE8M0 scales: encoded byte `s` represents `2 ** (s - 127)`. Sample
    # around 127 (= scale 1.0) so dequantized values stay within bf16 range.
    scales = torch.randint(
        120, 135, (num_pages, block_size * SCALE_DIM), generator=g
    ).to(torch.uint8)
    cache[:, block_size * TOKEN_DATA_SIZE :] = scales

    return cache.to(DEVICE)


def _make_normal_only_k_cache(
    num_pages: int, block_size: int, seed: int = 0
) -> torch.Tensor:
    """Like `_make_random_k_cache`, but every FP8 byte is a *normal* e4m3 value.

    The fused IEEE754 bit-construction path FTZs FP8 subnormals (exp == 0),
    while the native intrinsic decodes them to small finite values. To make
    the two paths bit-identical we therefore restrict the FP8 bytes to
    encodings where `exp_bits \u2208 [1, 14]` (excluding subnormals and the e4m3
    NaN codepoints 0x7F / 0xFF).
    """
    page_bytes = block_size * (TOKEN_DATA_SIZE + SCALE_DIM)
    g = torch.Generator(device="cpu").manual_seed(seed)
    cache = torch.zeros(num_pages, page_bytes, dtype=torch.uint8)

    shape = (num_pages, block_size, FP8_DIM)
    sign = torch.randint(0, 2, shape, generator=g, dtype=torch.int32)
    exp = torch.randint(1, 15, shape, generator=g, dtype=torch.int32)  # [1, 14]
    mant = torch.randint(0, 8, shape, generator=g, dtype=torch.int32)  # [0, 7]
    fp8 = ((sign << 7) | (exp << 3) | mant).to(torch.uint8)

    bf16 = (torch.randn((num_pages, block_size, BF16_DIM), generator=g) * 2.0).to(
        torch.bfloat16
    )
    bf16_bytes = (
        bf16.contiguous().view(torch.uint8).view(num_pages, block_size, BF16_DIM * 2)
    )

    token_data = torch.empty(num_pages, block_size, TOKEN_DATA_SIZE, dtype=torch.uint8)
    token_data[..., :FP8_DIM] = fp8
    token_data[..., FP8_DIM:TOKEN_DATA_SIZE] = bf16_bytes
    cache[:, : block_size * TOKEN_DATA_SIZE] = token_data.reshape(num_pages, -1)

    scales = torch.randint(
        120, 135, (num_pages, block_size * SCALE_DIM), generator=g
    ).to(torch.uint8)
    cache[:, block_size * TOKEN_DATA_SIZE :] = scales

    return cache.to(DEVICE)


def _build_block_table(
    num_seqs: int, max_blocks_per_seq: int, num_pages: int, seed: int = 0
) -> torch.Tensor:
    """Produce a `[num_seqs, max_blocks_per_seq]` int32 block table whose
    entries are valid (and per-row unique) physical page indices."""
    assert num_pages >= max_blocks_per_seq, "Need enough pages for one full row"
    g = torch.Generator(device="cpu").manual_seed(seed)
    bt = torch.empty(num_seqs, max_blocks_per_seq, dtype=torch.int32)
    for b in range(num_seqs):
        ids = torch.randperm(num_pages, generator=g)[:max_blocks_per_seq]
        bt[b] = ids.to(torch.int32)
    return bt.to(DEVICE)


def reference_dequantize_gather(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    gather_lens,
    block_size: int,
    offset: int,
) -> None:
    """Pure-PyTorch ground truth mirroring `dequantize_gather.cuh`.

    Writes results into `out[b, offset + i, :]` for each gathered token.
    Untouched output positions are left as-is (preserving caller-provided
    sentinels), matching the kernel's behavior.
    """
    B = seq_lens.shape[0]
    k_cache_cpu = k_cache.cpu()
    block_table_cpu = block_table.cpu()
    seq_lens_cpu = seq_lens.cpu()
    gather_lens_cpu = gather_lens.cpu() if gather_lens is not None else None

    out_cpu = out.cpu().clone()

    for b in range(B):
        seq_len = int(seq_lens_cpu[b].item())
        glen = (
            int(gather_lens_cpu[b].item()) if gather_lens_cpu is not None else seq_len
        )
        start_pos = seq_len - glen
        for i in range(glen):
            pos = start_pos + i
            blk_in_seq = pos // block_size
            pos_in_blk = pos - blk_in_seq * block_size
            phys = int(block_table_cpu[b, blk_in_seq].item())

            page = k_cache_cpu[phys]  # 1-D uint8 view of the physical page
            tok_off = pos_in_blk * TOKEN_DATA_SIZE
            scale_base = block_size * TOKEN_DATA_SIZE + pos_in_blk * SCALE_DIM

            # Stage 1: FP8 dequantization across 7 quant blocks.
            for qb in range(N_QUANT_BLOCKS):
                fp8_bytes = page[
                    tok_off + qb * QUANT_BLOCK : tok_off + (qb + 1) * QUANT_BLOCK
                ].contiguous()
                fp8_vals = fp8_bytes.view(torch.float8_e4m3fn)
                fp32_vals = fp8_vals.float()
                encoded_scale = int(page[scale_base + qb].item())
                scale = 2.0 ** (encoded_scale - 127)
                dequant = (fp32_vals * scale).to(torch.bfloat16)
                out_cpu[
                    b, offset + i, qb * QUANT_BLOCK : (qb + 1) * QUANT_BLOCK
                ] = dequant

            # Stage 2: direct BF16 copy of the rope tail.
            bf16_bytes = page[
                tok_off + FP8_DIM : tok_off + TOKEN_DATA_SIZE
            ].contiguous()
            bf16_vals = bf16_bytes.view(torch.bfloat16)
            out_cpu[b, offset + i, FP8_DIM:OUTPUT_DIM] = bf16_vals

    out.copy_(out_cpu.to(out.device))


def _run_compare(
    B: int,
    seq_lens_list,
    block_size: int,
    gather_lens_list=None,
    offset: int = 0,
    max_M: int = None,
    seed: int = 42,
) -> None:
    """Build inputs, run kernel + reference, and compare touched regions."""
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    if gather_lens_list is not None:
        gather_lens = torch.tensor(gather_lens_list, dtype=torch.int32, device=DEVICE)
    else:
        gather_lens = None

    max_seq_len = max(seq_lens_list)
    max_blocks = (max_seq_len + block_size - 1) // block_size
    # Slack so different rows can map to disjoint pages; the kernel also
    # only reads pages that block_table actually points to.
    num_pages = max_blocks * B + 4

    k_cache = _make_random_k_cache(num_pages, block_size, seed=seed)
    block_table = _build_block_table(B, max_blocks, num_pages, seed=seed + 1)

    if max_M is None:
        if gather_lens_list is not None:
            max_M = offset + max(gather_lens_list)
        else:
            max_M = offset + max_seq_len

    # Initialize output to a sentinel so we can verify the kernel only
    # touches the [offset, offset+gather_len) slice per row.
    sentinel = torch.full(
        (B, max_M, OUTPUT_DIM), float("nan"), dtype=torch.bfloat16, device=DEVICE
    )
    out_kernel = sentinel.clone()
    out_ref = sentinel.clone()

    dequantize_and_gather_k_cuda(
        out=out_kernel,
        k_cache=k_cache,
        seq_lens=seq_lens,
        block_table=block_table,
        offset=offset,
        gather_lens=gather_lens,
        block_size=block_size,
    )
    reference_dequantize_gather(
        out_ref, k_cache, seq_lens, block_table, gather_lens, block_size, offset
    )

    for b in range(B):
        seq_len = int(seq_lens[b].item())
        glen = int(gather_lens[b].item()) if gather_lens is not None else seq_len
        end = offset + glen

        a = out_kernel[b, offset:end].float()
        r = out_ref[b, offset:end].float()
        # BF16 round-tripping plus an FP32 multiply by exp2(s-127) gives a
        # tiny tolerance budget; allow a slightly looser margin to absorb
        # implementation differences in fp8->float promotion.
        torch.testing.assert_close(a, r, atol=1e-2, rtol=1e-2)

        # Untouched output positions must remain at the sentinel (NaN).
        if offset > 0:
            assert torch.isnan(out_kernel[b, :offset]).all(), (
                f"Row {b}: kernel wrote into pre-offset region"
            )
        if end < max_M:
            assert torch.isnan(out_kernel[b, end:]).all(), (
                f"Row {b}: kernel wrote past gather_len region"
            )


# ─── Test cases ────────────────────────────────────────────────────────────


def test_basic_full_gather():
    """gather_lens=None: gather full seq_len per request, block_size=128."""
    _run_compare(B=2, seq_lens_list=[200, 150], block_size=128, seed=11)


def test_partial_gather():
    """gather_lens specifies a strict suffix of each sequence."""
    _run_compare(
        B=2,
        seq_lens_list=[300, 200],
        block_size=128,
        gather_lens_list=[64, 100],
        seed=12,
    )


def test_block_size_64():
    """C4 path (block_size=64), with seq lengths around the page boundary."""
    _run_compare(B=3, seq_lens_list=[100, 64, 130], block_size=64, seed=13)


def test_with_offset():
    """Non-zero offset writes into the right output column slice."""
    _run_compare(
        B=2,
        seq_lens_list=[64, 96],
        block_size=128,
        offset=32,
        max_M=200,
        seed=14,
    )


def test_multi_batch():
    """Mix of batch sizes with both partial gather and varied seq lengths."""
    _run_compare(
        B=4,
        seq_lens_list=[50, 200, 130, 96],
        block_size=128,
        gather_lens_list=[50, 80, 130, 32],
        seed=15,
    )


def test_fused_fp8_path():
    """Test the `use_fp8_native=False` fused IEEE754 bit-construction path.

    The fused path is bit-identical to the native CUDA FP8 intrinsic for all
    FP8 *normal* values, so we compare the kernel output against the existing
    reference (which uses native conversion). Test data is restricted to FP8
    normal encodings so subnormal FTZ behavior does not introduce diffs.
    """
    B = 2
    seq_lens_list = [256, 128]
    block_size = 128
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)

    max_seq_len = max(seq_lens_list)
    max_blocks = (max_seq_len + block_size - 1) // block_size
    num_pages = max_blocks * B + 4

    k_cache = _make_normal_only_k_cache(num_pages, block_size, seed=21)
    block_table = _build_block_table(B, max_blocks, num_pages, seed=22)

    max_M = max_seq_len
    sentinel = torch.full(
        (B, max_M, OUTPUT_DIM), float("nan"), dtype=torch.bfloat16, device=DEVICE
    )
    out_kernel = sentinel.clone()
    out_ref = sentinel.clone()

    dequantize_and_gather_k_cuda(
        out=out_kernel,
        k_cache=k_cache,
        seq_lens=seq_lens,
        block_table=block_table,
        offset=0,
        gather_lens=None,
        block_size=block_size,
        use_fp8_native=False,
    )
    reference_dequantize_gather(
        out_ref, k_cache, seq_lens, block_table, None, block_size, 0
    )

    for b in range(B):
        seq_len = int(seq_lens[b].item())
        a = out_kernel[b, :seq_len].float()
        r = out_ref[b, :seq_len].float()
        # For FP8 normal values the fused path is bit-identical to native.
        torch.testing.assert_close(a, r, atol=0.0, rtol=0.0)


def test_fused_vs_native_consistency():
    """Run the kernel twice (native vs. fused) on identical inputs and assert
    bit-identical outputs over FP8 normal values. Catches discrepancies
    between the two device-side decode paths directly, without going through
    the CPU reference.
    """
    B = 3
    seq_lens_list = [200, 96, 130]
    block_size = 64
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)

    max_seq_len = max(seq_lens_list)
    max_blocks = (max_seq_len + block_size - 1) // block_size
    num_pages = max_blocks * B + 4

    k_cache = _make_normal_only_k_cache(num_pages, block_size, seed=31)
    block_table = _build_block_table(B, max_blocks, num_pages, seed=32)

    max_M = max_seq_len
    sentinel = torch.full(
        (B, max_M, OUTPUT_DIM), float("nan"), dtype=torch.bfloat16, device=DEVICE
    )
    out_native = sentinel.clone()
    out_fused = sentinel.clone()

    common_kwargs = dict(
        k_cache=k_cache,
        seq_lens=seq_lens,
        block_table=block_table,
        offset=0,
        gather_lens=None,
        block_size=block_size,
    )
    dequantize_and_gather_k_cuda(out=out_native, use_fp8_native=True, **common_kwargs)
    dequantize_and_gather_k_cuda(out=out_fused, use_fp8_native=False, **common_kwargs)

    for b in range(B):
        seq_len = int(seq_lens[b].item())
        a = out_fused[b, :seq_len].float()
        r = out_native[b, :seq_len].float()
        torch.testing.assert_close(a, r, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
