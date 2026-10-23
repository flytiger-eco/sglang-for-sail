"""Unit tests for the vectorized DSv4 paged K-cache dequantize kernel.

``dequantize_k_cache_paged`` dispatches to a vectorized kernel that splits each
token across a warp and issues 16-byte accesses, falling back to the original
one-thread-per-token scalar kernel when a buffer is not 16-byte aligned.

The invariant under test is that the vector kernel is bit-identical to the
scalar one across every axis the templates and the address arithmetic depend on:
threads-per-token instantiation, power-of-two vs. non-power-of-two page size
(shift vs. integer-division path), token counts around the grid-stride and warp
boundaries, page-table order and dtype, and output placement.

Runs on any CUDA device; no DSv4 checkpoint required.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsa.triton_kernel import _supports_fp8
from sglang.kernels.ops.attention.dsv4 import (
    dequantize_k_cache_paged_cuda,
    dequantize_k_cache_paged_v2_cuda,
)
from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    DIM_NOPE,
    DIM_ROPE,
    NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN,
    VECTORIZED_THREADS_PER_TOKEN,
    _vector_access_ok,
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_ref,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OUT_DIM = DIM_NOPE + DIM_ROPE
THREADS_PER_TOKEN = (8, 16, 32)
# 64/128 take the shift path; 48/96 are not powers of two and exercise the
# integer division fallback in the kernel.
PAGE_SIZES = (64, 128, 48, 96)
# Chosen so that num_tokens * threads_per_token straddles multiples of the
# 128-thread block for every threads-per-token value.
TOKEN_COUNTS = (
    0,
    1,
    2,
    3,
    4,
    5,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    257,
    1023,
    1025,
    4096,
    4097,
)


def bytes_per_page_for(page_size: int) -> int:
    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    return -(-raw // NOPE_ROPE_BYTES) * NOPE_ROPE_BYTES


def run_scalar(cache, table, page_size, out, native):
    dequantize_k_cache_paged_cuda(
        out=out,
        k_cache=cache,
        page_table=table,
        page_size=page_size,
        use_fp8_native=native,
    )
    return out


def run_vector(cache, table, page_size, out, native, tpt):
    dequantize_k_cache_paged_v2_cuda(
        out=out,
        k_cache=cache,
        page_table=table,
        page_size=page_size,
        use_fp8_native=native,
        threads_per_token=tpt,
    )
    return out


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDequantKCachePagedVectorized(CustomTestCase):
    def assertBitsEqual(self, actual, expected, context=""):
        self.assertEqual(actual.shape, expected.shape, context)
        self.assertEqual(actual.dtype, expected.dtype, context)
        # NaN == NaN is False, so identical NaNs must not count as mismatches.
        bad = (actual != expected) & ~(torch.isnan(actual) & torch.isnan(expected))
        count = int(bad.sum())
        self.assertEqual(count, 0, f"{count} mismatched elements {context}")

    def make_cache(self, page_size, num_pages, seed=0, scale_lo=0, scale_hi=256):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        bpp = bytes_per_page_for(page_size)
        cache = torch.randint(
            0,
            256,
            (num_pages, bpp),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        if (scale_lo, scale_hi) != (0, 256):
            self.restrict_scale_bytes(
                cache, page_size, num_pages, generator, scale_lo, scale_hi
            )
        return cache, bpp

    def restrict_scale_bytes(self, cache, page_size, num_pages, generator, lo, hi):
        """Overwrite the per-token scale region so no scale byte is 0 or 255.

        The scalar kernel and the torch reference legitimately disagree on a
        scale byte of 0 (the reference clamps 2**-127 to zero, the kernel does
        not), so comparisons against the reference must avoid it.
        """
        bpp = cache.shape[-1]
        scale_off = page_size * NOPE_ROPE_BYTES
        per_page = page_size * PADDED_SCALE_PER_TOKEN
        scales = torch.randint(
            lo,
            hi,
            (num_pages * per_page,),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        ).view(num_pages, per_page)
        idx = (torch.arange(num_pages, device="cuda") * bpp + scale_off).view(
            -1, 1
        ) + torch.arange(per_page, device="cuda").view(1, -1)
        cache.view(-1)[idx.view(-1)] = scales.view(-1)

    def make_table(self, num_tokens, num_pages, page_size, pattern, seed=0):
        capacity = num_pages * page_size
        generator = torch.Generator(device="cuda").manual_seed(seed)
        if pattern == "random":
            return torch.randint(
                0,
                capacity,
                (num_tokens,),
                dtype=torch.int32,
                device="cuda",
                generator=generator,
            )
        if pattern == "ascending":
            return torch.arange(num_tokens, device="cuda", dtype=torch.int32) % capacity
        if pattern == "descending":
            # CUDA tensors reject a negative slice step, so flip explicitly.
            return (
                torch.flip(
                    torch.arange(num_tokens, device="cuda", dtype=torch.int32), [0]
                )
                % capacity
            )
        if pattern == "duplicate":
            return torch.full(
                (num_tokens,), capacity // 2, dtype=torch.int32, device="cuda"
            )
        if pattern == "page_edges":
            # First and last token of every page: the extremes of the in-page
            # offset arithmetic.
            base = torch.arange(num_tokens, device="cuda", dtype=torch.int32)
            page = (base // 2) % num_pages
            in_page = (base % 2) * (page_size - 1)
            return page * page_size + in_page
        if pattern == "full_range":
            # Touches the lowest and highest addressable token.
            table = torch.zeros(num_tokens, dtype=torch.int32, device="cuda")
            table[0] = 0
            if num_tokens > 1:
                table[1] = capacity - 1
            if num_tokens > 2:
                table[2:] = torch.randint(
                    0,
                    capacity,
                    (num_tokens - 2,),
                    dtype=torch.int32,
                    device="cuda",
                    generator=generator,
                )
            return table
        raise ValueError(pattern)

    def out_buffer(self, num_tokens):
        return torch.empty(
            (num_tokens, 1, OUT_DIM), dtype=torch.bfloat16, device="cuda"
        )

    # -- the core invariant: vector == scalar -------------------------------

    def test_vector_matches_scalar_across_tpt_and_page_size(self):
        for page_size in PAGE_SIZES:
            num_pages = 256
            cache, bpp = self.make_cache(page_size, num_pages, seed=page_size)
            table = self.make_table(2000, num_pages, page_size, "random")
            expected = run_scalar(cache, table, page_size, self.out_buffer(2000), True)
            for tpt in THREADS_PER_TOKEN:
                with self.subTest(page_size=page_size, tpt=tpt):
                    self.assertTrue(
                        _vector_access_ok(cache, expected),
                        "test setup should exercise the vector path",
                    )
                    actual = run_vector(
                        cache, table, page_size, self.out_buffer(2000), True, tpt
                    )
                    self.assertBitsEqual(
                        actual,
                        expected,
                        f"page_size={page_size} tpt={tpt} bpp={bpp}",
                    )

    def test_vector_matches_scalar_across_token_counts(self):
        page_size, num_pages = 64, 512
        cache, _ = self.make_cache(page_size, num_pages, seed=11)
        for num_tokens in TOKEN_COUNTS:
            table = self.make_table(num_tokens, num_pages, page_size, "random")
            expected = run_scalar(
                cache, table, page_size, self.out_buffer(num_tokens), True
            )
            for tpt in THREADS_PER_TOKEN:
                with self.subTest(num_tokens=num_tokens, tpt=tpt):
                    actual = run_vector(
                        cache, table, page_size, self.out_buffer(num_tokens), True, tpt
                    )
                    self.assertBitsEqual(
                        actual, expected, f"num_tokens={num_tokens} tpt={tpt}"
                    )

    def test_vector_matches_scalar_across_page_table_patterns(self):
        page_size, num_pages, num_tokens = 64, 256, 1500
        cache, _ = self.make_cache(page_size, num_pages, seed=12)
        for pattern in (
            "random",
            "ascending",
            "descending",
            "duplicate",
            "page_edges",
            "full_range",
        ):
            table = self.make_table(num_tokens, num_pages, page_size, pattern)
            expected = run_scalar(
                cache, table, page_size, self.out_buffer(num_tokens), True
            )
            for tpt in THREADS_PER_TOKEN:
                with self.subTest(pattern=pattern, tpt=tpt):
                    actual = run_vector(
                        cache, table, page_size, self.out_buffer(num_tokens), True, tpt
                    )
                    self.assertBitsEqual(
                        actual, expected, f"pattern={pattern} tpt={tpt}"
                    )

    def test_page_table_dtype(self):
        page_size, num_pages, num_tokens = 128, 128, 1000
        cache, _ = self.make_cache(page_size, num_pages, seed=13)
        table32 = self.make_table(num_tokens, num_pages, page_size, "random")
        table64 = table32.to(torch.int64)
        expected = run_scalar(
            cache, table32, page_size, self.out_buffer(num_tokens), _supports_fp8()
        )
        # The public entry converts int64 down to int32 before launching.
        actual = dequantize_k_cache_paged(cache, table64, page_size)
        self.assertBitsEqual(actual, expected, "int64 page_table")

    def test_both_fp8_conversion_paths(self):
        """Cover both template instantiations regardless of host capability.

        The fused integer path flushes FP8 subnormals, so it does not match the
        torch reference; it must still match the scalar kernel run the same way.
        """
        page_size, num_pages, num_tokens = 64, 128, 1000
        cache, _ = self.make_cache(page_size, num_pages, seed=14)
        table = self.make_table(num_tokens, num_pages, page_size, "random")
        for native in (True, False):
            expected = run_scalar(
                cache, table, page_size, self.out_buffer(num_tokens), native
            )
            for tpt in THREADS_PER_TOKEN:
                with self.subTest(native=native, tpt=tpt):
                    actual = run_vector(
                        cache,
                        table,
                        page_size,
                        self.out_buffer(num_tokens),
                        native,
                        tpt,
                    )
                    self.assertBitsEqual(actual, expected, f"native={native} tpt={tpt}")

    def test_matches_torch_reference_when_scale_is_normal(self):
        if not _supports_fp8():
            self.skipTest("the fused fallback flushes FP8 subnormals")
        page_size, num_pages, num_tokens = 64, 128, 1000
        cache, _ = self.make_cache(
            page_size, num_pages, seed=15, scale_lo=1, scale_hi=255
        )
        table = self.make_table(num_tokens, num_pages, page_size, "random")
        expected = dequantize_k_cache_paged_ref(cache, table, page_size)
        actual = dequantize_k_cache_paged(cache, table, page_size)
        self.assertBitsEqual(actual, expected, "vector kernel vs torch reference")

    # -- output placement ---------------------------------------------------

    def test_out_workspace_slices(self):
        """The production call pattern: one workspace sliced into two regions."""
        page_size, num_pages = 64, 512
        cache, _ = self.make_cache(page_size, num_pages, seed=16)
        capacity = 8192
        workspace = torch.empty(
            (capacity, 1, OUT_DIM), dtype=torch.bfloat16, device="cuda"
        )
        for n_compressed in (0, 1, 7, 4096, 8191):
            num_tokens = capacity - n_compressed
            table = self.make_table(num_tokens, num_pages, page_size, "random")
            expected = run_scalar(
                cache, table, page_size, self.out_buffer(num_tokens), _supports_fp8()
            )
            for name, slot in (
                ("head", workspace[:num_tokens]),
                ("tail", workspace[capacity - num_tokens :]),
            ):
                with self.subTest(n_compressed=n_compressed, slot=name):
                    self.assertTrue(_vector_access_ok(cache, slot))
                    dequantize_k_cache_paged(cache, table, page_size, out=slot)
                    self.assertBitsEqual(slot, expected, f"{name} slice")

    def test_misaligned_out_falls_back_and_stays_correct(self):
        page_size, num_pages, num_tokens = 64, 128, 512
        cache, _ = self.make_cache(page_size, num_pages, seed=17)
        table = self.make_table(num_tokens, num_pages, page_size, "random")
        expected = run_scalar(
            cache, table, page_size, self.out_buffer(num_tokens), _supports_fp8()
        )

        # Shift by one bf16 element so data_ptr % 16 != 0. A whole-row slice
        # would stay aligned and exercise the vector path instead.
        backing = torch.empty(
            (num_tokens * OUT_DIM + 8,), dtype=torch.bfloat16, device="cuda"
        )
        odd = backing[1 : 1 + num_tokens * OUT_DIM].view(num_tokens, 1, OUT_DIM)
        self.assertNotEqual(odd.data_ptr() % 16, 0, "setup must be misaligned")
        self.assertFalse(_vector_access_ok(cache, odd))

        got = dequantize_k_cache_paged(cache, table, page_size, out=odd)
        self.assertBitsEqual(got, expected, "misaligned fallback")

    def test_out_none_allocates(self):
        page_size, num_pages, num_tokens = 64, 64, 333
        cache, _ = self.make_cache(page_size, num_pages, seed=18)
        table = self.make_table(num_tokens, num_pages, page_size, "random")
        allocated = dequantize_k_cache_paged(cache, table, page_size)
        self.assertEqual(allocated.shape, (num_tokens, 1, OUT_DIM))
        self.assertEqual(allocated.dtype, torch.bfloat16)
        expected = run_scalar(
            cache, table, page_size, self.out_buffer(num_tokens), _supports_fp8()
        )
        self.assertBitsEqual(allocated, expected, "out=None")

    def test_vector_path_is_taken_for_aligned_inputs(self):
        """Guard against a predicate regression silently disabling the kernel.

        Every test above would still pass if the dispatcher always fell back to
        the scalar path, so assert the predicate directly on realistic buffers.
        """
        for page_size in PAGE_SIZES:
            cache, bpp = self.make_cache(page_size, 64, seed=19)
            with self.subTest(page_size=page_size):
                self.assertEqual(bpp % 16, 0)
                self.assertEqual(cache.data_ptr() % 16, 0)
                for rows in (1, 1024):
                    ws = torch.empty(
                        (rows, 1, OUT_DIM), dtype=torch.bfloat16, device="cuda"
                    )
                    self.assertTrue(_vector_access_ok(cache, ws))
                    self.assertTrue(_vector_access_ok(cache, ws[: rows // 2 or 1]))

    def test_vector_kernel_is_faster_than_scalar_at_scale(self):
        """Correctness tests cannot see a performance regression.

        A bad threads-per-token default or a broken grid computation would still
        produce bit-identical output while collapsing back to scalar-kernel
        speed. The measured gap at this size is ~18x, so 2x leaves a wide margin
        for a noisy or differently-balanced device.
        """
        page_size, num_pages, num_tokens = 64, 512, 20096
        cache, _ = self.make_cache(page_size, num_pages, seed=21)
        table = self.make_table(num_tokens, num_pages, page_size, "random")
        out = self.out_buffer(num_tokens)

        def time_us(fn, iters=30, warmup=5):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / iters * 1000.0

        scalar_us = time_us(lambda: run_scalar(cache, table, page_size, out, True))
        vector_us = time_us(
            lambda: run_vector(
                cache, table, page_size, out, True, VECTORIZED_THREADS_PER_TOKEN
            )
        )
        self.assertGreater(
            scalar_us / vector_us,
            2.0,
            f"vector kernel only {scalar_us / vector_us:.2f}x faster "
            f"(scalar {scalar_us:.1f} us, vector {vector_us:.1f} us)",
        )

    def test_rejects_invalid_threads_per_token(self):
        page_size, num_pages, num_tokens = 64, 32, 128
        cache, _ = self.make_cache(page_size, num_pages, seed=20)
        table = self.make_table(num_tokens, num_pages, page_size, "random")
        out = self.out_buffer(num_tokens)
        for tpt in (0, 4, 64, 7):
            with self.subTest(tpt=tpt):
                with self.assertRaises(Exception):
                    run_vector(cache, table, page_size, out, True, tpt)

    # -- 64-bit addressing --------------------------------------------------

    def test_large_cache_uses_64bit_addressing(self):
        """A long-context cache exceeds 2**31 bytes; offsets must not truncate."""
        page_size = 64
        bpp = bytes_per_page_for(page_size)
        num_pages = (1 << 31) // bpp + 8
        need = num_pages * bpp
        free, _ = torch.cuda.mem_get_info()
        if free < need + (2 << 30):
            self.skipTest(f"needs {need / 2**30:.1f} GiB free, have {free / 2**30:.1f}")

        # Uninitialised bytes are fine: both kernels read the same buffer, and
        # the assertion is that they agree, not what the values are.
        cache = torch.empty((num_pages, bpp), dtype=torch.uint8, device="cuda")
        self.assertGreater(num_pages * bpp, 1 << 31)

        # Tokens in the highest pages, where a 32-bit product would wrap.
        table = torch.tensor(
            [
                num_pages * page_size - 1,
                (num_pages - 1) * page_size,
                (num_pages - 2) * page_size + 3,
                (1 << 31) // bpp * page_size,
            ],
            dtype=torch.int32,
            device="cuda",
        )
        num_tokens = table.shape[0]
        expected = run_scalar(
            cache, table, page_size, self.out_buffer(num_tokens), True
        )
        for tpt in THREADS_PER_TOKEN:
            with self.subTest(tpt=tpt):
                actual = run_vector(
                    cache, table, page_size, self.out_buffer(num_tokens), True, tpt
                )
                self.assertBitsEqual(actual, expected, f"large cache tpt={tpt}")
        del cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
