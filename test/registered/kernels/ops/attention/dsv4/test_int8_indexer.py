"""Unit tests for the DSv4.1 PPU INT8 indexer cache and E4M3FN emulation.

Covers the three hardware-independent pieces of the INT8/dense-FP4 indexer
support: the packed index-K cache (``store_int8_index_k``/
``gather_int8_index_k`` addressing, FP32 scale byte packing, the
``slot > 0`` reservation), ``quantize_query``'s per-head Q-scale folding into
the head weights, and the software E4M3FN helpers that replace the
``float8e4nv`` casts Triton rejects on PPU below SM89 (bit-exact against
torch's E4M3FN cast/decode). The DeepGEMM ``int8_*_mqa_logits`` scoring
itself is covered by E2E accuracy runs.

Runs on any CUDA device; no DSv4 checkpoint required.
"""

import unittest

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.fp4_rope_fake_quant import (
    _round_positive_e4m3_scale,
)
from sglang.kernels.ops.attention.dsv4.int8_indexer import (
    gather_int8_index_k,
    int8_logits_rows_per_chunk,
    quantize_query,
    store_int8_index_k,
)
from sglang.kernels.ops.embeddings.engram_gather import _e4m3fn_to_float
from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 64


@triton.jit
def _drive_round_scale(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, _round_positive_e4m3_scale(x), mask=mask)


@triton.jit
def _drive_e4m3_decode(b_ptr, y_ptr, N, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    payload = tl.load(b_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, _e4m3fn_to_float(payload), mask=mask)


class TestInt8IndexerCache(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.device = "cuda"

    def _zeros_cache(self, pages):
        return torch.zeros(
            pages, PAGE_SIZE * 132, dtype=torch.uint8, device=self.device
        )

    def test_store_gather_roundtrip(self):
        torch.manual_seed(0)
        rows = 5
        loc = torch.tensor([0, 1, 63, 64, 255], dtype=torch.int64, device=self.device)
        key = torch.randn(rows, 128, dtype=torch.bfloat16, device=self.device)
        cache = self._zeros_cache(4)

        store_int8_index_k(key, cache, loc, PAGE_SIZE)
        k, scales = gather_int8_index_k(cache, loc, PAGE_SIZE)
        torch.cuda.synchronize()

        ref_q, ref_s = per_token_quant_int8(key.contiguous())
        self.assertTrue(torch.equal(k[1:], ref_q.view(rows, 128)[1:]))
        self.assertTrue(torch.equal(scales[1:], ref_s.view(rows).to(torch.float32)[1:]))
        # Slot 0 is reserved: nothing was written and its row stays zero.
        self.assertTrue(bool((k[0] == 0).all()) and scales[0].item() == 0.0)

    def test_scale_bytes_match_fp32_bitcast(self):
        torch.manual_seed(1)
        loc = torch.tensor([130], dtype=torch.int64, device=self.device)
        key = torch.randn(1, 128, dtype=torch.bfloat16, device=self.device)
        cache = self._zeros_cache(3)

        store_int8_index_k(key, cache, loc, PAGE_SIZE)
        torch.cuda.synchronize()

        ref_q, ref_s = per_token_quant_int8(key.contiguous())
        page, off = int(loc[0]) // PAGE_SIZE, int(loc[0]) % PAGE_SIZE
        scale_region = cache[:, PAGE_SIZE * 128 :].view(torch.float32)
        self.assertEqual(scale_region[page, off].item(), ref_s.view(-1)[0].item())
        payload = cache[:, : PAGE_SIZE * 128].view(torch.int8)
        payload = payload.reshape(cache.shape[0], PAGE_SIZE, 128)
        self.assertTrue(torch.equal(payload[page, off], ref_q.view(128)))

    def test_rows_per_chunk_matches_budget_arithmetic(self):
        for width in (1, 128, 257, 1024):
            for budget_mb in (1, 16, 4096):
                budget = budget_mb * 1024 * 1024
                stride = (width + 256 + 3) // 4 * 4
                expect = max(1, budget // (stride * 4) // 4 * 4)
                self.assertEqual(int8_logits_rows_per_chunk(width, budget), expect)

    def test_quantize_query_folds_q_scales_into_weights(self):
        torch.manual_seed(2)
        q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device=self.device)
        weights = torch.rand(3, 4, dtype=torch.float32, device=self.device)

        q_int8, folded = quantize_query(q, weights)
        ref_q, ref_s = per_token_quant_int8(q.flatten(0, 1).contiguous())
        self.assertEqual(tuple(q_int8.shape), (3, 4, 128))
        self.assertTrue(torch.equal(q_int8, ref_q.view(3, 4, 128)))
        self.assertTrue(torch.equal(folded, weights * ref_s.view(3, 4)))
        # Dequantized error stays within one rounding step of the per-token scale.
        dequant = q_int8.float() * ref_s.view(3, 4, 1)
        err = (dequant - q.float()).abs()
        self.assertTrue((err <= ref_s.view(3, 4, 1)).all())


class TestE4M3Emulation(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.device = "cuda"

    def _drive(self, kernel, inp, out_dtype):
        n = inp.numel()
        out = torch.empty(n, dtype=out_dtype, device=self.device)
        kernel[(triton.cdiv(n, 256),)](inp, out, n, BLOCK=256)
        torch.cuda.synchronize()
        return out

    def test_round_scale_matches_torch_fp8_cast(self):
        values = torch.cat(
            [
                torch.linspace(2.0**-9, 2.0**-6 * 1.98, 512),
                torch.linspace(2.0**-6, 448.0, 1024),
                torch.tensor([2.0**-9, 2.0**-7, 1.0, 2.0, 448.0]),
            ]
        ).to(self.device)
        got = self._drive(_drive_round_scale, values, torch.float32)
        expect = values.to(torch.float8_e4m3fn).to(torch.float32)
        self.assertTrue(torch.equal(got, expect))

    def test_decode_matches_torch_fp8_view(self):
        payload = torch.arange(256, dtype=torch.uint8, device=self.device)
        got = self._drive(_drive_e4m3_decode, payload, torch.float32)
        expect = payload.view(torch.float8_e4m3fn).to(torch.float32)
        nan = torch.isnan(expect)
        self.assertTrue(bool(nan[0x7F]) and bool(nan[0xFF]))
        self.assertTrue(bool(got[nan].isnan().all()))
        self.assertTrue(torch.equal(got[~nan], expect[~nan]))


if __name__ == "__main__":
    unittest.main()
