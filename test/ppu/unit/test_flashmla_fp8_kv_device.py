"""PPU device numerics for the SGLang FlashMLA wrapper (no model required).

Run on the verified PPU environment with the repository's python/ on PYTHONPATH.
Reference attention uses the quantized inputs, separating kernel correctness
from model-level FP8 quantization error.
"""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


@unittest.skipUnless(torch.cuda.is_available(), "PPU runtime required")
class TestFlashMlaFp8Device(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / (
            "python/sglang/srt/hardware_backend/ppu/attention/flash_mla.py"
        )
        spec = importlib.util.spec_from_file_location("ppu_flashmla_device_ops", path)
        cls.ops = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.ops)
        torch.manual_seed(42)

    def dense_inputs(self, sq=1, page=64):
        b, h, length = 2, 8, 128
        q = (torch.randn(b, sq, h, 576, device="cuda") * 0.2).to(torch.float8_e4m3fn)
        kv = (torch.randn(b * length // page, page, 1, 576, device="cuda") * 0.2).to(
            torch.float8_e4m3fn
        )
        table = torch.arange(b * length // page, device="cuda", dtype=torch.int32).view(
            b, -1
        )
        lens = torch.tensor([length - 7, length], device="cuda", dtype=torch.int32)
        return q, kv, table, lens

    def reference(self, q, keys, causal=False):
        # q: S,H,D; keys: T,D. MLA V is the first 512 K dimensions.
        scores = torch.einsum("shd,td->hst", q.float(), keys.float()) / 576**0.5
        if causal:
            sq, sk = q.shape[0], keys.shape[0]
            mask = torch.arange(sk, device=q.device)[None, :] > (
                sk - sq + torch.arange(sq, device=q.device)[:, None]
            )
            scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
        out = torch.einsum("hst,td->shd", scores.softmax(-1), keys[:, :512].float())
        return out, scores.logsumexp(-1)

    def run_dense(self, inputs, meta):
        q, kv, table, lens = inputs
        return self.ops.flash_mla_with_kvcache(
            q,
            kv,
            table,
            lens,
            512,
            meta,
            causal=True,
            is_fp8_kvcache=q.dtype == torch.float8_e4m3fn,
        )

    def check_dense(self, inputs, out, lse):
        q, kv, table, lens = inputs
        self.assertEqual(out.dtype, torch.bfloat16)
        for b in range(q.shape[0]):
            keys = kv[table[b].long()].reshape(-1, 576)[: int(lens[b])]
            ref, ref_lse = self.reference(q[b], keys, causal=q.shape[1] > 1)
            torch.testing.assert_close(out[b].float(), ref, atol=2e-3, rtol=2e-2)
            torch.testing.assert_close(lse[b], ref_lse, atol=2e-3, rtol=2e-3)

    def test_real_backend_uses_raw_fp8_query(self):
        from sglang.srt.layers.attention.flashmla_backend import FlashMLABackend

        q = torch.tensor([0.5, 1.0, -2.0, 4.0], device="cuda").reshape(1, 1, 1, 4)
        q8, dq, dk = FlashMLABackend._prepare_fp8_query(
            None, q, SimpleNamespace(k_scale=torch.tensor([4.0], device="cuda"))
        )
        torch.testing.assert_close(q8.float(), q, atol=0, rtol=0)
        self.assertIsNone(dq)
        self.assertIsNone(dk)

    def test_real_ppu_configuration_checks(self):
        from sglang.srt.arg_groups.overrides import _ppu_flashmla_kv_cache_checks
        from sglang.srt.utils import is_ppu

        self.assertTrue(
            is_ppu(), "This suite must exercise the PPU configuration branch"
        )
        config = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3",
            attention_backend="dsa",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            dsa_prefill_backend="flashmla_sparse",
            dsa_decode_backend="flashmla_kv",
            dcp_size=1,
        )
        self.assertEqual(_ppu_flashmla_kv_cache_checks(config), {})
        config.dsa_decode_backend = "flashmla_sparse"
        with self.assertRaisesRegex(ValueError, "flashmla_kv"):
            _ppu_flashmla_kv_cache_checks(config)

    def test_dense_decode_and_multi_query(self):
        for sq, page in ((1, 64), (3, 64), (1, 128)):
            with self.subTest(sq=sq, page=page):
                inputs = self.dense_inputs(sq, page)
                meta, _ = self.ops.get_mla_metadata(is_fp8_kvcache=True)
                out, lse = self.run_dense(inputs, meta)
                self.check_dense(inputs, out, lse)

    def test_dense_bf16_regression(self):
        q, kv, table, lens = self.dense_inputs(sq=3)
        inputs = (q.bfloat16(), kv.bfloat16(), table, lens)
        meta, _ = self.ops.get_mla_metadata()
        out, lse = self.run_dense(inputs, meta)
        self.check_dense(inputs, out, lse)

    def test_dense_graph_capture_and_replay(self):
        inputs = self.dense_inputs()
        # Warm up code loading, then capture lazy metadata allocation itself.
        meta, _ = self.ops.get_mla_metadata()
        self.run_dense(inputs, meta)
        torch.cuda.synchronize()
        meta, _ = self.ops.get_mla_metadata()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse = self.run_dense(inputs, meta)
        for _ in range(3):
            q = inputs[0]
            q.copy_((torch.randn_like(q.float()) * 0.2).to(q.dtype))
            graph.replay()
            torch.cuda.synchronize()
            self.check_dense(inputs, out, lse)

    def sparse_inputs(self):
        length, heads = 64, 64
        q = (torch.randn(1, 1, heads, 576, device="cuda") * 0.2).bfloat16()
        nope = (torch.randn(length, 512, device="cuda") * 0.2).to(torch.float8_e4m3fn)
        rope = (torch.randn(length, 64, device="cuda") * 0.2).bfloat16()
        scales = torch.tensor([0.5, 2.0, 4.0, 0.25], device="cuda").repeat(length, 1)
        packed = torch.cat(
            [nope.view(torch.uint8), scales.view(torch.uint8), rope.view(torch.uint8)],
            -1,
        ).reshape(1, length, 1, 656)
        indices = torch.arange(length, device="cuda", dtype=torch.int32).view(1, 1, -1)
        keys = torch.cat(
            [nope.float() * scales.repeat_interleave(128, dim=1), rope.float()], -1
        )
        return q, packed, indices, keys

    def test_sparse_prefix_cache_dequantization(self):
        from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
            dequantize_k_cache_paged,
        )

        _, packed, _, keys = self.sparse_inputs()
        # Prefix sharing repeats a cache slot; also check non-monotonic gather.
        slots = torch.tensor([63, 2, 2, 0, 40], device="cuda", dtype=torch.int32)
        result = dequantize_k_cache_paged(packed.view(torch.float8_e4m3fn), slots)
        torch.testing.assert_close(
            result[:, 0], keys[slots.long()].bfloat16(), atol=0, rtol=0
        )

    def test_sparse_packed_fp8_decode(self):
        q, packed, indices, keys = self.sparse_inputs()
        ref, ref_lse = self.reference(q[0], keys)
        for carrier in (torch.uint8, torch.float8_e4m3fn):
            with self.subTest(carrier=carrier):
                meta, _ = self.ops.get_mla_metadata()
                out, lse = self.ops.flash_mla_with_kvcache(
                    q,
                    packed.view(carrier),
                    None,
                    None,
                    512,
                    meta,
                    indices=indices,
                    is_fp8_kvcache=True,
                )
                torch.testing.assert_close(out[0].float(), ref, atol=2e-3, rtol=2e-2)
                torch.testing.assert_close(lse[0], ref_lse, atol=2e-3, rtol=2e-3)

    def test_ragged_bf16_topk_matches_torch_scores(self):
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
            DSATopKBackend,
            TopkTransformMethod,
        )

        scores = torch.randn(2, 4096, device="cuda").bfloat16()
        lengths = torch.tensor([3000, 3500], device="cuda", dtype=torch.int32)
        starts = torch.tensor([0, 37], device="cuda", dtype=torch.int32)
        offsets = torch.tensor([0, 5000], device="cuda", dtype=torch.int32)
        result = DSATopKBackend.SGL_KERNEL.topk_transform(
            scores,
            lengths,
            2048,
            TopkTransformMethod.RAGGED,
            SimpleNamespace(page_table_1=torch.empty(0, device="cuda")),
            topk_indices_offset=offsets,
            row_starts=starts,
        )
        for row in range(2):
            local = result[row].long() - offsets[row].long()
            begin, count = int(starts[row]), int(lengths[row])
            self.assertTrue(bool(((local >= 0) & (local < count)).all()))
            self.assertEqual(local.unique().numel(), 2048)
            actual = scores[row, local + begin].float().sort().values
            expected = (
                scores[row, begin : begin + count]
                .float()
                .topk(2048)
                .values.sort()
                .values
            )
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_sparse_prefill_requires_bf16_input(self):
        q = torch.zeros(1, 64, 576, device="cuda", dtype=torch.bfloat16)
        kv = torch.zeros(64, 1, 576, device="cuda", dtype=torch.bfloat16)
        indices = torch.arange(64, device="cuda", dtype=torch.int32).view(1, 1, 64)
        for fp8_q in (False, True):
            with self.subTest(fp8_q=fp8_q), self.assertRaises(RuntimeError):
                self.ops.flash_mla_sparse_fwd(
                    q.to(torch.float8_e4m3fn) if fp8_q else q,
                    kv if fp8_q else kv.to(torch.float8_e4m3fn),
                    indices,
                    576**-0.5,
                )


if __name__ == "__main__":
    unittest.main()
