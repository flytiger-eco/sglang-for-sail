"""Accuracy of the PPU MegaAttention path of DeepSeek-V4.1 against a CPU fp32 reference.

Covers what the PPU path changes: wq_b rows permuted into the kernel's query
order, the per-token FP8 output of flash_mla_mega_prefill on both the generic
(SWA-only, top-k 128) and the hs64 (compressed, top-k 640) kernels, and wo_a run
as a per-token x per-channel FP8 einsum.
"""

import unittest

import torch

from sglang.srt.utils import is_sail
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import CustomTestCase

register_ppu_ci(est_time=120, suite="stage-b-test-1-gpu-ppu")

HEADS, HEAD_DIM, ROPE_DIM = 64, 512, 64
GROUPS, O_LORA_RANK, Q_LORA_RANK = 8, 1024, 1280
SWA_WINDOW, INDEX_TOPK = 128, 512
TOKENS, N_COMPRESSED, MAX_POS = 256, 8192, 16384
# (layer kind, attention top-k): SWA-only layers and compressed layers reach
# different MegaAttention kernels.
LAYER_KINDS = (("swa", SWA_WINDOW), ("compressed", SWA_WINDOW + INDEX_TOPK))
# Bounds against the fp32 reference: one FP8 e4m3 per-token quantization measures
# ~2.6% relative error, two (FP8 activation x FP8 weight) ~3.8%.
MIN_COS = 0.999
MAX_REL_ONE_FP8 = 3.5e-2
MAX_REL_TWO_FP8 = 5e-2


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().double(), b.flatten().double(), dim=0
    ).item()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _rope_cpu(x: torch.Tensor, freqs: torch.Tensor, inverse: bool) -> torch.Tensor:
    """Interleaved-pair RoPE on the last ROPE_DIM dims; ``freqs`` is [T, ROPE_DIM // 2]."""
    x = x.clone()
    rot = torch.view_as_complex(x[..., -ROPE_DIM:].reshape(*x.shape[:-1], -1, 2))
    rot = rot * (freqs.conj() if inverse else freqs)[:, None]
    x[..., -ROPE_DIM:] = torch.view_as_real(rot).flatten(-2)
    return x


def _attention_cpu(q, kv, indices, lens, positions, sink, freqs_cis, scale):
    """fp32 reference of RoPE(q) -> sparse attention with sink -> inverse RoPE."""
    q, kv = q.float().cpu(), kv.float().cpu()[:, 0]
    indices, lens = indices.cpu()[:, 0], lens.cpu()
    freqs = freqs_cis.cpu()[positions.cpu().long()]
    sink = sink.float().cpu()
    q = _rope_cpu(q, freqs, inverse=False)
    out = torch.empty_like(q)
    for t in range(q.shape[0]):
        k = kv[indices[t, : lens[t]].long()]
        logits = torch.cat([(q[t] @ k.T) * scale, sink[:, None]], dim=-1)
        out[t] = logits.softmax(-1)[:, :-1] @ k
    return _rope_cpu(out, freqs, inverse=True)


def _make_attention_inputs(topk: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n_comp = 0 if topk == SWA_WINDOW else N_COMPRESSED
    q = (torch.randn(TOKENS, HEADS, HEAD_DIM, device="cuda", generator=gen) / 4).to(
        torch.bfloat16
    )
    kv = (
        torch.randn(
            n_comp + SWA_WINDOW + TOKENS, 1, HEAD_DIM, device="cuda", generator=gen
        )
        / 4
    ).to(torch.bfloat16)
    # Workspace rows are [compressed | swa]: each query reads its SWA window plus
    # the indexer's top-k compressed rows.
    swa = (
        n_comp
        + torch.arange(TOKENS, device="cuda")[:, None]
        + torch.arange(1, SWA_WINDOW + 1, device="cuda")[None]
    )
    if topk > SWA_WINDOW:
        compressed = (
            torch.randint(
                0, n_comp, (TOKENS, topk - SWA_WINDOW), device="cuda", generator=gen
            )
            .sort(-1)
            .values
        )
        swa = torch.cat([compressed, swa], dim=-1)
    indices = swa.to(torch.int32).view(TOKENS, 1, topk).contiguous()
    lens = torch.full((TOKENS,), topk, device="cuda", dtype=torch.int32)
    positions = torch.arange(
        MAX_POS - TOKENS, MAX_POS, device="cuda", dtype=torch.int32
    )
    sink = torch.randn(HEADS, device="cuda", generator=gen) * 0.1
    return q, kv, indices, lens, positions, sink


@unittest.skipUnless(
    torch.cuda.is_available() and is_sail(),
    "PPU MegaAttention (FLASHMLA_FUSED_OUTPUT_CHANNELWISE=1 build)",
)
class TestMegaAttentionPpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # As at server startup: the SAIL platform registers the PPU FlashMLA
        # hooks and load_plugins applies them.
        import sglang.srt.platforms  # noqa: F401
        from sglang.srt.plugins import load_plugins

        load_plugins()

        inv = 1.0 / (10000 ** (torch.arange(0, ROPE_DIM, 2).double() / ROPE_DIM))
        angles = torch.arange(MAX_POS).double()[:, None] * inv[None]
        cls.freqs_cis = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)
        # precompute_cos_sin_table's layout: [cos(32) | sin(32)] per position.
        cls.cos_sin = (
            torch.view_as_real(cls.freqs_cis)
            .transpose(1, 2)
            .reshape(MAX_POS, -1)
            .contiguous()
            .cuda()
        )
        cls.scale = HEAD_DIM**-0.5

    def _mega_attention(self, q, kv, indices, lens, positions, sink):
        from sglang.kernels.ops.attention.flash_mla_interface import (
            flash_mla_mega_prefill,
        )
        from sglang.srt.layers.attention.deepseek_v4_mega_backend import pack_query

        out_fp8, out_sf, _, _ = flash_mla_mega_prefill(
            pack_query(q, HEADS),
            kv,
            indices,
            positions,
            self.cos_sin,
            self.scale,
            attn_sink=sink,
            topk_length=lens,
        )
        return out_fp8, out_sf

    def test_wq_b_row_permutation(self):
        from sglang.srt.layers.attention.deepseek_v4_mega_backend import (
            interleave_query_rows,
        )
        from sglang.srt.layers.quantization.w8a8_fp8 import (
            W8A8Fp8Config,
            W8A8Fp8LinearMethod,
        )

        torch.manual_seed(0)
        out_rows = HEADS * HEAD_DIM
        weight = (torch.randn(out_rows, Q_LORA_RANK, device="cuda") / 8).to(
            torch.float8_e4m3fn
        )
        scale = torch.rand(out_rows, 1, device="cuda") * 1e-2 + 1e-3
        x = torch.randn(512, Q_LORA_RANK, device="cuda", dtype=torch.bfloat16)

        method = W8A8Fp8LinearMethod(W8A8Fp8Config(is_checkpoint_fp8_serialized=True))
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale.clone(), requires_grad=False)
        # As MQALayer.setup_mega_attention: rows and their scales together.
        for param in (layer.weight, layer.weight_scale):
            param.data.copy_(interleave_query_rows(param.data, HEADS))
        method.process_weights_after_loading(layer)
        actual = method.apply(layer, x).float().cpu()

        w_cpu = weight.float().cpu() * scale.cpu()
        expected = (x.float().cpu() @ w_cpu.T).view(-1, HEADS, 32, 16)
        # The kernel's query order: [32 tiles][heads][16].
        expected = expected.transpose(1, 2).reshape(-1, out_rows)
        self.assertGreater(_cos(actual, expected), MIN_COS)
        self.assertLess(_rel(actual, expected), MAX_REL_ONE_FP8)

    def test_attention_output(self):
        for kind, topk in LAYER_KINDS:
            with self.subTest(layer=kind):
                q, kv, indices, lens, positions, sink = _make_attention_inputs(
                    topk, seed=1
                )
                out_fp8, out_sf = self._mega_attention(
                    q, kv, indices, lens, positions, sink
                )
                # Dequantize, then undo the [16 tiles][8 heads][32] column order.
                actual = (out_fp8.float() * out_sf).cpu()
                actual = (
                    actual.view(TOKENS, GROUPS, HEAD_DIM // 32, HEADS // GROUPS, 32)
                    .transpose(2, 3)
                    .reshape(TOKENS, HEADS, HEAD_DIM)
                )
                expected = _attention_cpu(
                    q, kv, indices, lens, positions, sink, self.freqs_cis, self.scale
                )
                self.assertGreater(_cos(actual, expected), MIN_COS)
                self.assertLess(_rel(actual, expected), MAX_REL_ONE_FP8)

    def test_wo_a_projection(self):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_quant_fp8,
        )
        from sglang.srt.layers.attention.deepseek_v4_mega_backend import (
            interleave_output_columns,
        )
        from sglang.srt.models.deepseek_v4 import _fp8_wo_a_einsum

        gen = torch.Generator(device="cuda").manual_seed(2)
        wo_a = (
            torch.randn(
                GROUPS * O_LORA_RANK,
                HEADS // GROUPS * HEAD_DIM,
                device="cuda",
                generator=gen,
            )
            / 64
        ).to(torch.bfloat16)
        # As MQALayer.setup_mega_attention on PPU.
        w_fp8, w_scale = sglang_per_token_quant_fp8(wo_a)
        w_fp8 = interleave_output_columns(w_fp8.view(GROUPS, O_LORA_RANK, -1))
        w_scale = w_scale.view(GROUPS, O_LORA_RANK, 1)
        for kind, topk in LAYER_KINDS:
            with self.subTest(layer=kind):
                q, kv, indices, lens, positions, sink = _make_attention_inputs(
                    topk, seed=3
                )
                out_fp8, out_sf = self._mega_attention(
                    q, kv, indices, lens, positions, sink
                )
                actual = (
                    _fp8_wo_a_einsum(
                        out_fp8[:, :GROUPS],
                        out_sf[:, :GROUPS],
                        w_fp8,
                        w_scale,
                        recipe=(1, 1, wo_a.shape[-1]),
                    )
                    .float()
                    .cpu()
                )
                o = _attention_cpu(
                    q, kv, indices, lens, positions, sink, self.freqs_cis, self.scale
                ).view(TOKENS, GROUPS, -1)
                expected = torch.einsum(
                    "tgd,grd->tgr",
                    o,
                    wo_a.float().cpu().view(GROUPS, O_LORA_RANK, -1),
                )
                self.assertGreater(_cos(actual, expected), MIN_COS)
                self.assertLess(_rel(actual, expected), MAX_REL_TWO_FP8)


if __name__ == "__main__":
    unittest.main()
