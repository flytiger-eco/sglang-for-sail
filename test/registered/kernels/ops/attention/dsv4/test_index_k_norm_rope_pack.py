"""The index-K Triton launch must write the same bytes with the k_norm stage
folded in as the separate k_norm -> RoPE/fq4 -> fp4 pack -> paged store chain.

Coverage is deliberately broad because the fused launch has to stand in for the
chain on every shape the model can produce: any even rope_dim up to 128 (and no
RoPE at all), the two launch geometries the caller can pick, both norm weight
dtypes, page sizes from one slot per page to the pool's own, and input
distributions that sit on the fp4 grid boundaries, at zero, saturated and near
the amax floor.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
    INDEX_K_SLOT_BYTES,
    index_k_rope_pack,
    quantize_fp4_indexer_tensor,
    store_index_k_norm_rope_fp4,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import RMSNorm, _rope_fq4
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 128
ROPE_DIM = 64
EPS = 1e-6
MAX_POS = 4096
# Boundary values of the e2m1 grid and its half-way points, plus zero.
GRID_VALUES = (
    0.0,
    0.25,
    0.75,
    1.25,
    1.75,
    2.5,
    3.5,
    5.0,
    6.0,
    7.0,
    -0.25,
    -0.75,
    -1.25,
    -1.75,
    -2.5,
    -3.5,
    -5.0,
    -6.0,
    -7.0,
)


@unittest.skipIf(is_hip(), "ROCm keeps the split payload/scale layout and the torched chain")
class TestIndexKNormRopePack(CustomTestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.freqs = torch.polar(
            torch.ones(MAX_POS, ROPE_DIM // 2, device="cuda"),
            torch.randn(MAX_POS, ROPE_DIM // 2, device="cuda"),
        )

    def _reference(self, x, weight, positions, rope_dim=ROPE_DIM):
        # The chain takes the per-token freqs already gathered, exactly like the
        # production caller; the fused launch gathers the same entries in-kernel.
        norm = RMSNorm(HEAD_DIM, EPS).cuda()
        with torch.no_grad():
            norm.weight.copy_(weight)
        rotated = _rope_fq4(norm(x), self.freqs[positions], rope_dim)
        return quantize_fp4_indexer_tensor(rotated, rne=True)

    def _assert_same_bytes(self, got, ref, what):
        payload, scale = got
        ref_payload, ref_scale = ref
        self.assertTrue(
            torch.equal(payload, ref_payload),
            f"{what}: {int((payload != ref_payload).sum())} payload bytes differ",
        )
        self.assertTrue(
            torch.equal(scale, ref_scale),
            f"{what}: {int((scale != ref_scale).sum())} scale words differ",
        )

    @staticmethod
    def _make_x(rows, kind):
        if kind == "normal":
            return torch.randn(rows, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        if kind == "grid":
            values = torch.tensor(GRID_VALUES, device="cuda", dtype=torch.float32)
            pick = torch.randint(
                0, values.numel(), (rows * HEAD_DIM,), device="cuda"
            )
            return values[pick].view(rows, HEAD_DIM).to(torch.bfloat16)
        if kind == "zeros":
            return torch.zeros(rows, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        if kind == "tiny":
            # Down to 1e-30: below the fake-quant amax floor but above bf16's min
            # normal, so the floor and the denormal path both get exercised.
            return (
                torch.rand(rows, HEAD_DIM, device="cuda", dtype=torch.float32)
                .pow(4)
                .mul(1e-24)
                .to(torch.bfloat16)
            )
        raise AssertionError(kind)

    def test_packed_bytes_match_unfused_chain(self):
        # Rows span the empty launch, a partial warp, a full warp and beyond the
        # batched quantizer's Blackwell threshold.
        for rows in (0, 1, 7, 33, 512, 4096):
            for rope_dim in (0, 2, 64):
                for num_warps in (1, 4):
                    with self.subTest(rows=rows, rope_dim=rope_dim, num_warps=num_warps):
                        x = self._make_x(rows, "normal")
                        weight = (
                            torch.randn(HEAD_DIM, device="cuda") * 0.5
                        ).to(torch.bfloat16)
                        positions = torch.randint(
                            0, MAX_POS, (rows,), device="cuda", dtype=torch.int64
                        )
                        got = store_index_k_norm_rope_fp4(
                            x,
                            weight,
                            EPS,
                            self.freqs[positions],
                            rope_dim,
                            num_warps=num_warps,
                        )
                        ref = self._reference(x, weight, positions, rope_dim)
                        if rows == 0:
                            self.assertEqual(got[0].numel(), 0)
                            continue
                        self._assert_same_bytes(
                            got, ref, f"rows={rows} rope_dim={rope_dim} w={num_warps}"
                        )

    def test_rotate_whole_row(self):
        # rope_dim == 128 rotates every element, so the alternating real/imag
        # gather covers the whole row instead of its tail half. The freqs table
        # must then be 128/2 complex wide: with a narrower one both sides read
        # past their row.
        rows = 4096
        freqs = torch.polar(
            torch.ones(MAX_POS, HEAD_DIM // 2, device="cuda"),
            torch.randn(MAX_POS, HEAD_DIM // 2, device="cuda"),
        )
        x = self._make_x(rows, "normal")
        weight = (torch.randn(HEAD_DIM, device="cuda") * 0.5).to(torch.bfloat16)
        positions = torch.randint(0, MAX_POS, (rows,), device="cuda", dtype=torch.int64)
        norm = RMSNorm(HEAD_DIM, EPS).cuda()
        with torch.no_grad():
            norm.weight.copy_(weight)
        ref = quantize_fp4_indexer_tensor(
            _rope_fq4(norm(x), freqs[positions], HEAD_DIM), rne=True
        )
        for num_warps in (1, 4):
            with self.subTest(num_warps=num_warps):
                got = store_index_k_norm_rope_fp4(
                    x,
                    weight,
                    EPS,
                    freqs[positions],
                    HEAD_DIM,
                    num_warps=num_warps,
                )
                self._assert_same_bytes(got, ref, f"whole row w={num_warps}")

    def test_weight_dtypes(self):
        rows = 37
        x = self._make_x(rows, "normal")
        positions = torch.randint(0, MAX_POS, (rows,), device="cuda", dtype=torch.int64)
        for weight_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(weight_dtype=weight_dtype):
                weight = (torch.randn(HEAD_DIM, device="cuda") * 0.5).to(
                    weight_dtype
                )
                got = store_index_k_norm_rope_fp4(
                    x, weight, EPS, self.freqs[positions], ROPE_DIM
                )
                ref = self._reference(x, weight, positions)
                self._assert_same_bytes(got, ref, f"{weight_dtype}")

    def test_input_distributions(self):
        # The two quantization stages round differently at their boundaries, so
        # the fused launch has to reproduce the chain on grid-valued, zero and
        # denormal-scale rows as well as on random ones, and with a weight large
        # enough to saturate the e2m1 range after the norm.
        rows = 64
        positions = torch.randint(0, MAX_POS, (rows,), device="cuda", dtype=torch.int64)
        for kind in ("normal", "grid", "zeros", "tiny"):
            for weight_scale in (0.5, 50.0):
                with self.subTest(kind=kind, weight_scale=weight_scale):
                    weight = (
                        torch.randn(HEAD_DIM, device="cuda") * weight_scale
                    ).to(torch.bfloat16)
                    x = self._make_x(rows, kind)
                    got = store_index_k_norm_rope_fp4(
                        x, weight, EPS, self.freqs[positions], ROPE_DIM
                    )
                    ref = self._reference(x, weight, positions)
                    self._assert_same_bytes(got, ref, f"{kind} w*{weight_scale}")

    def test_rope_pack_without_norm_matches_chain(self):
        # The no-norm entry point shares the two quantization stages and the
        # store with the fused one; CUDA's fused-compress path calls it with the
        # norm already applied.
        for rows in (1, 33, 512):
            with self.subTest(rows=rows):
                x = self._make_x(rows, "normal")
                positions = torch.randint(
                    0, MAX_POS, (rows,), device="cuda", dtype=torch.int64
                )
                got = index_k_rope_pack(x, self.freqs[positions], ROPE_DIM)
                ref = quantize_fp4_indexer_tensor(
                    _rope_fq4(x, self.freqs[positions], ROPE_DIM), rne=True
                )
                self._assert_same_bytes(got, ref, f"no norm rows={rows}")

    def test_paged_write_matches_scatter(self):
        rows = 300
        x = self._make_x(rows, "normal")
        weight = (torch.randn(HEAD_DIM, device="cuda") * 0.5).to(torch.bfloat16)
        positions = torch.randint(0, MAX_POS, (rows,), device="cuda", dtype=torch.int64)
        payload, scale = store_index_k_norm_rope_fp4(
            x, weight, EPS, self.freqs[positions], ROPE_DIM
        )

        for page_size in (1, 16, 64):
            with self.subTest(page_size=page_size):
                pages = 4 * rows // page_size + 2
                slots = pages * page_size
                # Slot 0 is the reserved dummy the caller pads with; it is written
                # like any other slot, so it must land where the scatter puts it.
                loc = torch.randperm(slots - 1, device="cuda")[: rows - 1].to(
                    torch.int64
                )
                loc = torch.cat([torch.zeros(1, dtype=torch.int64, device="cuda"), loc + 1])
                cache = torch.zeros(
                    (pages, page_size * INDEX_K_SLOT_BYTES),
                    dtype=torch.uint8,
                    device="cuda",
                )
                store_index_k_norm_rope_fp4(
                    x,
                    weight,
                    EPS,
                    self.freqs[positions],
                    ROPE_DIM,
                    cache=cache,
                    loc=loc,
                )

                page = loc // page_size
                offset = loc % page_size
                expected = torch.zeros_like(cache)
                expected[
                    page[:, None],
                    offset[:, None] * 64 + torch.arange(64, device="cuda")[None, :],
                ] = payload.view(torch.uint8)
                expected[
                    page[:, None],
                    page_size * 64
                    + offset[:, None] * 4
                    + torch.arange(4, device="cuda")[None, :],
                ] = scale.view(torch.uint8).view(rows, 4)
                self.assertTrue(torch.equal(cache, expected))


if __name__ == "__main__":
    unittest.main()
