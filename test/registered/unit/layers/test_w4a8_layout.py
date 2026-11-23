"""Unit tests for the W4A8 packed-int4 -> DeepGEMM rhs conversion.

The expected values come from an independent, byte-by-byte transcription of
``common_preprocess.hpp`` (``PreprocessMode::W4A8_INT8``) at DeepGEMM
``674cfa5``; the vectorised implementation must match it exactly.
"""

import unittest

import torch

from sglang.srt.layers.quantization.w4a8_layout import (
    A8W4_TILE_PERMUTATION_STORED_TO_LOGICAL,
    repack_a8w4_ckpt_to_deepgemm,
    swizzle_a8w4_k32,
    unpack_a8w4_pairs,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _ref_steps_2_3(group):
    """Literal transcription of the preprocess converter-group repack + sign flip."""
    nibbles = [((b & 0x0F), ((b >> 4) & 0x0F)) for b in group]
    a, c, e, g = (nibbles[i][0] for i in range(4))
    b, d, f, h = (nibbles[i][1] for i in range(4))
    packed = [a | (e << 4), b | (f << 4), c | (g << 4), d | (h << 4)]
    return [value ^ 0x88 for value in packed]


def _ref_swizzle(row):
    """Literal transcription of preprocess step 1 for one ``K // 2`` byte row."""
    out = list(row)
    for group in range(len(row) // 16):
        base = group * 16
        for unit in range(8):
            j1, j2 = unit // 4, unit % 4
            dst = 2 * (2 * j2 + j1)
            out[base + dst : base + dst + 2] = row[base + 2 * unit : base + 2 * unit + 2]
    return out


def _tile_marker_roundtrip():
    """A portable K32 group where every 4-element tile is a recognisable byte."""
    # tile u -> two identical bytes so the tile is visible after unpacking
    row = []
    for unit in range(8):
        row.extend([unit, unit])
    return row


class TestA8W4Conversion(unittest.TestCase):
    NUM_EXPERTS = 3
    NUM_CHANNELS = 16
    NUM_K = 64

    def _random_packed(self, dtype=torch.uint8):
        generator = torch.Generator().manual_seed(20261118)
        packed = torch.randint(
            0, 256, (self.NUM_EXPERTS, self.NUM_CHANNELS, self.NUM_K // 2), generator=generator
        ).to(dtype)
        return packed

    def test_steps_2_3_match_byte_level_reference(self):
        packed = self._random_packed()
        converted = repack_a8w4_ckpt_to_deepgemm(packed)
        self.assertEqual(converted.dtype, torch.uint8)
        self.assertEqual(converted.shape, packed.shape)

        expected = packed.tolist()
        for expert in range(self.NUM_EXPERTS):
            for channel in range(self.NUM_CHANNELS):
                row = expected[expert][channel]
                out = []
                for group in range(0, self.NUM_K // 2, 4):
                    out.extend(_ref_steps_2_3(row[group : group + 4]))
                expected[expert][channel] = out
        self.assertEqual(converted.tolist(), expected)

    def test_int8_storage_is_reinterpreted_not_converted(self):
        packed = self._random_packed()
        as_int8 = packed.view(torch.int8)
        self.assertTrue(
            torch.equal(
                repack_a8w4_ckpt_to_deepgemm(as_int8),
                repack_a8w4_ckpt_to_deepgemm(packed),
            )
        )

    def test_input_is_not_modified(self):
        packed = self._random_packed()
        before = packed.clone()
        repack_a8w4_ckpt_to_deepgemm(packed)
        self.assertTrue(torch.equal(packed, before))
        self.assertFalse(packed.data_ptr() == repack_a8w4_ckpt_to_deepgemm(packed).data_ptr())

    def test_chunking_does_not_change_output(self):
        import sglang.srt.layers.quantization.w4a8_layout as layout

        packed = self._random_packed()
        whole = repack_a8w4_ckpt_to_deepgemm(packed)
        saved = layout.A8W4_REPACK_CHUNK_BYTES
        try:
            layout.A8W4_REPACK_CHUNK_BYTES = 1
            chunked = repack_a8w4_ckpt_to_deepgemm(packed)
        finally:
            layout.A8W4_REPACK_CHUNK_BYTES = saved
        self.assertTrue(torch.equal(whole, chunked))

    def test_offset8_encoding_covers_signed_int4_range(self):
        # Two's-complement nibble v -> (v + 8) mod 16 after the sign flip.
        packed = torch.zeros(1, 16, 32, dtype=torch.uint8)
        values = list(range(-8, 8))
        for i, value in enumerate(values):
            packed[0, i, :4] = torch.tensor(
                [(value & 0x0F) | ((value & 0x0F) << 4)] * 4, dtype=torch.uint8
            )
        converted = repack_a8w4_ckpt_to_deepgemm(packed)
        for i, value in enumerate(values):
            expected = (value + 8) & 0x0F
            self.assertEqual(int(converted[0, i, 0]) & 0x0F, expected)
            self.assertEqual(int(converted[0, i, 0]) >> 4, expected)


class TestA8W4Swizzle(unittest.TestCase):
    def test_produces_the_observed_k32_tile_order(self):
        row = _tile_marker_roundtrip()
        swizzled = _ref_swizzle(row)
        # Each stored 2-byte unit reports which logical 4-element tile it holds.
        stored_order = tuple(swizzled[2 * unit] for unit in range(8))
        self.assertEqual(stored_order, A8W4_TILE_PERMUTATION_STORED_TO_LOGICAL)
        # The vectorised implementation agrees with the reference.
        tensor = torch.tensor([row], dtype=torch.uint8)
        vectorised = swizzle_a8w4_k32(tensor)
        self.assertEqual(vectorised[0].tolist(), swizzled)

    def test_rotation_has_order_three(self):
        generator = torch.Generator().manual_seed(7)
        packed = torch.randint(
            0, 256, (2, 8, 16), generator=generator, dtype=torch.uint8
        )
        once = swizzle_a8w4_k32(packed)
        twice = swizzle_a8w4_k32(once)
        thrice = swizzle_a8w4_k32(twice)
        self.assertFalse(torch.equal(once, packed))
        self.assertFalse(torch.equal(twice, packed))
        self.assertTrue(torch.equal(thrice, packed))

    def test_swizzle_preserves_the_k32_group_bytes(self):
        generator = torch.Generator().manual_seed(11)
        packed = torch.randint(
            0, 256, (2, 8, 32), generator=generator, dtype=torch.uint8
        )
        swizzled = swizzle_a8w4_k32(packed)
        self.assertTrue(
            torch.equal(
                packed.reshape(-1).sort().values,
                swizzled.reshape(-1).sort().values,
            )
        )


class TestA8W4VerificationLogic(unittest.TestCase):
    """Locks the expected probe verdict without needing DeepGEMM on the host."""

    def test_probe_separates_portable_from_converted_buffers(self):
        import sys
        import types

        from sglang.srt.layers.quantization import w4a8_layout

        def fake_preprocess(weight):
            # deep_gemm.preprocess_a8w4_weight == step 1 + steps 2+3.
            swizzled = w4a8_layout.swizzle_a8w4_k32(weight)
            groups = swizzled.reshape(-1, 4)
            out = torch.empty_like(groups)
            for index, group in enumerate(groups.tolist()):
                out[index] = torch.tensor(_ref_steps_2_3(group), dtype=torch.uint8)
            return out.reshape(swizzled.shape)

        stub = types.ModuleType("deep_gemm")
        stub.preprocess_a8w4_weight = fake_preprocess
        saved = sys.modules.get("deep_gemm")
        sys.modules["deep_gemm"] = stub
        try:
            converter = torch.Generator().manual_seed(5)
            portable = torch.randint(0, 256, (2, 8, 32), generator=converter, dtype=torch.uint8)
            # A checkpoint sits past step 1, i.e. it is the swizzled form.
            checkpoint = w4a8_layout.swizzle_a8w4_k32(portable)
            report = w4a8_layout._verify_against_deep_gemm(checkpoint)
        finally:
            if saved is None:
                sys.modules.pop("deep_gemm", None)
            else:
                sys.modules["deep_gemm"] = saved

        self.assertTrue(report["steps_2_3_bit_exact"])
        self.assertFalse(report["matches_full_preprocess"])


class TestA8W4Unpack(unittest.TestCase):
    def test_unpack_reads_low_nibble_as_even_index(self):
        packed = torch.zeros(1, 8, 4, dtype=torch.uint8)
        # low nibble 0x1 -> k even, high nibble 0xF -> k odd (-1)
        packed[0, 0, 0] = 0xF1
        values = unpack_a8w4_pairs(packed)
        self.assertEqual(values[0, 0, :2].tolist(), [1, -1])


class TestA8W4ShapeGuards(unittest.TestCase):
    def test_rejects_non_multiple_of_32_k(self):
        packed = torch.zeros(2, 8, 8, dtype=torch.uint8)  # K = 16
        with self.assertRaises(ValueError):
            repack_a8w4_ckpt_to_deepgemm(packed)

    def test_rejects_non_multiple_of_8_n(self):
        packed = torch.zeros(2, 4, 16, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            repack_a8w4_ckpt_to_deepgemm(packed)

    def test_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            repack_a8w4_ckpt_to_deepgemm(torch.zeros(8, 16, dtype=torch.uint8))


if __name__ == "__main__":
    unittest.main()
