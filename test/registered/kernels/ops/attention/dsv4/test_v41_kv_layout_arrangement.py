"""Addressing geometry of the V4.1 KV layouts in both arrangements.

The block-split layouts (``V41`` / ``V41_FP4``) place a page's data rows before
its scale rows; the interleaved ones (``V41_NEW`` / ``V41_FP4_NEW``) put each
token's scales right after that token's data, which is what the FlashMLA kernels
used on PPU read. This test pins ``row_addressing`` / ``page_bytes`` / the
arrangement helpers so a writer and a reader can never silently disagree.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import (
    KVLayout,
    is_valid_kv_layout_pair,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 256


class TestKvLayoutArrangement(unittest.TestCase):
    def test_block_split_addressing(self):
        for layout, data, scale, page_align in (
            (KVLayout.V4, 576, 8, 576),
            (KVLayout.V41, 512, 16, 512),
            (KVLayout.V41_FP4, 256, 32, 256),
        ):
            with self.subTest(layout=layout):
                # slot i: data at i * data_bytes, scales in the page-level block.
                self.assertEqual(
                    layout.row_addressing(PAGE), (data, PAGE * data, scale)
                )
                self.assertEqual(layout.page_align, page_align)
                raw = PAGE * (data + scale)
                self.assertEqual(
                    layout.page_bytes(PAGE), -(-raw // page_align) * page_align
                )

    def test_interleaved_addressing(self):
        for layout, bpt, data in (
            (KVLayout.V41_NEW, 528, 512),
            (KVLayout.V41_FP4_NEW, 288, 256),
        ):
            with self.subTest(layout=layout):
                # slot i: data at i * bpt, scales at i * bpt + data_bytes.
                self.assertEqual(layout.row_addressing(PAGE), (bpt, data, bpt))
                self.assertEqual(layout.page_align, bpt)
                # No page padding: the page is page_size contiguous token rows.
                self.assertEqual(layout.page_bytes(PAGE), PAGE * bpt)

    def test_arrangements_share_page_extent(self):
        for split, interleaved in (
            (KVLayout.V41, KVLayout.V41_NEW),
            (KVLayout.V41_FP4, KVLayout.V41_FP4_NEW),
        ):
            with self.subTest(split=split):
                self.assertEqual(split.page_bytes(PAGE), interleaved.page_bytes(PAGE))
                self.assertEqual(split.bytes_per_token, interleaved.bytes_per_token)

    def test_arrangement_helpers(self):
        self.assertFalse(KVLayout.V41.is_interleaved)
        self.assertTrue(KVLayout.V41_NEW.is_interleaved)
        # encoding folds the arrangement away so byte-level code stays shared.
        self.assertIs(KVLayout.V41_NEW.encoding, KVLayout.V41)
        self.assertIs(KVLayout.V41_FP4_NEW.encoding, KVLayout.V41_FP4)
        self.assertIs(KVLayout.V41.encoding, KVLayout.V41)
        # with_interleaved round-trips.
        for layout in (KVLayout.V41, KVLayout.V41_FP4):
            with self.subTest(layout=layout):
                new = layout.with_interleaved(True)
                self.assertTrue(new.is_interleaved)
                self.assertIs(new.encoding, layout)
                self.assertIs(new.with_interleaved(False), layout)

    def test_is_fp4_spans_arrangements(self):
        self.assertTrue(KVLayout.V41_FP4.is_fp4)
        self.assertTrue(KVLayout.V41_FP4_NEW.is_fp4)
        self.assertFalse(KVLayout.V41.is_fp4)
        self.assertFalse(KVLayout.V41_NEW.is_fp4)

    def test_valid_layout_pairs(self):
        # A V4.1 fp8 main cache may pair with a V4.1 fp4 extra cache of the same
        # arrangement, and identical layouts always pair.
        self.assertTrue(is_valid_kv_layout_pair(KVLayout.V41, KVLayout.V41_FP4))
        self.assertTrue(is_valid_kv_layout_pair(KVLayout.V41_NEW, KVLayout.V41_FP4_NEW))
        for layout in (
            KVLayout.V4,
            KVLayout.V41,
            KVLayout.V41_FP4,
            KVLayout.V41_NEW,
            KVLayout.V41_FP4_NEW,
        ):
            with self.subTest(layout=layout):
                self.assertTrue(is_valid_kv_layout_pair(layout, layout))
        # Mixed arrangements would make one geometry read both caches.
        self.assertFalse(is_valid_kv_layout_pair(KVLayout.V41_NEW, KVLayout.V41_FP4))
        self.assertFalse(is_valid_kv_layout_pair(KVLayout.V41, KVLayout.V41_FP4_NEW))
        self.assertFalse(is_valid_kv_layout_pair(KVLayout.V41_NEW, KVLayout.V41))

    def test_parse_and_cpp_name(self):
        self.assertIs(KVLayout.parse("v41_new"), KVLayout.V41_NEW)
        self.assertIs(KVLayout.parse("v41_fp4_new"), KVLayout.V41_FP4_NEW)
        self.assertEqual(KVLayout.V41_NEW.cpp_name, "V41_NEW")

    def test_compressed_cache_follows_main_arrangement(self):
        try:
            from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
                resolve_compressed_kv_layout,
            )
        except Exception as error:  # pragma: no cover - torch-only environments
            self.skipTest(f"memory pool import unavailable: {error!r}")
        for main in (
            KVLayout.V41,
            KVLayout.V41_NEW,
        ):
            with self.subTest(main=main):
                for ratio, fp4 in ((1, True), (2, True)):
                    resolved = resolve_compressed_kv_layout(main, ratio, None)
                    self.assertIs(resolved.is_interleaved, main.is_interleaved)
                    self.assertIs(resolved.is_fp4, fp4)

    def _select_with(self, *, flashmla_supports: bool, mode: str) -> KVLayout:
        """Run select_dsv4_kv_layout() as if on a SAIL sm89 board."""
        import sglang.srt.mem_cache.deepseek_v4_memory_pool as pool
        from sglang.srt.environ import envs

        class _Kernel:
            dsv4_attn_backend = "flashmla"

        class _Exec:
            kernel = _Kernel()

        previous = envs.SGLANG_DSV4_KV_LAYOUT.get()
        envs.SGLANG_DSV4_KV_LAYOUT.set(mode)
        try:
            with (
                mock.patch.object(pool, "get_exec", lambda: _Exec()),
                mock.patch.object(pool, "is_sail", lambda: True),
                mock.patch.object(
                    pool, "get_platform", lambda: SimpleNamespace(device_sm=89)
                ),
                mock.patch.object(pool, "_is_hip", False),
                mock.patch.object(
                    pool,
                    "flashmla_supports_v41_kv_layouts",
                    lambda: flashmla_supports,
                ),
            ):
                return pool.select_dsv4_kv_layout()[0]
        finally:
            envs.SGLANG_DSV4_KV_LAYOUT.set(previous)

    def test_selection_asserts_without_flashmla_v41_support(self):
        try:
            import sglang.srt.mem_cache.deepseek_v4_memory_pool  # noqa: F401
        except Exception as error:  # pragma: no cover - torch-only environments
            self.skipTest(f"memory pool import unavailable: {error!r}")
        # A FlashMLA build without the V4.1 formats must fail at selection time,
        # not on the first request.
        for mode in ("v41", "v41_new", "auto"):
            with self.subTest(mode=mode):
                with self.assertRaises(AssertionError):
                    self._select_with(flashmla_supports=False, mode=mode)
        # And the same board selects the interleaved layout when it is supported.
        self.assertIs(
            self._select_with(flashmla_supports=True, mode="auto"),
            KVLayout.V41_NEW,
        )


if __name__ == "__main__":
    unittest.main()


class TestSelectDsv4KvLayout(unittest.TestCase):
    """Which arrangement a new pool gets, per platform and SGLANG_DSV4_KV_LAYOUT.

    The PPU flash_mla wheel decodes the V4.1 token-interleaved rows natively, so
    SAIL defaults to ``V41_NEW``; CUDA keeps the block-split ``V41`` that pairs
    with the sgl-kernel build, and non-SM100 boards stay on ``V4``.
    """

    def _select(self, mode, *, sail=False, capability=(8, 9), cuda_available=False):
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache import deepseek_v4_memory_pool as pool

        exec_ns = SimpleNamespace(kernel=SimpleNamespace(dsv4_attn_backend="flashmla"))
        with (
            mock.patch.object(envs.SGLANG_DSV4_KV_LAYOUT, "get", lambda: mode),
            mock.patch.object(pool, "get_exec", lambda: exec_ns),
            mock.patch.object(pool, "is_sail", lambda: sail),
            mock.patch.object(
                pool,
                "get_platform",
                lambda: SimpleNamespace(device_sm=capability[0] * 10 + capability[1]),
            ),
            mock.patch.object(pool, "_is_hip", False),
            mock.patch.object(pool, "flashmla_supports_v41_kv_layouts", lambda: True),
            mock.patch.object(torch.cuda, "is_available", lambda: cuda_available),
            mock.patch.object(
                torch.cuda, "get_device_capability", lambda *a: capability
            ),
            mock.patch.object(torch.version, "cuda", "13.0"),
        ):
            return pool.select_dsv4_kv_layout()

    def test_ppu_defaults_to_interleaved(self):
        for mode in ("auto", "v41_new"):
            with self.subTest(mode=mode):
                layout, option = self._select(mode, sail=True)
                self.assertIs(layout, KVLayout.V41_NEW)
                self.assertEqual(option, "auto")

    def test_ppu_explicit_block_split(self):
        layout, _ = self._select("v41", sail=True)
        self.assertIs(layout, KVLayout.V41)

    def test_cuda_sm100_pairs_block_split_by_default(self):
        layout, _ = self._select("auto", capability=(10, 0), cuda_available=True)
        self.assertIs(layout, KVLayout.V41)
        layout, _ = self._select("v41_new", capability=(10, 0), cuda_available=True)
        self.assertIs(layout, KVLayout.V41_NEW)

    def test_small_cuda_boards_stay_on_v4(self):
        layout, option = self._select("auto", capability=(8, 9), cuda_available=True)
        self.assertIs(layout, KVLayout.V4)
        self.assertIsNone(option)

    def test_sm100_asserts_without_cuda(self):
        with self.assertRaises(AssertionError):
            self._select("v41", capability=(8, 0), cuda_available=True)
