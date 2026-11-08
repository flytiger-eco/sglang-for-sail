import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4 import ppu_logits_workspace as module
from sglang.srt.layers.attention.dsv4.ppu_logits_workspace import LogitsWorkspace
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLogitsWorkspace(unittest.TestCase):
    def test_mask_fixed_storage_exact_values_and_lease_limits(self):
        arena = LogitsWorkspace(
            device="cpu", capacity_bytes=8 * 1024 * 4, preallocate_mask=True
        )
        arena.initialize()
        pointer = arena.mask_storage.data_ptr()
        self.assertEqual(arena.mask_storage.numel(), arena.capacity_bytes // 4)
        with self.assertRaisesRegex(RuntimeError, "live logits lease"):
            arena.causal_mask(1, 8)
        with patch.object(torch, "empty", side_effect=AssertionError("no allocation")):
            for rows, width in [(1, 8), (5, 24), (8, 1024), (3, 700)]:
                with arena.lease(rows, width):
                    mask = arena.causal_mask(rows, width)
                    self.assertEqual(mask.data_ptr(), pointer)
                    self.assertEqual(mask.stride(), (width, 1))
                    columns = torch.arange(width)[None, :]
                    lens = torch.arange(rows)[:, None] * width // max(rows - 1, 1)
                    torch.ge(columns, lens, out=mask)
                    self.assertTrue(torch.equal(mask, columns >= lens))
                    with self.assertRaisesRegex(ValueError, "active logits tile"):
                        arena.causal_mask(9, 2048)

    def test_mask_flag_requires_logits_and_startup_keeps_fallback(self):
        with patch.object(
            module.envs.SGLANG_PPU_INDEXER_PREALLOCATE_MASK, "get", return_value=True
        ):
            with patch.object(
                module.envs.SGLANG_PPU_INDEXER_PREALLOCATE_LOGITS,
                "get",
                return_value=False,
            ):
                with self.assertRaisesRegex(ValueError, "requires logits"):
                    module.register_logits_workspace(torch.device("cuda:0"))
            with patch.object(module, "_serving", False):
                self.assertIsNone(
                    module.causal_mask_workspace(device="cuda:0", rows=4, width=8)
                )

    def test_fixed_storage_current_stride_and_no_reallocation(self):
        workspace = LogitsWorkspace(device="cpu", capacity_bytes=8 * 1024 * 4)
        workspace.initialize()
        pointer = workspace.storage.data_ptr()
        with patch.object(
            torch, "empty", side_effect=AssertionError("no allocation after init")
        ):
            workspace.initialize()
            for rows, width in [(1, 19), (5, 257), (8, 1024), (3, 700)]:
                with workspace.lease(rows, width) as out:
                    shape = ((rows + 3) // 4 * 4, (width + 255) // 256 * 256)
                    self.assertEqual(tuple(out.shape), shape)
                    self.assertEqual(out.stride(), (shape[1], 1))
                    self.assertEqual(out.data_ptr(), pointer)
                    out.fill_(rows)
                    self.assertTrue(torch.all(out == rows))

    def test_lease_spans_yield_and_releases_on_close_or_exception(self):
        workspace = LogitsWorkspace(device="cpu", capacity_bytes=4096)
        workspace.initialize()

        def producer():
            with workspace.lease(4, 256) as out:
                yield out

        generator = producer()
        next(generator)
        with self.assertRaisesRegex(RuntimeError, "Nested"):
            with workspace.lease(4, 256):
                pass
        generator.close()
        with self.assertRaisesRegex(ValueError, "consumer error"):
            with workspace.lease(4, 256):
                raise ValueError("consumer error")
        with workspace.lease(4, 256) as out:
            self.assertEqual(out.data_ptr(), workspace.storage.data_ptr())

    def test_rejects_uninitialized_or_oversize_without_fallback(self):
        workspace = LogitsWorkspace(device="cpu", capacity_bytes=4096)
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            with workspace.lease(4, 256):
                pass
        workspace.initialize()
        with patch.object(
            torch, "empty", side_effect=AssertionError("no fallback allocation")
        ):
            with self.assertRaisesRegex(ValueError, "capacity"):
                with workspace.lease(5, 257):
                    pass

    def test_startup_fallback_and_strict_serving_registration(self):
        with patch.object(
            module.envs.SGLANG_PPU_INDEXER_PREALLOCATE_LOGITS, "get", return_value=True
        ):
            with patch.object(module, "_serving", False):
                with module.logits_workspace(device="cuda:0", rows=4, width=256) as out:
                    self.assertIsNone(out)
            with (
                patch.object(module, "_serving", True),
                patch.object(module, "_workspaces", {}),
            ):
                with self.assertRaisesRegex(RuntimeError, "No preallocated"):
                    with module.logits_workspace(device="cuda:0", rows=4, width=256):
                        pass

    def test_flat_logits_producer_passes_out_and_holds_tile_lease(self):
        from sglang.kernels.ops.attention.dsv4 import index_logits

        arena = LogitsWorkspace(device="cpu", capacity_bytes=1024 * 1024)
        arena.initialize()
        calls = []

        def produce(q, kv, weights, starts, ends, clean, width, *, out):
            self.assertEqual(out.data_ptr(), arena.storage.data_ptr())
            self.assertEqual(out.stride(0), (width + 255) // 256 * 256)
            calls.append(tuple(out.shape))
            out.fill_(len(calls))
            return out[: q[0].shape[0], :width]

        q = (
            torch.zeros(8, 32, 64, dtype=torch.int8),
            torch.zeros(8, 32, dtype=torch.int32),
        )
        with (
            patch.dict(
                "sys.modules",
                {"deep_gemm": SimpleNamespace(fp8_fp4_mqa_logits=produce)},
            ),
            patch.object(index_logits, "logits_out_function", return_value=produce),
            patch.object(index_logits, "is_ppu", return_value=True),
            patch.object(
                module.envs.SGLANG_PPU_INDEXER_PREALLOCATE_LOGITS,
                "get",
                return_value=True,
            ),
            patch.object(
                module.envs.SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB, "get", return_value=1
            ),
            patch.object(module, "_serving", True),
            patch.object(module, "_workspaces", {torch.device("cpu"): arena}),
        ):
            generator = index_logits.flat_index_logits_tiles(
                q=q,
                kv=(None, None),
                weights=torch.zeros(8, 32),
                starts=torch.zeros(8, dtype=torch.int32),
                lengths=torch.full((8,), 50000, dtype=torch.int32),
                context_lengths=[50000],
                budget_bytes=1024 * 1024,
                width_align=8,
            )
            tile, scores = next(generator)
            self.assertEqual((tile.start, tile.stop), (0, 4))
            self.assertTrue(torch.all(scores == 1))
            with self.assertRaisesRegex(RuntimeError, "Nested"):
                with arena.lease(4, 50000):
                    pass
            del scores
            tile, scores = next(generator)
            self.assertEqual((tile.start, tile.stop), (4, 8))
            self.assertTrue(torch.all(scores == 2))
            generator.close()
            with arena.lease(4, 50000):
                pass
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
