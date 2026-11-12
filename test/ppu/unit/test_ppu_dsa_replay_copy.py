"""PPU FlashMLA replay skips copying opaque scheduler metadata."""

import unittest
from types import SimpleNamespace

from sglang.srt.hardware_backend.ppu.attention.ppu_dsa_hooks import (
    _ppu_dsa_copy_replay_metadata_from_sibling,
    _ppu_dsa_init_forward_metadata_replay_cuda_graph_from_precomputed,
)


class TestPPUDSAReplayCopy(unittest.TestCase):
    def test_opaque_metadata_is_hidden_from_sibling_copy_and_restored(self):
        opaque = object()
        precomputed = SimpleNamespace(flashmla_metadata=opaque)
        backend = SimpleNamespace(dsa_decode_impl="flashmla_kv")
        source_backend = object()
        forward_mode = SimpleNamespace(is_decode_or_idle=lambda: True)
        calls = []

        def sibling_copy(owner, source, bs, data, mode):
            self.assertIs(owner, backend)
            self.assertIs(source, source_backend)
            self.assertEqual(bs, 2)
            self.assertIs(data, precomputed)
            self.assertIs(mode, forward_mode)
            self.assertIsNone(data.flashmla_metadata)
            calls.append(True)

        result = _ppu_dsa_copy_replay_metadata_from_sibling(
            sibling_copy, backend, source_backend, 2, precomputed, forward_mode
        )
        self.assertIsNone(result)
        self.assertEqual(calls, [True])
        self.assertIs(precomputed.flashmla_metadata, opaque)

    def test_sparse_backend_is_unchanged(self):
        opaque = object()
        precomputed = SimpleNamespace(flashmla_metadata=opaque)
        backend = SimpleNamespace(dsa_decode_impl="flashmla_sparse")
        forward_mode = SimpleNamespace(is_decode_or_idle=lambda: True)

        def sibling_copy(owner, source, bs, data, mode):
            self.assertIs(data.flashmla_metadata, opaque)

        _ppu_dsa_copy_replay_metadata_from_sibling(
            sibling_copy, backend, object(), 1, precomputed, forward_mode
        )
        self.assertIs(precomputed.flashmla_metadata, opaque)

    def test_non_decode_mode_is_unchanged(self):
        opaque = object()
        precomputed = SimpleNamespace(flashmla_metadata=opaque)
        backend = SimpleNamespace(dsa_decode_impl="flashmla_kv")
        forward_mode = SimpleNamespace(is_decode_or_idle=lambda: False)

        def sibling_copy(owner, source, bs, data, mode):
            self.assertIs(data.flashmla_metadata, opaque)

        _ppu_dsa_copy_replay_metadata_from_sibling(
            sibling_copy, backend, object(), 1, precomputed, forward_mode
        )
        self.assertIs(precomputed.flashmla_metadata, opaque)

    def test_absent_flashmla_metadata_is_unchanged(self):
        precomputed = SimpleNamespace(flashmla_metadata=None)
        backend = SimpleNamespace(dsa_decode_impl="flashmla_kv")
        forward_mode = SimpleNamespace(is_decode_or_idle=lambda: True)

        def sibling_copy(owner, source, bs, data, mode):
            self.assertIsNone(data.flashmla_metadata)

        _ppu_dsa_copy_replay_metadata_from_sibling(
            sibling_copy, backend, object(), 1, precomputed, forward_mode
        )
        self.assertIsNone(precomputed.flashmla_metadata)

    def test_ordinary_replay_hook_remains(self):
        opaque = object()
        precomputed = SimpleNamespace(flashmla_metadata=opaque)

        def ordinary_copy(owner, bs, data, mode):
            self.assertIsNone(data.flashmla_metadata)

        _ppu_dsa_init_forward_metadata_replay_cuda_graph_from_precomputed(
            ordinary_copy, object(), 1, precomputed, object()
        )
        self.assertIs(precomputed.flashmla_metadata, opaque)


if __name__ == "__main__":
    unittest.main()
