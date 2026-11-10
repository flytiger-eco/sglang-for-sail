"""The PPU flash_mla wrapper must pass the V4.1 interleaved layouts through.

The PPU flash_mla wheel (2.1.0) follows the upstream API: it has no ``use_legacy``
switch and decodes the V4.1 rows (512 B E4M3 + 16 B ue8m0 per token, i.e. the
interleaved ``V41_NEW`` arrangement) natively. ``use_legacy=False`` therefore has
to reach the wheel instead of being rejected by the wrapper.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.hardware_backend.ppu.attention import flash_mla as wrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _fake_flash_mla(calls):
    def with_kvcache(*args, **kwargs):
        calls["kvcache"] = (args, kwargs)
        return torch.zeros(1), torch.zeros(1)

    def sparse_fwd(*args, **kwargs):
        calls["sparse"] = (args, kwargs)
        return torch.zeros(1), torch.zeros(1), torch.zeros(1)

    return SimpleNamespace(
        flash_mla_with_kvcache=with_kvcache, flash_mla_sparse_fwd=sparse_fwd
    )


class TestPpuFlashMlaWrapper(unittest.TestCase):
    def test_has_no_legacy_only_guard(self):
        self.assertFalse(hasattr(wrapper, "_require_legacy_kernels"))

    def test_kvcache_accepts_interleaved_layouts(self):
        calls = {}
        stub = SimpleNamespace(
            flash_mla_with_kvcache=_fake_flash_mla(calls).flash_mla_with_kvcache
        )
        with (
            mock.patch.object(wrapper, "_flashmla", stub),
            mock.patch.object(wrapper, "_import_error", None),
        ):
            out, lse = wrapper.flash_mla_with_kvcache(
                q=torch.zeros((1, 1, 2, 64), dtype=torch.float32),
                k_cache=torch.zeros((1, 1, 1, 64), dtype=torch.bfloat16),
                block_table=None,
                cache_seqlens=None,
                head_dim_v=512,
                tile_scheduler_metadata=object(),
                use_legacy=False,
            )
        self.assertIsNotNone(out)
        # The switch is wrapper-only: the wheel never sees it.
        self.assertNotIn("use_legacy", calls["kvcache"][1])

    def test_sparse_fwd_accepts_interleaved_layouts(self):
        calls = {}
        stub = SimpleNamespace(
            flash_mla_sparse_fwd=_fake_flash_mla(calls).flash_mla_sparse_fwd
        )
        with (
            mock.patch.object(wrapper, "_flashmla", stub),
            mock.patch.object(wrapper, "_import_error", None),
        ):
            wrapper.flash_mla_sparse_fwd(
                q=torch.zeros((1, 2, 64), dtype=torch.float32),
                kv=torch.zeros((1, 1, 64), dtype=torch.bfloat16),
                indices=torch.zeros((1, 1, 1), dtype=torch.int32),
                sm_scale=0.1,
                use_legacy=False,
            )
        self.assertNotIn("use_legacy", calls["sparse"][1])


if __name__ == "__main__":
    unittest.main()
