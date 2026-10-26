"""CPU API-contract checks and opt-in PPU numerical checks.

Run directly with Python so CPU contract checks need no SGLang GPU extensions.
On a configured PPU host the same command also compares PLA with Triton.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import torch

_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "python/sglang/srt/layers/attention/linear/kernels/gdn_pla.py"
)
_SPEC = importlib.util.spec_from_file_location("gdn_pla_adapter", _SOURCE)
adapter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adapter)


class TestPLAContract(unittest.TestCase):
    def test_indexed_state_and_return_contract(self):
        for dtype in (torch.float32, torch.bfloat16):
            for strided in (False, True):
                with self.subTest(dtype=dtype, strided=strided):
                    self._check_contract(dtype, strided)

    def _check_contract(self, dtype, strided):
        # Rectangular states catch accidental use of the legacy K,V layout.
        backing = torch.arange(5 * 2 * 4 * 3 * 2).reshape(5, 2, 4, 3, 2)
        state = backing[..., 0].to(dtype).contiguous()
        if strided:
            storage = torch.zeros(5, 2, 4, 3, 2, dtype=dtype)
            storage[..., 0] = state
            state = storage[..., 0]
        before = state.clone()
        indices = torch.tensor([3, 1], dtype=torch.int64)
        cu = torch.tensor([0, 3, 5], dtype=torch.int64)
        q = torch.randn(1, 5, 2, 6)[..., ::2]
        g = torch.randn(1, 5, 2)
        output = torch.randn(1, 5, 2, 4)

        def forward(**kw):
            self.assertEqual(kw["raw_cu_seqlens"], (0, 3, 5))
            self.assertTrue(kw["normalize_qk"])
            self.assertFalse(kw["output_final_state"])
            self.assertEqual(kw["final_state_dtype"], dtype)
            self.assertEqual(kw["initial_state"].dtype, dtype)
            self.assertEqual(tuple(kw["initial_state"].shape[-2:]), (4, 3))
            for name in ("q", "k", "v", "g", "beta", "initial_state"):
                self.assertTrue(kw[name].is_contiguous(), name)
            for name in ("initial_state_indices", "cu_seqlens"):
                self.assertEqual(kw[name].dtype, torch.int32)
            self.assertTrue(torch.equal(kw["q"], q))
            self.assertTrue(
                torch.equal(
                    kw["initial_state"][kw["initial_state_indices"]], before[indices]
                )
            )
            kw["initial_state"][kw["initial_state_indices"]] += 2
            return object(), object(), output, None, None

        pla = ModuleType("pla")
        prefill = ModuleType("pla.prefill")
        flashqla = ModuleType("pla.prefill.flashqla")
        flashqla.chunk_gated_delta_rule_fwd = Mock(side_effect=forward)
        with patch.dict(
            sys.modules,
            {"pla": pla, "pla.prefill": prefill, "pla.prefill.flashqla": flashqla},
        ):
            result = adapter.pla_gdn_prefill(
                q,
                q,
                q,
                g,
                g,
                ssm_states=state,
                cache_indices=indices,
                query_start_loc=cu,
                seq_lens_cpu=[3, 2],
            )
        self.assertIs(result[0], output)
        self.assertEqual(result[1:], (None, None))
        expected = before.clone()
        expected[indices] += 2
        self.assertTrue(torch.equal(state, expected))


@unittest.skipUnless(torch.cuda.is_available(), "Requires a configured PPU host")
class TestPPUPrefill(unittest.TestCase):
    def test_switch_outputs_and_state(self):
        from sglang.srt.environ import envs
        from sglang.srt.layers.attention.linear.kernels import gdn_triton
        from sglang.srt.utils import is_ppu

        if not is_ppu():
            self.skipTest("Requires PPU")
        torch.manual_seed(42)
        kernel = gdn_triton.TritonGDNKernel()
        device = "cuda"
        lengths = [65, 37]
        total = sum(lengths)
        q, k = [
            torch.randn(1, total, 2, 128, device=device, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        v = torch.randn(1, total, 4, 128, device=device, dtype=torch.bfloat16)
        g = -torch.rand(1, total, 4, device=device)
        beta = torch.rand_like(g)
        indices = torch.tensor([3, 1], device=device, dtype=torch.int64)
        cu = torch.tensor([0, 65, total], device=device, dtype=torch.int64)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                initial = torch.randn(5, 4, 128, 128, device=device, dtype=dtype) * 0.05
                pools = [initial.clone(), initial.clone()]
                results = []
                for enabled, pool in zip((False, True), pools):
                    with patch.object(
                        envs.SGLANG_SAIL_PLA_CUDA, "get", return_value=enabled
                    ):
                        results.append(
                            kernel.extend(
                                q, k, v, g, beta,
                                ssm_states=pool,
                                cache_indices=indices,
                                query_start_loc=cu,
                                seq_lens_cpu=lengths,
                            )
                        )
                torch.testing.assert_close(
                    results[0][0], results[1][0], atol=2e-2, rtol=1e-2
                )
                torch.testing.assert_close(pools[0], pools[1], atol=2e-2, rtol=1e-2)
                torch.testing.assert_close(
                    pools[1][[0, 2, 4]], initial[[0, 2, 4]], atol=0, rtol=0
                )
                self.assertEqual(results[1][1:], (None, None))
                # Checkpoint batches must retain Triton's actual intermediate h.
                with patch.object(envs.SGLANG_SAIL_PLA_CUDA, "get", return_value=True):
                    with patch.object(
                        gdn_triton,
                        "chunk_gated_delta_rule",
                        return_value=(None, None, "h"),
                    ) as triton:
                        result = kernel.extend(
                            q, k, v, g, beta,
                            ssm_states=initial.clone(),
                            cache_indices=indices,
                            query_start_loc=cu,
                            requires_intermediate_states=True,
                        )
                        self.assertEqual(result[2], "h")
                        triton.assert_called_once()


if __name__ == "__main__":
    unittest.main()
