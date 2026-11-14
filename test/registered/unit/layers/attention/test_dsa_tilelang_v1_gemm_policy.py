"""CPU contract checks for the v1 sparse-MLA specialization's GEMM policy.

The real factory body is executed with a no-op prim_func decorator. No TIR is
lowered and no TileLang/device compiler is imported by these tests.
"""

import ast
import copy
import inspect
import unittest
from collections import Counter
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Policy(IntEnum):
    FullRow = 1
    FullCol = 2


class TestDSATileLangV1GemmPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (
            Path(__file__).resolve().parents[5]
            / "python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py"
        )
        cls.factory_ast = next(
            node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "sparse_attention_fwd_kernel_v1"
        )

    def make_specialization(
        self, *, ppu, heads, dim, tail_dim, block_i=64, threads=256, topk=128
    ):
        factory = copy.deepcopy(self.factory_ast)
        factory.decorator_list = []
        module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
        namespace = {
            "is_ppu": lambda: ppu,
            "tilelang": SimpleNamespace(
                math=SimpleNamespace(
                    next_power_of_2=lambda n: 1 << (n - 1).bit_length()
                ),
                cdiv=lambda a, b: (a + b - 1) // b,
            ),
            "T": SimpleNamespace(
                symbolic=lambda _: 1,
                Tensor=lambda *_: object(),
                prim_func=lambda fn: fn,
                GemmWarpPolicy=_Policy,
            ),
        }
        exec(compile(module, str(__file__), "exec"), namespace)
        return namespace[factory.name](
            heads,
            dim,
            tail_dim,
            topk,
            block_I=block_i,
            threads=threads,
            num_stages=1,
        )

    def test_glm53_small_head_specialization_limits_ppu_threads(self):
        for heads in (8, 16):
            with self.subTest(heads=heads):
                kernel = self.make_specialization(
                    ppu=True, heads=heads, dim=512, tail_dim=0, topk=2048
                )
                self.assertEqual(
                    inspect.getclosurevars(kernel).nonlocals["threads"], 128
                )

    def test_ppu_qk_and_pv_use_the_entire_thread_budget(self):
        cases = (
            (8, 512, 64, 256, 128),
            (16, 256, 64, 512, 128),
            (24, 512, 64, 256, 256),
            (32, 512, 64, 512, 256),
            (64, 512, 64, 256, 256),
            (128, 512, 64, 256, 256),
            (8, 512, 32, 256, 64),
            (8, 64, 128, 256, 128),
            (8, 192, 128, 256, 128),
            (8, 448, 64, 256, 128),
            (8, 192, 48, 96, 96),
        )
        for heads, dim, block_i, requested, expected in cases:
            with self.subTest(heads=heads, dim=dim, block_i=block_i):
                kernel = self.make_specialization(
                    ppu=True,
                    heads=heads,
                    dim=dim,
                    tail_dim=0,
                    block_i=block_i,
                    threads=requested,
                    topk=block_i * 2,
                )
                closure = inspect.getclosurevars(kernel).nonlocals
                self.assertEqual(closure["threads"], expected)
                warps = expected // 32
                m_warps = min(closure["H_per_block"] // 16, warps)
                n_warps = warps // m_warps
                self.assertEqual(m_warps * n_warps, warps)
                self.assertEqual(closure["H_per_block"] % (16 * m_warps), 0)
                self.assertEqual(block_i % (16 * n_warps), 0)
                self.assertEqual(dim % (16 * n_warps), 0)

    def test_non_ppu_keeps_the_requested_thread_count(self):
        for threads in (128, 256, 512):
            kernel = self.make_specialization(
                ppu=False, heads=8, dim=512, tail_dim=0, threads=threads
            )
            self.assertEqual(
                inspect.getclosurevars(kernel).nonlocals["threads"], threads
            )

    def test_ppu_rejects_a_tile_without_an_m16n16_partition(self):
        with self.assertRaisesRegex(ValueError, "m16n16 partition"):
            self.make_specialization(ppu=True, heads=8, dim=512, tail_dim=0, block_i=8)

    def test_ppu_uses_full_row_for_main_and_zero_tail_geometries(self):
        for heads in (16, 32, 64, 128):
            for dim, tail_dim in ((256, 0), (512, 64)):
                with self.subTest(heads=heads, dim=dim, tail_dim=tail_dim):
                    kernel = self.make_specialization(
                        ppu=True, heads=heads, dim=dim, tail_dim=tail_dim
                    )
                    self.assertIs(
                        inspect.getclosurevars(kernel).nonlocals["gemm_policy"],
                        _Policy.FullRow,
                    )

    def test_non_ppu_specializations_keep_full_col(self):
        for heads in (16, 32, 64, 128):
            for dim, tail_dim in ((256, 0), (512, 64)):
                with self.subTest(heads=heads, dim=dim, tail_dim=tail_dim):
                    kernel = self.make_specialization(
                        ppu=False, heads=heads, dim=dim, tail_dim=tail_dim
                    )
                    self.assertIs(
                        inspect.getclosurevars(kernel).nonlocals["gemm_policy"],
                        _Policy.FullCol,
                    )

    def test_both_qk_gemms_and_pv_share_the_specialization_policy(self):
        main = next(
            node
            for node in self.factory_ast.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        gemms = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "T"
            and node.func.attr == "gemm"
        ]
        self.assertEqual(len(gemms), 3)
        self.assertEqual(
            Counter(node.args[2].id for node in gemms), {"acc_s": 2, "acc_o": 1}
        )
        for gemm in gemms:
            keywords = {kw.arg: kw.value for kw in gemm.keywords}
            self.assertIsInstance(keywords["policy"], ast.Name)
            self.assertEqual(keywords["policy"].id, "gemm_policy")
            if gemm.args[2].id == "acc_s":
                self.assertIsInstance(keywords["transpose_B"], ast.Constant)
                self.assertIs(keywords["transpose_B"].value, True)


if __name__ == "__main__":
    unittest.main()
