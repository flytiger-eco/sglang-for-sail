"""CPU-only sparse-indexer dispatch regression tests.

Execute the actual gate functions with mocked device and DeepGEMM boundaries;
kernel numerics and graph replay still require PPU hardware.
"""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
LAYERS = ROOT / "python/sglang/srt/layers"


def load_function(path, name, namespace):
    nodes = ast.parse(path.read_text()).body
    function = next(
        n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class TestDSV41SparseIndexerSupport(unittest.TestCase):
    def supports(self, *, ppu, sm, enabled=True, blackwell=False, missing=None):
        deep_gemm = ModuleType("deep_gemm")
        for name in (
            "get_paged_sparse_mqa_logits_metadata",
            "fp8_fp4_paged_sparse_mqa_logits",
        ):
            setattr(deep_gemm, name, None if missing == name else Mock())
        gate = load_function(
            LAYERS / "deep_gemm_wrapper/configurer.py",
            "_supports_paged_sparse_mqa_logits",
            {
                "_is_ppu": ppu,
                "ENABLE_JIT_DEEPGEMM": enabled,
                "DEEPGEMM_BLACKWELL": blackwell,
                "get_device_sm": lambda: sm,
            },
        )
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            return gate()

    def factory(
        self,
        *,
        ppu,
        sm,
        supported=True,
        topk=16,
        cp=1,
        prefill=True,
        varlen=True,
        int8=False,
    ):
        package = "sglang.srt.layers.attention.dsv4.v41_indexer"
        config_name = "sglang.srt.layers.deep_gemm_wrapper.configurer"
        config = ModuleType(config_name)
        config.DEEPGEMM_PAGED_SPARSE_MQA_LOGITS = supported
        config.DEEPGEMM_SPARSE_MQA_LOGITS = varlen
        dense = ModuleType(package + ".dense_blocks")
        dense.DenseBlocksBackend = Mock()
        sparse = ModuleType(package + ".sparse_table")
        sparse.SparseTableBackend = Mock()
        varlen_module = ModuleType(package + ".sparse_varlen")
        varlen_module.VarlenSparsePrefillBackend = Mock()
        runtime = ModuleType("sglang.srt.runtime_context")
        runtime.get_parallel = lambda: SimpleNamespace(attn_cp_size=cp)
        factory = load_function(
            LAYERS / "attention/dsv4/v41_indexer/__init__.py",
            "make_candidate_indexer",
            {
                "__package__": package,
                "is_ppu": lambda: ppu,
                "get_device_sm": lambda: sm,
                "is_sm100_or_newer": lambda: sm >= 100,
                "_use_deep_gemm_prefill": lambda: prefill,
                "use_dsv41_int8_indexer": lambda: int8,
                "logger": Mock(),
            },
        )
        with patch.dict(
            sys.modules,
            {
                config_name: config,
                dense.__name__: dense,
                sparse.__name__: sparse,
                varlen_module.__name__: varlen_module,
                runtime.__name__: runtime,
            },
        ):
            result = factory(
                token_to_kv_pool=None,
                req_to_token=None,
                page_size=64,
                candidate_topk_blocks=topk,
                candidate_block_size=8,
            )
        return (
            result,
            sparse.SparseTableBackend,
            dense.DenseBlocksBackend,
            varlen_module.VarlenSparsePrefillBackend,
        )

    def test_ppu_sm89_and_newer_select_sparse_engine(self):
        for sm in (89, 90, 100):
            with self.subTest(sm=sm):
                self.assertTrue(self.supports(ppu=True, sm=sm))
                result, constructor, _, varlen = self.factory(ppu=True, sm=sm)
                self.assertEqual(
                    result, (varlen.return_value, constructor.return_value)
                )
                constructor.assert_called_once_with(
                    token_to_kv_pool=None,
                    req_to_token=None,
                    page_size=64,
                    candidate_topk_blocks=16,
                    candidate_block_size=8,
                )

    def test_ppu_older_devices_keep_fallback(self):
        for sm in (80, 86, 88):
            with self.subTest(sm=sm):
                self.assertFalse(self.supports(ppu=True, sm=sm))
                result, constructor, dense, _ = self.factory(ppu=True, sm=sm)
                self.assertEqual(result, (dense.return_value, dense.return_value))
                constructor.assert_not_called()

    def test_missing_paged_api_keeps_varlen_prefill(self):
        result, constructor, dense, varlen = self.factory(
            ppu=True, sm=89, supported=False
        )
        self.assertEqual(result, (varlen.return_value, dense.return_value))
        constructor.assert_not_called()

    def test_missing_both_apis_falls_back_to_dense(self):
        result, constructor, dense, varlen = self.factory(
            ppu=True, sm=89, supported=False, varlen=False
        )
        self.assertEqual(result, (dense.return_value, dense.return_value))
        constructor.assert_not_called()

    def test_nvidia_gate_is_preserved(self):
        self.assertFalse(self.supports(ppu=False, sm=90))
        result, constructor, dense, _ = self.factory(ppu=False, sm=90)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        constructor.assert_not_called()
        self.assertTrue(self.supports(ppu=False, sm=100, blackwell=True))
        result, constructor, _, varlen = self.factory(ppu=False, sm=100)
        self.assertEqual(result, (varlen.return_value, constructor.return_value))

    def test_disabled_candidate_budget_skips_engine(self):
        result, constructor, dense, _ = self.factory(ppu=True, sm=89, topk=0)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        constructor.assert_not_called()

    def test_torch_prefill_keeps_dense_prefill_sparse_decode(self):
        result, sparse, dense, _ = self.factory(ppu=True, sm=89, prefill=False)
        self.assertEqual(result, (dense.return_value, sparse.return_value))

    def test_varlen_prefill_backend_is_selected(self):
        for cp in (1, 8):
            with self.subTest(cp=cp):
                result, sparse, dense, varlen = self.factory(ppu=True, sm=89, cp=cp)
                self.assertEqual(result, (varlen.return_value, sparse.return_value))
        varlen.assert_called()
        varlen.assert_called_with(
            token_to_kv_pool=None,
            req_to_token=None,
            candidate_topk_blocks=16,
        )

    def test_missing_varlen_api_falls_back_to_the_paged_table(self):
        result, sparse, dense, varlen = self.factory(
            ppu=True, sm=89, cp=8, varlen=False
        )
        self.assertEqual(result, (sparse.return_value, sparse.return_value))
        varlen.assert_not_called()

    def test_int8_format_keeps_dense_for_both_phases(self):
        result, sparse, dense, _ = self.factory(ppu=True, sm=89, int8=True)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        sparse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
