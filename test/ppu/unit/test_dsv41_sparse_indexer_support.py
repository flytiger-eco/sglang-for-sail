"""CPU-only sparse-indexer dispatch regression tests.

Execute the actual gate functions with mocked device and DeepGEMM boundaries;
kernel numerics and graph replay still require PPU hardware.
"""

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

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
    def supports(self, *, sail, sm, enabled=True, blackwell=False, missing=None):
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
                "_is_sail": sail,
                "ENABLE_JIT_DEEPGEMM": enabled,
                "DEEPGEMM_BLACKWELL": blackwell,
                "get_device_sm": lambda: sm,
            },
        )
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            return gate()

    def factory(
        self, *, sail, sm, supported=True, topk=16, cp=1, prefill=True, int8=False
    ):
        package = "sglang.srt.layers.attention.dsv4.v41_indexer"
        config_name = "sglang.srt.layers.deep_gemm_wrapper.configurer"
        config = ModuleType(config_name)
        config.DEEPGEMM_PAGED_SPARSE_MQA_LOGITS = supported
        dense = ModuleType(package + ".dense_blocks")
        dense.DenseBlocksBackend = Mock()
        sparse = ModuleType(package + ".sparse_table")
        sparse.SparseTableBackend = Mock()
        runtime = ModuleType("sglang.srt.runtime_context")
        runtime.get_parallel = lambda: SimpleNamespace(attn_cp_size=cp)
        quant = ModuleType("sglang.srt.layers.attention.dsv4.indexer_quant")
        quant.use_dsv41_int8_indexer = lambda: int8
        factory = load_function(
            LAYERS / "attention/dsv4/v41_indexer/__init__.py",
            "make_candidate_indexer",
            {
                "__package__": package,
                "is_sail": lambda: sail,
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
                runtime.__name__: runtime,
                quant.__name__: quant,
            },
        ):
            result = factory(
                token_to_kv_pool=None,
                req_to_token=None,
                page_size=64,
                candidate_topk_blocks=topk,
                candidate_block_size=8,
            )
        return result, sparse.SparseTableBackend, dense.DenseBlocksBackend

    def test_sail_sm89_and_newer_select_sparse_engine(self):
        for sm in (89, 90, 100):
            with self.subTest(sm=sm):
                self.assertTrue(self.supports(sail=True, sm=sm))
                result, constructor, dense = self.factory(sail=True, sm=sm)
                self.assertEqual(
                    result, (constructor.return_value, constructor.return_value)
                )
                constructor.assert_called_once_with(
                    token_to_kv_pool=None,
                    req_to_token=None,
                    page_size=64,
                    candidate_topk_blocks=16,
                    candidate_block_size=8,
                )

    def test_sail_older_devices_keep_fallback(self):
        for sm in (80, 86, 88):
            with self.subTest(sm=sm):
                self.assertFalse(self.supports(sail=True, sm=sm))
                result, constructor, dense = self.factory(sail=True, sm=sm)
                self.assertEqual(result, (dense.return_value, dense.return_value))
                constructor.assert_not_called()

    def test_deepgemm_disabled_or_missing_api_falls_back_to_dense(self):
        self.assertFalse(self.supports(sail=True, sm=89, enabled=False))
        for api in (
            "get_paged_sparse_mqa_logits_metadata",
            "fp8_fp4_paged_sparse_mqa_logits",
        ):
            with self.subTest(api=api):
                self.assertFalse(self.supports(sail=True, sm=89, missing=api))
        result, constructor, dense = self.factory(sail=True, sm=89, supported=False)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        constructor.assert_not_called()

    def test_unloadable_deepgemm_does_not_fail_the_probe(self):
        gate = load_function(
            LAYERS / "deep_gemm_wrapper/configurer.py",
            "_supports_paged_sparse_mqa_logits",
            {
                "_is_sail": True,
                "ENABLE_JIT_DEEPGEMM": True,
                "get_device_sm": lambda: 89,
            },
        )
        original_import = __import__
        for error in (ImportError, OSError, RuntimeError):
            with self.subTest(error=error):

                def import_module(name, *args, **kwargs):
                    if name == "deep_gemm":
                        raise error("unavailable DeepGEMM build")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=import_module):
                    self.assertFalse(gate())

    def test_nvidia_gate_is_preserved(self):
        self.assertFalse(self.supports(sail=False, sm=90))
        result, constructor, dense = self.factory(sail=False, sm=90)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        constructor.assert_not_called()
        self.assertTrue(self.supports(sail=False, sm=100, blackwell=True))
        result, constructor, dense = self.factory(sail=False, sm=100)
        self.assertEqual(result, (constructor.return_value, constructor.return_value))

    def test_disabled_candidate_budget_skips_engine(self):
        result, constructor, dense = self.factory(sail=True, sm=89, topk=0)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        constructor.assert_not_called()

    def test_cp_and_torch_prefill_keep_dense_prefill_sparse_decode(self):
        for cp, prefill in ((2, True), (1, False)):
            with self.subTest(cp=cp, prefill=prefill):
                result, sparse, dense = self.factory(
                    sail=True, sm=89, cp=cp, prefill=prefill
                )
                self.assertEqual(result, (dense.return_value, sparse.return_value))

    def test_int8_format_keeps_dense_for_both_phases(self):
        result, sparse, dense = self.factory(sail=True, sm=89, int8=True)
        self.assertEqual(result, (dense.return_value, dense.return_value))
        sparse.assert_not_called()

    def test_sm90_decode_respects_platform_and_cache_format(self):
        """SAIL SM90 must use its INT8/DeepGEMM scorer, not Hopper's FP4 scorer."""
        scoring_path = LAYERS / "attention/dsv4/v41_indexer/scoring.py"
        for sail, int8, selected in (
            (True, True, "int8"),
            (True, False, "deep_gemm_fp4"),
            (False, False, "hopper_fp4"),
        ):
            with self.subTest(sail=sail, int8=int8):
                tensor = MagicMock(shape=(1, 64 * (132 if int8 else 68)))
                metadata = MagicMock(max_compressed_seq_len=8, compressed_page_size=64)
                metadata.row_chunks.return_value = [(slice(0, 1), Mock())]
                inputs = MagicMock(
                    compress_ratio=4,
                    req_rows=tensor,
                    positions=tensor,
                    paged_metadata=metadata,
                )
                inputs.indexer.use_int8 = int8
                inputs.indexer.queries.return_value = tensor
                inputs.indexer.head_weights.return_value = tensor
                pool = Mock()
                pool.get_index_k_with_scale_buffer.return_value = tensor
                kernels = {
                    name: MagicMock(name=name)
                    for name in ("int8", "deep_gemm_fp4", "hopper_fp4")
                }
                int8_module = ModuleType(
                    "sglang.kernels.ops.attention.dsv4.int8_indexer"
                )
                fp4_module = ModuleType(
                    "sglang.kernels.ops.attention.dsv4.index_logits"
                )
                int8_module.int8_paged_index_logits = kernels["int8"]
                fp4_module.deep_gemm_fp4_paged_mqa_logits = kernels["deep_gemm_fp4"]
                columns = MagicMock()
                columns.__getitem__.return_value.__ge__.return_value = tensor
                namespace = {
                    "torch": MagicMock(
                        arange=Mock(return_value=columns), inf=float("inf")
                    ),
                    "get_platform": lambda: SimpleNamespace(is_sm90=True),
                    "is_sail": lambda: sail,
                    "DecodeScores": SimpleNamespace,
                    "PagedDecodeScores": SimpleNamespace,
                    "plan_topk_v2": Mock(),
                    "fp4_index_logits_paged": kernels["hopper_fp4"],
                    "fp4_index_logits_decode": Mock(),
                    "quantize_index_q": Mock(return_value=(tensor, tensor)),
                    "get_index_k_cache": Mock(return_value=tensor),
                }
                decode = load_function(scoring_path, "decode_score_chunks", namespace)
                with patch.dict(
                    sys.modules,
                    {
                        int8_module.__name__: int8_module,
                        fp4_module.__name__: fp4_module,
                    },
                ):
                    outputs = list(
                        decode(
                            inputs=inputs,
                            token_to_kv_pool=pool,
                            req_to_token=tensor,
                        )
                    )

                kernels[selected].assert_called_once()
                for name, kernel in kernels.items():
                    if name != selected:
                        kernel.assert_not_called()
                namespace["fp4_index_logits_decode"].assert_not_called()
                self.assertEqual(len(outputs), 1)
                self.assertEqual((outputs[0].bs, outputs[0].lmax), (1, 8))
                scores = kernels[selected].return_value
                if sail:
                    scores = scores.__getitem__.return_value
                self.assertIs(outputs[0].scores, scores)


if __name__ == "__main__":
    unittest.main()
