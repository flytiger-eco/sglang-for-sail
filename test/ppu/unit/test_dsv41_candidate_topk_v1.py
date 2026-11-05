"""CPU dispatch regression: both candidate publisher top-k calls use v1.

Execute the real Python wrapper, block selector, and publisher with mocked
kernel/stream boundaries. No PyTorch or device runtime is needed on this host.
Device numerics and graph replay are covered separately by test_topk_v1.py.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call


def _load_definition(path, name, namespace):
    nodes = ast.parse(path.read_text()).body
    definition = next(node for node in nodes if getattr(node, "name", None) == name)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            definition,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class TestCandidateTopkV1(unittest.TestCase):
    def test_publish_and_block_selection_need_no_v2_metadata(self):
        source = Path(__file__).resolve().parents[3] / "python/sglang"
        candidate_file = (
            source / "srt/layers/attention/dsv4/v41_indexer/sparse_table.py"
        )
        for use_v2 in (False, True):
            for with_raw in (False, True):
                with self.subTest(use_v2=use_v2, with_raw=with_raw):
                    seq_lens = Mock(name="seq_lens")
                    seq_lens.reshape.return_value = seq_lens
                    metadata = SimpleNamespace(
                        compressed_seq_lens=seq_lens,
                        page_table=Mock(name="page_table"),
                        deep_gemm_metadata=Mock(name="deep_gemm_plan"),
                        max_compressed_seq_len=40000,
                        compressed_page_size=64,
                        use_topk_v2=use_v2,
                        # Deliberately omit topk_metadata: v1 must never read it.
                    )
                    kernel = Mock(name="v1_kernel")
                    logits = Mock(name="logits", shape=(2, 40000))
                    keys = logits.new_empty.return_value
                    blocks = Mock(name="blocks")
                    nblocks = Mock(name="nblocks")
                    alt_stream = Mock(name="alt_stream")
                    main_stream = Mock(name="main_stream")
                    empty = Mock(return_value=blocks)
                    namespace = {
                        "torch": SimpleNamespace(
                            int32="int32",
                            empty=empty,
                            cuda=SimpleNamespace(
                                current_stream=Mock(return_value=main_stream),
                                stream=Mock(return_value=MagicMock()),
                                Event=Mock(return_value=Mock(name="ready")),
                            ),
                        ),
                        "is_hip_runtime": lambda: False,
                        "is_xpu": lambda: False,
                        "_jit_topk_v1_module": Mock(
                            return_value=SimpleNamespace(topk_transform=kernel)
                        ),
                        "deep_gemm_fp4_paged_mqa_logits": Mock(return_value=logits),
                        "candidate_row_lens": Mock(return_value=(nblocks, Mock())),
                        "amax8_varlen": Mock(),
                        "sort_candidate_blocks": Mock(return_value=Mock()),
                        "build_sparse_indexer_schedule": Mock(return_value=Mock()),
                        "_SparseTable": SimpleNamespace,
                        "CANDIDATE_BLOCK_SIZE": 8,
                    }
                    _load_definition(
                        source / "kernels/ops/attention/dsv4/topk.py",
                        "topk_transform_paged",
                        namespace,
                    )
                    _load_definition(
                        source / "kernels/ops/attention/dsv4/candidate_blocks.py",
                        "amax_topk_blocks",
                        namespace,
                    )
                    cls = _load_definition(
                        candidate_file, "SparseTableBackend", namespace
                    )
                    publisher = cls.__new__(cls)
                    publisher.topk_blocks = 2048
                    publisher.alt_stream = alt_stream
                    publisher._get_request_ids = Mock(return_value=Mock())
                    inputs = SimpleNamespace(
                        paged_metadata=metadata,
                        q_fp4=Mock(),
                        q_sf=Mock(),
                        k_cache=Mock(),
                        weights=Mock(),
                        request_ids=None,
                        num_rows=2,
                    )
                    pages = Mock(name="page_indices")
                    raw = Mock(name="raw_indices") if with_raw else None
                    namespace["get_deep_gemm_decode_data"] = lambda *args: (
                        SimpleNamespace(
                            q_fp4=Mock(shape=(2, 1, 64, 64)),
                            q_sf=Mock(),
                            k_cache=Mock(),
                            weights=Mock(),
                        )
                    )
                    publisher.token_to_kv_pool = Mock()
                    table = publisher.publish_decode(
                        inputs, SimpleNamespace(page_indices=pages, raw_indices=raw)
                    )

                    self.assertEqual(
                        kernel.call_args_list,
                        [
                            call(logits, seq_lens, metadata.page_table, pages, 64, raw),
                            call(keys, nblocks, None, blocks, 1, None),
                        ],
                    )
                    logits.new_empty.assert_called_once_with(2, 5000)
                    empty.assert_called_once_with(
                        2, 2048, dtype="int32", device=logits.device
                    )
                    namespace["amax8_varlen"].assert_called_once_with(
                        logits, seq_lens, out=keys
                    )
                    self.assertIs(table.blocks, blocks)
                    table.ready.record.assert_called_once_with(alt_stream)
                    logits.record_stream.assert_called_once_with(alt_stream)

    def test_chunked_publish_uses_v1_without_topk_plans(self):
        source = Path(__file__).resolve().parents[3] / "python/sglang"
        candidate_file = (
            source / "srt/layers/attention/dsv4/v41_indexer/sparse_table.py"
        )
        for with_raw in (False, True):
            with self.subTest(with_raw=with_raw):
                lens = MagicMock()
                tensor = MagicMock(shape=(2, 1, 64, 64))
                chunks = [(slice(0, 1), Mock()), (slice(1, 2), Mock())]
                metadata = SimpleNamespace(
                    compressed_seq_lens=lens,
                    deep_gemm_metadata=[plan for _, plan in chunks],
                    page_table=MagicMock(),
                    compressed_page_size=64,
                    max_compressed_seq_len=40000,
                    row_chunks=lambda: chunks,
                    # No topk_metadata_chunks: the PPU v1 path needs no plan.
                )
                data = SimpleNamespace(
                    q_fp4=tensor, q_sf=tensor, k_cache=tensor, weights=tensor
                )
                kernel = Mock()
                namespace = {
                    "torch": SimpleNamespace(
                        cat=Mock(return_value=tensor),
                        bfloat16="bf16",
                        cuda=SimpleNamespace(current_stream=Mock(), Event=Mock()),
                    ),
                    "get_deep_gemm_decode_data": lambda *args: data,
                    "deep_gemm_fp4_paged_mqa_logits": Mock(return_value=tensor),
                    "topk_transform_paged": kernel,
                    "candidate_row_lens": Mock(return_value=(tensor, tensor)),
                    "amax_topk_blocks": Mock(return_value=tensor),
                    "sort_candidate_blocks": Mock(return_value=tensor),
                    "build_sparse_indexer_schedule": Mock(return_value=tensor),
                    "_SparseTable": SimpleNamespace,
                }
                cls = _load_definition(candidate_file, "SparseTableBackend", namespace)
                publisher = cls.__new__(cls)
                publisher.token_to_kv_pool = Mock()
                publisher.topk_blocks = 2048
                publisher._get_request_ids = Mock()
                out = SimpleNamespace(
                    page_indices=MagicMock(),
                    raw_indices=MagicMock() if with_raw else None,
                )
                publisher.publish_decode(SimpleNamespace(paged_metadata=metadata), out)
                self.assertEqual(kernel.call_count, 2)
                for (rows, _), launch in zip(chunks, kernel.call_args_list):
                    self.assertEqual(
                        launch.args,
                        (
                            tensor,
                            lens.reshape.return_value[rows],
                            metadata.page_table[rows],
                            out.page_indices[rows],
                            64,
                        ),
                    )
                    self.assertEqual(
                        launch.kwargs,
                        {
                            "out_raw_indices": out.raw_indices[rows]
                            if with_raw
                            else None
                        },
                    )
                self.assertEqual(namespace["amax_topk_blocks"].call_count, 2)


if __name__ == "__main__":
    unittest.main()
