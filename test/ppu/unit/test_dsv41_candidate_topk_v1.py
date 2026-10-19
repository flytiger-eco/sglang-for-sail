"""CPU dispatch regression for the target's DecodeInputs-owned output buffers.

Execute the real Python wrapper, block selector, and publisher with mocked
kernel/stream boundaries. No PyTorch or device runtime is needed on this host.
Device numerics and graph replay are covered separately by test_topk_v1.py.
"""

import ast
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang"
CANDIDATE_FILE = SOURCE / "srt/layers/attention/dsv4/v41_indexer/sparse_table.py"


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


class _Rows:
    """Small row-view buffer: slices share their output rows with the parent."""

    dtype = "packed"
    device = "device"

    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]) if data else 0)

    def __getitem__(self, rows):
        return _Rows(self.data[rows])

    def reshape(self, *_):
        return self


class TestCandidateTopkV1(unittest.TestCase):
    def test_publish_and_block_selection_need_no_v2_metadata(self):
        for use_v2 in (False, True):
            with self.subTest(use_v2=use_v2):
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
                    SOURCE / "kernels/ops/attention/dsv4/topk.py",
                    "topk_transform_paged",
                    namespace,
                )
                _load_definition(
                    SOURCE / "kernels/ops/attention/dsv4/candidate_blocks.py",
                    "amax_topk_blocks",
                    namespace,
                )
                cls = _load_definition(CANDIDATE_FILE, "SparseTableBackend", namespace)
                publisher = cls.__new__(cls)
                publisher.topk_blocks = 2048
                publisher.alt_stream = alt_stream
                publisher._get_request_ids = Mock(return_value=Mock())
                publisher.token_to_kv_pool = Mock()
                pages = Mock(name="page_indices")
                inputs = SimpleNamespace(
                    paged_metadata=metadata,
                    out_page_indices=pages,
                    # DecodeInputs has no raw-index output or separate out object.
                )
                namespace["get_deep_gemm_decode_data"] = lambda *args: SimpleNamespace(
                    q_fp4=Mock(shape=(2, 1, 64, 64)),
                    q_sf=Mock(),
                    k_cache=Mock(),
                    weights=Mock(),
                )
                table = publisher.publish_decode(inputs)
                self.assertEqual(
                    kernel.call_args_list,
                    [
                        call(logits, seq_lens, metadata.page_table, pages, 64, None),
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
                alt_stream.wait_stream.assert_called_once_with(main_stream)
                for tensor in (
                    table.blocks,
                    table.schedule,
                    table.phys_blocks,
                    table.valid_lens,
                ):
                    tensor.record_stream.assert_called_once_with(main_stream)

    def test_chunked_publish_keeps_local_rows_and_releases_logits(self):
        # Five rank-local rows, not the global CP batch. Include a short last chunk.
        tensor = _Rows([[i] for i in range(5)])
        pages = _Rows([[-99] for _ in range(5)])
        chunks = [
            (slice(0, 2), object()),
            (slice(2, 4), object()),
            (slice(4, 5), object()),
        ]
        metadata = SimpleNamespace(
            compressed_seq_lens=tensor,
            deep_gemm_metadata=[plan for _, plan in chunks],
            page_table=tensor,
            compressed_page_size=64,
            max_compressed_seq_len=40000,
            row_chunks=lambda: chunks,
            # No topk_metadata_chunks: the SAIL v1 path needs no plan.
        )
        data = SimpleNamespace(
            q_fp4=tensor, q_sf=tensor, k_cache=tensor, weights=tensor
        )
        scored = []
        last_logits = None

        def score(q, cache, weights, lens, table, plan, width):
            nonlocal last_logits
            if last_logits is not None:
                self.assertIsNone(last_logits(), "previous chunk's logits stayed live")
            rows, expected_plan = chunks[len(scored)]
            self.assertIs(plan, expected_plan)
            for operand in (*q, weights, lens, table):
                self.assertEqual(operand.data, tensor.data[rows])
            self.assertIs(cache, tensor)
            self.assertEqual(width, 40000)
            scored.append(len(q[0].data))
            logits = _Rows([row.copy() for row in q[0].data])
            last_logits = weakref.ref(logits)
            return logits

        def topk(logits, lens, table, out, page_size):
            self.assertEqual(lens.data, logits.data)
            self.assertEqual(table.data, logits.data)
            self.assertEqual(page_size, 64)
            for src, dst in zip(logits.data, out.data):
                dst[:] = src

        def concatenate(parts):
            return _Rows([row for part in parts for row in part.data])

        main_stream = Mock()
        namespace = {
            "torch": SimpleNamespace(
                cat=concatenate,
                cuda=SimpleNamespace(
                    current_stream=lambda: main_stream,
                    Event=Mock(),
                ),
            ),
            "get_deep_gemm_decode_data": lambda *args: data,
            "deep_gemm_fp4_paged_mqa_logits": score,
            "topk_transform_paged": topk,
            "candidate_row_lens": lambda lens, k: (lens, lens),
            "amax_topk_blocks": lambda logits, *args: _Rows(
                [row.copy() for row in logits.data]
            ),
            "sort_candidate_blocks": lambda blocks, *args: blocks,
            "build_sparse_indexer_schedule": lambda *args: object(),
            "_SparseTable": SimpleNamespace,
        }
        cls = _load_definition(CANDIDATE_FILE, "SparseTableBackend", namespace)
        publisher = cls.__new__(cls)
        publisher.token_to_kv_pool = object()
        publisher.topk_blocks = 2048
        publisher._get_request_ids = lambda inputs, rows, device: tensor
        # No alt_stream: the bounded path must stay on the main stream.
        inputs = SimpleNamespace(paged_metadata=metadata, out_page_indices=pages)
        table = publisher.publish_decode(inputs)
        self.assertEqual(scored, [2, 2, 1])
        self.assertEqual(pages.data, tensor.data)
        self.assertEqual(table.blocks.data, tensor.data)
        self.assertEqual(table.phys_blocks.data, tensor.data)
        self.assertEqual(table.valid_lens.data, tensor.data)
        self.assertIsNone(last_logits())
        table.ready.record.assert_called_once_with(main_stream)


if __name__ == "__main__":
    unittest.main()
