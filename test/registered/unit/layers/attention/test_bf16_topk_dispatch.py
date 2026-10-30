import ast
import importlib.util
import os
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


def _find_repo_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "python/sglang/srt/layers/attention/dsa").is_dir():
            return parent
    raise RuntimeError(f"Cannot locate the sglang repository root from {__file__}")


def _load_ci_register(root):
    # ci_register.py is stdlib-only; loading it by path keeps this test free of
    # the sglang package import (and its runtime dependencies) on CPU hosts.
    name = "_bf16_topk_dispatch_ci_register"
    spec = importlib.util.spec_from_file_location(
        name, root / "python/sglang/test/ci/ci_register.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TEST_REPO = _find_repo_root()
_ci_register = _load_ci_register(TEST_REPO)
register_cpu_ci = _ci_register.register_cpu_ci
register_ppu_ci = _ci_register.register_ppu_ci

# Hardware-free source-level dispatch checks, so the CPU suite owns them. Also
# registered on the PPU per-commit chain: PR CI for this repository dispatches
# only pr-test-ppu.yml, and this file is the only guard for the BF16/v2 top-k
# dispatch order that the PPU kernel tests rely on.
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_ppu_ci(est_time=10, suite="stage-b-test-1-gpu-ppu")

# Production sources are parsed from this checkout by default. The optional
# override points the same checks at another tree (e.g. a BASE checkout in a
# manual A/B comparison); it is never required.
REPO = Path(os.environ.get("SGLANG_MIGRATION_REPO") or TEST_REPO)
DSA = REPO / "python/sglang/srt/layers/attention/dsa"


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    if len(nodes) != len(names):
        raise AssertionError(f"Missing production definitions: {names}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            )
        ]
        + nodes,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


def flag(value):
    return SimpleNamespace(get=lambda: value)


class BF16TopKDispatchTest(unittest.TestCase):
    def setUp(self):
        self.envs = SimpleNamespace(
            SGLANG_DSA_FUSE_TOPK=flag(True),
            SGLANG_DSA_USE_DSV4_BF16_TOPK=flag(True),
            SGLANG_OPT_USE_TOPK_V2=flag(True),
        )
        self.ns = load_definitions(
            DSA / "dsa_topk_backend.py",
            {"TopkTransformMethod", "DSATopKBackend", "_build_packed_row_to_batch"},
            dict(
                torch=torch,
                Enum=Enum,
                IntEnum=IntEnum,
                auto=auto,
                envs=self.envs,
                _is_xpu=False,
            ),
        )
        self.backend = self.ns["DSATopKBackend"].SGL_KERNEL
        self.method = self.ns["TopkTransformMethod"]
        self.kernel = Mock()
        self.legacy = SimpleNamespace(
            fast_topk_v2=Mock(return_value="unfused"),
            fast_topk_transform_fused=Mock(return_value="paged-legacy"),
            fast_topk_transform_ragged_fused=Mock(return_value="ragged-legacy"),
        )
        self.v2_paged = Mock(return_value="paged-v2")
        self.v2_ragged = Mock(return_value="ragged-v2")
        self.v2_packed = Mock(return_value="packed-v2")
        self.ns.update(
            _topk_transform_v2_paged=self.v2_paged,
            _topk_transform_v2_ragged=self.v2_ragged,
            _topk_transform_v2_packed=self.v2_packed,
        )
        self.modules = patch.dict(
            sys.modules,
            {
                "sglang.kernels.ops.attention.dsv4": SimpleNamespace(
                    top_k_per_row_prefill_bf16=self.kernel
                ),
                "sglang.kernels.ops.attention.fast_topk": self.legacy,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.scores = torch.randn(6, 32, dtype=torch.bfloat16)
        self.lengths = torch.tensor([0, 1, 7, 12, 15, 20], dtype=torch.int32)
        self.starts = torch.tensor([0, 1, 3, 5, 7, 9], dtype=torch.int32)
        self.mapping = torch.tensor([2, 0, 1, 2, 1, 0], dtype=torch.int32)
        self.cu = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
        self.page = torch.arange(3 * 32, dtype=torch.int32).view(3, 32)
        self.offsets = torch.tensor([101, 203, 307, 409, 503, 601], dtype=torch.int32)
        self.metadata = SimpleNamespace(
            real_page_table=self.page,
            page_table_1=self.page,
            token_to_batch_idx=self.mapping,
            page_size=64,
        )

    def call(self, method, **kwargs):
        args = dict(
            logits=self.scores,
            lengths=self.lengths,
            topk=16,
            topk_transform_method=method,
            attn_metadata=self.metadata,
            cu_seqlens_q_topk=self.cu,
            row_starts=self.starts,
        )
        args.update(kwargs)
        return self.backend.topk_transform(**args)

    def test_paged_explicit_interleaved_map_keeps_compact_table(self):
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        with patch.object(
            torch,
            "searchsorted",
            side_effect=AssertionError("Unexpected row-map rebuild"),
        ):
            out = self.call(self.method.PAGED, batch_idx_list=self.mapping)
        args, kwargs = self.kernel.call_args
        self.assertIs(args[0], self.scores)
        self.assertIs(args[3], self.page)
        self.assertIs(args[1], self.starts)
        self.assertTrue(torch.equal(args[2], self.starts + self.lengths))
        self.assertIs(kwargs["row_to_page"], self.mapping)
        self.assertIsNone(kwargs["out_offsets"])
        self.assertEqual(out.shape, (6, 16))
        self.v2_paged.assert_not_called()
        self.legacy.fast_topk_transform_fused.assert_not_called()

    def test_strided_row_metadata_is_not_copied(self):
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        starts = torch.arange(12, dtype=torch.int32)[::2]
        mapping = torch.tensor(
            [2, 99, 0, 99, 1, 99, 2, 99, 1, 99, 0, 99], dtype=torch.int32
        )[::2]
        self.call(self.method.PAGED, row_starts=starts, batch_idx_list=mapping)
        args, kwargs = self.kernel.call_args
        self.assertIs(args[1], starts)
        self.assertTrue(args[2].is_contiguous())
        self.assertIs(kwargs["row_to_page"], mapping)

    def test_paged_derives_map_from_cu_seqlens(self):
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        self.call(self.method.PAGED)
        mapping = self.kernel.call_args.kwargs["row_to_page"]
        self.assertTrue(
            torch.equal(mapping, torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32))
        )

    def test_paged_request_list_is_not_a_row_map(self):
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        self.call(self.method.PAGED, batch_idx_list=[2, 0, 1])
        args, kwargs = self.kernel.call_args
        self.assertTrue(torch.equal(args[3], self.page[[2, 0, 1]]))
        self.assertTrue(
            torch.equal(
                kwargs["row_to_page"],
                torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32),
            )
        )

    def test_ragged_bf16_helper_needs_no_page_size_1_table(self):
        # RAGGED columns are already flattened-KV positions, so the helper takes
        # no page table at all: only the additive output offsets. v2 is disabled
        # here because the default configuration routes these shapes to v2 first.
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        self.metadata.page_table_1 = None
        self.call(self.method.RAGGED, topk_indices_offset=self.offsets)
        args, kwargs = self.kernel.call_args
        self.assertIsNone(args[3])
        self.assertIsNone(kwargs["row_to_page"])
        self.assertIs(kwargs["out_offsets"], self.offsets)
        self.v2_ragged.assert_not_called()
        self.legacy.fast_topk_transform_ragged_fused.assert_not_called()

    def test_ragged_cp_preserves_explicit_offsets(self):
        self.call(
            self.method.RAGGED,
            topk_indices_offset=self.offsets,
            batch_idx_list=[2, 0, 1],
        )
        self.assertTrue(
            torch.equal(self.kernel.call_args.kwargs["out_offsets"], self.offsets)
        )
        self.assertIsNone(self.kernel.call_args.kwargs["row_to_page"])

    def test_ragged_requires_offsets(self):
        with self.assertRaisesRegex(RuntimeError, "topk_indices_offset"):
            self.call(self.method.RAGGED)
        self.kernel.assert_not_called()

    def test_decode_bf16_helper_uses_real_page_table_when_v2_is_off(self):
        # With v2 disabled, decode-shaped inputs use the BF16 helper directly
        # with real_page_table, without requiring a page_size=1 table.
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        self.metadata.page_table_1 = None
        self.call(
            self.method.PAGED,
            logits=self.scores[:3],
            lengths=self.lengths[:3],
            row_starts=None,
            cu_seqlens_q_topk=None,
        )
        args, kwargs = self.kernel.call_args
        self.assertIs(args[3], self.metadata.real_page_table)
        self.assertEqual(args[5], 64)
        self.assertIsNone(kwargs["row_to_page"])
        self.v2_paged.assert_not_called()

    def test_bf16_default_decode_shapes_prefer_v2(self):
        # Decode and expanded verify/draft-extend PAGED inputs prefer v2 even
        # when the BF16 helper gate is enabled, fusing selection and mapping.
        cases = (
            ("decode", self.scores[:3], self.lengths[:3], self.page),
            (
                "verify_expanded",
                self.scores,
                self.lengths,
                torch.arange(6 * 32, dtype=torch.int32).view(6, 32),
            ),
        )
        for name, scores, lengths, page_table in cases:
            with self.subTest(shape=name):
                self.v2_paged.reset_mock()
                self.metadata.real_page_table = page_table
                self.metadata.page_table_1 = None
                self.assertEqual(
                    self.call(
                        self.method.PAGED,
                        logits=scores,
                        lengths=lengths,
                        row_starts=None,
                        cu_seqlens_q_topk=None,
                    ),
                    "paged-v2",
                )
                args = self.v2_paged.call_args.args
                self.assertIs(args[0], scores)
                self.assertIs(args[1], lengths)
                self.assertIs(args[3], self.metadata)
                self.kernel.assert_not_called()

    def test_decode_v2_dispatch_runs_before_bf16_helper(self):
        # Keep decode-shaped v2 dispatch before the BF16 helper so the default
        # BF16 configuration can reach v2.
        tree = ast.parse((DSA / "dsa_topk_backend.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DSATopKBackend"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "topk_transform"
        )

        def only_call_line(name, field):
            nodes = [
                n
                for n in ast.walk(method)
                if isinstance(n, ast.Call) and getattr(n.func, field, None) == name
            ]
            self.assertEqual(len(nodes), 1)
            return nodes[0].lineno

        self.assertLess(
            only_call_line("_topk_transform_v2_paged", "id"),
            only_call_line("_topk_transform_dsv4_bf16", "attr"),
        )

    def test_v2_decode_gate_rejects_cp_row_starts_and_row_map(self):
        # CP/chunked prefill bypasses decode paged; windowed inputs use packed v2.
        self.envs.SGLANG_DSA_USE_DSV4_BF16_TOPK = flag(False)
        row_map = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)
        for name, kwargs in (
            ("packed-v2", dict(row_starts=self.starts)),
            ("paged-legacy", dict(row_starts=None, batch_idx_list=row_map)),
        ):
            with self.subTest(case=name):
                self.assertEqual(self.call(self.method.PAGED, **kwargs), name)
        self.v2_paged.assert_not_called()
        self.kernel.assert_not_called()

    def test_disabled_bf16_gate_keeps_decode_v2_native_bf16(self):
        self.envs.SGLANG_DSA_USE_DSV4_BF16_TOPK = flag(False)
        self.metadata.page_table_1 = None
        scores = self.scores[:3]
        self.assertEqual(
            self.call(
                self.method.PAGED,
                logits=scores,
                lengths=self.lengths[:3],
                row_starts=None,
            ),
            "paged-v2",
        )
        self.assertIs(self.v2_paged.call_args.args[0], scores)
        self.kernel.assert_not_called()

    def test_disabled_bf16_gate_keeps_ragged_v2_native_bf16(self):
        self.envs.SGLANG_DSA_USE_DSV4_BF16_TOPK = flag(False)
        self.assertEqual(
            self.call(self.method.RAGGED, topk_indices_offset=self.offsets), "ragged-v2"
        )
        self.assertIs(self.v2_ragged.call_args.args[0], self.scores)
        self.kernel.assert_not_called()

    def test_v2_gate_off_keeps_bf16_helper_on_both_methods(self):
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        self.call(self.method.RAGGED, topk_indices_offset=self.offsets)
        self.assertIsNone(self.kernel.call_args.args[3])
        self.call(
            self.method.PAGED,
            logits=self.scores[:3],
            lengths=self.lengths[:3],
            row_starts=None,
        )
        self.assertIs(self.kernel.call_args.args[3], self.metadata.real_page_table)
        self.assertEqual(self.kernel.call_args.args[5], 64)
        self.v2_paged.assert_not_called()
        self.v2_ragged.assert_not_called()

    def test_bf16_dispatch_has_one_helper_call_site(self):
        tree = ast.parse((DSA / "dsa_topk_backend.py").read_text())
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "DSATopKBackend"
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "topk_transform"
        )
        calls = [
            n
            for n in ast.walk(method)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_topk_transform_dsv4_bf16"
        ]
        self.assertEqual(len(calls), 1)

    def test_fp32_v2_input_is_not_changed(self):
        scores = self.scores.to(torch.float32)
        self.assertEqual(
            self.call(
                self.method.RAGGED, logits=scores, topk_indices_offset=self.offsets
            ),
            "ragged-v2",
        )
        self.assertIs(self.v2_ragged.call_args.args[0], scores)
        self.kernel.assert_not_called()

    def test_legacy_paged_collapses_per_token_map_without_fp32_copy(self):
        self.envs.SGLANG_DSA_USE_DSV4_BF16_TOPK = flag(False)
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        mapping = torch.tensor([2, 2, 0, 0, 1, 1], dtype=torch.int32)
        self.assertEqual(
            self.call(self.method.PAGED, batch_idx_list=mapping), "paged-legacy"
        )
        args = self.legacy.fast_topk_transform_fused.call_args.kwargs
        self.assertIs(args["score"], self.scores)
        self.assertTrue(torch.equal(args["page_table_size_1"], self.page[[2, 0, 1]]))
        self.assertEqual(args["page_table_size_1"].shape[0] + 1, self.cu.numel())

    def test_legacy_paged_full_token_map_does_not_gather_page_table(self):
        # A full-forward call passes attn_metadata.token_to_batch_idx itself.
        # Reuse the request-level table without gathering. A trailing empty
        # query makes cu_seqlens_q[:-1] include an out-of-bounds row index.
        self.envs.SGLANG_DSA_USE_DSV4_BF16_TOPK = flag(False)
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        cu = torch.tensor([0, 2, 4, 6, 6], dtype=torch.int32)
        self.assertEqual(
            self.call(
                self.method.PAGED,
                batch_idx_list=self.metadata.token_to_batch_idx,
                cu_seqlens_q_topk=cu,
            ),
            "paged-legacy",
        )
        args = self.legacy.fast_topk_transform_fused.call_args.kwargs
        self.assertIs(args["page_table_size_1"], self.page)
        self.assertIs(args["cu_seqlens_q"], cu)

    def test_legacy_paged_chunk_map_still_collapses_to_requests(self):
        # Chunked prefill passes a distinct token_to_batch_idx slice, which
        # still needs collapsing to one page-table row per request.
        self.envs.SGLANG_DSA_USE_DSV4_BF16_TOPK = flag(False)
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        full_map = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int32)
        chunk_map = full_map[2:6]
        cu = torch.tensor([0, 2, 4], dtype=torch.int32)
        self.assertIsNot(chunk_map, self.metadata.token_to_batch_idx)
        self.assertEqual(
            self.call(
                self.method.PAGED,
                logits=self.scores[:4],
                lengths=self.lengths[:4],
                cu_seqlens_q_topk=cu,
                batch_idx_list=chunk_map,
            ),
            "paged-legacy",
        )
        args = self.legacy.fast_topk_transform_fused.call_args.kwargs
        self.assertTrue(torch.equal(args["page_table_size_1"], self.page[[1, 2]]))
        self.assertEqual(args["page_table_size_1"].shape[0] + 1, cu.numel())

    def test_legacy_ragged_does_not_make_fp32_copy(self):
        self.envs.SGLANG_DSA_USE_DSV4_BF16_TOPK = flag(False)
        self.envs.SGLANG_OPT_USE_TOPK_V2 = flag(False)
        self.assertEqual(
            self.call(self.method.RAGGED, topk_indices_offset=self.offsets),
            "ragged-legacy",
        )
        self.assertIs(
            self.legacy.fast_topk_transform_ragged_fused.call_args.kwargs["score"],
            self.scores,
        )
        self.kernel.assert_not_called()

    def test_unfused_paths_accept_preserved_bf16_logits_without_fp32_copy(self):
        for fuse, force in ((False, False), (True, True)):
            with self.subTest(fuse=fuse, force=force):
                self.envs.SGLANG_DSA_FUSE_TOPK = flag(fuse)
                self.assertEqual(
                    self.call(self.method.RAGGED, force_unfused_topk=force), "unfused"
                )
                self.assertIs(self.legacy.fast_topk_v2.call_args.args[0], self.scores)
        self.kernel.assert_not_called()

    def test_packed_preserves_interleaved_map_and_both_dtypes(self):
        self.metadata.page_table_1 = None
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                scores = self.scores.to(dtype)
                with patch.object(
                    torch,
                    "searchsorted",
                    side_effect=AssertionError("Unexpected interleaved map rebuild"),
                ):
                    self.assertEqual(
                        self.call(
                            self.method.PAGED,
                            logits=scores,
                            batch_idx_list=self.mapping,
                        ),
                        "packed-v2",
                    )
                args, kwargs = self.v2_packed.call_args
                self.assertIs(args[0], scores)
                self.assertIs(args[3], self.metadata)
                self.assertIs(kwargs["row_starts"], self.starts)
                self.assertIs(kwargs["row_to_batch"], self.mapping)
        self.kernel.assert_not_called()
        self.v2_paged.assert_not_called()

    def test_packed_cp_request_list_handles_empty_segments(self):
        cu = torch.tensor([0, 2, 2, 6, 6], dtype=torch.int32)
        self.metadata.page_table_1 = None
        self.assertEqual(
            self.call(
                self.method.PAGED, batch_idx_list=[2, 0, 1, 2], cu_seqlens_q_topk=cu
            ),
            "packed-v2",
        )
        mapping = self.v2_packed.call_args.kwargs["row_to_batch"]
        self.assertTrue(
            torch.equal(mapping, torch.tensor([2, 2, 1, 1, 1, 1], dtype=torch.int32))
        )

    def test_packed_cp_single_request_shares_table_for_2048_queries(self):
        self.metadata.real_page_table = self.page[:1]
        self.metadata.page_table_1 = None
        self.metadata.token_to_batch_idx = None
        self.assertEqual(
            self.call(
                self.method.PAGED,
                logits=torch.empty(2048, 32, dtype=torch.bfloat16),
                lengths=torch.ones(2048, dtype=torch.int32),
                row_starts=torch.zeros(2048, dtype=torch.int32),
                cu_seqlens_q_topk=torch.tensor([0, 2048], dtype=torch.int32),
            ),
            "packed-v2",
        )
        mapping = self.v2_packed.call_args.kwargs["row_to_batch"]
        self.assertTrue(torch.equal(mapping, torch.zeros(2048, dtype=torch.int32)))

    def test_packed_chunk_map_takes_priority_over_full_metadata(self):
        chunk = self.mapping[2:]
        self.assertEqual(
            self.call(
                self.method.PAGED,
                logits=self.scores[:4],
                lengths=self.lengths[:4],
                row_starts=self.starts[:4],
                batch_idx_list=chunk,
            ),
            "packed-v2",
        )
        self.assertIs(self.v2_packed.call_args.kwargs["row_to_batch"], chunk)

    def test_packed_passes_explicit_map_without_copy(self):
        self.call(self.method.PAGED, batch_idx_list=self.mapping)
        self.assertIs(self.v2_packed.call_args.kwargs["row_to_batch"], self.mapping)

    def test_packed_rejects_incomplete_cp_mapping(self):
        for kwargs in (
            dict(batch_idx_list=self.mapping[:-1]),
            dict(batch_idx_list=[2, 0], cu_seqlens_q_topk=self.cu),
            dict(batch_idx_list=[0], cu_seqlens_q_topk=None),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.call(self.method.PAGED, **kwargs)
        self.v2_packed.assert_not_called()

    def test_xpu_keeps_existing_paged_fallback(self):
        self.ns["_is_xpu"] = True
        self.envs.SGLANG_DSA_USE_DSV4_BF16_TOPK = flag(False)
        self.assertEqual(self.call(self.method.PAGED), "paged-legacy")
        self.v2_packed.assert_not_called()

    def test_unfused_paged_does_not_build_cp_mapping(self):
        self.assertEqual(
            self.call(
                self.method.PAGED,
                batch_idx_list=[0],
                cu_seqlens_q_topk=None,
                force_unfused_topk=True,
            ),
            "unfused",
        )
        self.v2_packed.assert_not_called()

    def test_packed_helper_keeps_compact_table_without_metadata_copy(self):
        kernel = Mock()
        ns = load_definitions(
            DSA / "dsa_topk_backend.py",
            {"_topk_transform_v2_packed"},
            dict(torch=torch),
        )
        with patch.dict(
            sys.modules,
            {
                "sglang.kernels.ops.attention.dsv4.topk": SimpleNamespace(
                    topk_transform_packed_v2=kernel
                )
            },
        ):
            for dtype in (torch.float32, torch.bfloat16):
                scores = self.scores.to(dtype)
                result = ns["_topk_transform_v2_packed"](
                    scores, self.lengths, 512, self.metadata, self.starts, self.mapping
                )
                args, kwargs = kernel.call_args
                self.assertIs(args[0], scores)
                self.assertIs(args[1], self.lengths)
                self.assertIs(args[2], self.metadata.real_page_table)
                self.assertEqual(result.shape, (6, 512))
                self.assertIs(kwargs["row_starts"], self.starts)
                self.assertIs(kwargs["row_to_batch"], self.mapping)

    def test_indexer_paged_mapping_tolerates_metadata_without_method(self):
        # The non-chunked indexer call site reads the transform method off the
        # metadata object, which BaseIndexerMetadata does not declare. Evaluate
        # the production expression itself: PAGED keeps the original per-token
        # map, RAGGED drops it, and a metadata object without the attribute must
        # degrade to "no map" instead of raising.
        source = DSA / "dsa_indexer.py"
        nodes = [
            n
            for n in ast.walk(ast.parse(source.read_text()))
            if isinstance(n, ast.keyword)
            and n.arg == "batch_idx_list"
            and isinstance(n.value, ast.IfExp)
        ]
        self.assertEqual(len(nodes), 1)
        code = compile(
            ast.fix_missing_locations(ast.Expression(body=nodes[0].value)),
            str(source),
            "eval",
        )

        def mapping_for(metadata):
            return eval(
                code,
                dict(
                    token_to_batch_idx=self.mapping,
                    metadata=metadata,
                    TopkTransformMethod=self.method,
                ),
            )

        self.assertIs(
            mapping_for(SimpleNamespace(topk_transform_method=self.method.PAGED)),
            self.mapping,
        )
        self.assertIsNone(
            mapping_for(SimpleNamespace(topk_transform_method=self.method.RAGGED))
        )
        self.assertIsNone(mapping_for(SimpleNamespace()))

    def test_paged_metadata_does_not_build_ragged_offsets(self):
        ns = load_definitions(
            DSA / "dsa_indexer_metadata.py",
            {"DSAIndexerMetadata"},
            dict(
                torch=torch,
                dataclass=dataclass,
                BaseIndexerMetadata=object,
                DSATopKBackend=self.ns["DSATopKBackend"],
                TopkTransformMethod=self.method,
                compute_cu_seqlens=lambda lengths: torch.cat(
                    (lengths.new_zeros(1), lengths.cumsum(0).to(torch.int32))
                ),
            ),
        )
        backend = SimpleNamespace(topk_transform=Mock(return_value="metadata"))
        metadata = ns["DSAIndexerMetadata"](
            attn_metadata=self.metadata,
            topk_transform_method=self.method.PAGED,
            topk_backend=backend,
        )
        with patch.object(
            torch,
            "repeat_interleave",
            side_effect=AssertionError("Unexpected ragged offset allocation"),
        ):
            metadata.topk_transform(
                self.scores,
                16,
                cu_seqlens_q=torch.tensor([2, 2, 2]),
                ke_offset=self.lengths,
            )
        self.assertIsNone(
            backend.topk_transform.call_args.kwargs["topk_indices_offset"]
        )


class FastTopKTransformWrapperTest(unittest.TestCase):
    """Shape-contract checks on the production fused-transform wrapper. The real
    input validator needs CUDA tensors, so only that helper is replaced; every
    other line comes from the production source and runs on CPU tensors."""

    def setUp(self):
        self.kernel = Mock()
        self.factory = Mock(
            return_value=SimpleNamespace(fast_topk_transform_fused=self.kernel)
        )
        self.transform = load_definitions(
            REPO / "python/sglang/kernels/ops/attention/fast_topk.py",
            {"fast_topk_transform_fused"},
            dict(
                torch=torch,
                _validate_topk_inputs=Mock(),
                _jit_fast_topk_module=self.factory,
            ),
        )["fast_topk_transform_fused"]

    def call(self, batch, page_rows, cu, **kwargs):
        score = torch.randn(batch, 32, dtype=torch.bfloat16)
        lengths = torch.full((batch,), 8, dtype=torch.int32)
        page = torch.arange(page_rows * 32, dtype=torch.int32).view(page_rows, 32)
        return self.transform(score, lengths, page, cu, 512, **kwargs)

    def test_more_page_rows_than_query_rows_is_accepted(self):
        # cu_seqlens_q = [0, 1, 1, 2]: the middle request owns no query row, so
        # two score rows come with three page-table rows.
        cu = torch.tensor([0, 1, 1, 2], dtype=torch.int32)
        out = self.call(2, 3, cu)
        self.assertEqual(out.shape, (2, 512))
        args = self.kernel.call_args.args
        self.assertEqual(args[3].shape[0], 3)
        self.assertIs(args[4], cu)

    def test_cu_seqlens_q_shape_mismatch_is_still_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "cu_seqlens_q"):
            self.call(2, 3, torch.tensor([0, 1, 2], dtype=torch.int32))
        self.kernel.assert_not_called()

    def test_empty_batch_with_pages_does_not_reach_the_kernel(self):
        self.assertEqual(
            self.call(0, 3, torch.zeros(4, dtype=torch.int32)).shape, (0, 512)
        )
        self.factory.assert_not_called()

    def test_non_empty_scores_still_require_a_page_row(self):
        with self.assertRaisesRegex(RuntimeError, "page-table row"):
            self.call(2, 0, torch.zeros(1, dtype=torch.int32))
        self.factory.assert_not_called()


class V2WrapperTest(unittest.TestCase):
    def test_jit_module_selects_input_dtype(self):
        class Args(tuple):
            def __str__(self):
                return ", ".join(map(str, self))

        cpp_args = Mock(side_effect=lambda *args: Args(args))
        load_jit = Mock()
        namespace = load_definitions(
            REPO / "python/sglang/kernels/ops/attention/dsv4/topk.py",
            {"_jit_topk_v2_module"},
            dict(
                torch=torch,
                cache_once=lambda fn: fn,
                is_arch_support_pdl=lambda: False,
                is_hip_runtime=lambda: False,
                make_cpp_args=cpp_args,
                make_name=lambda name: name,
                load_jit=load_jit,
            ),
        )
        occupancy = SimpleNamespace(
            NoSchedulableClustersError=ValueError, get_max_active_clusters=Mock()
        )
        with patch.dict(sys.modules, {"sglang.kernels.jit.utils.occupancy": occupancy}):
            namespace["_jit_topk_v2_module"]()
            self.assertEqual(cpp_args.call_args.args, (False,))
            self.assertEqual(load_jit.call_args.args[1:], (False,))
            namespace["_jit_topk_v2_module"](torch.bfloat16)
            self.assertEqual(cpp_args.call_args.args, (False, torch.bfloat16))
            self.assertEqual(load_jit.call_args.args[1:], (False, torch.bfloat16))
            self.assertIn(
                str(torch.bfloat16), load_jit.call_args.kwargs["cuda_wrappers"][0][1]
            )
            self.assertIn(
                "topk_transform_packed",
                dict(load_jit.call_args.kwargs["cuda_wrappers"]),
            )
            with self.assertRaises(TypeError):
                namespace["_jit_topk_v2_module"](torch.float16)

    def test_v2_wrappers_forward_bf16_and_fp32_without_cast(self):
        module = SimpleNamespace(
            topk_transform_paged=Mock(),
            topk_transform_ragged=Mock(),
            topk_transform_packed=Mock(),
        )
        factory = Mock(return_value=module)
        namespace = load_definitions(
            REPO / "python/sglang/kernels/ops/attention/dsv4/topk.py",
            {
                "topk_transform_paged_v2",
                "topk_transform_ragged_v2",
                "topk_transform_packed_v2",
            },
            dict(torch=torch, is_xpu=lambda: False, _jit_topk_v2_module=factory),
        )
        lengths = torch.ones(3, dtype=torch.int32)
        indices = torch.empty(3, 512, dtype=torch.int32)
        pages = torch.zeros(3, 1, dtype=torch.int32)
        plan = torch.empty(4, 2, dtype=torch.int32)
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                scores = torch.randn(3, 16, dtype=dtype)
                namespace["topk_transform_paged_v2"](
                    scores, lengths, pages, indices, 64, plan
                )
                self.assertEqual(factory.call_args.args, (dtype,))
                self.assertIs(module.topk_transform_paged.call_args.args[0], scores)
                namespace["topk_transform_ragged_v2"](
                    scores, lengths, out_offsets=lengths, out_indices=indices
                )
                self.assertEqual(factory.call_args.args, (dtype,))
                self.assertIs(module.topk_transform_ragged.call_args.args[0], scores)
                namespace["topk_transform_packed_v2"](
                    scores,
                    lengths,
                    pages[:1],
                    indices,
                    64,
                    row_starts=torch.zeros_like(lengths),
                    row_to_batch=torch.zeros_like(lengths),
                )
                self.assertEqual(factory.call_args.args, (dtype,))
                self.assertIs(module.topk_transform_packed.call_args.args[0], scores)
                self.assertEqual(
                    module.topk_transform_packed.call_args.args[3].shape[0], 1
                )


if __name__ == "__main__":
    unittest.main()
