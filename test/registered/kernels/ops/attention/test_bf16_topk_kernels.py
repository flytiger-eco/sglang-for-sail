import importlib
import unittest
from itertools import product
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_ppu_ci

# PPU numerical coverage for the BF16 prefill top-k helper, native BF16/FP32 v2
# top-k (paged, ragged, packed) and the DSA dispatch helpers that reach them.
# The optional timing comparison lives in
# benchmark/kernels/attention/bench_bf16_topk_compare.py.
register_ppu_ci(est_time=120, suite="stage-b-test-1-gpu-ppu")


class BF16TopKKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA/PPU device required")
        cls.topk = importlib.import_module("sglang.kernels.ops.attention.dsv4.topk")

    def make_inputs(self, topk, case):
        cases = {
            "boundaries": (
                8192,
                3,
                [0, 1, 3, 5, 7, 11],
                [0, 1, topk - 1, topk, topk + 1, 4096],
            ),
            "register_1pass": (16383, 5, [0, 1, 3, 7], [8192, 8193, 12000, 16376]),
            "register_16k_boundary": (20001, 0, [0, 1, 3], [16383, 16384, 16385]),
            "register_2pass": (32768, 0, [0, 1, 7], [32768, 32767, 32761]),
            "stream_mixed": (
                40001,
                2,
                [0, 1, 3, 5, 7, 11],
                [0, topk, 8193, 16385, 32769, 39990],
            ),
            "stream_ties": (40001, 2, [0, 1, 7], [40001, 33000, 32769]),
        }
        width, stride_pad, starts, lengths = cases[case]
        torch.manual_seed(42)
        base = torch.full(
            (len(lengths), width + stride_pad),
            512.0,
            dtype=torch.bfloat16,
            device="cuda",
        )
        scores = base[:, :width]
        for row, (start, length) in enumerate(zip(starts, lengths)):
            if case == "stream_ties":
                scores[row, start : start + length].fill_(1)
            else:
                scores[row, start : start + length] = torch.randn(
                    length, dtype=torch.bfloat16, device="cuda"
                )
        return (
            scores,
            torch.tensor(starts, dtype=torch.int32, device="cuda"),
            torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        )

    def make_transform(self, scores, mode):
        rows, width = scores.shape
        offsets = None
        mapping = None
        page_size = 64 if mode == "paged64" else 1
        if mode == "ragged":
            pages = None
            offsets = torch.arange(rows, dtype=torch.int32, device="cuda") * 100003 + 97
        else:
            page_rows = rows if mode == "paged" else 3
            num_pages = (width + page_size - 1) // page_size
            storage = torch.empty(
                (page_rows, num_pages + 3), dtype=torch.int32, device="cuda"
            )
            pages = storage[:, :num_pages]
            pages.copy_(
                torch.randperm(page_rows * num_pages, device="cuda")
                .reshape(page_rows, num_pages)
                .to(torch.int32)
                + 1009
            )
            if mode != "paged":
                mapping = (
                    torch.arange(rows, dtype=torch.int32, device="cuda") * 2 + 2
                ) % page_rows
        return pages, mapping, offsets, page_size

    def check_raw(self, scores, starts, lengths, raw, topk):
        for row in range(scores.shape[0]):
            start, length = int(starts[row]), int(lengths[row])
            selected = raw[row]
            valid = selected >= 0
            self.assertEqual(int(valid.sum()), min(length, topk))
            self.assertTrue(bool((selected[~valid] == -1).all()))
            positions = selected[valid].long()
            self.assertTrue(bool((positions < length).all()))
            self.assertEqual(positions.unique().numel(), min(length, topk))
            if length <= topk:
                self.assertTrue(
                    torch.equal(
                        selected[:length],
                        torch.arange(length, dtype=torch.int32, device="cuda"),
                    )
                )
                self.assertTrue(bool((selected[length:] == -1).all()))
            section = scores[row, start : start + length].float()
            got = section[positions].sort(descending=True).values
            expected = (
                section.topk(min(length, topk)).values.sort(descending=True).values
            )
            self.assertTrue(
                torch.equal(got, expected),
                f"Top-k values differ at row {row}, length={length}",
            )

    def check_output(
        self,
        scores,
        starts,
        lengths,
        out,
        raw,
        topk,
        pages,
        mapping,
        offsets,
        page_size,
    ):
        self.check_raw(scores, starts, lengths, raw, topk)
        for row in range(scores.shape[0]):
            valid = raw[row] >= 0
            positions = raw[row, valid].long()
            expected = torch.full_like(out[row], -1)
            if pages is None:
                expected[valid] = positions.to(torch.int32) + offsets[row]
            else:
                page_row = row if mapping is None else int(mapping[row])
                expected[valid] = pages[
                    page_row, positions // page_size
                ] * page_size + (positions % page_size).to(torch.int32)
            self.assertTrue(
                torch.equal(out[row], expected),
                f"Output transform differs at row {row}",
            )

    def recover_raw(self, out, pages, mapping, offsets, page_size):
        raw = torch.full_like(out, -1)
        for row in range(out.shape[0]):
            valid = out[row] != -1
            if pages is None:
                raw[row, valid] = out[row, valid] - offsets[row]
            else:
                page_row = row if mapping is None else int(mapping[row])
                sorted_pages, order = pages[page_row].sort()
                locations = torch.searchsorted(
                    sorted_pages, out[row, valid] // page_size
                )
                self.assertTrue(bool((locations < sorted_pages.numel()).all()))
                self.assertTrue(
                    torch.equal(sorted_pages[locations], out[row, valid] // page_size)
                )
                raw[row, valid] = (
                    order[locations].to(torch.int32) * page_size
                    + out[row, valid] % page_size
                )
        return raw

    def run_case(self, topk, mode, case):
        scores, starts, lengths = self.make_inputs(topk, case)
        pages, mapping, offsets, page_size = self.make_transform(scores, mode)
        out = torch.full((scores.shape[0], topk), 171, dtype=torch.int32, device="cuda")
        raw = torch.full_like(out, 171)
        self.topk.top_k_per_row_prefill_bf16(
            scores,
            starts,
            starts + lengths,
            pages,
            out,
            page_size,
            out_raw_indices=raw,
            row_to_page=mapping,
            out_offsets=offsets,
        )
        torch.cuda.synchronize()
        self.check_output(
            scores, starts, lengths, out, raw, topk, pages, mapping, offsets, page_size
        )

    def test_fast_topk_bf16_fallbacks(self):
        fast = importlib.import_module("sglang.kernels.ops.attention.fast_topk")
        for topk in (512, 2048):
            for case in ("boundaries", "stream_mixed"):
                with self.subTest(topk=topk, case=case):
                    scores, starts, lengths = self.make_inputs(topk, case)
                    raw = fast.fast_topk_v2(scores, lengths, topk, row_starts=starts)
                    torch.cuda.synchronize()
                    self.check_raw(scores, starts, lengths, raw, topk)

                    pages, mapping, offsets, page_size = self.make_transform(
                        scores, "paged"
                    )
                    cu = torch.arange(
                        scores.shape[0] + 1, dtype=torch.int32, device="cuda"
                    )
                    paged = fast.fast_topk_transform_fused(
                        scores, lengths, pages, cu, topk, row_starts=starts
                    )
                    torch.cuda.synchronize()
                    raw = self.recover_raw(paged, pages, mapping, offsets, page_size)
                    self.check_output(
                        scores,
                        starts,
                        lengths,
                        paged,
                        raw,
                        topk,
                        pages,
                        mapping,
                        offsets,
                        page_size,
                    )

                    pages, mapping, offsets, page_size = self.make_transform(
                        scores, "ragged"
                    )
                    ragged = fast.fast_topk_transform_ragged_fused(
                        scores, lengths, offsets, topk, row_starts=starts
                    )
                    torch.cuda.synchronize()
                    raw = self.recover_raw(ragged, pages, mapping, offsets, page_size)
                    self.check_output(
                        scores,
                        starts,
                        lengths,
                        ragged,
                        raw,
                        topk,
                        pages,
                        mapping,
                        offsets,
                        page_size,
                    )

    def test_optional_raw_output_is_not_required(self):
        for mode in ("paged", "indirect", "paged64", "ragged"):
            with self.subTest(mode=mode):
                scores, starts, lengths = self.make_inputs(512, "boundaries")
                pages, mapping, offsets, page_size = self.make_transform(scores, mode)
                out = torch.full(
                    (scores.shape[0], 512), 171, dtype=torch.int32, device="cuda"
                )
                self.topk.top_k_per_row_prefill_bf16(
                    scores,
                    starts,
                    starts + lengths,
                    pages,
                    out,
                    page_size,
                    row_to_page=mapping,
                    out_offsets=offsets,
                )
                torch.cuda.synchronize()
                raw = self.recover_raw(out, pages, mapping, offsets, page_size)
                self.check_output(
                    scores,
                    starts,
                    lengths,
                    out,
                    raw,
                    512,
                    pages,
                    mapping,
                    offsets,
                    page_size,
                )

    def test_optional_argument_validation(self):
        scores, starts, lengths = self.make_inputs(512, "boundaries")
        pages, mapping, offsets, page_size = self.make_transform(scores, "indirect")
        out = torch.empty((scores.shape[0], 512), dtype=torch.int32, device="cuda")
        row_offsets = torch.zeros_like(starts)
        cases = (
            (None, None, None),
            (pages, mapping, row_offsets),
            (None, mapping, row_offsets),
            (pages, mapping[:-1], None),
            (pages, mapping.long(), None),
            (None, None, row_offsets.long()),
            (None, None, row_offsets[:-1]),
        )
        for table, row_map, out_offsets in cases:
            with self.subTest(
                table=table is not None,
                map_dtype=getattr(row_map, "dtype", None),
                offsets_dtype=getattr(out_offsets, "dtype", None),
            ):
                with self.assertRaises(Exception):
                    self.topk.top_k_per_row_prefill_bf16(
                        scores,
                        starts,
                        starts + lengths,
                        table,
                        out,
                        page_size,
                        row_to_page=row_map,
                        out_offsets=out_offsets,
                    )

    def test_helper_rejects_noncontiguous_metadata_without_copy(self):
        backend_module = importlib.import_module(
            "sglang.srt.layers.attention.dsa.dsa_topk_backend"
        )
        backend = backend_module.DSATopKBackend.SGL_KERNEL
        scores, starts, lengths = self.make_inputs(512, "boundaries")
        pages, mapping, offsets, page_size = self.make_transform(scores, "indirect")
        backend._topk_transform_dsv4_bf16(
            scores, lengths, pages, None, 512, starts, row_to_page=mapping
        )
        torch.cuda.synchronize()
        bad_starts = torch.stack((starts, starts), dim=1).reshape(-1)[::2]
        bad_mapping = torch.stack((mapping, mapping), dim=1).reshape(-1)[::2]
        bad_pages = torch.stack((pages, pages), dim=-1).reshape(pages.shape[0], -1)[
            :, ::2
        ]
        offsets = (
            torch.arange(scores.shape[0], dtype=torch.int32, device="cuda") * 100003
            + 97
        )
        bad_offsets = torch.stack((offsets, offsets), dim=1).reshape(-1)[::2]
        cases = (
            (pages, bad_starts, mapping, None),
            (pages, starts, bad_mapping, None),
            (bad_pages, starts, mapping, None),
            (None, starts, None, bad_offsets),
        )
        for table, row_starts, row_map, row_offsets in cases:
            with self.subTest(
                table_stride=table.stride() if table is not None else None,
                starts_stride=row_starts.stride(),
                map_stride=getattr(row_map, "stride", lambda: None)(),
            ):
                with self.assertRaisesRegex(Exception, "(not contiguous|stride)"):
                    backend._topk_transform_dsv4_bf16(
                        scores,
                        lengths,
                        table,
                        None,
                        512,
                        row_starts,
                        row_to_page=row_map,
                        topk_indices_offset=row_offsets,
                    )

    def test_cuda_graph_replay_reads_updated_metadata(self):
        for mode in ("indirect", "ragged"):
            with self.subTest(mode=mode):
                scores, starts, lengths = self.make_inputs(512, "boundaries")
                pages, mapping, offsets, page_size = self.make_transform(scores, mode)
                ends = starts + lengths
                out = torch.full(
                    (scores.shape[0], 512), 171, dtype=torch.int32, device="cuda"
                )
                raw = torch.full_like(out, 171)

                def launch():
                    self.topk.top_k_per_row_prefill_bf16(
                        scores,
                        starts,
                        ends,
                        pages,
                        out,
                        page_size,
                        out_raw_indices=raw,
                        row_to_page=mapping,
                        out_offsets=offsets,
                    )

                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    launch()
                torch.cuda.current_stream().wait_stream(warmup)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                for replay in range(2):
                    lengths.copy_(
                        torch.tensor(
                            [1, 0, 100, 512, 1000, 2000],
                            dtype=torch.int32,
                            device="cuda",
                        )
                    )
                    ends.copy_(starts + lengths)
                    scores.neg_()
                    if offsets is not None:
                        offsets.add_(10007)
                    if pages is not None:
                        pages.add_(10007)
                        mapping.copy_((mapping + 1) % pages.shape[0])
                    out.fill_(171)
                    raw.fill_(171)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.check_output(
                        scores,
                        starts,
                        lengths,
                        out,
                        raw,
                        512,
                        pages,
                        mapping,
                        offsets,
                        page_size,
                    )


for _topk in (512, 2048):
    for _mode in ("paged", "indirect", "paged64", "ragged"):
        for _case in (
            "boundaries",
            "register_1pass",
            "register_16k_boundary",
            "register_2pass",
            "stream_mixed",
            "stream_ties",
        ):

            def _test(self, topk=_topk, mode=_mode, case=_case):
                self.run_case(topk, mode, case)

            setattr(BF16TopKKernelTest, f"test_{_mode}_{_case}_k{_topk}", _test)


_V2_CASES = {
    "naive": None,
    "register2_minus": 8191,
    "register2_edge": 8192,
    "register2_plus": 8193,
    "register4_minus": 16383,
    "register4_edge": 16384,
    "register4_plus": 16385,
    "stream64k": 65537,
    "stream_ties": 70001,
}


class V2TopKKernelTest(unittest.TestCase):
    check_raw = BF16TopKKernelTest.check_raw
    check_output = BF16TopKKernelTest.check_output
    recover_raw = BF16TopKKernelTest.recover_raw

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA/PPU device required")
        cls.topk = importlib.import_module("sglang.kernels.ops.attention.dsv4.topk")
        cls.backend_module = importlib.import_module(
            "sglang.srt.layers.attention.dsa.dsa_topk_backend"
        )
        cls.backend = cls.backend_module.DSATopKBackend.SGL_KERNEL

    def make_matrix(self, width, starts, lengths, dtype, ties=False):
        torch.manual_seed(7919 + width + len(lengths))
        stride = ((width + 3) & ~3) + 4
        storage = torch.full((len(lengths), stride), 512.0, dtype=dtype, device="cuda")
        scores = storage[:, :width]
        for row, (start, length) in enumerate(zip(starts, lengths)):
            self.assertLessEqual(start + length, width)
            if ties:
                scores[row, start : start + length].fill_(1)
            else:
                scores[row, start : start + length] = torch.randn(
                    length, dtype=dtype, device="cuda"
                )
        self.assertEqual(scores.dtype, dtype)
        self.assertEqual(scores.stride(0) % 4, 0)
        self.assertGreater(scores.stride(0), width)
        return (
            scores,
            torch.tensor(starts, dtype=torch.int32, device="cuda"),
            torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        )

    def make_v2_inputs(self, topk, mode, case, dtype):
        starts = [0, 1, 2, 3, 7, 11] if mode == "ragged" else [0] * 6
        boundary = _V2_CASES[case]
        if boundary is None:
            lengths = [0, 1, topk - 1, topk, topk + 1, topk + 3]
            width = max(s + n for s, n in zip(starts, lengths))
        elif mode == "ragged":
            lengths = [boundary - 1, boundary, boundary + 1] * 2
            width = max(s + n for s, n in zip(starts, lengths))
        else:
            width = boundary
            lengths = [0, 1, topk, topk + 1, boundary - 1, boundary]
        return self.make_matrix(width, starts, lengths, dtype, case == "stream_ties")

    def test_packed_cp_windows_bf16_and_fp32(self):
        for dtype in (torch.float32, torch.bfloat16):
            for topk in (512, 2048):
                for case in _V2_CASES:
                    with self.subTest(dtype=dtype, topk=topk, case=case):
                        scores, starts, lengths = self.make_v2_inputs(
                            topk, "ragged", case, dtype
                        )
                        reference = scores.clone()
                        pages, _, _, page_size = self.make_v2_transform(
                            scores[:2], "paged64"
                        )
                        mapping = torch.tensor(
                            [1, 0, 1, 0, 0, 1], dtype=torch.int32, device="cuda"
                        )
                        out = torch.empty(
                            (scores.shape[0], topk), dtype=torch.int32, device="cuda"
                        )
                        self.topk.topk_transform_packed_v2(
                            scores,
                            lengths,
                            pages,
                            out,
                            page_size,
                            row_starts=starts,
                            row_to_batch=mapping,
                        )
                        torch.cuda.synchronize()
                        raw = self.recover_raw(out, pages, mapping, None, page_size)
                        self.check_output(
                            reference,
                            starts,
                            lengths,
                            out,
                            raw,
                            topk,
                            pages,
                            mapping,
                            None,
                            page_size,
                        )
                        metadata = SimpleNamespace(
                            real_page_table=pages,
                            page_size=page_size,
                            page_table_1=None,
                            token_to_batch_idx=mapping,
                        )
                        out = self.backend.topk_transform(
                            scores,
                            lengths,
                            topk,
                            self.backend_module.TopkTransformMethod.PAGED,
                            metadata,
                            row_starts=starts,
                            batch_idx_list=mapping,
                        )
                        torch.cuda.synchronize()
                        raw = self.recover_raw(out, pages, mapping, None, page_size)
                        self.check_output(
                            reference,
                            starts,
                            lengths,
                            out,
                            raw,
                            topk,
                            pages,
                            mapping,
                            None,
                            page_size,
                        )

    def test_packed_2048_queries_share_one_page_table(self):
        rows, length, topk, start, page_size = 2048, 3073, 2048, 3, 64
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(2048)
                width = (start + length + 3) & ~3
                source = torch.full((1, width), 512.0, dtype=dtype, device="cuda")
                source[:, start : start + length] = torch.randn(
                    1, length, dtype=dtype, device="cuda"
                )
                # Share only the request page table; score rows need separate
                # storage and must not have a zero row stride.
                scores = source.repeat(rows, 1)
                pages, _, _, _ = self.make_v2_transform(scores[:1], "paged64")
                out = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
                self.topk.topk_transform_packed_v2(
                    scores,
                    torch.full((rows,), length, dtype=torch.int32, device="cuda"),
                    pages,
                    out,
                    page_size,
                    row_starts=torch.full(
                        (rows,), start, dtype=torch.int32, device="cuda"
                    ),
                    row_to_batch=torch.zeros(rows, dtype=torch.int32, device="cuda"),
                )
                torch.cuda.synchronize()
                sorted_pages, order = pages[0].sort()
                page_ids = out // page_size
                locations = torch.searchsorted(sorted_pages, page_ids)
                self.assertTrue(bool((locations < sorted_pages.numel()).all()))
                self.assertTrue(torch.equal(sorted_pages[locations], page_ids))
                positions = order[locations] * page_size + out % page_size
                self.assertTrue(bool(((positions >= 0) & (positions < length)).all()))
                self.assertTrue(
                    bool((positions.sort(dim=1).values.diff(dim=1) > 0).all())
                )
                got = (
                    scores.float()
                    .gather(1, positions.long() + start)
                    .sort(dim=1)
                    .values
                )
                expected = (
                    source[:, start : start + length]
                    .float()
                    .topk(topk)
                    .values.sort(dim=1)
                    .values
                )
                self.assertTrue(torch.equal(got, expected.expand_as(got)))

    def test_packed_graph_replay_updates_windows_and_request_map(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                scores, starts, lengths = self.make_matrix(
                    17004,
                    [0, 1, 2, 3, 7, 11],
                    [0, 1, 512, 513, 8192, 16384],
                    dtype,
                )
                pages, _, _, page_size = self.make_v2_transform(scores[:2], "paged64")
                mapping = torch.tensor(
                    [1, 0, 1, 0, 0, 1], dtype=torch.int32, device="cuda"
                )
                out = torch.empty((6, 512), dtype=torch.int32, device="cuda")

                def launch():
                    self.topk.topk_transform_packed_v2(
                        scores,
                        lengths,
                        pages,
                        out,
                        page_size,
                        row_starts=starts,
                        row_to_batch=mapping,
                    )

                graph = self.capture_launch(launch)
                for _ in range(2):
                    starts.add_(1)
                    lengths.copy_(
                        torch.tensor(
                            [1, 0, 511, 514, 8193, 16385],
                            dtype=torch.int32,
                            device="cuda",
                        )
                    )
                    mapping.copy_(1 - mapping)
                    pages.add_(10007)
                    scores.normal_()
                    reference = scores.clone()
                    graph.replay()
                    torch.cuda.synchronize()
                    raw = self.recover_raw(out, pages, mapping, None, page_size)
                    self.check_output(
                        reference,
                        starts,
                        lengths,
                        out,
                        raw,
                        512,
                        pages,
                        mapping,
                        None,
                        page_size,
                    )

    def test_packed_rejects_overlapping_rows_and_missing_map(self):
        scores, starts, lengths = self.make_matrix(
            4096, [0] * 6, [2048] * 6, torch.bfloat16
        )
        pages, _, _, page_size = self.make_v2_transform(scores[:1], "paged64")
        out = torch.empty((6, 512), dtype=torch.int32, device="cuda")
        mapping = torch.zeros(6, dtype=torch.int32, device="cuda")
        for source, row_map in ((scores[:1].expand(6, -1), mapping), (scores, None)):
            with (
                self.subTest(overlapping=source.stride(0) == 0),
                self.assertRaises(Exception),
            ):
                self.topk.topk_transform_packed_v2(
                    source,
                    lengths,
                    pages,
                    out,
                    page_size,
                    row_starts=starts,
                    row_to_batch=row_map,
                )

    def assert_disjoint_request_pages(self, pages):
        # Physical pages must be unique across request rows, so recover_raw
        # cannot accept a query that was mapped through another request.
        flat = pages.reshape(-1)
        self.assertEqual(flat.unique().numel(), flat.numel())

    def test_compact_table_keeps_request_level_cu_map(self):
        # PR187 comment 4419850010: three queries over a three-row compact
        # table must keep the request-level cu_seqlens_q map. Equal row counts
        # alone do not prove one query per table row.
        envs = self.backend_module.envs
        method = self.backend_module.TopkTransformMethod.PAGED
        layouts = (
            ([0, 2, 2, 3], [0, 0, 2]),
            # Trailing empty request.
            ([0, 1, 3, 3], [0, 1, 1]),
        )
        identity = torch.arange(3, dtype=torch.int32, device="cuda")
        for topk, page_size, use_v2, marker, layout, long_rows in product(
            (512, 2048), (1, 64), (False, True), (2, None), layouts, (False, True)
        ):
            cu_list, map_list = layout
            with self.subTest(
                topk=topk,
                page_size=page_size,
                use_v2=use_v2,
                max_seq_len_q=marker,
                cu=cu_list,
                long_rows=long_rows,
            ):
                if long_rows:
                    lengths_list = [topk + 1, topk + 63, 2 * topk + 65]
                else:
                    lengths_list = [1, topk // 4 + 1, topk]
                scores, starts, lengths = self.make_matrix(
                    max(lengths_list), [0] * 3, lengths_list, torch.bfloat16
                )
                reference = scores.clone()
                pages, _, _, _ = self.make_v2_transform(
                    scores, "paged1" if page_size == 1 else "paged64"
                )
                self.assertEqual(pages.shape[0], 3)
                self.assert_disjoint_request_pages(pages)
                mapping = torch.tensor(map_list, dtype=torch.int32, device="cuda")
                # No page_table_1 and no topk_v2_plan: the legacy transform and
                # the decode-shaped v2 paged entry must both stay unreachable.
                metadata = SimpleNamespace(
                    real_page_table=pages, page_size=page_size, page_table_1=None
                )
                if marker is not None:
                    metadata.max_seq_len_q = marker
                with (
                    envs.SGLANG_DSA_FUSE_TOPK.override(True),
                    envs.SGLANG_DSA_USE_DSV4_BF16_TOPK.override(True),
                    envs.SGLANG_OPT_USE_TOPK_V2.override(use_v2),
                ):
                    out = self.backend.topk_transform(
                        scores,
                        lengths,
                        topk,
                        method,
                        metadata,
                        cu_seqlens_q_topk=torch.tensor(
                            cu_list, dtype=torch.int32, device="cuda"
                        ),
                    )
                torch.cuda.synchronize()
                self.assertEqual(tuple(out.shape), (3, topk))
                self.assertEqual(out.dtype, torch.int32)
                raw = self.recover_raw(out, pages, mapping, None, page_size)
                self.check_output(
                    reference,
                    starts,
                    lengths,
                    out,
                    raw,
                    topk,
                    pages,
                    mapping,
                    None,
                    page_size,
                )
                if not long_rows:
                    expected = torch.full_like(out, -1)
                    for row, (request, length) in enumerate(
                        zip(map_list, lengths_list)
                    ):
                        positions = torch.arange(length, device="cuda")
                        expected[row, :length] = pages[
                            request, positions // page_size
                        ] * page_size + (positions % page_size).to(torch.int32)
                    self.assertTrue(torch.equal(out, expected))
                # Identity row mapping differs from the request map in one row;
                # the disjoint page tables must reject it.
                self.assertFalse(torch.equal(mapping, identity))
                with self.assertRaises(AssertionError):
                    self.recover_raw(out, pages, identity, None, page_size)

    def make_v2_transform(self, scores, mode):
        rows, width = scores.shape
        if mode == "ragged":
            offsets = torch.arange(rows, dtype=torch.int32, device="cuda") * 100003 + 97
            return None, None, offsets, 1
        page_size = 1 if mode == "paged1" else 64
        if mode == "raw":
            return None, None, None, page_size
        num_pages = (width + page_size - 1) // page_size
        storage = torch.empty((rows, num_pages + 3), dtype=torch.int32, device="cuda")
        pages = storage[:, :num_pages]
        pages.copy_(
            torch.randperm(rows * num_pages, device="cuda")
            .reshape(rows, num_pages)
            .to(torch.int32)
            + 1009
        )
        return pages, None, None, page_size

    def make_plan(self, lengths, threshold=-1):
        plan = self.topk.plan_topk_v2(lengths, static_threshold=threshold)
        torch.cuda.synchronize()
        return plan

    def launch_v2(
        self, scores, starts, lengths, out, mode, transform, plan=None, raw=None
    ):
        pages, mapping, offsets, page_size = transform
        if mode == "ragged":
            self.topk.topk_transform_ragged_v2(
                scores, lengths, out_offsets=offsets, out_indices=out, row_starts=starts
            )
        else:
            self.topk.topk_transform_paged_v2(
                scores, lengths, pages, out, page_size, plan, raw
            )

    def check_v2(self, scores, starts, lengths, out, topk, mode, transform, raw=None):
        if mode == "raw":
            self.check_raw(scores, starts, lengths, out, topk)
        else:
            if raw is None:
                raw = self.recover_raw(out, *transform)
            self.check_output(scores, starts, lengths, out, raw, topk, *transform)

    def run_v2_case(self, topk, mode, case, dtype):
        scores, starts, lengths = self.make_v2_inputs(topk, mode, case, dtype)
        reference = scores.clone()
        transform = self.make_v2_transform(scores, mode)
        plan = None if mode == "ragged" else self.make_plan(lengths)
        out = torch.full((scores.shape[0], topk), 171, dtype=torch.int32, device="cuda")
        raw = torch.full_like(out, 171) if mode.startswith("paged") else None
        self.launch_v2(scores, starts, lengths, out, mode, transform, plan, raw)
        torch.cuda.synchronize()
        self.check_v2(reference, starts, lengths, out, topk, mode, transform, raw)
        if mode.startswith("paged"):
            out.fill_(171)
            self.launch_v2(scores, starts, lengths, out, mode, transform, plan)
            torch.cuda.synchronize()
            self.check_v2(reference, starts, lengths, out, topk, mode, transform)
        if mode == "ragged":
            for row in range(scores.shape[0]):
                start = int(starts[row])
                allowed = torch.zeros(scores.shape[1], dtype=torch.bool, device="cuda")
                if int(lengths[row]) > topk:
                    allowed[start - start % 4 : start] = True
                self.assertTrue(bool(((scores[row] == reference[row]) | allowed).all()))

    def helper_calls(self, scores, starts, lengths, topk, mode, transform, plan):
        pages, mapping, offsets, page_size = transform
        metadata = SimpleNamespace(
            real_page_table=pages, page_size=page_size, topk_v2_plan=plan
        )

        def prefill():
            return self.backend._topk_transform_dsv4_bf16(
                scores,
                lengths,
                pages,
                None,
                topk,
                ks=starts if mode == "ragged" else None,
                topk_indices_offset=offsets,
                page_size=page_size,
            )

        def native_v2():
            if mode == "ragged":
                return self.backend_module._topk_transform_v2_ragged(
                    scores, lengths, topk, offsets, starts
                )
            return self.backend_module._topk_transform_v2_paged(
                scores, lengths, topk, metadata
            )

        return {"bf16_prefill": prefill, "v2_native_bf16": native_v2}

    def run_helper_case(self, topk, mode, case):
        scores, starts, lengths = self.make_v2_inputs(topk, mode, case, torch.bfloat16)
        reference = scores.clone()
        transform = self.make_v2_transform(scores, mode)
        plan = None if mode == "ragged" else self.make_plan(lengths)
        for name, call in self.helper_calls(
            scores, starts, lengths, topk, mode, transform, plan
        ).items():
            with self.subTest(helper=name):
                self.assertEqual(scores.dtype, torch.bfloat16)
                out = call()
                torch.cuda.synchronize()
                self.check_v2(reference, starts, lengths, out, topk, mode, transform)

    def run_no_row_starts(self, topk, dtype):
        lengths_list = [0, 1, topk, 8193, 16385, 65537]
        scores, starts, lengths = self.make_matrix(65537, [0] * 6, lengths_list, dtype)
        transform = self.make_v2_transform(scores, "ragged")
        out = torch.full((6, topk), 171, dtype=torch.int32, device="cuda")
        self.launch_v2(scores, None, lengths, out, "ragged", transform)
        torch.cuda.synchronize()
        self.check_v2(scores, starts, lengths, out, topk, "ragged", transform)

    def test_plan_shared_between_dtypes(self):
        for topk in (512, 2048):
            with self.subTest(topk=topk):
                lengths_list = [0, 1, topk - 1, topk, topk + 1, 8193, 16385, 65537] * 4
                lengths = torch.tensor(lengths_list, dtype=torch.int32, device="cuda")
                plan = self.make_plan(lengths)
                for dtype in (torch.bfloat16, torch.float32):
                    scores, starts, generated_lengths = self.make_matrix(
                        65537, [0] * 32, lengths_list, dtype
                    )
                    self.assertTrue(torch.equal(lengths, generated_lengths))
                    transform = self.make_v2_transform(scores, "paged64")
                    out = torch.full((32, topk), 171, dtype=torch.int32, device="cuda")
                    self.launch_v2(
                        scores, starts, lengths, out, "paged64", transform, plan
                    )
                    torch.cuda.synchronize()
                    self.check_v2(
                        scores, starts, lengths, out, topk, "paged64", transform
                    )

    def run_cluster_case(self, topk, dtype):
        from sglang.kernels.jit.utils import is_arch_support_pdl
        from sglang.kernels.jit.utils.occupancy import (
            NoSchedulableClustersError,
            get_max_active_clusters,
        )

        if not is_arch_support_pdl():
            self.skipTest("Device has no supported cluster path")
        try:
            capacity = get_max_active_clusters(8, occupancy=2)
        except NoSchedulableClustersError:
            self.skipTest("Device has no schedulable persistent clusters")
        rows = capacity + 1
        if not 2 <= rows <= 512:
            self.skipTest("Persistent cluster plan cannot be exercised on this device")
        lengths_list = [131073] * rows
        lengths_list[0] = 0
        scores, starts, lengths = self.make_matrix(
            131073, [0] * rows, lengths_list, dtype
        )
        plan = self.make_plan(lengths, 65536)
        self.assertEqual(int(plan[0, 0]), 65536)
        self.assertEqual(int(plan[0, 1]), rows - 1)
        items = plan[1:rows]
        self.assertTrue(
            torch.equal(
                items[:, 0].sort().values,
                torch.arange(1, rows, dtype=torch.int32, device="cuda"),
            )
        )
        self.assertTrue(bool((items[:, 1] == 131073).all()))
        transform = self.make_v2_transform(scores, "paged64")
        out = torch.full((rows, topk), 171, dtype=torch.int32, device="cuda")
        self.launch_v2(scores, starts, lengths, out, "paged64", transform, plan)
        torch.cuda.synchronize()
        self.check_v2(scores, starts, lengths, out, topk, "paged64", transform)

    def capture_launch(self, launch, launches=1):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(launches):
                launch()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        return graph

    def run_graph_case(self, topk, mode):
        starts_list = [0, 1, 2, 3, 7, 11] if mode == "ragged" else [0] * 6
        scores, starts, lengths = self.make_matrix(
            20001, starts_list, [0, 1, topk, 8192, 16384, 19990], torch.bfloat16
        )
        transform = self.make_v2_transform(scores, mode)
        plan = None if mode == "ragged" else self.make_plan(lengths)
        out = torch.full((6, topk), 171, dtype=torch.int32, device="cuda")
        raw = None if mode == "ragged" else torch.full_like(out, 171)
        graph = self.capture_launch(
            lambda: self.launch_v2(
                scores, starts, lengths, out, mode, transform, plan, raw
            )
        )
        updates = (
            [topk + 1, topk - 1, 0, 8193, 16385, 19001],
            [0, 1, topk, 8192, 16384, 19983],
        )
        for replay, new_lengths in enumerate(updates):
            lengths.copy_(torch.tensor(new_lengths, dtype=torch.int32, device="cuda"))
            if mode == "ragged":
                starts.copy_(
                    torch.tensor(
                        [3, 2, 1, 0, 11, 7] if replay == 0 else starts_list,
                        dtype=torch.int32,
                        device="cuda",
                    )
                )
                transform[2].add_(10007)
            else:
                transform[0].add_(10007)
                plan.copy_(self.make_plan(lengths))
            scores.copy_(torch.randn_like(scores))
            reference = scores.clone()
            out.fill_(171)
            if raw is not None:
                raw.fill_(171)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            self.check_v2(reference, starts, lengths, out, topk, mode, transform, raw)

    def make_equal_fp32_scores(self, scores):
        storage = torch.empty(
            (scores.shape[0], scores.stride(0)),
            dtype=torch.float32,
            device=scores.device,
        )
        converted = storage[:, : scores.shape[1]]
        converted.copy_(scores)
        self.assertEqual(converted.stride(0), scores.stride(0))
        return converted

    def test_same_value_fp32_bf16_topk(self):
        fast = importlib.import_module("sglang.kernels.ops.attention.fast_topk")
        for topk in (512, 2048):
            for mode in ("paged64", "ragged"):
                for case in _V2_CASES:
                    with self.subTest(topk=topk, mode=mode, case=case):
                        scores, starts, lengths = self.make_v2_inputs(
                            topk, mode, case, torch.bfloat16
                        )
                        reference = scores.clone()
                        scores_fp32 = self.make_equal_fp32_scores(scores)
                        self.assertTrue(
                            torch.equal(scores_fp32, scores.to(torch.float32))
                        )
                        transform = self.make_v2_transform(scores, mode)
                        plan = None if mode == "ragged" else self.make_plan(lengths)
                        for name, call in self.helper_calls(
                            scores, starts, lengths, topk, mode, transform, plan
                        ).items():
                            with self.subTest(implementation=name):
                                output = call()
                                torch.cuda.synchronize()
                                self.check_v2(
                                    reference,
                                    starts,
                                    lengths,
                                    output,
                                    topk,
                                    mode,
                                    transform,
                                )
                        output = torch.empty(
                            (scores.shape[0], topk), dtype=torch.int32, device="cuda"
                        )
                        self.launch_v2(
                            scores_fp32, starts, lengths, output, mode, transform, plan
                        )
                        torch.cuda.synchronize()
                        self.check_v2(
                            reference, starts, lengths, output, topk, mode, transform
                        )
                        if mode == "ragged":
                            output = fast.fast_topk_transform_ragged_fused(
                                scores_fp32,
                                lengths,
                                transform[2],
                                topk,
                                row_starts=starts,
                            )
                            torch.cuda.synchronize()
                            self.check_v2(
                                reference,
                                starts,
                                lengths,
                                output,
                                topk,
                                mode,
                                transform,
                            )


for _topk in (512, 2048):
    for _dtype_name, _dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        for _mode in ("paged1", "paged64", "raw", "ragged"):
            for _case in _V2_CASES:

                def _v2_test(self, topk=_topk, mode=_mode, case=_case, dtype=_dtype):
                    self.run_v2_case(topk, mode, case, dtype)

                setattr(
                    V2TopKKernelTest,
                    f"test_v2_{_dtype_name}_{_mode}_{_case}_k{_topk}",
                    _v2_test,
                )

        def _no_starts_test(self, topk=_topk, dtype=_dtype):
            self.run_no_row_starts(topk, dtype)

        setattr(
            V2TopKKernelTest,
            f"test_v2_{_dtype_name}_ragged_no_row_starts_k{_topk}",
            _no_starts_test,
        )

        def _cluster_test(self, topk=_topk, dtype=_dtype):
            self.run_cluster_case(topk, dtype)

        setattr(
            V2TopKKernelTest,
            f"test_v2_{_dtype_name}_cluster_plan_k{_topk}",
            _cluster_test,
        )
    for _mode in ("paged64", "ragged"):
        for _case in _V2_CASES:

            def _helper_test(self, topk=_topk, mode=_mode, case=_case):
                self.run_helper_case(topk, mode, case)

            setattr(
                V2TopKKernelTest,
                f"test_helpers_bf16_{_mode}_{_case}_k{_topk}",
                _helper_test,
            )
    for _mode in ("paged1", "paged64", "ragged"):

        def _graph_test(self, topk=_topk, mode=_mode):
            self.run_graph_case(topk, mode)

        setattr(V2TopKKernelTest, f"test_v2_bf16_graph_{_mode}_k{_topk}", _graph_test)


if __name__ == "__main__":
    unittest.main()
