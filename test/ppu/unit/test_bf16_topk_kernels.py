import importlib
import json
import os
import statistics
import time
import unittest
from types import SimpleNamespace

import torch


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

    def measure_pairs(self, calls, launches=1, wall=False):
        samples = {name: [] for name in calls}
        names = list(calls)
        for iteration in range(3):
            for name in names[:: 1 if iteration % 2 == 0 else -1]:
                calls[name]()
            torch.cuda.synchronize()
        events = (
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            if not wall
            else None
        )
        for iteration in range(7):
            for name in names[:: 1 if iteration % 2 == 0 else -1]:
                torch.cuda.synchronize()
                if wall:
                    start = time.perf_counter_ns()
                    result = calls[name]()
                    torch.cuda.synchronize()
                    elapsed_us = (time.perf_counter_ns() - start) / 1000
                    del result
                else:
                    events[0].record()
                    calls[name]()
                    events[1].record()
                    events[1].synchronize()
                    elapsed_us = events[0].elapsed_time(events[1]) * 1000 / launches
                samples[name].append(elapsed_us)
        return {name: statistics.median(values) for name, values in samples.items()}

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

    @unittest.skipUnless(
        os.environ.get("SGLANG_TOPK_COMPARE_BENCH") == "1",
        "Optional Top-K comparison benchmark disabled",
    )
    def test_optional_fp32_compare_benchmark(self):
        for mode, rows in (("paged64", 32), ("ragged", 32), ("ragged", 64)):
            for length in (4096, 65536):
                for topk in (512, 2048):
                    with self.subTest(mode=mode, rows=rows, length=length, topk=topk):
                        self.run_fp32_compare_benchmark(mode, rows, length, topk)

    def run_fp32_compare_benchmark(self, mode, rows, length, topk):
        starts_list = (
            [row % 4 for row in range(rows)] if mode == "ragged" else [0] * rows
        )
        width = (length + max(starts_list) + 3) & ~3
        scores, starts, lengths = self.make_matrix(
            width, starts_list, [length] * rows, torch.bfloat16
        )
        reference = scores.clone()
        scores_fp32 = self.make_equal_fp32_scores(scores)
        transform = self.make_v2_transform(scores, mode)
        pages, mapping, offsets, page_size = transform
        plan = None if mode == "ragged" else self.make_plan(lengths)
        metadata = SimpleNamespace(
            real_page_table=pages, page_size=page_size, topk_v2_plan=plan
        )
        fast = importlib.import_module("sglang.kernels.ops.attention.fast_topk")
        cu = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
        fast_pages = None
        if pages is not None:
            positions = torch.arange(width, device="cuda")
            fast_pages = pages[:, positions // page_size] * page_size + (
                positions % page_size
            ).to(torch.int32)
        ends = starts + lengths
        helpers = self.helper_calls(
            scores, starts, lengths, topk, mode, transform, plan
        )
        outputs = {
            name: torch.empty((rows, topk), dtype=torch.int32, device="cuda")
            for name in ("bf16_prefill", "v2_native_bf16", "v2_native_fp32")
        }
        kernels = {
            "bf16_prefill": lambda: self.topk.top_k_per_row_prefill_bf16(
                scores,
                starts,
                ends,
                pages,
                outputs["bf16_prefill"],
                page_size,
                out_offsets=offsets,
            ),
            "v2_native_bf16": lambda: self.launch_v2(
                scores,
                starts,
                lengths,
                outputs["v2_native_bf16"],
                mode,
                transform,
                plan,
            ),
            "v2_native_fp32": lambda: self.launch_v2(
                scores_fp32,
                starts,
                lengths,
                outputs["v2_native_fp32"],
                mode,
                transform,
                plan,
            ),
        }

        def v2_helper(source):
            if mode == "ragged":
                return self.backend_module._topk_transform_v2_ragged(
                    source, lengths, topk, offsets, starts
                )
            return self.backend_module._topk_transform_v2_paged(
                source, lengths, topk, metadata
            )

        def fast_helper(source):
            if mode == "ragged":
                return fast.fast_topk_transform_ragged_fused(
                    source, lengths, offsets, topk, row_starts=starts
                )
            return fast.fast_topk_transform_fused(source, lengths, fast_pages, cu, topk)

        helpers.update(
            {
                "v2_native_fp32": lambda: v2_helper(scores_fp32),
                "fast_native_bf16": lambda: fast_helper(scores),
                "fast_native_fp32": lambda: fast_helper(scores_fp32),
                "v2_fp32_with_bf16_cast": lambda: v2_helper(scores.to(torch.float32)),
                "fast_fp32_with_bf16_cast": lambda: fast_helper(
                    scores.to(torch.float32)
                ),
            }
        )
        for name, call in helpers.items():
            with self.subTest(helper=name):
                output = call()
                torch.cuda.synchronize()
                self.check_v2(reference, starts, lengths, output, topk, mode, transform)
        for name, call in kernels.items():
            call()
            torch.cuda.synchronize()
            self.check_v2(
                reference, starts, lengths, outputs[name], topk, mode, transform
            )
        graphs = {name: self.capture_launch(call, 32) for name, call in kernels.items()}
        kernel_us = self.measure_pairs(
            {name: graph.replay for name, graph in graphs.items()}, launches=32
        )
        helper_us = self.measure_pairs(helpers, wall=True)
        print(
            json.dumps(
                {
                    "comparison": "equal_value_bf16_vs_fp32",
                    "mode": mode,
                    "rows": rows,
                    "length": length,
                    "topk": topk,
                    "shape": list(scores.shape),
                    "input_source": "one_bf16_matrix; fp32_preallocated_copy_has_identical_values",
                    "input_dtypes": {
                        "bf16_prefill": str(scores.dtype),
                        "v2_native_bf16": str(scores.dtype),
                        "v2_native_fp32": str(scores_fp32.dtype),
                    },
                    "kernel_cuda_graph_median_us": kernel_us,
                    "helper_wall_median_us": helper_us,
                    "fp32_preparation_excluded_from_native_timing": True,
                    "cast_cost_included_only_in_helpers": [
                        "v2_fp32_with_bf16_cast",
                        "fast_fp32_with_bf16_cast",
                    ],
                    "fast_page_table_expansion_excluded": True,
                    "graph_launches": 32,
                    "warmup": 3,
                    "samples": 7,
                    "alternating_order": True,
                    "plan_preprocessing_excluded": True,
                    "correctness_checked_before_timing": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    @unittest.skipUnless(
        os.environ.get("SGLANG_TOPK_COMPARE_BENCH") == "1",
        "Optional Top-K comparison benchmark disabled",
    )
    def test_optional_compare_benchmark(self):
        for mode, rows in (("paged64", 32), ("ragged", 32), ("ragged", 64)):
            for length in (4096, 65536):
                for topk in (512, 2048):
                    with self.subTest(mode=mode, rows=rows, length=length, topk=topk):
                        self.run_compare_benchmark(mode, rows, length, topk)

    def run_compare_benchmark(self, mode, rows, length, topk):
        starts_list = (
            [row % 4 for row in range(rows)] if mode == "ragged" else [0] * rows
        )
        width = length + max(starts_list)
        scores, starts, lengths = self.make_matrix(
            width, starts_list, [length] * rows, torch.bfloat16
        )
        reference = scores.clone()
        transform = self.make_v2_transform(scores, mode)
        plan = None if mode == "ragged" else self.make_plan(lengths)
        helpers = self.helper_calls(
            scores, starts, lengths, topk, mode, transform, plan
        )
        ends = starts + lengths
        outputs = {
            name: torch.empty((rows, topk), dtype=torch.int32, device="cuda")
            for name in helpers
        }
        pages, mapping, offsets, page_size = transform

        def prefill_kernel():
            self.topk.top_k_per_row_prefill_bf16(
                scores,
                starts,
                ends,
                pages,
                outputs["bf16_prefill"],
                page_size,
                out_offsets=offsets,
            )

        kernels = {
            "bf16_prefill": prefill_kernel,
            "v2_native_bf16": lambda: self.launch_v2(
                scores,
                starts,
                lengths,
                outputs["v2_native_bf16"],
                mode,
                transform,
                plan,
            ),
        }
        for name, call in helpers.items():
            self.assertEqual(scores.dtype, torch.bfloat16)
            out = call()
            torch.cuda.synchronize()
            self.check_v2(reference, starts, lengths, out, topk, mode, transform)
        for name, call in kernels.items():
            self.assertEqual(scores.dtype, torch.bfloat16)
            call()
            torch.cuda.synchronize()
            self.check_v2(
                reference, starts, lengths, outputs[name], topk, mode, transform
            )
        launches = 32
        graphs = {
            name: self.capture_launch(call, launches) for name, call in kernels.items()
        }
        kernel_us = self.measure_pairs(
            {name: graph.replay for name, graph in graphs.items()}, launches=launches
        )
        helper_us = self.measure_pairs(helpers, wall=True)
        print(
            json.dumps(
                {
                    "mode": "ragged" if mode == "ragged" else "paged",
                    "rows": rows,
                    "shape": list(scores.shape),
                    "row_stride": scores.stride(0),
                    "length": length,
                    "topk": topk,
                    "page_size": page_size,
                    "input_dtypes": {name: str(scores.dtype) for name in helpers},
                    "same_input_matrix": True,
                    "full_scores_float_conversion": False,
                    "kernel_cuda_graph_median_us": kernel_us,
                    "helper_wall_median_us": helper_us,
                    "graph_launches": launches,
                    "warmup": 3,
                    "samples": 7,
                    "alternating_order": True,
                    "plan_preprocessing_excluded": True,
                    "plan_source": "lengths_only; ragged has no plan",
                    "correctness_checked_before_timing": True,
                },
                sort_keys=True,
            ),
            flush=True,
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
