"""Manual microbenchmark: BF16 prefill top-k helper vs. native BF16/FP32 v2 top-k.

Two comparisons, each over paged64 (32 rows) and ragged (32/64 rows) inputs,
lengths {4096, 65536} and top-k {512, 2048}:

* ``bf16``: ``top_k_per_row_prefill_bf16`` against native BF16
  ``topk_transform_{paged,ragged}_v2`` on the same BF16 score matrix, plus the
  matching DSA backend helpers.
* ``fp32``: the same BF16 kernels against native FP32 v2 on a preallocated FP32
  copy with identical values, plus the legacy ``fast_topk`` transforms and the
  helper paths that pay for a BF16->FP32 cast.

Every implementation is checked for correctness before it is timed. Kernel
times are CUDA-graph medians (32 launches per replay); helper times are wall
medians including host overhead. Warmup is 3 iterations, 7 samples, and the
call order alternates between samples. Plan preprocessing is excluded.

This is not a CI test; numerical coverage lives in
test/registered/kernels/ops/attention/test_bf16_topk_kernels.py.

Run::

    PYTHONPATH=python python benchmark/kernels/attention/bench_bf16_topk_compare.py

Requires a CUDA/PPU device. Each configuration prints one JSON line.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from types import SimpleNamespace

import torch

CASES = (("paged64", 32), ("ragged", 32), ("ragged", 64))
LENGTHS = (4096, 65536)
TOPKS = (512, 2048)
GRAPH_LAUNCHES = 32
WARMUP = 3
SAMPLES = 7


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


class TopKCompareBench:
    def __init__(self):
        self.topk = importlib.import_module("sglang.kernels.ops.attention.dsv4.topk")
        self.fast = importlib.import_module("sglang.kernels.ops.attention.fast_topk")
        self.backend_module = importlib.import_module(
            "sglang.srt.layers.attention.dsa.dsa_topk_backend"
        )
        self.backend = self.backend_module.DSATopKBackend.SGL_KERNEL

    def make_matrix(self, width, starts, lengths, dtype):
        torch.manual_seed(7919 + width + len(lengths))
        stride = ((width + 3) & ~3) + 4
        storage = torch.full((len(lengths), stride), 512.0, dtype=dtype, device="cuda")
        scores = storage[:, :width]
        for row, (start, length) in enumerate(zip(starts, lengths)):
            _require(start + length <= width, "Row window exceeds the matrix width")
            scores[row, start : start + length] = torch.randn(
                length, dtype=dtype, device="cuda"
            )
        return (
            scores,
            torch.tensor(starts, dtype=torch.int32, device="cuda"),
            torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        )

    def make_equal_fp32_scores(self, scores):
        # Keep the BF16 row stride so both dtypes read identically shaped rows.
        storage = torch.empty(
            (scores.shape[0], scores.stride(0)),
            dtype=torch.float32,
            device=scores.device,
        )
        converted = storage[:, : scores.shape[1]]
        converted.copy_(scores)
        _require(converted.stride(0) == scores.stride(0), "FP32 row stride differs")
        return converted

    def make_v2_transform(self, scores, mode):
        rows, width = scores.shape
        if mode == "ragged":
            offsets = torch.arange(rows, dtype=torch.int32, device="cuda") * 100003 + 97
            return None, None, offsets, 1
        page_size = 64
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

    def make_plan(self, lengths):
        plan = self.topk.plan_topk_v2(lengths, static_threshold=-1)
        torch.cuda.synchronize()
        return plan

    def launch_v2(self, scores, starts, lengths, out, mode, transform, plan=None):
        pages, mapping, offsets, page_size = transform
        if mode == "ragged":
            self.topk.topk_transform_ragged_v2(
                scores, lengths, out_offsets=offsets, out_indices=out, row_starts=starts
            )
        else:
            self.topk.topk_transform_paged_v2(
                scores, lengths, pages, out, page_size, plan, None
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
                page_ids = out[row, valid] // page_size
                locations = torch.searchsorted(sorted_pages, page_ids)
                _require(
                    bool((locations < sorted_pages.numel()).all())
                    and torch.equal(sorted_pages[locations], page_ids),
                    f"Output page id not in the page table at row {row}",
                )
                raw[row, valid] = (
                    order[locations].to(torch.int32) * page_size
                    + out[row, valid] % page_size
                )
        return raw

    def check_v2(self, scores, starts, lengths, out, topk, transform):
        pages, mapping, offsets, page_size = transform
        raw = self.recover_raw(out, *transform)
        for row in range(scores.shape[0]):
            start, length = int(starts[row]), int(lengths[row])
            selected = raw[row]
            valid = selected >= 0
            count = min(length, topk)
            positions = selected[valid].long()
            _require(int(valid.sum()) == count, f"Wrong selection size at row {row}")
            _require(bool((selected[~valid] == -1).all()), f"Bad padding at row {row}")
            _require(bool((positions < length).all()), f"Index out of row {row}")
            _require(positions.unique().numel() == count, f"Duplicates at row {row}")
            section = scores[row, start : start + length].float()
            got = section[positions].sort(descending=True).values
            expected = section.topk(count).values.sort(descending=True).values
            _require(torch.equal(got, expected), f"Top-k values differ at row {row}")
            expected_out = torch.full_like(out[row], -1)
            if pages is None:
                expected_out[valid] = positions.to(torch.int32) + offsets[row]
            else:
                page_row = row if mapping is None else int(mapping[row])
                expected_out[valid] = pages[
                    page_row, positions // page_size
                ] * page_size + (positions % page_size).to(torch.int32)
            _require(
                torch.equal(out[row], expected_out),
                f"Output transform differs at row {row}",
            )

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

    def capture_launch(self, launch, launches):
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

    def measure_pairs(self, calls, launches=1, wall=False):
        samples = {name: [] for name in calls}
        names = list(calls)
        for iteration in range(WARMUP):
            for name in names[:: 1 if iteration % 2 == 0 else -1]:
                calls[name]()
            torch.cuda.synchronize()
        events = (
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            if not wall
            else None
        )
        for iteration in range(SAMPLES):
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

    def check_all(self, reference, starts, lengths, topk, transform, helpers, kernels):
        for call in helpers.values():
            output = call()
            torch.cuda.synchronize()
            self.check_v2(reference, starts, lengths, output, topk, transform)
        for call, output in kernels.values():
            call()
            torch.cuda.synchronize()
            self.check_v2(reference, starts, lengths, output, topk, transform)

    def time_all(self, helpers, kernels):
        graphs = {
            name: self.capture_launch(call, GRAPH_LAUNCHES)
            for name, (call, _) in kernels.items()
        }
        kernel_us = self.measure_pairs(
            {name: graph.replay for name, graph in graphs.items()},
            launches=GRAPH_LAUNCHES,
        )
        helper_us = self.measure_pairs(helpers, wall=True)
        return kernel_us, helper_us

    def run_compare(self, mode, rows, length, topk):
        starts_list = (
            [row % 4 for row in range(rows)] if mode == "ragged" else [0] * rows
        )
        width = length + max(starts_list)
        scores, starts, lengths = self.make_matrix(
            width, starts_list, [length] * rows, torch.bfloat16
        )
        reference = scores.clone()
        transform = self.make_v2_transform(scores, mode)
        pages, mapping, offsets, page_size = transform
        plan = None if mode == "ragged" else self.make_plan(lengths)
        helpers = self.helper_calls(
            scores, starts, lengths, topk, mode, transform, plan
        )
        ends = starts + lengths
        outputs = {
            name: torch.empty((rows, topk), dtype=torch.int32, device="cuda")
            for name in helpers
        }
        kernels = {
            "bf16_prefill": (
                lambda: self.topk.top_k_per_row_prefill_bf16(
                    scores,
                    starts,
                    ends,
                    pages,
                    outputs["bf16_prefill"],
                    page_size,
                    out_offsets=offsets,
                ),
                outputs["bf16_prefill"],
            ),
            "v2_native_bf16": (
                lambda: self.launch_v2(
                    scores,
                    starts,
                    lengths,
                    outputs["v2_native_bf16"],
                    mode,
                    transform,
                    plan,
                ),
                outputs["v2_native_bf16"],
            ),
        }
        self.check_all(reference, starts, lengths, topk, transform, helpers, kernels)
        _require(scores.dtype == torch.bfloat16, "Scores were converted in place")
        kernel_us, helper_us = self.time_all(helpers, kernels)
        return {
            "comparison": "bf16_prefill_vs_v2_native_bf16",
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
            "graph_launches": GRAPH_LAUNCHES,
            "warmup": WARMUP,
            "samples": SAMPLES,
            "alternating_order": True,
            "plan_preprocessing_excluded": True,
            "plan_source": "lengths_only; ragged has no plan",
            "correctness_checked_before_timing": True,
        }

    def run_fp32_compare(self, mode, rows, length, topk):
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
        cu = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
        fast_pages = None
        if pages is not None:
            # The legacy paged transform takes a page_size=1 table; build it
            # outside the timed region.
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
            "bf16_prefill": (
                lambda: self.topk.top_k_per_row_prefill_bf16(
                    scores,
                    starts,
                    ends,
                    pages,
                    outputs["bf16_prefill"],
                    page_size,
                    out_offsets=offsets,
                ),
                outputs["bf16_prefill"],
            ),
            "v2_native_bf16": (
                lambda: self.launch_v2(
                    scores,
                    starts,
                    lengths,
                    outputs["v2_native_bf16"],
                    mode,
                    transform,
                    plan,
                ),
                outputs["v2_native_bf16"],
            ),
            "v2_native_fp32": (
                lambda: self.launch_v2(
                    scores_fp32,
                    starts,
                    lengths,
                    outputs["v2_native_fp32"],
                    mode,
                    transform,
                    plan,
                ),
                outputs["v2_native_fp32"],
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
                return self.fast.fast_topk_transform_ragged_fused(
                    source, lengths, offsets, topk, row_starts=starts
                )
            return self.fast.fast_topk_transform_fused(
                source, lengths, fast_pages, cu, topk
            )

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
        self.check_all(reference, starts, lengths, topk, transform, helpers, kernels)
        kernel_us, helper_us = self.time_all(helpers, kernels)
        return {
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
            "graph_launches": GRAPH_LAUNCHES,
            "warmup": WARMUP,
            "samples": SAMPLES,
            "alternating_order": True,
            "plan_preprocessing_excluded": True,
            "correctness_checked_before_timing": True,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--comparison",
        choices=("bf16", "fp32", "all"),
        default="all",
        help="Which comparison to run (default: all).",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA/PPU device is required")
    bench = TopKCompareBench()
    runners = []
    if args.comparison in ("bf16", "all"):
        runners.append(bench.run_compare)
    if args.comparison in ("fp32", "all"):
        runners.append(bench.run_fp32_compare)
    for run in runners:
        for mode, rows in CASES:
            for length in LENGTHS:
                for topk in TOPKS:
                    print(
                        json.dumps(run(mode, rows, length, topk), sort_keys=True),
                        flush=True,
                    )


if __name__ == "__main__":
    main()
