# SPDX-License-Identifier: Apache-2.0
"""CUDA Graph / Kineto GPU-duration timing, with complete replay attribution."""

import gzip
import json
import statistics
import time
from pathlib import Path

import torch


def stats(x):
    return dict(
        median=statistics.median(x), min=min(x), max=max(x), mean=statistics.mean(x)
    )


def profile(fn, label, folder, rounds=5, iters=20, warmup=30, soak_seconds=0):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        keep = fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    end = time.monotonic() + soak_seconds
    while time.monotonic() < end:
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
    rows = []
    for r in range(rounds):
        start, stop = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iters):
            graph.replay()
        stop.record()
        stop.synchronize()
        event_us = start.elapsed_time(stop) * 1000 / iters
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        ) as p:
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            for _ in range(iters):
                graph.replay()
            torch.cuda.synchronize()
        path = folder / f"{label}_r{r}.json"
        p.export_chrome_trace(str(path))
        trace = json.loads(path.read_text())
        ev = trace["traceEvents"]
        launches = sorted(
            [
                e
                for e in ev
                if e.get("cat") == "cuda_runtime" and e.get("name") == "cudaGraphLaunch"
            ],
            key=lambda e: e["ts"],
        )
        assert len(launches) == iters + 5, (label, len(launches))
        cs = [e["args"]["correlation"] for e in launches[-iters:]]
        groups = [
            sorted(
                [
                    e
                    for e in ev
                    if e.get("cat") in ["kernel", "gpu_memcpy", "gpu_memset"]
                    and e.get("args", {}).get("correlation") == c
                    and e.get("ph") == "X"
                ],
                key=lambda e: e["ts"],
            )
            for c in cs
        ]
        assert all(groups), (label, "Missing graph GPU events")
        sequence = [(e["cat"], e["name"]) for e in groups[-1]]
        for g in groups:
            assert [(e["cat"], e["name"]) for e in g] == sequence, (
                label,
                "Incomplete event sequence",
            )
            for x, y in zip(g, g[1:]):
                assert x["ts"] + x["dur"] <= y["ts"] + 0.1, (
                    label,
                    "Overlapping GPU events",
                )
        times = [sum(e["dur"] for e in g) for g in groups]
        stages = [
            dict(
                name=name,
                category=cat,
                per_call_us=[
                    sum(e["dur"] for e in g if e["name"] == name and e["cat"] == cat)
                    for g in groups
                ],
            )
            for cat, name in dict.fromkeys(sequence)
        ]
        rows.append(
            dict(
                round=r,
                gpu_us=times,
                correlations=cs,
                event_us=event_us,
                sequence=sequence,
                stages=stages,
                trace=path.name + ".gz",
            )
        )
        with gzip.open(str(path) + ".gz", "wb") as f:
            f.write(path.read_bytes())
        path.unlink()
    result = dict(
        gpu_us=stats([statistics.median(r["gpu_us"]) for r in rows]),
        rounds=rows,
        iters=iters,
        warmup=warmup,
        soak_seconds=soak_seconds,
    )
    del graph, keep
    return result
