# SPDX-License-Identifier: Apache-2.0
"""AIM image-stage metrics attached to the existing serving collectors.

No Prometheus objects are created at import or per request. The tokenizer or
encoder collector installs the process recorder after multiprocess metrics are
configured. Executor threads share its thread-safe counter/histogram handles.
"""

import math
from contextlib import contextmanager
from time import perf_counter

_recorder = None
_STAGES = frozenset(
    {
        "redkv_get",
        "redkv_mget",
        "redkv_cache",
        "redkv_fallback_http",
        "image_http",
        "image_fetch",
        "image_load",
        "image_decode",
        "image_load_single",
        "image_decode_single",
        "image_fallback",
        "jpeg_cpu_decode",
        "image_tokenization",
        "encoder_prepare_queue",
        "encoder_prepare",
    }
)
_RESULTS = frozenset(
    {
        "success",
        "error",
        "cancel",
        "hit",
        "miss",
        "all_hit",
        "all_miss",
        "partial_miss",
        "fallback",
    }
)


def install_image_metrics(recorder):
    """Install once when an existing collector is constructed in this process."""
    global _recorder
    _recorder = recorder


def record_image_event(stage, result, seconds, count=1):
    if _recorder is not None:
        _recorder(stage, result, seconds, count)


@contextmanager
def image_stage(stage, *, enabled=True):
    if not enabled:
        yield
        return
    started = perf_counter()
    try:
        yield
    except BaseException as exc:
        result = "cancel" if type(exc).__name__ == "CancelledError" else "error"
        record_image_event(stage, result, perf_counter() - started)
        raise
    else:
        record_image_event(stage, "success", perf_counter() - started)


class ImageIOMetrics:
    """Use collector-owned metric classes, preserving custom stat-logger sinks.

    Durations are nested: image_decode is inside image_load and HTTP/cache I/O
    may be inside image_fetch. Do not add these stage histograms to compute TTFT.
    jpeg_cpu_decode is aggregate worker time; image_decode is batch wall time.
    Existing encoder preprocess/forward and request TTFT metrics remain the
    source of truth for end-to-end latency.
    """

    def __init__(self, labels, role, counter_cls, histogram_cls):
        if role not in ("tokenizer", "encoder"):
            raise ValueError("Image metric role must be tokenizer or encoder")
        self.labels = dict(labels)
        labelnames = [*self.labels, "stage", "result"]
        # Each existing role collector is constructed once in its process.
        # Distinct role names also allow both collectors in integration tests.
        self.items = counter_cls(
            name=f"sglang:aim_{role}_mm_items_total",
            documentation="AIM image items by stage and result; cache errors are not misses.",
            labelnames=labelnames,
        )
        self.seconds = histogram_cls(
            name=f"sglang:aim_{role}_mm_stage_seconds",
            documentation="AIM multimodal stage durations; nested stages must not be summed.",
            labelnames=labelnames,
            buckets=[
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,
                2,
                5,
                10,
                30,
                60,
            ],
        )

    def record(self, stage, result, seconds, count=1):
        # Reject unbounded labels, non-finite values and no-work events without
        # disrupting an otherwise valid serving request.
        if stage not in _STAGES or result not in _RESULTS:
            return
        if count <= 0 or seconds < 0 or not math.isfinite(seconds):
            return
        labels = dict(self.labels, stage=stage, result=result)
        self.items.labels(**labels).inc(count)
        # redkv_cache is a hit/miss/error count, not an extra timed operation.
        if stage not in ("redkv_cache", "image_fallback"):
            self.seconds.labels(**labels).observe(seconds)
