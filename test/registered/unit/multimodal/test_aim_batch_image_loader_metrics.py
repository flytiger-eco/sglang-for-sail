"""Whole-batch timing and cancellation labels use the single-image contracts."""

import importlib.util
import sys
import threading
import unittest
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/multimodal/aim_image_loader.py"
)
SPEC = importlib.util.spec_from_file_location(
    "aim_image_loader_metrics_test", MODULE_PATH
)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


class TestImageBatchMetrics(unittest.TestCase):
    def setUp(self):
        self.clock = 0.0
        self.events = []
        self.key = "imagetransfer_https://images/one"
        self.pool = ThreadPoolExecutor(1)
        connector = SimpleNamespace(
            is_redkv_url=lambda value: value.startswith("imagetransfer_http"),
            fallback_url=lambda value: value[len("imagetransfer_") :],
        )
        self.modules = patch.dict(
            sys.modules, {"sglang.srt.connector.redkv": connector}
        )
        self.timer = patch.object(loader, "perf_counter", lambda: self.clock)
        self.modules.start()
        self.timer.start()

    def tearDown(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.timer.stop()
        self.modules.stop()

    def load(self, sources=None, **kwargs):
        return loader.load_images_batch(
            [self.key] if sources is None else sources,
            executor=self.pool,
            max_in_flight=1,
            record=lambda *event: self.events.append(event),
            **kwargs,
        )

    def test_image_load_includes_mget_and_loading_time(self):
        def mget(keys):
            self.clock += 5.0
            return SimpleNamespace(values={keys[0]: b"image"}, error=None)

        def load_one(value):
            self.clock += 2.0
            return value

        self.assertEqual(
            self.load(enable_mget=True, mget=mget, load_one=load_one), [b"image"]
        )
        self.assertIn(("redkv_mget", "all_hit", 5.0, 1), self.events)
        self.assertIn(("image_load", "success", 7.0, 1), self.events)

    def test_mget_exception_records_total_error_and_preserves_exception(self):
        failure = RuntimeError("cache adapter failed")

        def mget(keys):
            self.clock += 5.0
            raise failure

        with self.assertRaises(RuntimeError) as caught:
            self.load(enable_mget=True, mget=mget, load_one=self.fail)
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.events, [("image_load", "error", 5.0, 1)])

    def test_cancel_after_mget_records_total_cancel_and_does_not_load(self):
        cancel = threading.Event()

        def mget(keys):
            self.clock += 5.0
            cancel.set()
            return SimpleNamespace(values={keys[0]: b"image"}, error=None)

        with self.assertRaises(CancelledError):
            self.load(
                enable_mget=True, mget=mget, load_one=self.fail, cancel_event=cancel
            )
        self.assertIn(("image_load", "cancel", 5.0, 1), self.events)
        self.assertFalse(any(event[1] == "error" for event in self.events))

    def test_already_cancelled_request_records_cancel_without_mget(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            self.load(
                enable_mget=True,
                mget=self.fail,
                load_one=self.fail,
                cancel_event=cancel,
            )
        self.assertEqual(self.events, [("image_load", "cancel", 0.0, 1)])

    def test_cpu_fetch_cancel_is_not_counted_as_error_or_fallback(self):
        def fetch(source):
            self.clock += 3.0
            raise CancelledError()

        with self.assertRaises(CancelledError):
            self.load(
                ["https://images/one"],
                cpu_jpeg_decode=True,
                get_bytes=fetch,
                on_error=lambda source, error: self.fail("cancel cannot fallback"),
                load_one=self.fail,
            )
        self.assertEqual(
            self.events,
            [("image_fetch", "cancel", 3.0, 1), ("image_load", "cancel", 3.0, 1)],
        )

    def test_cpu_decode_cancel_is_not_counted_as_error_or_retried(self):
        def decode(source):
            self.clock += 2.0
            raise CancelledError()

        with self.assertRaises(CancelledError):
            self.load(
                [b"\xff\xd8image"],
                cpu_jpeg_decode=True,
                get_bytes=self.fail,
                decode_cpu_jpeg=decode,
                load_one=self.fail,
            )
        self.assertEqual(
            self.events,
            [("image_decode", "cancel", 2.0, 1), ("image_load", "cancel", 2.0, 1)],
        )


if __name__ == "__main__":
    unittest.main()
