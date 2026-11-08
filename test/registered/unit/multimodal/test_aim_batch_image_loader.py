"""CPU-only tests; direct module loading avoids SGLang's optional GPU imports."""

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
    "aim_image_loader_under_test", MODULE_PATH
)
loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(loader)


class CountingExecutor(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submissions = 0
        self.lock = threading.Lock()

    def submit(self, *args, **kwargs):
        with self.lock:
            self.submissions += 1
        return super().submit(*args, **kwargs)


class TestBoundedImageLoader(unittest.TestCase):
    def test_out_of_order_completion_preserves_order_and_submission_bound(self):
        releases = [threading.Event() for _ in range(5)]
        entered = [threading.Event() for _ in range(5)]

        def work(index):
            entered[index].set()
            self.assertTrue(releases[index].wait(3))
            return index * 2

        with CountingExecutor(max_workers=5) as pool, ThreadPoolExecutor(1) as caller:
            future = caller.submit(
                loader.bounded_map,
                work,
                list(range(5)),
                executor=pool,
                max_in_flight=2,
            )
            self.assertTrue(entered[0].wait(3))
            self.assertTrue(entered[1].wait(3))
            self.assertEqual(pool.submissions, 2)
            releases[1].set()
            self.assertTrue(entered[2].wait(3))
            self.assertEqual(pool.submissions, 3)
            for event in releases:
                event.set()
            self.assertEqual(future.result(3), [0, 2, 4, 6, 8])
            self.assertEqual(
                pool.submit(lambda: "still usable").result(), "still usable"
            )

    def test_error_stops_submission_and_preserves_error(self):
        failure = ValueError("bad image")

        def work(_):
            raise failure

        with CountingExecutor(max_workers=1) as pool:
            with self.assertRaises(ValueError) as caught:
                loader.bounded_map(work, range(100), executor=pool, max_in_flight=1)
            self.assertIs(caught.exception, failure)
            self.assertEqual(pool.submissions, 1)

    def test_cancelled_batch_does_not_submit_more_or_own_executor(self):
        started, release, cancel = (threading.Event() for _ in range(3))
        calls = []

        def work(index):
            calls.append(index)
            started.set()
            release.wait(3)

        with CountingExecutor(max_workers=1) as pool, ThreadPoolExecutor(1) as caller:
            future = caller.submit(
                loader.bounded_map,
                work,
                range(100),
                executor=pool,
                max_in_flight=4,
                cancel_event=cancel,
            )
            try:
                self.assertTrue(started.wait(3))
                cancel.set()
                with self.assertRaises(CancelledError):
                    future.result(3)
                self.assertLessEqual(pool.submissions, 4)
            finally:
                release.set()
            self.assertEqual(pool.submit(lambda: 42).result(3), 42)
            self.assertEqual(calls, [0])

    def test_nested_coordinator_is_rejected(self):
        with ThreadPoolExecutor(1) as pool:
            pool.submit(
                lambda: None
            ).result()  # Ensure worker membership is registered.
            future = pool.submit(
                loader.bounded_map, str, [1], executor=pool, max_in_flight=1
            )
            with self.assertRaisesRegex(RuntimeError, "cannot run in its I/O executor"):
                future.result(3)

    def test_cancelled_before_loading_does_not_fetch(self):
        cancel = threading.Event()
        cancel.set()
        with CountingExecutor(max_workers=1) as pool:
            with self.assertRaises(CancelledError):
                loader.load_images_batch(
                    ["https://image"],
                    executor=pool,
                    max_in_flight=1,
                    load_one=self.fail,
                    cancel_event=cancel,
                )
            self.assertEqual(pool.submissions, 0)

    def test_mget_hit_miss_and_duplicates_keep_order_without_second_get(self):
        key_a = "imagetransfer_https://images/a"
        key_b = "imagetransfer_https://images/b"
        observed_keys, records = [], []
        coordinator = threading.get_ident()

        def mget(keys):
            observed_keys.append(keys)
            return SimpleNamespace(values={key_a: b"a", key_b: None}, error=None)

        def record(*args):
            self.assertEqual(threading.get_ident(), coordinator)
            records.append(args)

        connector = SimpleNamespace(
            is_redkv_url=lambda value: value.startswith("imagetransfer_http"),
            fallback_url=lambda value: value[len("imagetransfer_") :],
        )
        with patch.dict(sys.modules, {"sglang.srt.connector.redkv": connector}):
            with ThreadPoolExecutor(2) as pool:
                result = loader.load_images_batch(
                    [key_b, SimpleNamespace(url=key_a), "ordinary", key_a],
                    executor=pool,
                    max_in_flight=2,
                    load_one=lambda value: (value, None),
                    enable_mget=True,
                    mget=mget,
                    record=record,
                )
        self.assertEqual(observed_keys, [[key_b, key_a]])
        self.assertEqual(
            [item[0] for item in result], ["https://images/b", b"a", "ordinary", b"a"]
        )
        self.assertEqual(records[0][:2], ("redkv_mget", "partial_miss"))
        self.assertIn(("redkv_cache", "hit", 0.0, 2), records)
        self.assertIn(("redkv_cache", "miss", 0.0, 1), records)

    def test_mget_error_falls_back_but_is_not_counted_as_miss(self):
        records = []
        connector = SimpleNamespace(
            is_redkv_url=lambda value: value.startswith("imagetransfer_http"),
            fallback_url=lambda value: value[len("imagetransfer_") :],
        )
        with patch.dict(sys.modules, {"sglang.srt.connector.redkv": connector}):
            with ThreadPoolExecutor(1) as pool:
                result = loader.load_images_batch(
                    ["imagetransfer_https://images/a"],
                    executor=pool,
                    max_in_flight=1,
                    load_one=lambda value: value,
                    enable_mget=True,
                    mget=lambda keys: SimpleNamespace(values={}, error="connection"),
                    record=lambda *args: records.append(args),
                )
        self.assertEqual(result, ["https://images/a"])
        self.assertEqual(records[0][:2], ("redkv_mget", "error"))
        self.assertIn(("redkv_cache", "error", 0.0, 1), records)
        self.assertFalse(any(item[1] == "miss" for item in records))


if __name__ == "__main__":
    unittest.main()
