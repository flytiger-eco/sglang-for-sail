# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

path = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/multimodal/aim_metrics.py"
)
spec = importlib.util.spec_from_file_location("aim_metrics_under_test", path)
metrics = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = metrics
spec.loader.exec_module(metrics)


class TestAIMMetrics(unittest.TestCase):
    def tearDown(self):
        metrics.install_image_metrics(None)

    def make_metrics(self, role="tokenizer"):
        self.counter = Mock()
        self.histogram = Mock()
        return metrics.ImageIOMetrics(
            {"model_name": "qwen38"}, role, self.counter, self.histogram
        )

    def test_records_cache_outcomes_and_nested_stage_separately(self):
        recorder = self.make_metrics()
        for outcome in ["hit", "miss", "error"]:
            recorder.record("redkv_cache", outcome, 0, 2)
        self.assertEqual(self.counter.return_value.labels.call_count, 3)
        self.histogram.return_value.labels.assert_not_called()
        recorder.record("image_load", "success", 0.5, 4)
        recorder.record("image_decode", "success", 0.1, 4)
        observed = (
            self.histogram.return_value.labels.return_value.observe.call_args_list
        )
        self.assertEqual([call.args[0] for call in observed], [0.5, 0.1])

    def test_worker_uses_installed_recorder_without_registration(self):
        recorder = self.make_metrics()
        metrics.install_image_metrics(recorder.record)
        workers = [
            threading.Thread(
                target=metrics.record_image_event,
                args=("image_http", "success", 0.2, 1),
            )
            for _ in range(5)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(self.counter.call_count, 1)
        self.assertEqual(self.histogram.call_count, 1)
        self.assertEqual(self.counter.return_value.labels.call_count, 5)

    def test_rejects_unbounded_labels_and_nonfinite_observations(self):
        recorder = self.make_metrics()
        for stage, result, seconds, count in [
            ("https://private/key", "success", 1, 1),
            ("image_load", "request-id", 1, 1),
            ("image_load", "success", float("nan"), 1),
            ("image_load", "success", 1, 0),
        ]:
            recorder.record(stage, result, seconds, count)
        self.counter.return_value.labels.assert_not_called()

    def test_timer_reports_failures_and_does_not_swallow_them(self):
        record = Mock()
        metrics.install_image_metrics(record)
        with self.assertRaisesRegex(ValueError, "decode"):
            with metrics.image_stage("image_decode"):
                raise ValueError("decode")
        self.assertEqual(record.call_args.args[:2], ("image_decode", "error"))
        self.assertGreaterEqual(record.call_args.args[2], 0)

    def test_plain_text_timer_is_disabled_and_placeholder_is_count_only(self):
        record = Mock()
        metrics.install_image_metrics(record)
        with metrics.image_stage("image_tokenization", enabled=False):
            pass
        record.assert_not_called()
        recorder = self.make_metrics()
        recorder.record("image_fallback", "fallback", 0, 1)
        self.counter.return_value.labels.return_value.inc.assert_called_once_with(1)
        self.histogram.return_value.labels.assert_not_called()

    def test_optional_prometheus_registry_handles_both_roles(self):
        try:
            from prometheus_client import (
                CollectorRegistry,
                Counter,
                Histogram,
                generate_latest,
            )
        except ImportError:
            self.skipTest("prometheus_client is required for registry integration")
        registry = CollectorRegistry()
        counter = lambda **kwargs: Counter(registry=registry, **kwargs)
        histogram = lambda **kwargs: Histogram(registry=registry, **kwargs)
        for role in ["tokenizer", "encoder"]:
            recorder = metrics.ImageIOMetrics(
                {"model_name": "qwen38"}, role, counter, histogram
            )
            for _ in range(3):
                recorder.record("redkv_get", "error", 0.1, 1)
        output = generate_latest(registry).decode()
        self.assertIn('result="error"', output)
        self.assertIn("aim_tokenizer_mm_items_total", output)
        self.assertIn("aim_encoder_mm_items_total", output)
        self.assertNotIn('result="miss"', output)


if __name__ == "__main__":
    unittest.main()
