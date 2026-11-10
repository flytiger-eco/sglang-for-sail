"""Exercise real serving collectors and image metrics in a SGLang CPU runtime."""

import os
import unittest
from functools import partial
from unittest.mock import patch

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from sglang.srt.multimodal.aim_metrics import (
    install_image_metrics,
    record_image_event,
)
from sglang.srt.observability.metrics_collector import (
    EncoderMetricsCollector,
    TokenizerMetricsCollector,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import load_image
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAIMCollectorIntegration(unittest.TestCase):
    def tearDown(self):
        install_image_metrics(None)

    def make_collector(self, role):
        registry = CollectorRegistry()
        labels = {"model_name": "aim-test"}
        cls = (
            TokenizerMetricsCollector
            if role == "tokenizer"
            else EncoderMetricsCollector
        )
        with (
            patch.object(cls, "_counter_cls", partial(Counter, registry=registry)),
            patch.object(cls, "_gauge_cls", partial(Gauge, registry=registry)),
            patch.object(cls, "_histogram_cls", partial(Histogram, registry=registry)),
        ):
            if role == "tokenizer":
                from sglang.srt.runtime_context import publish
                args = ServerArgs(model_path="dummy")
                publish(args, role="tokenizer")
                collector = cls(server_args=args, labels=labels)
            else:
                collector = cls(labels=labels)
        return registry, collector

    def test_real_collectors_install_recorder_without_per_request_registration(self):
        for role in ("tokenizer", "encoder"):
            with self.subTest(role=role):
                registry, collector = self.make_collector(role)
                for _ in range(3):
                    record_image_event("redkv_cache", "error", 0, 2)
                self.assertEqual(
                    registry.get_sample_value(
                        f"sglang:aim_{role}_mm_items_total",
                        {
                            "model_name": "aim-test",
                            "stage": "redkv_cache",
                            "result": "error",
                        },
                    ),
                    6,
                )
                self.assertIsNotNone(collector.aim_image_metrics)

    def test_real_loader_records_placeholder_separately_from_completion(self):
        registry, _ = self.make_collector("tokenizer")
        with patch.dict(
            os.environ, {"SGLANG_MM_RETURN_1PX_IMAGE_ON_LOAD_FAILURE": "true"}
        ):
            image, _ = load_image(b"not an image", gpu_image_decode=False)
        self.assertEqual(image.size, (1, 1))
        for stage, result in (
            ("image_fallback", "fallback"),
            ("image_load_single", "success"),
        ):
            self.assertEqual(
                registry.get_sample_value(
                    "sglang:aim_tokenizer_mm_items_total",
                    {"model_name": "aim-test", "stage": stage, "result": result},
                ),
                1,
            )



    def test_deferred_processor_tokenization_records_once(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        import torch
        from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
        events = []
        install_image_metrics(lambda *event: events.append(event))
        processor = SimpleNamespace(_tokenizer=SimpleNamespace(encode=Mock(return_value=[1, 7, 3])),
                                    IM_TOKEN_ID=7, VIDEO_TOKEN_ID=8, spatial_merge_size=2)
        grid = torch.tensor([[1, 2, 2]])
        BaseMultimodalProcessor.build_input_ids(processor, "raw image prompt", img_grid_thw=grid)
        self.assertEqual([(event[0], event[1], event[3]) for event in events],
                         [("image_tokenization", "success", 1)])
        BaseMultimodalProcessor.build_input_ids(processor, [1, 7, 3], img_grid_thw=grid)
        self.assertEqual(len(events), 1)

if __name__ == "__main__":
    unittest.main()
