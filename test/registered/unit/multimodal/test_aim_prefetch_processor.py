"""Execute actual BaseMultimodalProcessor loading methods without GPU imports.

Only external model/framework types are stubbed; the class methods are compiled
directly from the maintained source, and use the real bounded batch loader.
"""

import ast
import asyncio
import concurrent.futures
import importlib.util
import io
import logging
import re
import threading
import unittest
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
BASE_PATH = ROOT / "python/sglang/srt/multimodal/processors/base_processor.py"
LOADER_PATH = ROOT / "python/sglang/srt/multimodal/aim_image_loader.py"
SPEC = importlib.util.spec_from_file_location("aim_processor_batch_loader", LOADER_PATH)
batch_loader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch_loader)


class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()


class InputFormat(Enum):
    PROCESSOR_OUTPUT = auto()
    PRECOMPUTED_EMBEDDING = auto()


class Tokens:
    image_token = "<image>"
    video_token = "<video>"
    audio_token = "<audio>"

    def get_combined_regex(self):
        return re.compile(r"(<image>|<video>|<audio>)")

    def get_modality_of_token(self, token):
        return {
            self.image_token: Modality.IMAGE,
            self.video_token: Modality.VIDEO,
            self.audio_token: Modality.AUDIO,
        }.get(token)


class FakeTensor:
    pass


def env(value):
    return SimpleNamespace(get=lambda: value)


def compile_loading_methods():
    methods = {
        "_load_single_item",
        "_get_preprocessed_input_format",
        "_is_preprocessed_input",
        "_all_mm_data_is_preprocessed",
        "_aim_image_batch_enabled",
        "load_images_only_async",
        "_submit_mm_data_loading_tasks_simple",
        "submit_data_loading_tasks",
        "_validate_one_modality",
        "validate_mm_data",
        "_process_loaded_mm_data",
        "load_mm_data",
        "fast_load_mm_data",
        "legacy_load_mm_data",
    }
    source = ast.parse(BASE_PATH.read_text())
    cls = next(
        node
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "BaseMultimodalProcessor"
    )
    cls.bases = []
    cls.decorator_list = []
    cls.body = [node for node in cls.body if getattr(node, "name", None) in methods]
    namespace = {
        "asyncio": asyncio,
        "concurrent": concurrent,
        "threading": threading,
        "re": re,
        "torch": SimpleNamespace(Tensor=FakeTensor),
        "Image": Image,
        "Modality": Modality,
        "MultimodalInputFormat": InputFormat,
        "BaseMultiModalProcessorOutput": SimpleNamespace,
        "CLIENT_MEDIA_EXCEPTIONS": (ValueError, OSError),
        "logger": logging.getLogger(__name__),
        "load_images_batch": batch_loader.load_images_batch,
        "envs": SimpleNamespace(
            SGLANG_ENABLE_REDKV_MGET=env(False),
            SGLANG_ENABLE_BATCH_JPEG_DECODE=env(False),
            SGLANG_ENABLE_JPEG_GPU_DECODE=env(True),
            SGLANG_JPEG_CPU_DECODE_WORKERS=env(1),
            SGLANG_MM_RETURN_1PX_IMAGE_ON_LOAD_FAILURE=env(False),
        ),
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, str(BASE_PATH), "exec"), namespace)
    return namespace["BaseMultimodalProcessor"], namespace


class TestAimProcessorImageLoading(unittest.TestCase):
    def setUp(self):
        self.cls, self.namespace = compile_loading_methods()
        self.cls.gpu_image_decode = True
        self.cls.smart_rgb_conversion = False
        self.processor = self.cls()
        self.processor.mm_io_worker_num = 2
        self.processor.io_executor = concurrent.futures.ThreadPoolExecutor(2)
        self.processor.skip_tokenizer_init = False
        self.processor._tokenizer = SimpleNamespace(decode=lambda ids: "<image>")
        self.calls = []

        def load_image(source, gpu):
            self.calls.append((source, gpu))
            if isinstance(source, Image.Image):
                return source, source.size
            if isinstance(source, bytes):
                return Image.open(io.BytesIO(source)), None
            raise ValueError("image should not have been fetched")

        self.namespace["load_image"] = load_image
        self.namespace["load_video"] = lambda source, limit: ("video", source)
        self.namespace["load_audio"] = lambda source, rate: ("audio", source)

    def tearDown(self):
        self.processor.io_executor.shutdown(wait=True, cancel_futures=True)

    def test_prefetch_applies_existing_rgb_policy_on_io_worker(self):
        image = Image.new("RGBA", (2, 3), (10, 20, 30, 40))
        result = asyncio.run(self.processor.load_images_only_async([image]))
        self.assertEqual(result[0].mode, "RGB")
        self.assertEqual(result[0].size, (2, 3))
        self.assertEqual(len(self.calls), 1)

    def test_prefetched_tensor_is_returned_without_image_decode(self):
        tensor = FakeTensor()
        result = asyncio.run(self.processor.load_images_only_async([tensor]))
        self.assertIs(result[0], tensor)
        self.assertEqual(self.calls, [])

    def test_fast_path_reuses_images_and_keeps_other_modalities(self):
        first, second = Image.new("RGB", (1, 1)), FakeTensor()
        result = asyncio.run(
            self.processor.load_mm_data(
                "a<image>b<video>c<image>d<audio>",
                Tokens(),
                image_data=["unused first URL", "unused second URL"],
                video_data=["clip"],
                audio_data=["sound"],
                _preloaded_images=[first, second],
            )
        )
        self.assertEqual(result.images, [first, second])
        self.assertEqual(result.videos, [("video", "clip")])
        self.assertEqual(result.audios, [("audio", "sound")])
        self.assertEqual(result.input_text, "a<image>b<video>c<image>d<audio>")
        self.assertEqual(self.calls, [])

    def test_legacy_dynamic_frames_expand_in_original_token_order(self):
        self.processor.support_dynamic_frame_expansion = True
        frames = [Image.new("RGB", (1, 1)), Image.new("RGB", (2, 2))]
        result = asyncio.run(
            self.processor.load_mm_data(
                "before<image>after",
                Tokens(),
                image_data=["raw dynamic source"],
                _preloaded_images=[frames],
            )
        )
        self.assertEqual(result.images, frames)
        self.assertEqual(result.input_text, "before<image><image>after")
        self.assertEqual(self.calls, [])

    def test_legacy_precomputed_dictionary_keeps_metadata_and_token(self):
        self.processor.skip_tokenizer_init = True
        data = {"format": "precomputed_embedding", "hash": "retained", "feature": 42}
        result = asyncio.run(
            self.processor.load_mm_data(
                "before<image>after",
                Tokens(),
                image_data=[data],
                _preloaded_images=[data],
            )
        )
        self.assertIs(result.images[0], data)
        self.assertEqual(result.input_text, "before<image>after")
        self.assertEqual(self.calls, [])

    def test_preprocessed_input_ids_early_return_keeps_original_objects(self):
        data = {"format": "processor_output", "input_ids": [1, 2]}
        result = asyncio.run(
            self.processor.load_mm_data([1, 2], Tokens(), image_data=[data])
        )
        self.assertIs(result.images[0], data)
        self.assertEqual(result.input_ids, [1, 2])
        self.assertEqual(self.calls, [])

    def test_preloaded_length_mismatch_is_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "Preloaded image count"):
            asyncio.run(
                self.processor.load_mm_data(
                    "<image>", Tokens(), image_data=["url"], _preloaded_images=[]
                )
            )
        self.assertEqual(self.calls, [])

    def test_batch_switch_uses_prefetch_once(self):
        self.namespace["envs"].SGLANG_ENABLE_REDKV_MGET = env(True)
        first, second = object(), object()
        prefetched = []

        async def prefetch(image_data, discard_alpha_channel=True):
            prefetched.append(image_data)
            return [first, second]

        self.processor.load_images_only_async = prefetch
        result = asyncio.run(
            self.processor.load_mm_data(
                "<image><image>", Tokens(), image_data=["first", "second"]
            )
        )
        self.assertEqual(prefetched, [["first", "second"]])
        self.assertEqual(result.images, [first, second])
        self.assertEqual(self.calls, [])

    def test_legacy_extra_unused_images_are_not_prefetched(self):
        self.namespace["envs"].SGLANG_ENABLE_REDKV_MGET = env(True)
        image = Image.new("RGB", (1, 1))

        async def unexpected_prefetch(*args, **kwargs):
            self.fail("mismatched legacy tokens must keep the original loading path")

        self.processor.load_images_only_async = unexpected_prefetch
        result = asyncio.run(
            self.processor.load_mm_data(
                "<image>", Tokens(), image_data=[image, "unused invalid image"]
            )
        )
        self.assertEqual(result.images, [image])
        self.assertEqual(len(self.calls), 1)

    def test_cancel_signals_background_coordinator(self):
        started = threading.Event()
        cancelled = threading.Event()

        def blocked_batch(*args, **kwargs):
            started.set()
            if kwargs["cancel_event"].wait(3):
                cancelled.set()
            return []

        async def scenario():
            task = asyncio.create_task(self.processor.load_images_only_async(["url"]))
            await asyncio.to_thread(started.wait, 3)
            self.assertTrue(started.is_set())
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        with patch.dict(self.namespace, {"load_images_batch": blocked_batch}):
            asyncio.run(scenario())
        self.assertTrue(cancelled.is_set())


if __name__ == "__main__":
    unittest.main()
