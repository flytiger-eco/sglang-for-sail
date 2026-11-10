"""Execute the real TM/Qwen methods with CPU fakes for their serving dependencies.

The AST loader isolates unchanged method bodies from CUDA/model imports; tests
exercise task scheduling, cancellation, request output, and Qwen's processing
flow instead of merely checking that symbols occur in source.
"""

import __future__
import ast
import asyncio
import logging
import re
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from PIL import Image

_ROOT = Path(__file__).resolve().parents[4]


def load_method(path, class_name, method_name, namespace):
    tree = ast.parse((_ROOT / path).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and node.name == method_name
    )
    module = ast.Module(body=[method], type_ignores=[])
    exec(
        compile(
            module,
            str(_ROOT / path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[method_name]


class GenerateReqInput(types.SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(
            text="describe <image>",
            session_id=None,
            session_params=None,
            input_embeds=None,
            input_ids=None,
            image_data="https://images.example/image.jpg",
            video_data=None,
            audio_data=None,
            mm_content_hashes=None,
            mm_hashes=None,
            need_wait_for_mm_inputs=False,
        )
        self.__dict__.update(kwargs)

    def contains_mm_input(self):
        return bool(self.image_data or self.video_data or self.audio_data)


class EmbeddingReqInput(GenerateReqInput):
    is_cross_encoder_request = True


class ImageData(types.SimpleNamespace):
    content_hash = None


class Flag:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class TestTTFTParallel(unittest.IsolatedAsyncioTestCase):
    def make_manager(self):
        self.flags = types.SimpleNamespace(
            SGLANG_ENABLE_TTFT_PARALLEL=Flag(True),
            SGLANG_MM_PRECOMPUTE_HASH=Flag(False),
        )
        method = load_method(
            "python/sglang/srt/managers/tokenizer_manager.py",
            "TokenizerManager",
            "_tokenize_one_request",
            {
                "asyncio": asyncio,
                "GenerateReqInput": GenerateReqInput,
                "EmbeddingReqInput": EmbeddingReqInput,
                "ImageData": ImageData,
                "Image": Image,
                "envs": self.flags,
                "get_memory": lambda: manager.server_args,
                "get_disagg": lambda: manager.server_args,
                "DisaggregationMode": types.SimpleNamespace(PREFILL="prefill", DECODE="decode"),
                "logger": logging.getLogger(__name__),
            },
        )
        self.loaded = [Image.new("RGB", (3, 2))]
        self.mm_result = types.SimpleNamespace(
            input_ids=None, token_type_ids=None, mm_items=[]
        )
        processor = types.SimpleNamespace(
            mm_preprocess_cache=types.SimpleNamespace(enabled=False),
            uses_media_artifacts_without_cache=False,
            supports_aim_image_prefetch=True,
            prefer_tokenized_input=False,
            load_images_only_async=AsyncMock(return_value=self.loaded),
            process_mm_data_async=AsyncMock(return_value=self.mm_result),
        )
        processor.mm_tokens = types.SimpleNamespace(
            get_combined_regex=lambda: re.compile(r"(<image>|<video>)"),
            get_modality_of_token=lambda part: (
                "image" if part == "<image>" else "video" if part == "<video>" else None
            ),
        )
        can_prefetch = load_method(
            "python/sglang/srt/multimodal/processors/qwen_vl.py",
            "QwenVLImageProcessor",
            "can_prefetch_images",
            {"Modality": types.SimpleNamespace(IMAGE="image")},
        )
        processor.can_prefetch_images = types.MethodType(can_prefetch, processor)
        manager = types.SimpleNamespace(
            disaggregation_mode=None,
            tokenizer=object(),
            mm_processor=processor,
            server_args=types.SimpleNamespace(
                disable_radix_cache=True,
                language_only=False,
                language_model_only=False,
                encoder_transfer_backend="mooncake",
            ),
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(architectures=[])
            ),
            max_req_input_len=4096,
            _tokenize_texts=AsyncMock(return_value=([10, 20], [0, 0])),
            _validate_mm_limits=Mock(),
            _normalize_mm_content_hashes=Mock(),
            _validate_one_request=Mock(),
            _create_tokenized_object=lambda obj, text, ids, embeds, mm, types: (
                text, ids, types, mm
            ),
        )
        manager._tokenize_one_request = types.MethodType(method, manager)
        return manager

    async def test_raw_prompt_processor_keeps_redundant_tokenization_skipped(self):
        manager = self.make_manager()
        manager.mm_processor.generates_input_ids_from_raw_prompt = True
        self.mm_result.input_ids = [12, 34]
        result = await manager._tokenize_one_request(GenerateReqInput())
        manager._tokenize_texts.assert_not_awaited()
        manager.mm_processor.load_images_only_async.assert_not_awaited()
        self.assertEqual(result[1], [12, 34])

    async def test_artifact_cache_paths_do_not_duplicate_image_loading(self):
        manager = self.make_manager()
        manager.mm_processor.mm_preprocess_cache.enabled = True
        await manager._tokenize_one_request(GenerateReqInput())
        manager.mm_processor.load_images_only_async.assert_not_awaited()
        manager.mm_processor.mm_preprocess_cache.enabled = False
        manager.mm_processor.uses_media_artifacts_without_cache = True
        await manager._tokenize_one_request(GenerateReqInput())
        manager.mm_processor.load_images_only_async.assert_not_awaited()

    async def test_image_worker_overlaps_synchronous_tokenizer_and_is_reused(self):
        manager = self.make_manager()
        tokenizer_running = threading.Event()
        image_started = threading.Event()

        async def preload(**kwargs):
            def fetch():
                self.assertTrue(tokenizer_running.wait(2))
                image_started.set()
                return self.loaded

            return await asyncio.to_thread(fetch)

        async def tokenize(*args):
            # No await: matches the regular tokenizer's synchronous execution.
            tokenizer_running.set()
            self.assertTrue(
                image_started.wait(2), "preload did not start during tokenization"
            )
            tokenizer_running.clear()
            return [10, 20], [0, 0]

        manager.mm_processor.load_images_only_async.side_effect = preload
        manager._tokenize_texts.side_effect = tokenize
        request = GenerateReqInput()
        result = await manager._tokenize_one_request(request)
        self.assertEqual(result[:3], (request.text, [10, 20], [0, 0]))
        manager.mm_processor.load_images_only_async.assert_awaited_once_with(
            image_data=request.image_data
        )
        call = manager.mm_processor.process_mm_data_async.await_args.kwargs
        self.assertIs(call["_preloaded_images"], self.loaded)
        self.assertEqual(request.image_data, ["https://images.example/image.jpg"])
        manager._validate_mm_limits.assert_called_once_with(request)
        manager._normalize_mm_content_hashes.assert_called_once_with(request)

    async def test_tokenization_failure_cancels_and_drains_preload(self):
        manager = self.make_manager()
        cleaned = asyncio.Event()

        async def preload(**kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        manager.mm_processor.load_images_only_async.side_effect = preload
        manager._tokenize_texts.side_effect = ValueError("tokenizer failed")
        with self.assertRaisesRegex(ValueError, "tokenizer failed"):
            await manager._tokenize_one_request(GenerateReqInput())
        self.assertTrue(cleaned.is_set())
        manager.mm_processor.process_mm_data_async.assert_not_awaited()

    async def test_request_cancellation_drains_preload(self):
        manager = self.make_manager()
        started = asyncio.Event()
        cleaned = asyncio.Event()

        async def preload(**kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        async def tokenize(*args):
            await asyncio.Event().wait()

        manager.mm_processor.load_images_only_async.side_effect = preload
        manager._tokenize_texts.side_effect = tokenize
        task = asyncio.create_task(manager._tokenize_one_request(GenerateReqInput()))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(cleaned.is_set())
        manager.mm_processor.process_mm_data_async.assert_not_awaited()

    async def test_preload_failure_is_propagated_without_second_fetch(self):
        manager = self.make_manager()
        manager.mm_processor.load_images_only_async.side_effect = ValueError(
            "bad image"
        )
        with self.assertRaisesRegex(ValueError, "bad image"):
            await manager._tokenize_one_request(GenerateReqInput())
        manager.mm_processor.load_images_only_async.assert_awaited_once()
        manager.mm_processor.process_mm_data_async.assert_not_awaited()

    async def test_completed_preload_error_is_retrieved_on_tokenizer_failure(self):
        manager = self.make_manager()
        manager.mm_processor.load_images_only_async.side_effect = ValueError(
            "bad image"
        )
        tokenizer_error = RuntimeError("tokenizer failed")
        manager._tokenize_texts.side_effect = tokenizer_error
        with self.assertRaises(RuntimeError) as raised:
            await manager._tokenize_one_request(GenerateReqInput())
        self.assertIs(raised.exception, tokenizer_error)
        manager.mm_processor.process_mm_data_async.assert_not_awaited()

    async def test_limits_fail_before_any_prefetch(self):
        manager = self.make_manager()
        manager._validate_mm_limits.side_effect = ValueError("too many images")
        with self.assertRaisesRegex(ValueError, "too many images"):
            await manager._tokenize_one_request(GenerateReqInput())
        manager.mm_processor.load_images_only_async.assert_not_called()
        manager._tokenize_texts.assert_not_awaited()

    async def test_unsupported_input_paths_do_not_prefetch(self):
        cases = (
            {"video_data": ["video.mp4"]},
            {"audio_data": ["audio.wav"]},
            {"image_data": [{"format": "processor_output"}]},
            {"image_data": [["image1", "image2"]]},
            {"image_data": [ImageData(url="image", content_hash="ab" * 32)]},
            {"mm_content_hashes": ["ab" * 32]},
            {"input_ids": [1, 2]},
            {"input_embeds": [[1.0]], "input_ids": [1]},
            {"text": ""},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                manager = self.make_manager()
                await manager._tokenize_one_request(GenerateReqInput(**kwargs))
                manager.mm_processor.load_images_only_async.assert_not_called()
                call = manager.mm_processor.process_mm_data_async.await_args.kwargs
                self.assertNotIn("_preloaded_images", call)

    async def test_unused_bad_image_does_not_trigger_prefetch(self):
        manager = self.make_manager()
        # One placeholder consumes the first image in the legacy loader; the
        # unreachable second URL must not cause an early fetch or failure.
        request = GenerateReqInput(
            text="describe <image>",
            image_data=[
                "https://images.example/valid.jpg",
                "https://unreachable.invalid/unused.jpg",
            ],
        )
        manager.mm_processor.load_images_only_async.side_effect = ValueError(
            "unused URL must not be fetched"
        )
        result = await manager._tokenize_one_request(request)
        self.assertIs(result[3], self.mm_result)
        manager.mm_processor.load_images_only_async.assert_not_called()
        call = manager.mm_processor.process_mm_data_async.await_args.kwargs
        self.assertNotIn("_preloaded_images", call)
        self.assertEqual(call["image_data"], request.image_data)

    async def test_placeholder_count_gate_uses_image_modality_only(self):
        manager = self.make_manager()
        processor = manager.mm_processor
        self.assertTrue(processor.can_prefetch_images("<image> <image>", 2))
        self.assertFalse(processor.can_prefetch_images("<image> <video>", 2))
        self.assertFalse(processor.can_prefetch_images("text only", 1))
        self.assertFalse(processor.can_prefetch_images("<image> <image>", 1))

    async def test_disabled_unsupported_and_language_only_do_not_prefetch(self):
        for variant in ("flag", "processor", "language_only", "language_model_only"):
            with self.subTest(variant=variant):
                manager = self.make_manager()
                request = GenerateReqInput()
                if variant == "flag":
                    self.flags.SGLANG_ENABLE_TTFT_PARALLEL.value = False
                elif variant == "processor":
                    manager.mm_processor.supports_aim_image_prefetch = False
                else:
                    setattr(manager.server_args, variant, True)
                    request.need_wait_for_mm_inputs = True
                if variant == "language_model_only":
                    with self.assertRaises(ValueError):
                        await manager._tokenize_one_request(request)
                else:
                    await manager._tokenize_one_request(request)
                manager.mm_processor.load_images_only_async.assert_not_called()

    async def test_prefer_tokenized_input_and_rebuilt_ids_are_preserved(self):
        manager = self.make_manager()
        manager.mm_processor.prefer_tokenized_input = True
        self.mm_result.input_ids = [10, 999, 20]
        self.mm_result.token_type_ids = [1, 1, 1]
        result = await manager._tokenize_one_request(GenerateReqInput())
        call = manager.mm_processor.process_mm_data_async.await_args.kwargs
        self.assertEqual(call["input_text"], [10, 20])
        self.assertEqual(result[1:3], ([10, 999, 20], [1, 1, 1]))

    async def test_actual_text_tokenizer_crossencoder_kwargs_are_preserved(self):
        manager = self.make_manager()
        method = load_method(
            "python/sglang/srt/managers/tokenizer_manager.py",
            "TokenizerManager",
            "_tokenize_texts",
            {
                "InputFormat": types.SimpleNamespace(SINGLE_STRING="single"),
                "logger": logging.getLogger(__name__),
            },
        )
        manager._tokenize_texts = types.MethodType(method, manager)
        manager._detect_input_format = lambda *_: "cross_encoder"
        manager._prepare_tokenizer_input = lambda texts, _: texts
        manager._extract_tokenizer_results = lambda ids, types, *_: (ids[0], types[0])
        manager.async_dynamic_batch_tokenizer = None
        manager.model_config.is_embedding_gemma = False
        manager.tokenizer = Mock(
            return_value={"input_ids": [[10, 20]], "token_type_ids": [[0, 1]]}
        )
        request = EmbeddingReqInput(text=[["query", "passage"]])
        result = await manager._tokenize_one_request(request)
        manager.tokenizer.assert_called_once_with(
            [["query", "passage"]], return_token_type_ids=True
        )
        self.assertEqual(result[1:3], ([10, 20], [0, 1]))
        manager.mm_processor.load_images_only_async.assert_not_called()


class FakeTensor:
    ndim = 2
    dtype = "int64"
    device = "cpu"

    def __init__(self, values):
        self.values = values

    def flatten(self):
        return self

    def numel(self):
        return len(self.values)

    def tolist(self):
        return list(self.values)


class TestQwenPrefetchForwarding(unittest.IsolatedAsyncioTestCase):
    async def test_real_qwen_method_reuses_loaded_images_and_preserves_mrope(self):
        method = load_method(
            "python/sglang/srt/multimodal/processors/qwen_vl.py",
            "QwenVLImageProcessor",
            "_process_mm_data_uncached",
            {
                "time": time,
                "logger": logging.getLogger(__name__),
                "torch": types.SimpleNamespace(Tensor=FakeTensor),
                "Modality": types.SimpleNamespace(IMAGE="image", VIDEO="video"),
                "MultimodalProcessorOutput": types.SimpleNamespace,
                "_get_processor_video_config": lambda *_: None,
            },
        )
        loaded = [Image.new("RGB", (3, 2))]
        base_output = types.SimpleNamespace(images=loaded, videos=[], audios=[])
        positions = FakeTensor([1, 2, 3])
        processor = types.SimpleNamespace(
            load_mm_data=AsyncMock(return_value=base_output),
            process_and_combine_mm_data_async=AsyncMock(return_value=(
                [], FakeTensor([10, 999, 20]), {"padded_input_ids": [10, 999, 20]}
            )),
            _mark_cuda_ipc_features_for_deferred_reconstruction=Mock(),
            _get_processor_output_value=lambda output, key: output.get(key),
            _get_grid_from_output_or_items=lambda *args: None,
            _get_precomputed_mrope_from_output=lambda output: (positions, 2),
            hf_config=types.SimpleNamespace(model_type="qwen4_exp"),
            model_type="qwen4_exp",
            video_config={},
            vision_start_token_id=1,
            vision_end_token_id=2,
            mm_tokens=types.SimpleNamespace(
                image_token_id=999, video_token_id=998, audio_token_id=997
            ),
        )
        request = GenerateReqInput()
        result = await method(
            processor,
            image_data=[request.image_data],
            input_text=request.text,
            request_obj=request,
            _preloaded_images=loaded,
        )
        call = processor.load_mm_data.await_args.kwargs
        self.assertIs(call["_preloaded_images"], loaded)
        processor.load_mm_data.assert_awaited_once()
        self.assertIs(
            processor.process_and_combine_mm_data_async.await_args.args[0].images,
            loaded,
        )
        self.assertEqual(result.input_ids, [10, 999, 20])
        self.assertEqual(result.padded_input_ids, [10, 999, 20])
        self.assertIs(result.mrope_positions, positions)
        self.assertEqual(result.mrope_position_delta, 2)


if __name__ == "__main__":
    unittest.main()
