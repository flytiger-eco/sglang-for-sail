"""Text encoding stays identical and leaves the request loop responsive."""
import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from sglang.srt.disaggregation.encoder.text_overlap import prepare_text


class TextOverlapTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_does_not_block_the_event_loop_or_change_prompt_ids(self):
        entered, release = threading.Event(), threading.Event()
        prompt = "文本<|image_pad|>🙂"
        def encode(text):
            entered.set()
            self.assertTrue(release.wait(2))
            return [ord(x) for x in text]
        processor = type("QwenVLImageProcessor", (), {})()
        processor._tokenizer = SimpleNamespace(encode=encode)
        with patch.dict("os.environ", {"SGLANG_ENCODER_TEXT_OVERLAP": "1"}):
            task = prepare_text(processor, prompt, [{"modality": "image"}])
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                self.assertFalse(task.done())
                await asyncio.sleep(0)
            finally:
                release.set()
            self.assertEqual(await task, [ord(x) for x in prompt])

    async def test_disabled_and_unsupported_media_do_not_start_encoding(self):
        processor = type("QwenVLImageProcessor", (), {})()
        with patch.dict("os.environ", {"SGLANG_ENCODER_TEXT_OVERLAP": "0"}):
            self.assertIsNone(prepare_text(processor, "x", [{"modality": "image"}]))
        with patch.dict("os.environ", {"SGLANG_ENCODER_TEXT_OVERLAP": "1"}):
            self.assertIsNone(prepare_text(processor, "x", [{"modality": "video"}]))
            self.assertIsNone(prepare_text(processor, [1], [{"modality": "image"}]))
            self.assertIsNone(prepare_text(SimpleNamespace(), "x", [{"modality": "image"}]))
