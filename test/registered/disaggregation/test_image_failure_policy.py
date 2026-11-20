"""CPU-only regression checks for the deployed explicit 1px policy."""
import io
import os
import unittest
from unittest.mock import patch

import requests
from PIL import Image
from sglang.srt.multimodal.aim_image import load_image_with_fallback
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.managers.schedule_batch import Modality


class ImageFailurePolicy(unittest.TestCase):
    def load(self, payload, enabled=True):
        with patch.dict(os.environ, {"SGLANG_MM_RETURN_1PX_IMAGE_ON_LOAD_FAILURE": "1" if enabled else "0"}):
            return BaseMultimodalProcessor._load_single_item(payload, Modality.IMAGE, gpu_image_decode=False)

    def png(self, mode, color):
        b = io.BytesIO()
        Image.new(mode, (32, 32), color).save(b, format="PNG")
        return b.getvalue()

    def test_rgb_unchanged(self):
        p = self.png("RGB", (255, 0, 0))
        self.assertEqual(self.load(p).tobytes(), self.load(p, False).tobytes())

    def test_grayscale_to_rgb_unchanged(self):
        p = self.png("L", 128)
        on, off = self.load(p), self.load(p, False)
        self.assertEqual(on.mode, "RGB")
        self.assertEqual(on.getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(on.tobytes(), off.tobytes())

    def test_corrupt_image_replaced(self):
        im = self.load(b"not an image")
        self.assertEqual((im.mode, im.size), ("RGB", (1, 1)))

    def test_corrupt_image_strict_mode(self):
        with self.assertRaises(ValueError):
            self.load(b"not an image", False)

    def test_http_403_is_replaced_without_retry(self):
        calls = []
        def fail():
            calls.append(1)
            response = requests.Response()
            response.status_code = 403
            raise requests.HTTPError("Forbidden", response=response)
        im, size = load_image_with_fallback(fail, enabled=True)
        self.assertEqual((im.mode, im.size, size, len(calls)), ("RGB", (1, 1), (1, 1), 1))

    def test_runtime_and_cuda_faults_still_propagate(self):
        for error in (RuntimeError("processor bug"), MemoryError("out of memory"), ValueError("CUDA error 700: illegal memory access")):
            with self.subTest(error=type(error).__name__):
                def fail(): raise error
                with self.assertRaises(type(error)):
                    load_image_with_fallback(fail, enabled=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
