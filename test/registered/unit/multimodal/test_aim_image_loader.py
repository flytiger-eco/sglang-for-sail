"""AIM image loader integration tests for a CPU SGLang environment."""

import io
import os
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from torchvision.io import ImageReadMode

from sglang.srt.connector import redkv
from sglang.srt.utils import common
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def jpeg_bytes(mode="L"):
    stream = io.BytesIO()
    Image.new(mode, (7, 5), 63).save(stream, format="JPEG")
    return stream.getvalue()


class TestAIMImageLoader(unittest.TestCase):
    def test_http_uses_existing_download_policy(self):
        url = "https://images.example/image.jpg"
        with (
            patch.object(common, "download_remote_media", return_value=b"image") as fetch,
            patch.dict(os.environ, {"REQUEST_TIMEOUT": "7"}),
        ):
            self.assertEqual(common.get_image_bytes(url), b"image")
        fetch.assert_called_once_with(url, timeout=7)

    def test_redkv_fallback_uses_existing_download_policy(self):
        key = "imagetransfer_https://images.example/image.jpg"

        def resolve(source, *, http_fetch):
            self.assertEqual(source, key)
            return http_fetch(source.removeprefix("imagetransfer_"))

        with (
            patch.object(redkv, "resolve_image_bytes", side_effect=resolve),
            patch.object(common, "download_remote_media", return_value=b"image") as fetch,
            patch.dict(os.environ, {"REQUEST_TIMEOUT": "7"}),
        ):
            self.assertEqual(common.get_image_bytes(key), b"image")
        fetch.assert_called_once_with("https://images.example/image.jpg", timeout=7)

    def test_grayscale_gpu_requests_rgb(self):
        expected = torch.zeros((3, 5, 7), dtype=torch.uint8)
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch("torchvision.io.decode_jpeg", return_value=expected) as decode,
        ):
            image, _ = common.load_image(jpeg_bytes(), gpu_image_decode=True)
        self.assertIs(image, expected)
        self.assertEqual(decode.call_args.kwargs["mode"], ImageReadMode.RGB)
        self.assertEqual(decode.call_args.kwargs["device"], "cuda")

    def test_failed_gpu_decode_uses_real_cpu_image(self):
        data = jpeg_bytes()
        with (
            patch.object(common, "is_cuda", return_value=True),
            patch("torchvision.io.decode_jpeg", side_effect=RuntimeError("bad JPEG")),
        ):
            image, _ = common.load_image(data, gpu_image_decode=True)
        reference = Image.open(io.BytesIO(data))
        self.assertEqual(image.tobytes(), reference.tobytes())
        self.assertEqual(image.size, (7, 5))

    def test_cuda_errors_never_fallback_to_cpu_or_placeholder(self):
        for error in (
            torch.OutOfMemoryError("out of memory"),
            RuntimeError("CUDA error: an illegal memory access was encountered"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(common, "is_cuda", return_value=True),
                patch("torchvision.io.decode_jpeg", side_effect=error),
                patch.dict(
                    os.environ, {"SGLANG_MM_RETURN_1PX_IMAGE_ON_LOAD_FAILURE": "true"}
                ),
            ):
                with self.assertRaises(type(error)) as raised:
                    common.load_image(jpeg_bytes(), gpu_image_decode=True)
                self.assertIs(raised.exception, error)

    def test_invalid_bytes_use_explicit_fallback_only(self):
        with patch.dict(
            os.environ, {"SGLANG_MM_RETURN_1PX_IMAGE_ON_LOAD_FAILURE": "false"}
        ):
            with self.assertRaises(ValueError):
                common.load_image(b"bad image", gpu_image_decode=False)
        with patch.dict(
            os.environ, {"SGLANG_MM_RETURN_1PX_IMAGE_ON_LOAD_FAILURE": "true"}
        ):
            image, size = common.load_image(b"bad image", gpu_image_decode=False)
        self.assertEqual((image.mode, image.size, size), ("RGB", (1, 1), (1, 1)))


if __name__ == "__main__":
    unittest.main()
