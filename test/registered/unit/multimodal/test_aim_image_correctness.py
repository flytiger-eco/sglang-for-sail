"""CPU image contracts; runnable without importing SGLang's CUDA runtime.

Run directly with ``python test_aim_image_correctness.py``. A full SGLang
installation can additionally run test_base_processor_image_decode.py to cover
the ordinary processor integration and its GPU decoder dispatch.
"""

import base64
import concurrent.futures
import importlib.util
import io
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "aim_image_under_test",
    _ROOT / "python/sglang/srt/multimodal/aim_image.py",
)
aim_image = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aim_image)


def make_image_bytes(mode="RGB", image_format="PNG"):
    image = Image.new(mode, (7, 5), 63)
    stream = io.BytesIO()
    image.save(stream, format=image_format)
    return stream.getvalue()


class TestImageInputContracts(unittest.TestCase):
    def test_placeholder_records_exactly_one_fallback_not_a_cache_miss(self):
        record = Mock()
        image, size = aim_image.load_image_with_fallback(
            Mock(side_effect=ValueError("bad image")), enabled=True, record=record
        )
        self.assertEqual((image.size, size), ((1, 1), (1, 1)))
        record.assert_called_once_with("image_fallback", "fallback", 0.0, 1)

    def setUp(self):
        self.payload = make_image_bytes()
        self.fetch = Mock(return_value=self.payload)

    def read(self, source):
        return aim_image.image_bytes_from_source(source, http_fetch=self.fetch)

    def test_bytes_base64_and_data_uri_have_identical_content(self):
        encoded = base64.b64encode(self.payload).decode()
        for source in (self.payload, encoded, "data:image/png;base64," + encoded):
            with self.subTest(source_type=type(source).__name__):
                self.assertEqual(self.read(source), self.payload)
        self.fetch.assert_not_called()

    def test_bare_jpeg_base64_starting_with_slash_is_content(self):
        payload = make_image_bytes(image_format="JPEG")
        encoded = base64.b64encode(payload).decode()
        self.assertTrue(encoded.startswith("/9j"))
        self.assertEqual(self.read(encoded), payload)
        self.fetch.assert_not_called()

    def test_url_delegates_to_existing_fetch_policy(self):
        source = "https://image.example/test.png?signature=private"
        self.assertEqual(self.read(source), self.payload)
        self.fetch.assert_called_once_with(source)

    def test_http_errors_propagate(self):
        self.fetch.side_effect = requests.HTTPError("404")
        with self.assertRaises(requests.HTTPError):
            self.read("https://image.example/missing.png")

    def test_file_uri_and_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "image with space.heic")
            path.write_bytes(self.payload)
            for source in (str(path), path.as_uri(), os.path.relpath(path)):
                with self.subTest(source=source):
                    self.assertEqual(self.read(source), self.payload)

    def test_missing_file_is_client_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Could not read image file"):
                self.read(str(Path(directory, "missing.png")))

    def test_malformed_input_rejected(self):
        for source in ("data:image/png;base64", "%%%", object(), None):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    self.read(source)

    def test_corrupt_image_fails_during_decode(self):
        for payload in (b"", b"not an image", self.payload[:35]):
            with self.subTest(payload_length=len(payload)):
                with self.assertRaises(ValueError):
                    aim_image.open_image_bytes(payload)

    def test_decode_preserves_pixels_and_processor_alpha_policy(self):
        for mode in ("RGB", "RGBA", "L", "P"):
            with self.subTest(mode=mode):
                payload = make_image_bytes(mode)
                actual = aim_image.open_image_bytes(payload)
                reference = Image.open(io.BytesIO(payload))
                self.assertEqual(actual.mode, reference.mode)
                self.assertEqual(actual.tobytes(), reference.tobytes())
                # Eager loading releases the decoder file handle; the model's
                # discard_alpha_channel flag still decides whether to convert.
                self.assertIsNone(actual.fp)

    def test_grayscale_jpeg_cpu_matches_reference_rgb(self):
        payload = make_image_bytes("L", "JPEG")
        actual = aim_image.open_image_bytes(payload).convert("RGB")
        reference = Image.open(io.BytesIO(payload)).convert("RGB")
        self.assertEqual(actual.mode, "RGB")
        self.assertEqual(actual.tobytes(), reference.tobytes())


class TestImageFailurePolicy(unittest.TestCase):
    def test_fatal_cuda_error_classification(self):
        for message in (
            "CUDA error: an illegal memory access was encountered",
            "CUDA_ERROR_ILLEGAL_ADDRESS",
            "CUDA error: device-side assert triggered",
            "failed to synchronize CUDA stream: 700",
            "CUDA error 717",
        ):
            with self.subTest(message=message):
                self.assertTrue(aim_image.is_fatal_cuda_error(RuntimeError(message)))
        self.assertFalse(aim_image.is_fatal_cuda_error(RuntimeError("bad JPEG")))

    def test_strict_default_preserves_error(self):
        with self.assertRaisesRegex(ValueError, "bad image"):
            aim_image.load_image_with_fallback(
                Mock(side_effect=ValueError("bad image")), enabled=False
            )

    def test_explicit_fallback_is_fresh_rgb_image(self):
        results = []
        for error in (ValueError("bad"), OSError("truncated"), requests.Timeout()):
            image, size = aim_image.load_image_with_fallback(
                Mock(side_effect=error), enabled=True
            )
            self.assertEqual((image.mode, image.size, size), ("RGB", (1, 1), (1, 1)))
            results.append(image)
        self.assertIsNot(results[0], results[1])

    def test_runtime_memory_and_programming_errors_never_fallback(self):
        for error in (
            RuntimeError("CUDA error: an illegal memory access was encountered"),
            MemoryError("out of memory"),
            TypeError("loader bug"),
        ):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    aim_image.load_image_with_fallback(
                        Mock(side_effect=error), enabled=True
                    )

    def test_error_handler_preserves_original_error_without_refetch(self):
        error = requests.Timeout("fetch timed out")
        image, size = aim_image.image_load_error_fallback(error, enabled=True)
        self.assertEqual((image.size, size), ((1, 1), (1, 1)))
        with self.assertRaises(requests.Timeout) as raised:
            aim_image.image_load_error_fallback(error, enabled=False)
        self.assertIs(raised.exception, error)

    def test_fatal_context_error_cannot_be_a_placeholder(self):
        with self.assertRaises(ValueError):
            aim_image.image_load_error_fallback(
                ValueError("CUDA error: an illegal memory access was encountered"),
                enabled=True,
            )

    def test_missing_heif_dependency_never_becomes_placeholder(self):
        heif_header = b"\x00\x00\x00\x18ftypheic" + b"\0" * 12
        with (
            patch.object(aim_image, "_heif_registered", False),
            patch.dict(sys.modules, {"pillow_heif": None}),
        ):
            with self.assertRaisesRegex(RuntimeError, "require pillow-heif"):
                aim_image.load_image_with_fallback(
                    lambda: aim_image.open_image_bytes(heif_header), enabled=True
                )


class TestHeifSupport(unittest.TestCase):
    def test_detects_actual_bytes_independent_of_source_name(self):
        self.assertTrue(aim_image.is_heif_bytes(b"\0\0\0\x18ftypheic"))
        self.assertTrue(aim_image.is_heif_bytes(b"\0\0\0\x18ftypmif1"))
        self.assertFalse(aim_image.is_heif_bytes(b"\0\0\0\x18ftypavif"))
        self.assertFalse(aim_image.is_heif_bytes(b"ftypheic"))
        self.assertFalse(aim_image.is_heif_bytes(make_image_bytes()))

    def test_concurrent_registration_happens_once(self):
        register = Mock()
        fake_module = types.SimpleNamespace(register_heif_opener=register)
        with (
            patch.object(aim_image, "_heif_registered", False),
            patch.dict(sys.modules, {"pillow_heif": fake_module}),
            concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool,
        ):
            list(pool.map(lambda _: aim_image.require_heif_support(), range(32)))
        register.assert_called_once_with()

    def test_heif_round_trip_when_dependency_installed(self):
        try:
            import pillow_heif
        except ImportError:
            self.skipTest("pillow-heif is not installed in this CPU test environment")
        pillow_heif.register_heif_opener()
        stream = io.BytesIO()
        # This is a decoder test, not an x265 throughput test. Bound sample
        # generation on high-core-count hosts with a small container CPU quota.
        Image.new("RGB", (16, 12), (40, 80, 120)).save(
            stream,
            format="HEIF",
            enc_params={"x265:pools": "none", "x265:frame-threads": "1"},
        )
        data = stream.getvalue()
        self.assertTrue(aim_image.is_heif_bytes(data))
        actual = aim_image.open_image_bytes(data)
        self.assertEqual(actual.size, (16, 12))
        self.assertEqual(actual.convert("RGB").size, (16, 12))


if __name__ == "__main__":
    unittest.main()
