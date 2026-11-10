# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CPU-only image input contracts shared by the ordinary and Encoder paths."""

from __future__ import annotations

import base64
import logging
import os
import threading
from io import BytesIO
from typing import Callable
from urllib.parse import unquote, urlparse

import requests
from PIL import Image

logger = logging.getLogger(__name__)

_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}
_CUDA_FATAL_ERROR_TOKENS = (
    "illegal memory access",
    "cudaerrorillegaladdress",
    "cuda error 700",
    "cuda runtime error (700)",
    "failed to synchronize cuda stream: 700",
    "operation not supported on global/shared address space",
    "cuda error 717",
    "cuda runtime error (717)",
    "failed to synchronize cuda stream: 717",
    "device-side assert",
    "unspecified launch failure",
    "misaligned address",
)
_heif_registered = False
_heif_lock = threading.Lock()


def image_bytes_from_source(
    source: str | bytes,
    *,
    http_fetch: Callable[[str], bytes],
    base64_decode: Callable = base64.b64decode,
) -> bytes:
    """Read one image without bypassing the caller's remote URL policy.

    Relative files are supported, as in AIM. Content inputs retain strict
    base64 validation. Input errors are ValueErrors, so both serving paths
    classify them as client errors rather than failed inference.
    """
    if isinstance(source, bytes):
        return source
    if not isinstance(source, str):
        raise ValueError(f"Unsupported image input type: {type(source).__name__}")
    if source.startswith(("http://", "https://")):
        return http_fetch(source)
    if source.startswith("data:"):
        try:
            _, encoded = source.split(",", 1)
        except ValueError as exc:
            raise ValueError("Image data URI is missing its payload separator") from exc
        return base64_decode(encoded, validate=True)

    path = unquote(urlparse(source).path) if source.startswith("file://") else source
    if source.startswith(("file://", "/")) or os.path.isfile(path):
        try:
            with open(path, "rb") as image_file:
                return image_file.read()
        except OSError as exc:
            # Do not echo a potentially private file path into the API response.
            raise ValueError("Could not read image file") from exc
    return base64_decode(source, validate=True)


def is_heif_bytes(data: bytes) -> bool:
    """Detect HEIF from its ftyp brand, even if the CDN URL advertises JPEG."""
    return len(data) >= 12 and data[4:8] == b"ftyp" and data[8:12] in _HEIF_BRANDS


def is_fatal_cuda_error(error: BaseException) -> bool:
    """A poisoned CUDA context must not be hidden by a CPU image fallback."""
    message = str(error).lower().replace("_", "")
    return any(token in message for token in _CUDA_FATAL_ERROR_TOKENS)


def require_heif_support() -> None:
    """Register Pillow's optional HEIF opener once per process."""
    global _heif_registered
    if _heif_registered:
        return
    with _heif_lock:
        if _heif_registered:
            return
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except (ImportError, OSError) as exc:
            # A missing serving dependency is not a malformed client image and
            # must remain an error even when the 1px fallback is enabled.
            raise RuntimeError(
                "HEIF/HEIC images require pillow-heif in the serving environment"
            ) from exc
        _heif_registered = True


def open_image_bytes(data: bytes) -> Image.Image:
    """Decode on the I/O worker and preserve the processor's color/alpha policy."""
    if is_heif_bytes(data):
        require_heif_support()
    try:
        image = Image.open(BytesIO(data))
        image.load()
        return image
    except OSError as exc:
        # Pillow raises OSError for truncated payloads as well as unknown
        # formats. Let both load_image callers return the same client error.
        raise ValueError(f"Could not decode image: {exc}") from exc


def image_load_error_fallback(error: Exception, *, enabled: bool, record=None):
    """Handle a failed fetch/decode without fetching the same image again."""
    if (
        not enabled
        or is_fatal_cuda_error(error)
        or not isinstance(
            error, (ValueError, OSError, requests.exceptions.RequestException)
        )
    ):
        raise error
    logger.warning(
        "[MM-IMAGE-FALLBACK] Image load failed (%s); returning a 1x1 RGB "
        "image because SGLANG_MM_RETURN_1PX_IMAGE_ON_LOAD_FAILURE is enabled",
        type(error).__name__,
    )
    if record is not None:
        record("image_fallback", "fallback", 0.0, 1)
    return Image.new("RGB", (1, 1)), (1, 1)


def load_image_with_fallback(loader: Callable, *, enabled: bool, record=None):
    """AIM's explicit 1px policy, limited to input/transport/decode failures.

    RuntimeError (including CUDA faults), MemoryError and programming errors
    always propagate. The strict default continues to reject malformed input.
    """
    try:
        return loader()
    except (ValueError, OSError, requests.exceptions.RequestException) as exc:
        return image_load_error_fallback(exc, enabled=enabled, record=record)
