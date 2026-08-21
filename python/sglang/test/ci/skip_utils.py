"""PPU CI skip guard utilities.

Provides decorators and helpers for conditionally skipping tests
when required models are missing or hardware capabilities are unavailable.

Usage:
    from sglang.test.ci.skip_utils import skip_if_model_missing, skip_if_no_fp8

    @skip_if_model_missing("meta-llama/Llama-3.1-8B-Instruct")
    class TestMyFeature(unittest.TestCase):
        ...

    @skip_if_no_fp8()
    class TestFP8Feature(unittest.TestCase):
        ...
"""

import functools
import os
import unittest

# NAS base path for model checkpoints
NAS_MODEL_BASE = os.environ.get(
    "SGLANG_NAS_MODEL_BASE", "/nas_aisw/datasets/checkpoints/LLM"
)


def _on_ppu() -> bool:
    """Whether we are running on PPU.

    Every guard below is scoped to PPU: each answers a question that only makes
    sense there (is the model on the PPU CI's NAS share, does this PPU part have
    FP8/FP4). Off PPU the answers are meaningless and, worse, wrong in the
    skip direction — the NAS path does not exist on a CUDA/AMD runner, and
    PPU_SUPPORTS_FP8/FP4 default to "0" because only the PPU workflows set them.
    Without this gate every guarded class silently skips on those runners, and a
    skip is not a failure, so CI stays green with no signal at all.

    Mirrors sglang.srt.utils.is_ppu() inline rather than importing it, to keep
    this module importable without torch.
    """
    return "PPU_SDK" in os.environ


def model_exists(model_name: str) -> bool:
    """Check if a model exists on NAS or via environment variable override.

    Priority:
    1. If SGLANG_TEST_MODEL_<sanitized_name> env var is set, use that path
    2. Fall back to checking NAS_MODEL_BASE/<model_name> exists

    Args:
        model_name: HuggingFace-style model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        True if model is available, False otherwise
    """
    # Check env var override (sanitize model name for env var)
    env_key = (
        "SGLANG_TEST_MODEL_" + model_name.replace("/", "_").replace("-", "_").upper()
    )
    env_path = os.environ.get(env_key)
    if env_path:
        return os.path.exists(env_path)

    # Fall back to NAS path
    nas_path = os.path.join(NAS_MODEL_BASE, model_name)
    return os.path.exists(nas_path)


def get_model_path(model_name: str) -> str:
    """Get the resolved model path (env var or NAS).

    Args:
        model_name: HuggingFace-style model name

    Returns:
        Resolved path string (may or may not exist)
    """
    env_key = (
        "SGLANG_TEST_MODEL_" + model_name.replace("/", "_").replace("-", "_").upper()
    )
    env_path = os.environ.get(env_key)
    if env_path:
        return env_path
    return os.path.join(NAS_MODEL_BASE, model_name)


def skip_if_model_missing(model_name: str, reason: str = None):
    """Decorator to skip a test class/function if the required model is not available.

    Args:
        model_name: HuggingFace-style model name to check
        reason: Custom skip reason (optional)

    Returns:
        Decorator that skips the test if model is missing
    """
    skip_reason = reason or f"Model {model_name} not available on NAS"

    def decorator(obj):
        if not _on_ppu() or model_exists(model_name):
            return obj
        if isinstance(obj, type):
            # Class decorator
            return unittest.skip(skip_reason)(obj)
        else:
            # Function decorator
            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                raise unittest.SkipTest(skip_reason)

            return wrapper

    return decorator


def skip_if_no_fp8(reason: str = None):
    """Decorator to skip tests requiring FP8 hardware capability.

    Checks PPU_SUPPORTS_FP8 environment variable.
    Default is "0" (not supported), set to "1" to enable.

    Args:
        reason: Custom skip reason (optional)

    Returns:
        Decorator that skips the test if FP8 is not supported
    """
    skip_reason = reason or "PPU does not support FP8 (PPU_SUPPORTS_FP8 != 1)"
    fp8_supported = os.environ.get("PPU_SUPPORTS_FP8", "0") == "1"

    def decorator(obj):
        if not _on_ppu() or fp8_supported:
            return obj
        if isinstance(obj, type):
            return unittest.skip(skip_reason)(obj)
        else:

            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                raise unittest.SkipTest(skip_reason)

            return wrapper

    return decorator


def skip_if_no_fp4(reason: str = None):
    """Decorator to skip tests requiring FP4/NVFP4 hardware capability.

    Checks PPU_SUPPORTS_FP4 environment variable.
    Default is "0" (not supported), set to "1" to enable.

    Args:
        reason: Custom skip reason (optional)

    Returns:
        Decorator that skips the test if FP4 is not supported
    """
    skip_reason = reason or "PPU does not support FP4/NVFP4 (PPU_SUPPORTS_FP4 != 1)"
    fp4_supported = os.environ.get("PPU_SUPPORTS_FP4", "0") == "1"

    def decorator(obj):
        if not _on_ppu() or fp4_supported:
            return obj
        if isinstance(obj, type):
            return unittest.skip(skip_reason)(obj)
        else:

            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                raise unittest.SkipTest(skip_reason)

            return wrapper

    return decorator
