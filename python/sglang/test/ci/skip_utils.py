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

The module name is backend-neutral; the behaviour is not. All three
decorators are gated on _on_ppu(), so they only ever skip when PPU_SDK is in
the environment — on a CUDA or AMD runner each one returns the decorated
class or function unchanged. So do not reach for them as general capability
gates: @skip_if_no_fp8() on a CUDA-only test is silently dead code, and
nothing warns you. Gate non-PPU tests on an explicit predicate instead
(is_cuda(), is_hip(), a real capability probe).

The gate is also not removable to make them universal. See _on_ppu() for
why each question is only answerable on PPU; the consequence is that
dropping it makes every guarded class here skip on CUDA/AMD runners, and
because a skip is not a failure, CI stays green with no signal at all.
"""

import functools
import glob
import os
import unittest

# NAS base path for model checkpoints
NAS_MODEL_BASE = os.environ.get(
    "SGLANG_NAS_MODEL_BASE", "/nas_aisw/datasets/checkpoints/LLM"
)

# The NAS layout is NAS_MODEL_BASE/<org>/<version>/<model> with a
# DIFFERENT version string per org (v1.0 / v2.0 / v3.1 / v3.8 ... across
# 411 org directories), so NAS_MODEL_BASE/<org>/<model> misses every
# versioned model. Version-layer resolution (<org>/*/<model>) fixes the
# lookup, but turning it on unconditionally would suddenly make 17
# guarded models runnable -- among them 120B/119B checkpoints -- on a
# nightly that already runs 4 hours with 73% of the load on 1-GPU
# runners. The allowlist therefore controls which models may be
# resolved through the version layer; empty (the default) keeps the
# exact pre-fix behaviour. Comma-separated org/model names, or "*" to
# allow all. Read at call time so tests can vary it per process.
ALLOWLIST_ENV_VAR = "SGLANG_NAS_VERSION_LAYER_ALLOWLIST"


def _version_layer_allowlist() -> set:
    raw = os.environ.get(ALLOWLIST_ENV_VAR, "").strip()
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def _resolve_version_layer(model_name: str) -> str:
    """Resolve <org>/<model> under the org's version layer, if allowed.

    Returns NAS_MODEL_BASE/<org>/<version>/<model> for the first version
    directory containing the model (versions are not enumerated because
    they differ per org), or "" when the model is not allowlisted for
    version-layer resolution or no version directory contains it.
    """
    allowlist = _version_layer_allowlist()
    if not allowlist or (model_name not in allowlist and "*" not in allowlist):
        return ""
    org, _, model = model_name.partition("/")
    if not org or not model:
        return ""
    matches = glob.glob(os.path.join(NAS_MODEL_BASE, org, "*", model))
    return sorted(matches)[0] if matches else ""


def _resolve_model_path(model_name: str) -> str:
    """Shared resolution for model_exists() and get_model_path().

    Priority:
    1. SGLANG_TEST_MODEL_<sanitized_name> env override path
    2. Direct path NAS_MODEL_BASE/<model_name>
    3. Version layer NAS_MODEL_BASE/<org>/*/<model_name>, only for
       models in SGLANG_NAS_VERSION_LAYER_ALLOWLIST (empty default =
       this layer is off and behaviour matches the pre-fix code)
    """
    env_key = (
        "SGLANG_TEST_MODEL_" + model_name.replace("/", "_").replace("-", "_").upper()
    )
    env_path = os.environ.get(env_key)
    if env_path:
        return env_path

    direct_path = os.path.join(NAS_MODEL_BASE, model_name)
    if os.path.exists(direct_path):
        return direct_path

    version_path = _resolve_version_layer(model_name)
    return version_path or direct_path


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

    Resolution priority (see _resolve_model_path):
    1. If SGLANG_TEST_MODEL_<sanitized_name> env var is set, use that path
    2. Direct path NAS_MODEL_BASE/<model_name>
    3. Version layer NAS_MODEL_BASE/<org>/*/<model_name>, only for models
       listed in SGLANG_NAS_VERSION_LAYER_ALLOWLIST (default empty = off)

    Args:
        model_name: HuggingFace-style model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        True if model is available, False otherwise
    """
    return os.path.exists(_resolve_model_path(model_name))


def get_model_path(model_name: str) -> str:
    """Get the resolved model path (env var or NAS).

    Uses the same resolution priority as model_exists(), so for an
    allowlisted model this returns the version-layer path that actually
    exists, not the version-less NAS_MODEL_BASE/<model_name> join.

    Args:
        model_name: HuggingFace-style model name

    Returns:
        Resolved path string (may or may not exist)
    """
    return _resolve_model_path(model_name)


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
