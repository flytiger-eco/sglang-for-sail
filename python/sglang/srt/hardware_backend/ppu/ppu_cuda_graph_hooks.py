"""PPU breakable-CUDA-graph capture hooks.

Uses ``@plugin_hook`` to keep the breakable prefill CUDA graph capturing on PPU
without modifying the community source.

Registered hooks:

1. ``...runner_backend_utils.breakable_cuda_graph._is_stream_capturing``
   (AROUND) — the community path asks cuda-python's ``cudaStreamGetCaptureInfo``
   whether a stream is capturing. On PPU that runtime binding has no driver
   behind it and returns error 35 (cudaErrorInsufficientDriver) for every
   stream, so any capture that forks or joins a side stream aborts with
   ``RuntimeError: CUDA error 35``. The hook falls back to the device module's
   portable query (the one the ROCm/XPU branches already use) and remembers
   that the binding is unusable.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_cuda_bindings_unusable = False


def _portable_is_stream_capturing(stream) -> bool:
    from sglang.srt.utils import get_device_module

    with get_device_module().stream(stream):
        return get_device_module().is_current_stream_capturing()


@plugin_hook(
    "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"
    ".breakable_cuda_graph._is_stream_capturing",
    type=HookType.AROUND,
)
def _ppu_is_stream_capturing(original_fn, stream):
    """Answer the capture query through the device module when cuda-python fails."""
    global _cuda_bindings_unusable
    if _cuda_bindings_unusable:
        return _portable_is_stream_capturing(stream)
    try:
        return original_fn(stream)
    except RuntimeError:
        _cuda_bindings_unusable = True
        return _portable_is_stream_capturing(stream)
