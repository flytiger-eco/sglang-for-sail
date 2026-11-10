"""Process-local fatal-CUDA state shared by image loading and Encoder health."""

import logging
import math
import os
import re
import threading

logger = logging.getLogger(__name__)

_FATAL_DIAGNOSTIC = re.compile(
    r"^(?:"
    r"(?:cuda error:\s*)?(?:an )?(?:illegal memory access|misaligned address|"
    r"device-side assert|unspecified launch failure|"
    r"operation not supported on global/shared address space)\b"
    r"|cudaerrorillegaladdress\b"
    r"|cuda error (?:700|717)\b"
    r"|cuda runtime error \((?:700|717)\)"
    r"|failed to synchronize cuda stream: (?:700|717)\b"
    r")"
)


def _is_runtime_cuda_failure(error):
    # Preprocessing also receives HTTP/Pillow/client exceptions. Their message
    # may contain an arbitrary URL, including CUDA error keywords. Only native
    # RuntimeError diagnostics at the start of the message may poison a process.
    return isinstance(error, RuntimeError) and bool(
        _FATAL_DIAGNOSTIC.match(str(error).lower().replace("_", "").strip())
    )


class CudaFailureState:
    def __init__(
        self, *, classifier=None, exit_process=os._exit, timer_factory=threading.Timer
    ):
        self._classifier = classifier
        self._exit_process = exit_process
        self._timer_factory = timer_factory
        self._failed = False
        self._exit_scheduled = False
        self._lock = threading.Lock()

    @property
    def failed(self):
        return self._failed

    def reset_after_fork(self):
        # Locks held by vanished parent threads must not be acquired here.
        self._lock = threading.Lock()
        self._failed = False
        self._exit_scheduled = False

    def record(self, error, *, exit_enabled=False, exit_delay=0):
        classifier = self._classifier
        if classifier is None:
            from sglang.srt.multimodal.aim_image import is_fatal_cuda_error

            classifier = is_fatal_cuda_error
        seen = set()
        fatal = False
        while error is not None and id(error) not in seen and len(seen) < 16:
            seen.add(id(error))
            if _is_runtime_cuda_failure(error) and classifier(error):
                fatal = True
                break
            error = error.__cause__ or error.__context__
        if not fatal:
            return False
        with self._lock:
            self._failed = True
            if exit_enabled and not self._exit_scheduled:
                if not math.isfinite(exit_delay) or exit_delay < 0:
                    raise ValueError("CUDA exit delay must be finite and non-negative")
                timer = self._timer_factory(exit_delay, lambda: self._exit_process(1))
                timer.daemon = True
                timer.start()
                self._exit_scheduled = True
        return True


_state = CudaFailureState()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_state.reset_after_fork)


def cuda_failure_detected():
    """A known poisoned CUDA context cannot pass a busy/liveness shortcut."""
    return _state.failed


def record_cuda_failure(error, *, encoder=False):
    """Record fatal errors without hiding the original caller exception.

    Shutdown is explicitly opt-in. The marker remains set even when scheduling
    shutdown fails; health checks must still reject this process. Input errors
    and recoverable OOM are not treated as poisoned CUDA contexts.
    """
    from sglang.srt.environ import envs

    was_failed = _state.failed
    encoder_exit = encoder and envs.SGLANG_ENCODER_EXIT_ON_CUDA_FATAL.get()
    exit_enabled = encoder_exit or envs.SGLANG_EXIT_ON_CUDA_FATAL.get()
    exit_delay = (
        envs.SGLANG_ENCODER_CUDA_FATAL_EXIT_DELAY_S.get()
        if encoder_exit
        else envs.SGLANG_CUDA_FATAL_EXIT_DELAY_S.get()
    )
    try:
        fatal = _state.record(error, exit_enabled=exit_enabled, exit_delay=exit_delay)
    except (ValueError, RuntimeError):
        logger.error("Failed to schedule fatal-CUDA exit; process remains unhealthy")
        fatal = _state.failed
    if fatal and not was_failed:
        # Do not echo image URLs or arbitrary exception payloads into this log.
        logger.critical("Fatal CUDA error detected; process marked unhealthy")
    return fatal
