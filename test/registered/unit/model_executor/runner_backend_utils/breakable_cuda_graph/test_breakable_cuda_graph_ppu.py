"""CPU checks for the PPU capture-status fallback in breakable CUDA graph.

Bug regression: on PPU (CUDA-compat runtime), breakable prefill capture
aborted with CUDA error 36 (cudaErrorNotSupported) whenever the captured
forward forked a side stream — e.g. the Qwen3.5 dual-stream MoE path doing
``current_stream.wait_stream(alt_stream)`` — and again at segment-end
auto-join. The hook answered "is this side stream capturing?" via
cuda-python's cudaStreamGetCaptureInfo, which the PPU runtime does not
implement. The fix answers from the hook's own fork bookkeeping on PPU and
keeps the runtime query everywhere else.
"""

import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)


class _FakeStream:
    """Minimal torch.cuda.Stream stand-in: identity hash plus the two
    attributes the hook reads (stream_id, cuda_stream)."""

    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.cuda_stream = stream_id


class _FakeGraph:
    def __init__(self):
        self.ended = False

    def capture_end(self):
        self.ended = True


@pytest.fixture()
def ppu_hook_env():
    """Run the hook on the PPU path with a status query that hard-fails —
    the PPU compat runtime's error-36 behavior — and record every wait."""
    capture_stream = _FakeStream(1)
    forked = set()
    wait_calls = []

    def _record_wait(self_stream, other_stream):
        wait_calls.append((self_stream, other_stream))

    tokens = (
        bcg._current_stream_var.set(capture_stream),
        bcg._forked_streams_var.set(forked),
    )
    stack = ExitStack()
    # Force the non-HIP/XPU branch regardless of the runner's torch build so
    # the PPU branch is what actually runs; the status query hard-fails,
    # matching the PPU compat runtime's error-36 behavior.
    stack.enter_context(patch.object(bcg, "_is_ppu", True))
    stack.enter_context(patch.object(bcg, "is_hip", return_value=False))
    stack.enter_context(patch.object(bcg, "_is_xpu", False))
    stack.enter_context(patch.object(bcg, "_original_wait_stream", _record_wait))
    stack.enter_context(
        patch.object(
            bcg,
            "_capture_status",
            side_effect=AssertionError("cudaStreamGetCaptureInfo queried"),
        )
    )
    try:
        yield SimpleNamespace(
            capture_stream=capture_stream, forked=forked, wait_calls=wait_calls
        )
    finally:
        stack.close()
        bcg._current_stream_var.reset(tokens[0])
        bcg._forked_streams_var.reset(tokens[1])


def test_ppu_dual_stream_fork_join_cycle(ppu_hook_env):
    """The exact Qwen3.5 dual-stream sequence: the alt stream waits on the
    capture stream (fork), then the capture stream waits back (join). Both
    waits must pass through and the side stream must leave the fork set."""
    side = _FakeStream(2)

    bcg._hooked_wait_stream(side, ppu_hook_env.capture_stream)
    assert side in ppu_hook_env.forked

    bcg._hooked_wait_stream(ppu_hook_env.capture_stream, side)
    assert side not in ppu_hook_env.forked

    assert ppu_hook_env.wait_calls == [
        (side, ppu_hook_env.capture_stream),
        (ppu_hook_env.capture_stream, side),
    ]


def test_ppu_join_with_unforked_stream_is_skipped(ppu_hook_env):
    """Waiting on a stream that never forked off this capture is not
    capturing; the wait must be skipped (it would be an illegal cross-stream
    dependency during capture) without any status query."""
    outsider = _FakeStream(3)
    bcg._hooked_wait_stream(ppu_hook_env.capture_stream, outsider)
    assert ppu_hook_env.wait_calls == []
    assert outsider not in ppu_hook_env.forked


def test_ppu_segment_end_auto_joins_forked_streams(ppu_hook_env):
    """Segment end must join every still-forked side stream back onto the
    capture stream and clear the fork set (the second error-36 site in the
    original traceback, reached via __exit__)."""
    graph = bcg.BreakableCUDAGraph()
    capture = bcg.BreakableCUDAGraphCapture(graph)
    capture._current_graph = _FakeGraph()
    side = _FakeStream(2)
    ppu_hook_env.forked.add(side)

    capture._end_current_segment()

    assert ppu_hook_env.wait_calls == [(ppu_hook_env.capture_stream, side)]
    assert ppu_hook_env.forked == set()
    assert capture._current_graph is None
    assert len(graph._segments) == 1
    assert graph._segments[0].ended


def test_ppu_status_defaults_to_capturing_without_hook_context(ppu_hook_env):
    """With no hook context the conservative answer is True: a spurious
    extra join is harmless, a skipped join breaks capture_end."""
    token = bcg._forked_streams_var.set(None)
    try:
        assert bcg._is_stream_capturing(_FakeStream(9))
    finally:
        bcg._forked_streams_var.reset(token)


def test_non_ppu_keeps_runtime_capture_status_query():
    """Non-PPU platforms must keep answering from the CUDA runtime query;
    the PPU bookkeeping fallback must not leak onto them."""
    sentinel_active = object()
    fake_rt = SimpleNamespace(
        cudaStreamCaptureStatus=SimpleNamespace(
            cudaStreamCaptureStatusActive=sentinel_active
        )
    )
    stream = _FakeStream(1)
    with ExitStack() as stack:
        stack.enter_context(patch.object(bcg, "_is_ppu", False))
        stack.enter_context(patch.object(bcg, "is_hip", return_value=False))
        stack.enter_context(patch.object(bcg, "_is_xpu", False))
        stack.enter_context(patch.object(bcg, "rt", fake_rt))
        query = stack.enter_context(
            patch.object(bcg, "_capture_status", return_value=sentinel_active)
        )
        assert bcg._is_stream_capturing(stream)
        query.assert_called_once_with(stream.cuda_stream)
        query.reset_mock()
        query.return_value = object()
        assert not bcg._is_stream_capturing(stream)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
