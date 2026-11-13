"""CPU regression tests for the process-local fatal-CUDA policy."""

import concurrent.futures
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "python/sglang/srt/multimodal" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


image = load_module("aim_image")
health = load_module("aim_cuda_health")


class TestCudaFailureState(unittest.TestCase):
    def test_fatal_marks_unhealthy_without_exit(self):
        exits = []
        state = health.CudaFailureState(
            classifier=image.is_fatal_cuda_error, exit_process=exits.append
        )
        self.assertFalse(state.failed)
        self.assertTrue(state.record(RuntimeError("CUDA error: illegal memory access")))
        self.assertTrue(state.failed)
        self.assertEqual(exits, [])

    def test_client_errors_and_oom_do_not_poison_the_process(self):
        state = health.CudaFailureState(classifier=image.is_fatal_cuda_error)
        for exc in (
            ValueError("invalid image"),
            MemoryError(),
            RuntimeError("CUDA out of memory"),
        ):
            self.assertFalse(state.record(exc))
        self.assertFalse(state.failed)

    def test_client_payload_cannot_trigger_process_failure_or_exit(self):
        timers = []
        state = health.CudaFailureState(
            classifier=image.is_fatal_cuda_error,
            timer_factory=lambda *args: SimpleNamespace(
                start=lambda: timers.append(args)
            ),
        )
        for exc in (
            OSError(
                "404 Client Error: https://cdn.example/cudaerrorillegaladdress.jpg"
            ),
            ValueError("CUDA error: misaligned address"),
            RuntimeError(
                "Failed to load https://cdn.example/cudaerrorillegaladdress.jpg"
            ),
            RuntimeError("invalid input: illegal memory access"),
        ):
            self.assertFalse(state.record(exc, exit_enabled=True))
        self.assertFalse(state.failed)
        self.assertEqual(timers, [])

    def test_native_cuda_diagnostics_are_recognized(self):
        for message in (
            "CUDA error: an illegal memory access was encountered",
            "CUDA error: misaligned address",
            "CUDA error: device-side assert triggered",
            "CUDA error: unspecified launch failure",
            "CUDA error: operation not supported on global/shared address space",
            "cudaErrorIllegalAddress",
            "CUDA error 700: an illegal memory access was encountered",
            "CUDA runtime error (717)",
            "Failed to synchronize CUDA stream: 700",
        ):
            with self.subTest(message=message):
                state = health.CudaFailureState(classifier=image.is_fatal_cuda_error)
                self.assertTrue(state.record(RuntimeError(message)))

    def test_wrapped_http_failure_cannot_poison_state(self):
        state = health.CudaFailureState(classifier=image.is_fatal_cuda_error)
        wrapped = RuntimeError("Image loading failed")
        wrapped.__cause__ = OSError("https://cdn.example/cudaerrorillegaladdress.jpg")
        self.assertFalse(state.record(wrapped))
        self.assertFalse(state.failed)

    def test_wrapped_fatal_is_detected_and_cycles_terminate(self):
        state = health.CudaFailureState(classifier=image.is_fatal_cuda_error)
        ordinary = ValueError("ordinary")
        ordinary.__cause__ = ordinary
        self.assertFalse(state.record(ordinary))
        wrapped = RuntimeError("encoder failed")
        wrapped.__cause__ = RuntimeError("CUDA error: misaligned address")
        self.assertTrue(state.record(wrapped))
        self.assertTrue(state.failed)

    def test_explicit_exit_is_delayed_and_scheduled_once(self):
        exits, timers = [], []

        class Timer:
            def __init__(self, delay, callback):
                self.delay, self.callback = delay, callback
                self.daemon = False

            def start(self):
                timers.append(self)

        state = health.CudaFailureState(
            classifier=image.is_fatal_cuda_error,
            exit_process=exits.append,
            timer_factory=Timer,
        )
        fatal = RuntimeError("CUDA error: illegal memory access")
        for _ in range(3):
            self.assertTrue(state.record(fatal, exit_enabled=True, exit_delay=2.5))
        self.assertEqual(len(timers), 1)
        self.assertEqual(timers[0].delay, 2.5)
        self.assertTrue(timers[0].daemon)
        self.assertEqual(exits, [])
        timers[0].callback()
        self.assertEqual(exits, [1])

    def test_invalid_delay_does_not_clear_failure_or_schedule_exit(self):
        for delay in (float("inf"), float("nan"), -1):
            calls = []
            state = health.CudaFailureState(
                classifier=image.is_fatal_cuda_error,
                timer_factory=lambda *args: calls.append(args),
            )
            with self.assertRaises(ValueError):
                state.record(
                    RuntimeError("misaligned address"),
                    exit_enabled=True,
                    exit_delay=delay,
                )
            self.assertTrue(state.failed)
            self.assertEqual(calls, [])

    def test_concurrent_fatal_errors_schedule_one_exit(self):
        timers = []

        class Timer:
            def __init__(self, *args):
                self.daemon = False

            def start(self):
                timers.append(self)

        state = health.CudaFailureState(
            classifier=image.is_fatal_cuda_error, timer_factory=Timer
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(
                    lambda _: state.record(
                        RuntimeError("misaligned address"), exit_enabled=True
                    ),
                    range(50),
                )
            )
        self.assertTrue(all(results))
        self.assertEqual(len(timers), 1)

    def test_fork_reset_clears_parent_failure_state(self):
        state = health.CudaFailureState(classifier=image.is_fatal_cuda_error)
        state.record(RuntimeError("illegal memory access"))
        state.reset_after_fork()
        self.assertFalse(state.failed)
        self.assertTrue(state.record(RuntimeError("misaligned address")))


if __name__ == "__main__":
    unittest.main()
