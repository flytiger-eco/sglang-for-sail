"""No transfer failure may outlive its outstanding-write accounting."""

import concurrent.futures
import threading
import unittest
from unittest.mock import Mock

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _RunningWrite(concurrent.futures.Future):
    def __init__(self):
        super().__init__()
        self.waited = threading.Event()
        self.set_running_or_notify_cancel()

    def result(self, timeout=None):
        if not self.done():
            self.waited.set()
        return super().result(timeout=timeout)


class TestTransferFuturesDrain(CustomTestCase):
    def _assert_waits(self, operation, write):
        completed = threading.Event()
        result = {}

        def call():
            try:
                result["status"] = operation()
            except BaseException as error:
                result["error"] = error
            finally:
                completed.set()

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        try:
            self.assertTrue(
                write.waited.wait(timeout=3),
                "failure escaped before waiting for the running write",
            )
            self.assertFalse(completed.is_set())
        finally:
            write.set_result(0)
            thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        return result

    def test_exception_drains_running_writes_with_and_without_deferred_release(self):
        for enabled in (False, True):
            for error_type in (RuntimeError, ValueError):
                with self.subTest(enabled=enabled, error_type=error_type):
                    manager = MooncakeKVManager.__new__(MooncakeKVManager)
                    manager.enable_deferred_decode_kv_release = enabled
                    failed = concurrent.futures.Future()
                    failed.set_exception(error_type("write failed"))
                    write = _RunningWrite()
                    result = self._assert_waits(
                        lambda: manager._await_transfer_futures([failed, write]), write
                    )
                    self.assertIsInstance(result["error"], error_type)

    def test_nonzero_status_also_drains_running_writes(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                manager = MooncakeKVManager.__new__(MooncakeKVManager)
                manager.enable_deferred_decode_kv_release = enabled
                failed = concurrent.futures.Future()
                failed.set_result(-1)
                write = _RunningWrite()
                result = self._assert_waits(
                    lambda: manager._await_transfer_futures([failed, write]), write
                )
                self.assertEqual(result, {"status": -1})

    def test_partial_submission_drains_before_reraising_original_error(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.enable_deferred_decode_kv_release = True
        write = _RunningWrite()
        error = RuntimeError("executor stopped during submission")
        executor = Mock(submit=Mock(side_effect=[write, error]))
        result = self._assert_waits(
            lambda: manager._submit_transfer_tasks(
                executor, [(lambda: 0, ()), (lambda: 0, ())]
            ),
            write,
        )
        self.assertIs(result["error"], error)

    def test_interrupted_result_wait_still_drains_the_interrupted_write(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        write = _RunningWrite()
        original = write.result
        calls = []

        def interrupt_once(timeout=None):
            calls.append(True)
            if len(calls) == 1:
                raise KeyboardInterrupt()
            return original(timeout)

        write.result = interrupt_once
        result = self._assert_waits(
            lambda: manager._await_transfer_futures([write]), write
        )
        self.assertIsInstance(result["error"], KeyboardInterrupt)

    def test_cancelled_future_removed_from_executor_queue_cannot_hang_drain(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.enable_deferred_decode_kv_release = True
        cancelled = concurrent.futures.Future()
        cancelled.cancel()
        completed = threading.Event()
        result = []

        def call():
            result.append(manager._await_transfer_futures([cancelled]))
            completed.set()

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        try:
            self.assertTrue(completed.wait(timeout=3))
        finally:
            if not completed.is_set():
                cancelled.set_running_or_notify_cancel()
            thread.join(timeout=3)
        self.assertEqual(result, [-1])

    def test_external_cancellation_fails_after_draining_other_writes(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        cancelled = concurrent.futures.Future()
        cancelled.cancel()
        write = _RunningWrite()
        result = self._assert_waits(
            lambda: manager._await_transfer_futures([cancelled, write]), write
        )
        self.assertEqual(result, {"status": -1})

    def test_cancellation_preserves_a_prior_nonzero_status(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        failed = concurrent.futures.Future()
        failed.set_result(17)
        cancelled = concurrent.futures.Future()
        cancelled.cancel()
        self.assertEqual(manager._await_transfer_futures([failed, cancelled]), 17)


if __name__ == "__main__":
    unittest.main()
