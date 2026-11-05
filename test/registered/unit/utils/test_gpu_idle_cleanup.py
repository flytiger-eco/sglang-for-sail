import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.gpu_idle_cleanup import cleanup_residual_gpu_processes
from sglang.test.test_utils import _wait_for_gpu_idle_in_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeNVML:
    def __init__(self, processes=None):
        self.processes = processes or {}
        self.NVMLError = RuntimeError

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [
            SimpleNamespace(pid=pid, usedGpuMemory=3 << 30)
            for pid in self.processes.get(handle, [])
        ]


class TestGPUIdleCleanup(unittest.TestCase):
    @patch("sglang.test.gpu_idle_cleanup.kill_process_tree")
    @patch("sglang.test.gpu_idle_cleanup.os.getpid", return_value=999)
    def test_cleanup_excludes_self_and_init(self, mock_getpid, mock_kill):
        pynvml = FakeNVML({0: [999, 1, 101], 1: [102]})

        killed, errors = cleanup_residual_gpu_processes(pynvml, [0, 1])

        self.assertEqual(killed, [101, 102])
        self.assertEqual(errors, [])
        self.assertEqual(
            [call.args[0] for call in mock_kill.call_args_list], [101, 102]
        )
        self.assertEqual(
            [call.kwargs["wait_timeout"] for call in mock_kill.call_args_list],
            [10, 10],
        )

    @patch("sglang.test.gpu_idle_cleanup.kill_process_tree")
    def test_cleanup_continues_after_error(self, mock_kill):
        mock_kill.side_effect = [RuntimeError("denied"), None]
        pynvml = FakeNVML({0: [101, 102]})

        killed, errors = cleanup_residual_gpu_processes(pynvml, [0])

        self.assertEqual(killed, [102])
        self.assertEqual(len(errors), 1)
        self.assertIn("pid=101: denied", errors[0])

    def test_wait_returns_after_cleanup_releases_gpu(self):
        with self._wait_patches(
            reports=[["busy"], []], cleanup_result=([101], [])
        ) as cleanup:
            _wait_for_gpu_idle_in_ci(timeout=0, cleanup_timeout=0, poll_interval=0)
        cleanup.assert_called_once()

    def test_wait_can_disable_cleanup(self):
        with self._wait_patches(
            reports=[["busy"]], cleanup_enabled=False
        ) as cleanup:
            with self.assertRaisesRegex(RuntimeError, "still not idle"):
                _wait_for_gpu_idle_in_ci(
                    timeout=0, cleanup_timeout=0, poll_interval=0
                )
        cleanup.assert_not_called()

    def test_wait_reports_cleanup_errors(self):
        with self._wait_patches(
            reports=[["busy"]], cleanup_result=([], ["pid=101: denied"])
        ):
            with self.assertRaisesRegex(RuntimeError, "cleanup errors.*denied"):
                _wait_for_gpu_idle_in_ci(
                    timeout=0, cleanup_timeout=0, poll_interval=0
                )

    def _wait_patches(self, reports, cleanup_result=None, cleanup_enabled=True):
        pynvml = FakeNVML()
        return _WaitPatches(
            pynvml,
            reports,
            cleanup_result or ([101], []),
            cleanup_enabled,
        )


class _WaitPatches:
    def __init__(self, pynvml, reports, cleanup_result, cleanup_enabled):
        self.pynvml = pynvml
        self.reports = reports
        self.cleanup_result = cleanup_result
        self.cleanup_enabled = cleanup_enabled
        self.patches = []

    def __enter__(self):
        self.patches = [
            patch("sglang.test.test_utils.is_in_ci", return_value=True),
            patch.dict(sys.modules, {"pynvml": self.pynvml}),
            patch("sglang.test.test_utils._visible_gpu_indices", return_value=[0]),
            patch(
                "sglang.test.test_utils._collect_busy_gpu_reports",
                side_effect=self.reports,
            ),
            patch(
                "sglang.test.test_utils.get_bool_env_var",
                return_value=self.cleanup_enabled,
            ),
            patch(
                "sglang.test.test_utils.cleanup_residual_gpu_processes",
                return_value=self.cleanup_result,
            ),
        ]
        for item in self.patches:
            item.start()
        return self.patches[-1].new

    def __exit__(self, exc_type, exc_value, traceback):
        for item in reversed(self.patches):
            item.stop()
        return False


if __name__ == "__main__":
    unittest.main()
