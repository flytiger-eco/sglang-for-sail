"""CPU regressions for the DeepEP CUDA-graph adapter layout handoff.

capture() records the dispatch mode, capture_complete() records the
low-latency layout the buffer settled on during capture, and replay() must
restore both. A runner that captures without completing replays with
layout=None, which silently reuses the layout the global state currently
holds (e.g. the target's) and skips cleaning the draft's signaling regions.

The adapter class is loaded via AST with injected globals so this suite runs
without deep_ep or CUDA.
"""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_SRT = Path(importlib.util.find_spec("sglang").submodule_search_locations[0]) / "srt"
_ADAPTER = _SRT / "model_executor/runner_utils/deepep_adapter.py"
_MODE = "low_latency"
_LAYOUT = ("draft", 4096, 256)


def _load_adapter_class(namespace):
    nodes = ast.parse(_ADAPTER.read_text(encoding="utf-8")).body
    cls = next(
        node
        for node in nodes
        if isinstance(node, ast.ClassDef)
        and node.name == "DeepEPCudaGraphRunnerAdapter"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(_ADAPTER), "exec"), namespace)
    return namespace["DeepEPCudaGraphRunnerAdapter"]


class TestDeepEPGraphAdapterLayout(CustomTestCase):
    def _make(self, *, deepep=True):
        buffer_cls = MagicMock()
        buffer_cls._state.return_value = SimpleNamespace(low_latency_layout=_LAYOUT)
        mode = MagicMock()
        mode.resolve.return_value = _MODE
        namespace = {
            "DeepEPBuffer": buffer_cls,
            "get_deepep_mode": MagicMock(return_value=mode),
            "get_moe_a2a_backend": MagicMock(
                return_value=SimpleNamespace(is_deepep=lambda: deepep)
            ),
        }
        return _load_adapter_class(namespace)(), buffer_cls

    def test_replay_restores_the_captured_layout(self):
        adapter, buffer_cls = self._make()
        adapter.capture(is_extend_in_batch=False)
        adapter.capture_complete()
        # Another runner dispatches between this runner's capture and replay.
        buffer_cls._state.return_value.low_latency_layout = (512, 7168, 512)
        buffer_cls.set_dispatch_mode.reset_mock()

        adapter.replay()

        buffer_cls.set_dispatch_mode.assert_called_once_with(_MODE, _LAYOUT)

    def test_recapture_discards_the_previous_layout(self):
        adapter, buffer_cls = self._make()
        adapter.capture(is_extend_in_batch=False)
        adapter.capture_complete()
        adapter.capture(is_extend_in_batch=True)
        self.assertIsNone(adapter._captured_deepep_layout)
        new_layout = (512, 7168, 512)
        buffer_cls._state.return_value.low_latency_layout = new_layout
        adapter.capture_complete()
        adapter.replay()
        buffer_cls.set_dispatch_mode.assert_called_with(_MODE, new_layout)

    def test_missing_capture_complete_replays_without_a_layout(self):
        # Characterizes the hazard: without capture_complete() replay passes
        # layout=None, so the buffer keeps whatever layout is currently set.
        adapter, buffer_cls = self._make()
        adapter.capture(is_extend_in_batch=False)
        buffer_cls.set_dispatch_mode.reset_mock()

        adapter.replay()

        buffer_cls.set_dispatch_mode.assert_called_once_with(_MODE, None)

    def test_non_deepep_is_a_no_op(self):
        adapter, buffer_cls = self._make(deepep=False)

        adapter.capture(is_extend_in_batch=False)
        adapter.capture_complete()
        adapter.replay()

        buffer_cls.set_dispatch_mode.assert_not_called()

    def test_every_adapter_user_completes_capture(self):
        offenders = []
        for path in _SRT.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if (
                "deepep_adapter.capture(" in source
                and "deepep_adapter.capture_complete(" not in source
            ):
                offenders.append(str(path.relative_to(_SRT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
