"""Unit tests for the DeepEP low-latency buffer multi-layout bookkeeping.

DSpark registers both the target and the draft MoE layout before the first
low-latency dispatch. The process-wide buffer must be sized for the largest
registered layout, later layouts must be validated against the live buffer
instead of triggering an unsafe growth, and switching between layouts must
clean the low-latency counters with the NEW layout's expert count. All of
this is pure state-machine arithmetic and is exercised here with a fake
``Buffer``; real DeepEP communication is covered by E2E runs.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.runtime_context import get_resources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _install_deepep_stubs():
    deep_ep = types.ModuleType("deep_ep")

    class _StubBuffer:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def get_low_latency_rdma_size_hint(*args, **kwargs):
            return 0

    class _StubConfig:
        def __init__(self, *args, **kwargs):
            pass

    deep_ep.Buffer = _StubBuffer
    deep_ep.Config = _StubConfig

    mxfp = types.ModuleType("triton_kernels.numerics_details.mxfp")
    mxfp.downcast_to_mxfp = lambda *args, **kwargs: None
    numerics = types.ModuleType("triton_kernels.numerics_details")
    numerics.mxfp = mxfp
    triton_kernels = types.ModuleType("triton_kernels")
    triton_kernels.numerics_details = numerics
    return {
        "deep_ep": deep_ep,
        "triton_kernels": triton_kernels,
        "triton_kernels.numerics_details": numerics,
        "triton_kernels.numerics_details.mxfp": mxfp,
    }


# The layout bookkeeping is pure state-machine arithmetic over a fake Buffer;
# only the extension modules need stubbing so this CPU suite actually runs.
with mock.patch.dict(sys.modules, _install_deepep_stubs()):
    from sglang.srt.layers.moe.token_dispatcher import deepep

WORLD = 4


def _rdma_hint(tokens, hidden, experts):
    return tokens * hidden * (experts // WORLD)


class _FakeBuffer:
    instances = []

    def __init__(self, group, num_nvl_bytes, num_rdma_bytes, **kwargs):
        self.num_rdma_bytes = num_rdma_bytes
        self.low_latency_mode = kwargs.get("low_latency_mode", False)
        self.num_qps_per_rank = kwargs.get("num_qps_per_rank", 0)
        self.cleaned = []
        _FakeBuffer.instances.append(self)

    @staticmethod
    def get_low_latency_rdma_size_hint(max_tokens, hidden, world, experts):
        assert world == WORLD
        return _rdma_hint(max_tokens, hidden, experts)

    def clean_low_latency_buffer(self, max_tokens, hidden, experts):
        self.cleaned.append((max_tokens, hidden, experts))


class TestDeepEPBufferLayouts(CustomTestCase):
    def setUp(self):
        _FakeBuffer.instances.clear()
        self.resources = get_resources()
        self.group = SimpleNamespace(size=lambda: WORLD)
        self.mode = deepep.DeepEPMode.LOW_LATENCY
        self.set_qp_depth = mock.Mock()
        patches = [
            mock.patch.dict(self.resources.buffers, {}, clear=True),
            mock.patch.object(deepep, "Buffer", _FakeBuffer),
            mock.patch.object(
                deepep.DeepEPConfig,
                "get_instance",
                return_value=SimpleNamespace(
                    normal_dispatch_config=None,
                    normal_combine_config=None,
                    num_sms=4,
                ),
            ),
            mock.patch.object(deepep, "_set_nvshmem_qp_depth", self.set_qp_depth),
            mock.patch.object(deepep, "_is_npu", False),
            mock.patch.object(deepep, "_is_sail", False),
            mock.patch.object(deepep, "_is_mnnvl_fabric_supported", lambda: False),
            mock.patch.object(deepep, "is_tbo_enabled", lambda: False),
            mock.patch.object(deepep, "DEEPEP_SUPPORT_TIMEOUT_CONTROL", False),
            mock.patch.object(
                deepep.torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(multi_processor_count=132),
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def state(self):
        return deepep.DeepEPBuffer._state()

    def seed_buffer(self, layout, *, low_latency_mode=True, dispatch_mode=None):
        rdma = _rdma_hint(*layout)
        buffer = _FakeBuffer(
            None,
            0,
            rdma,
            low_latency_mode=low_latency_mode,
            num_qps_per_rank=layout[2] // WORLD,
        )
        state = self.state()
        state.buffer = buffer
        state.num_rdma_bytes = rdma
        state.num_qps_per_rank = layout[2] // WORLD
        state.low_latency_layout = layout
        state.num_max_dispatch_tokens_per_rank, state.hidden_size, state.num_experts = (
            layout
        )
        state.dispatch_mode = dispatch_mode or deepep.DeepEPDispatchMode.LOW_LATENCY
        return buffer

    def get_buffer(self, tokens, hidden, experts):
        return deepep.DeepEPBuffer.get_deepep_buffer(
            self.group,
            hidden,
            param_bytes=4,
            deepep_mode=self.mode,
            num_max_dispatch_tokens_per_rank=tokens,
            num_experts=experts,
        )

    def test_allocation_sizes_for_all_registered_layouts(self):
        draft = (64, 4096, 128)
        self.state().registered_layouts.add(draft)
        buffer = self.get_buffer(32, 2048, 64)

        expect_rdma = max(_rdma_hint(*draft), _rdma_hint(32, 2048, 64))
        self.assertEqual(buffer.num_rdma_bytes, expect_rdma)
        self.assertEqual(buffer.num_qps_per_rank, max(128 // WORLD, 64 // WORLD))
        self.assertTrue(buffer.low_latency_mode)
        self.assertEqual(self.state().registered_layouts, {draft})
        self.set_qp_depth.assert_called_once_with(draft[0])

    def test_auto_allocation_reserves_normal_and_all_low_latency_layouts(self):
        self.mode = deepep.DeepEPMode.AUTO
        target = (512, 4096, 128)
        self.state().registered_layouts.add(target)
        normal_config = SimpleNamespace(
            get_nvl_buffer_size_hint=lambda *_: 1024,
            get_rdma_buffer_size_hint=lambda *_: 2048,
        )
        with mock.patch.object(
            deepep.DeepEPConfig,
            "get_instance",
            return_value=SimpleNamespace(
                normal_dispatch_config=normal_config,
                normal_combine_config=normal_config,
                num_sms=64,
            ),
        ):
            buffer = self.get_buffer(32, 2048, 64)
        self.assertEqual(buffer.num_rdma_bytes, _rdma_hint(*target))
        self.assertEqual(buffer.num_qps_per_rank, 64)
        self.set_qp_depth.assert_called_once_with(target[0])

    def test_existing_buffer_rejects_invalid_expert_partition(self):
        self.seed_buffer((32, 2048, 64))
        with self.assertRaises(AssertionError):
            self.get_buffer(32, 2048, 65)

    def test_existing_buffer_rejects_larger_layout(self):
        self.seed_buffer((32, 2048, 64))
        with self.assertRaisesRegex(RuntimeError, "cannot accommodate"):
            self.get_buffer(64, 4096, 128)
        self.assertEqual(len(_FakeBuffer.instances), 1)

    def test_existing_buffer_rejects_more_experts_than_qps(self):
        self.seed_buffer((32, 2048, 64))
        with self.assertRaisesRegex(RuntimeError, "cannot accommodate"):
            self.get_buffer(16, 1024, 256)

    def test_existing_buffer_accepts_fitting_layout(self):
        seeded = self.seed_buffer((64, 4096, 128))
        buffer = self.get_buffer(32, 2048, 64)
        self.assertIs(buffer, seeded)
        self.assertEqual(len(_FakeBuffer.instances), 1)

    def test_existing_normal_buffer_rejected_for_low_latency(self):
        self.seed_buffer((32, 2048, 64), low_latency_mode=False)
        with self.assertRaisesRegex(RuntimeError, "cannot accommodate"):
            self.get_buffer(32, 2048, 64)

    def test_same_layout_does_not_clean(self):
        seeded = self.seed_buffer((32, 2048, 64))
        deepep.DeepEPBuffer.set_dispatch_mode_as_low_latency((32, 2048, 64))
        self.assertEqual(seeded.cleaned, [])
        self.assertEqual(
            self.state().dispatch_mode, deepep.DeepEPDispatchMode.LOW_LATENCY
        )

    def test_layout_change_cleans_with_new_layout(self):
        seeded = self.seed_buffer((32, 2048, 64))
        new_layout = (64, 4096, 128)
        deepep.DeepEPBuffer.set_dispatch_mode_as_low_latency(new_layout)

        self.assertEqual(seeded.cleaned, [new_layout])
        state = self.state()
        self.assertEqual(state.low_latency_layout, new_layout)
        self.assertEqual(
            (
                state.num_max_dispatch_tokens_per_rank,
                state.hidden_size,
                state.num_experts,
            ),
            new_layout,
        )

    def test_normal_to_low_latency_cleans(self):
        seeded = self.seed_buffer(
            (32, 2048, 64), dispatch_mode=deepep.DeepEPDispatchMode.NORMAL
        )
        deepep.DeepEPBuffer.set_dispatch_mode(self.mode, (32, 2048, 64))
        self.assertEqual(seeded.cleaned, [(32, 2048, 64)])

    def test_no_buffer_only_sets_mode(self):
        layout = (32, 2048, 64)
        deepep.DeepEPBuffer.set_dispatch_mode_as_low_latency(layout)
        state = self.state()
        self.assertEqual(state.dispatch_mode, deepep.DeepEPDispatchMode.LOW_LATENCY)
        self.assertIsNone(state.buffer)
        # Without a buffer the call is a mode-only early return; the layout is
        # recorded once the buffer exists and the mode is set again.
        self.assertIsNone(state.low_latency_layout)


if __name__ == "__main__":
    unittest.main()
