"""CPU checks for PPU-only Kimi-K3 backend restrictions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from sglang.kernels.jit.utils import arch as jit_arch
from sglang.srt.arg_groups import overrides as overrides_module
from sglang.srt.layers.attention.linear import kda_backend
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16DeepGemmMoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPpuKimiK3Constraints(CustomTestCase):
    def test_kda_dispatcher_forces_every_mode_to_triton_on_ppu(self):
        triton_kernel = SimpleNamespace(supports_packed_decode=False)
        with (
            patch.object(kda_backend, "is_sail", return_value=True),
            patch.object(kda_backend, "TritonKDAKernel", return_value=triton_kernel),
            patch.object(kda_backend, "rank0_log"),
        ):
            dispatcher = kda_backend.KDAKernelDispatcher(
                LinearAttnKernelBackend.FLASHINFER,
                LinearAttnKernelBackend.NVIDIA_KDA,
                LinearAttnKernelBackend.NV_CUTEDSL,
            )

        self.assertIs(dispatcher.decode_kernel, triton_kernel)
        self.assertIs(dispatcher.extend_kernel, triton_kernel)
        self.assertIs(dispatcher.verify_kernel, triton_kernel)

    def test_ppu_disables_pdl(self):
        with patch.object(jit_arch, "is_sail_runtime", return_value=True):
            self.assertFalse(jit_arch.is_arch_support_pdl.__wrapped__())

    def test_ppu_rejects_megamoe(self):
        with patch("sglang.srt.utils.common.is_sail", return_value=True):
            with self.assertRaisesRegex(ValueError, "MegaMoE is not supported on PPU"):
                overrides_module._a2a_backend_overrides(
                    SimpleNamespace(moe_a2a_backend="megamoe")
                )

    def test_ppu_w4a16_deep_gemm_moe_accepts_k3_situ(self):
        scheme = object.__new__(CompressedTensorsWNA16DeepGemmMoE)
        scheme.moe_runner_config = SimpleNamespace(activation="situ")
        scheme.runner = Mock()
        scheme.runner.run.return_value = sentinel.combine_input
        scheme.get_deep_gemm_quant_info = Mock(return_value=sentinel.quant_info)
        result = scheme.apply_weights(sentinel.layer, sentinel.dispatch_output)
        self.assertIs(result, sentinel.combine_input)
        scheme.get_deep_gemm_quant_info.assert_called_once_with(sentinel.layer)
        scheme.runner.run.assert_called_once_with(
            sentinel.dispatch_output, sentinel.quant_info
        )


if __name__ == "__main__":
    unittest.main()
