"""PPU device operations for the SRT platform layer.

T-HEAD PPU exposes a CUDA-compatible API through its torch backend, so
``PPUDeviceMixin`` inherits all device ops from ``CudaDeviceMixin`` and
only overrides identity (``_enum``, ``device_name``).

PPU presence is detected via the ``PPU_SDK`` environment variable.
"""

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

_ZW810E_NAME = "ZW810E"
_810_TAG = "810"


class PPUDeviceMixin(CudaDeviceMixin):
    """PPU device ops — identical surface to CUDA via T-HEAD's torch backend."""

    _enum: PlatformEnum = PlatformEnum.PPU
    device_name: str = "ppu"
    # device_type stays "cuda" — torch.device("cuda") is the only valid
    # device-type string for PPU devices in PyTorch.

    def is_cuda(self) -> bool:
        # PPU exposes a CUDA-compatible API, so it is also considered "cuda"
        return True


class PPUSRTPlatform(PPUDeviceMixin, SRTPlatform):
    """Default in-tree T-HEAD PPU SRT platform.

    Inherits CUDA capability flags since PPU exposes CUDA-compatible APIs
    (fp8, cuda graph, piecewise cuda graph are supported on PPU).
    """

    def supports_fp8(self) -> bool:
        return self.get_device_capability().to_int() >= 89

    def support_cuda_graph(self) -> bool:
        return True

    def support_piecewise_cuda_graph(self) -> bool:
        return True

    def get_jit_cuda_arch_suffix(self) -> str:
        # PPU 810/810e devices require the "sm80a" arch variant for JIT compilation.
        return "a" if _810_TAG in self.get_device_name() else ""

    def is_zw810e(self) -> bool:
        return _ZW810E_NAME in self.get_device_name()

    def get_device_num_tensorcores(self, device_id: int = 0) -> int:
        if self.is_zw810e():
            return 20
        return super().get_device_num_tensorcores(device_id)


# Load PPU FA3 ops hook when running on PPU hardware.
# The module-level @plugin_hook decorator registers an AROUND hook on
# FlashAttentionBackend.__init__ so PPU-specific FA3 ops are used transparently.
import os

if "PPU_SDK" in os.environ:
    import sglang.srt.hardware_backend.ppu.attention.ppu_fa3_hooks  # noqa: F401
