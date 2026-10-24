"""SAIL device operations for the SRT platform layer.

SAIL exposes a CUDA-compatible API through its torch backend, so
``SailDeviceMixin`` inherits all device ops from ``CudaDeviceMixin`` and
only overrides identity (``_enum``, ``device_name``).

SAIL presence is detected by matching the ``ZW`` tag in the CUDA device name,
keeping the check consistent with ``sglang.srt.utils.is_sail``.
"""

from functools import lru_cache

import torch

from sglang.srt.platforms.cuda import CudaDeviceMixin
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import PlatformCapabilities, SRTPlatform

_ZW_TAG = "ZW"
_ZW810E_NAME = "ZW810E"


@lru_cache(maxsize=1)
def is_sail_available() -> bool:
    """Return True when the local device is a SAIL (device name holds "ZW")."""
    return bool(torch.cuda.is_available() and _ZW_TAG in torch.cuda.get_device_name())


class SailDeviceMixin(CudaDeviceMixin):
    """SAIL device ops — identical surface to CUDA via its torch backend."""

    _enum: PlatformEnum = PlatformEnum.SAIL
    device_name: str = "sail"
    # device_type stays "cuda" — torch.device("cuda") is the only valid
    # device-type string for SAIL devices in PyTorch.

    def is_cuda(self) -> bool:
        # SAIL exposes a CUDA-compatible API, so it is also considered "cuda"
        return True


class SailSRTPlatform(SailDeviceMixin, SRTPlatform):
    """Default in-tree SAIL SRT platform.

    Inherits CUDA capability flags since SAIL exposes CUDA-compatible APIs
    (fp8, cuda graph, piecewise cuda graph are supported on SAIL).
    """

    capabilities = PlatformCapabilities(
        supports_triton=True,
        graph_capture=True,
        piecewise_graph=True,
    )

    def is_zw810e(self) -> bool:
        return _ZW810E_NAME in self.get_device_name()

    def get_device_num_tensorcores(self, device_id: int = 0) -> int:
        if self.is_zw810e():
            return 20
        return super().get_device_num_tensorcores(device_id)


# Load SAIL hooks when running on SAIL hardware.
# Each module registers @plugin_hook decorators that inject SAIL-native ops
# into the respective backends transparently.
if is_sail_available():
    import sglang.srt.hardware_backend.ppu.attention.ppu_dsa_hooks  # noqa: F401
    import sglang.srt.hardware_backend.ppu.attention.ppu_dsv41_cp_hooks  # noqa: F401
    import sglang.srt.hardware_backend.ppu.attention.ppu_fa3_hooks  # noqa: F401
    import sglang.srt.hardware_backend.ppu.attention.ppu_flashmla_hooks  # noqa: F401
    import sglang.srt.hardware_backend.ppu.attention.ppu_msa_hook  # noqa: F401
    import sglang.srt.hardware_backend.ppu.moe.ppu_deepep_hooks  # noqa: F401
