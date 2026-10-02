"""SAIL device operations for the SRT platform layer.

T-HEAD SAIL exposes a CUDA-compatible API through its torch backend, so
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
    """Return True when the local device is a T-HEAD SAIL (device name holds "ZW")."""
    return bool(torch.cuda.is_available() and _ZW_TAG in torch.cuda.get_device_name())


class SailDeviceMixin(CudaDeviceMixin):
    """SAIL device ops — identical surface to CUDA via T-HEAD's torch backend."""

    _enum: PlatformEnum = PlatformEnum.SAIL
    device_name: str = "sail"
    # device_type stays "cuda" — torch.device("cuda") is the only valid
    # device-type string for SAIL devices in PyTorch.

    def is_cuda(self) -> bool:
        # SAIL exposes a CUDA-compatible API, so it is also considered "cuda"
        return True


class SailSRTPlatform(SailDeviceMixin, SRTPlatform):
    """Default in-tree T-HEAD SAIL SRT platform.

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


# Load SAIL FA3 ops hook when running on SAIL hardware.
# The module-level @plugin_hook decorator registers an AROUND hook on
# FlashAttentionBackend.__init__ so SAIL-specific FA3 ops are used transparently.
if is_sail_available():
    import sglang.srt.hardware_backend.ppu.attention.ppu_fa3_hooks  # noqa: F401
