"""One resident FP32 logits allocation per PPU worker/device.

Loading/warmup retains the original allocating path. The scheduler allocates the
registered arenas outside graph capture before admitting requests. Leases cover
the complete producer/consumer lifetime, including a score generator's yield.
"""

from __future__ import annotations

import inspect
import logging
import threading
from contextlib import contextmanager

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class LogitsWorkspace:
    def __init__(
        self,
        *,
        device: torch.device,
        capacity_bytes: int,
        preallocate_mask: bool = False,
    ):
        if capacity_bytes <= 0 or capacity_bytes % 4:
            raise ValueError(
                "FP32 logits workspace capacity must be positive and divisible by 4"
            )
        self.device = torch.device(device)
        self.capacity_bytes = capacity_bytes
        self.storage = None
        self.preallocate_mask = preallocate_mask
        self.mask_storage = None
        self._leased_elements = 0
        self._lock = threading.Lock()
        self._owner = None
        self._last_stream = None

    def initialize(self):
        if self.storage is not None:
            return
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Cannot initialize logits workspace during graph capture"
                    )
        self.storage = torch.empty(
            self.capacity_bytes // 4, dtype=torch.float32, device=self.device
        )
        if self.preallocate_mask:
            self.mask_storage = torch.empty(
                self.capacity_bytes // 4, dtype=torch.bool, device=self.device
            )
        logger.info(
            "PPU indexer workspace allocated: device=%s logits_bytes=%d mask_bytes=%d",
            self.device,
            self.capacity_bytes,
            self.mask_storage.numel() if self.mask_storage is not None else 0,
        )

    @contextmanager
    def lease(self, rows: int, width: int):
        if self.storage is None:
            raise RuntimeError("Logits workspace was not initialized before serving")
        aligned_rows = (rows + 3) // 4 * 4
        aligned_width = (width + 255) // 256 * 256
        required_bytes = aligned_rows * aligned_width * 4
        if rows <= 0 or width <= 0 or required_bytes > self.capacity_bytes:
            raise ValueError(
                f"Logits shape {(rows, width)} needs {required_bytes} bytes, "
                f"workspace capacity is {self.capacity_bytes}"
            )
        owner = threading.get_ident()
        if self._owner == owner:
            raise RuntimeError(
                "Nested logits workspace lease would overwrite live scores"
            )
        self._lock.acquire()
        stream = None
        try:
            self._owner = owner
            self._leased_elements = required_bytes // 4
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            "Resident prefill logits workspace is not graph-capture scratch"
                        )
                    stream = torch.cuda.current_stream(self.device)
                    if self._last_stream is not None and stream != self._last_stream:
                        # Only stream transitions need an event; the normal
                        # sequential prefill path adds no device synchronization.
                        stream.wait_stream(self._last_stream)
            # A flat prefix has the SAME row stride as the original producer's
            # current-shape torch.empty, including its 1024-byte row alignment.
            yield self.storage[: required_bytes // 4].view(aligned_rows, aligned_width)
        finally:
            if stream is not None:
                self._last_stream = stream
            self._owner = None
            self._leased_elements = 0
            self._lock.release()

    def causal_mask(self, rows: int, width: int):
        # The logits lease also protects this co-live mask and its CUDA stream.
        if self._owner != threading.get_ident():
            raise RuntimeError(
                "Causal mask requires the current thread's live logits lease"
            )
        if self.mask_storage is None:
            raise RuntimeError("Causal mask workspace was not initialized")
        if rows <= 0 or width <= 0 or rows * width > self._leased_elements:
            raise ValueError("Causal mask shape exceeds the active logits tile")
        return self.mask_storage[: rows * width].view(rows, width)


_workspaces: dict[torch.device, LogitsWorkspace] = {}
_serving = False


def logits_out_function():
    # The PPU package's top-level symbol is a pybind dispatcher without out.
    # Use its existing Python attention frontend explicitly for resident output.
    from deep_gemm.jit_kernels.attention import fp8_fp4_mqa_logits

    return fp8_fp4_mqa_logits


def register_logits_workspace(device: torch.device):
    preallocate_mask = envs.SGLANG_PPU_INDEXER_PREALLOCATE_MASK.get()
    if not envs.SGLANG_PPU_INDEXER_PREALLOCATE_LOGITS.get():
        if preallocate_mask:
            raise ValueError("PPU mask preallocation requires logits preallocation")
        return
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError(
            "PPU logits workspace requires an explicit CUDA-compatible device"
        )
    capacity = envs.SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB.get() * 1024 * 1024
    if capacity <= 0:
        raise ValueError("SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB must be positive")
    if device in _workspaces:
        if (
            _workspaces[device].capacity_bytes != capacity
            or _workspaces[device].preallocate_mask != preallocate_mask
        ):
            raise RuntimeError(
                "Cannot change workspace configuration after registration"
            )
        return
    if _serving:
        raise RuntimeError(
            "Logits workspace must be registered during model initialization"
        )
    if "out" not in inspect.signature(logits_out_function()).parameters:
        raise RuntimeError(
            "PPU logits preallocation needs DeepGEMM's optional out interface; "
            "apply test/ppu/patches/deep_gemm_mqa_logits_out.patch first"
        )
    _workspaces[device] = LogitsWorkspace(
        device=device, capacity_bytes=capacity, preallocate_mask=preallocate_mask
    )


def initialize_logits_workspaces():
    global _serving
    for workspace in _workspaces.values():
        workspace.initialize()
    _serving = True


@contextmanager
def logits_workspace(*, device: torch.device, rows: int, width: int):
    if not envs.SGLANG_PPU_INDEXER_PREALLOCATE_LOGITS.get() or not _serving:
        # Startup warmups/captures keep their original allocation semantics.
        yield None
        return
    device = torch.device(device)
    if device not in _workspaces:
        raise RuntimeError(f"No preallocated logits workspace registered for {device}")
    workspace = _workspaces[device]
    if (
        workspace.capacity_bytes
        != envs.SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB.get() * 1024 * 1024
    ):
        raise RuntimeError("Logits budget changed after workspace initialization")
    with workspace.lease(rows, width) as out:
        yield out


def causal_mask_workspace(*, device: torch.device, rows: int, width: int):
    if not envs.SGLANG_PPU_INDEXER_PREALLOCATE_MASK.get() or not _serving:
        return None
    device = torch.device(device)
    if (
        not envs.SGLANG_PPU_INDEXER_PREALLOCATE_LOGITS.get()
        or device not in _workspaces
    ):
        raise RuntimeError(
            "Causal mask workspace requires registered logits preallocation"
        )
    return _workspaces[device].causal_mask(rows, width)
