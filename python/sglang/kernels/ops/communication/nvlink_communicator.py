"""NVLink plane ABI from upstream, alongside the release all-reduce ABI."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch
import tvm_ffi

from sglang.kernels.jit.utils import (
    cache_once,
    empty_sentinel,
    lazy_register_class,
    load_jit,
)


@cache_once
def _init_communicator() -> None:
    module = load_jit(
        "nvlink_communicator",
        cuda_files=["distributed/nvlink_registry.cuh"],
        cuda_wrappers=[("register_communicator", "register_nvlink_communicator")],
    )
    module.register_communicator()


@lazy_register_class("sgl.distributed.PushPlane", _init_communicator)
class PushPlane(tvm_ffi.Object):
    """Lamport push plane: a zero-filled symmetric workspace + a local counter.

    All buffers are owned by the caller; this object only validates and
    records them.
    """

    # C++ interface
    if TYPE_CHECKING:
        rank: int
        world_size: int

    def __init__(
        self,
        rank: int,
        world_size: int,
        *,
        workspaces: List[torch.Tensor],
        counter: torch.Tensor,
        mc_workspace: int | None = None,
    ) -> None:
        """
        :param workspaces: per-rank ``[2 * world_size, slot_bytes]`` uint8
                           views of symmetric memory. The local rank's view
                           MUST be zero-filled before first use -- the
                           kernels poll for a pos-zero marker.
        :param counter: local ``[num_blocks, 4]`` uint8 tensor, zero-filled.
        :param mc_workspace: multicast VA of the local workspace, or None.
        """
        self.__ffi_init__(rank, world_size, workspaces, counter, mc_workspace or 0)


@lazy_register_class("sgl.distributed.PullPlane", _init_communicator)
class PullPlane(tvm_ffi.Object):
    """Symmetric per-rank buffers plus the per-block semaphores guarding them.

    Either half may be omitted; the plane then holds a 0-element tensor in its
    place and any kernel needing that half fails with a clear message. The K3
    fused collectives take semaphores only -- they reduce in place on the
    caller's own symmetric input -- while the generic all-reduce takes both,
    since it stages plain tensors through the buffers before reducing.
    """

    # C++ interface
    if TYPE_CHECKING:
        rank: int
        world_size: int

    def __init__(
        self,
        rank: int,
        world_size: int,
        *,
        workspaces: List[torch.Tensor] | None = None,
        semaphores: List[torch.Tensor] | None = None,
        mc_workspace: int | None = None,
        mc_semaphore: int | None = None,
    ) -> None:
        """
        :param workspaces: per-rank ``[num_bytes]`` uint8 views of symmetric
                           memory, or None when the caller brings its own.
        :param semaphores: per-rank ``[num_blocks, 128]`` uint8 views of
                           symmetric memory, zero-filled before first use, or
                           None when the caller never barriers on this plane.
        :param mc_workspace: multicast VA of the local workspace, or None.
        :param mc_semaphore: multicast VA of the local semaphores, or None.
        """
        if workspaces is None:
            device = torch.device("cuda", torch.cuda.current_device())
            sentinel = empty_sentinel(device, torch.uint8).view(-1)
            workspaces = [sentinel for _ in range(world_size)]
        if semaphores is None:
            device = torch.device("cuda", torch.cuda.current_device())
            sentinel = empty_sentinel(device, torch.uint8).view(-1, 128)
            semaphores = [sentinel for _ in range(world_size)]

        mc_workspace = mc_workspace or 0
        mc_semaphore = mc_semaphore or 0
        self.__ffi_init__(
            rank, world_size, workspaces, semaphores, mc_workspace, mc_semaphore
        )


@lazy_register_class("sgl.distributed.Communicator", _init_communicator)
class Communicator(tvm_ffi.Object):
    """The planes shared by every kernel in ``kernels.ops.communication``.

    Pass ``None`` for a plane the owner never uses; kernels that need it then
    fail with a clear message instead of reading a placeholder buffer.
    """

    if TYPE_CHECKING:
        # C++ interface
        def get_rank(self) -> int: ...
        def get_world_size(self) -> int: ...
        def get_push(self) -> PushPlane | None: ...
        def get_pull(self) -> PullPlane | None: ...
        def set_pull_blocks(self, num_blocks: int | None) -> None: ...
        def set_pull_multicast_blocks(self, num_blocks: int | None) -> None: ...

    def __init__(
        self,
        push: PushPlane | None = None,
        pull: PullPlane | None = None,
    ) -> None:
        self.__ffi_init__(push, pull)

    @property
    def rank(self) -> int:
        return self.get_rank()

    @property
    def world_size(self) -> int:
        return self.get_world_size()

    @property
    def push(self) -> PushPlane | None:
        """The push plane, or None for a pull-only communicator."""
        return self.get_push()

    @property
    def pull(self) -> PullPlane | None:
        """The pull plane, or None for a push-only communicator."""
        return self.get_pull()


def from_release_all_reduce(owner) -> Communicator:
    """View release-owned buffers through the upstream plane ABI.

    Called collectively at model initialization, outside graph capture. The
    owner retains the symmetric allocation; no existing communicator is replaced.
    """
    if getattr(owner.obj, "push", None) is not None:
        return owner.obj
    cached = getattr(owner, "_nvlink_communicator", None)
    if cached is not None:
        return cached
    from torch._C._distributed_c10d import _SymmetricMemory

    slab = owner._symm_tensor
    memory = _SymmetricMemory.rendezvous(slab)
    rank, size = owner.rank, owner.world_size
    push_bytes = 2 * size * owner.max_push_size
    pull_bytes = owner.max_pull_size
    num_pull_blocks = owner.config.num_pull_blocks
    sem_offset = push_bytes + pull_bytes
    peers = [memory.get_buffer(i, [slab.numel()], torch.uint8) for i in range(size)]
    mc = int(memory.multicast_ptr)
    push = PushPlane(
        rank,
        size,
        workspaces=[p[:push_bytes].view(2 * size, owner.max_push_size) for p in peers],
        counter=owner._push_counter.view(-1, 1).view(torch.uint8),
        mc_workspace=mc,
    )
    pull = PullPlane(
        rank,
        size,
        workspaces=[p[push_bytes:sem_offset] for p in peers],
        semaphores=[
            p[sem_offset : sem_offset + num_pull_blocks * 128].view(
                num_pull_blocks, 128
            )
            for p in peers
        ],
        mc_workspace=mc + push_bytes if mc else 0,
        mc_semaphore=mc + sem_offset if mc else 0,
    )
    owner._nvlink_communicator = Communicator(push, pull)
    return owner._nvlink_communicator
