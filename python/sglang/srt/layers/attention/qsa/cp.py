"""QSA-only CP 的 token ownership 与层边界通信。"""

from dataclasses import dataclass

import torch

from sglang.srt.runtime_context import get_parallel


@dataclass(frozen=True)
class QSAOnlyCPContext:
    """query 按 rank::world_size 分片，KV 保持请求级全局视图。"""

    total_rows: int
    local_capacity: int
    rank: int
    world_size: int
    hc_interleaved: bool = False

    @property
    def local_rows(self) -> int:
        return self.total_rows // self.world_size + int(
            self.total_rows % self.world_size > self.rank
        )

    def shard(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[self.rank :: self.world_size][: self.local_rows].contiguous()


def qsa_only_cp_gather_rows(tensor: torch.Tensor, context: QSAOnlyCPContext):
    """聚合本地行并恢复全局顺序；不暴露 scheduler 的 padding 行。"""
    if tensor.shape[0] > context.local_capacity:
        raise ValueError("QSA-only CP tensor exceeds its local capacity")
    if tensor.shape[0] < context.local_capacity:
        padded = tensor.new_zeros((context.local_capacity, *tensor.shape[1:]))
        padded[: tensor.shape[0]].copy_(tensor)
    else:
        padded = tensor.contiguous()
    gathered = tensor.new_empty(
        (context.world_size * context.local_capacity, *tensor.shape[1:])
    )
    get_parallel().attn_tp_group.all_gather_into_tensor(gathered, padded)
    return (
        gathered.reshape(context.world_size, context.local_capacity, *tensor.shape[1:])
        .transpose(0, 1)
        .reshape(context.world_size * context.local_capacity, *tensor.shape[1:])[
            : context.total_rows
        ]
        .contiguous()
    )


def qsa_only_cp_to_interleaved(tensor: torch.Tensor, context: QSAOnlyCPContext):
    """将连续 HC-SP ownership 转为 QSA 的 round-robin ownership。"""
    if context.world_size == 1 or context.hc_interleaved:
        return tensor
    if tensor.shape[0] != context.local_capacity:
        raise ValueError("QSA-only CP input must match the HC-SP local capacity")
    if context.local_capacity % context.world_size:
        # 不规则尾块通过 gather 保持正确性，不要求 scheduler 桶对齐。
        full = tensor.new_empty(
            (context.world_size * context.local_capacity, *tensor.shape[1:])
        )
        get_parallel().attn_tp_group.all_gather_into_tensor(full, tensor.contiguous())
        local = context.shard(full[: context.total_rows])
        output = torch.zeros_like(tensor)
        output[: local.shape[0]].copy_(local)
        return output
    rows_per_peer = context.local_capacity // context.world_size
    send = (
        tensor.reshape(rows_per_peer, context.world_size, *tensor.shape[1:])
        .transpose(0, 1)
        .contiguous()
        .reshape_as(tensor)
    )
    output = torch.empty_like(send)
    get_parallel().attn_tp_group.all_to_all_single(output, send)
    return output


def qsa_only_cp_from_interleaved(tensor: torch.Tensor, context: QSAOnlyCPContext):
    """恢复 HC-SP ownership；布局已一致时无需 collective。"""
    if context.world_size == 1 or context.hc_interleaved:
        return tensor
    if context.local_capacity % context.world_size:
        full = qsa_only_cp_gather_rows(tensor, context)
        start = context.rank * context.local_capacity
        local = full[start : start + context.local_capacity]
        output = torch.zeros_like(tensor)
        output[: local.shape[0]].copy_(local)
        return output
    output = torch.empty_like(tensor)
    get_parallel().attn_tp_group.all_to_all_single(output, tensor.contiguous())
    return (
        output.reshape(
            context.world_size,
            context.local_capacity // context.world_size,
            *tensor.shape[1:],
        )
        .transpose(0, 1)
        .contiguous()
        .reshape_as(tensor)
    )
