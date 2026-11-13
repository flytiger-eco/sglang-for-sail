"""HC sequence parallel 的 interleaved 布局变换与 padding 融合。"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _interleaved_rows_kernel(
    src,
    dst,
    num_tokens,
    output_rows,
    WIDTH: tl.constexpr,
    LOCAL_ROWS,
    WORLD_SIZE: tl.constexpr,
    RANK: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    if MODE == 0:  # scatter：全局行 -> 本 rank 的 round-robin 行。
        src_rows = rows * WORLD_SIZE + RANK
        src_valid = src_rows < num_tokens
    elif MODE == 1:  # pack：全局行 -> Reduce-Scatter 的 rank-major 输入。
        ranks = rows // LOCAL_ROWS
        src_rows = (rows % LOCAL_ROWS) * WORLD_SIZE + ranks
        src_valid = src_rows < num_tokens
    else:  # unpack：All-Gather 的 rank-major 输出 -> 全局行。
        src_rows = (rows % WORLD_SIZE) * LOCAL_ROWS + rows // WORLD_SIZE
        src_valid = rows < num_tokens
    valid = (rows < output_rows)[:, None] & (cols < WIDTH)[None, :]
    values = tl.load(
        src + src_rows[:, None] * WIDTH + cols[None, :],
        mask=valid & src_valid[:, None],
        other=0.0,
    )
    tl.store(dst + rows[:, None] * WIDTH + cols[None, :], values, mask=valid)


def _transform(x, num_tokens, local_tokens, world_size, rank, mode):
    if x.ndim < 2:
        raise ValueError("HC-SP tensors require token and feature dimensions")
    rows = local_tokens if mode == 0 else local_tokens * world_size
    if mode == 2:
        rows = num_tokens
    output = x.new_empty((rows, *x.shape[1:]))
    if rows == 0:
        return output
    if not x.is_cuda:
        indices = torch.arange(rows, device=x.device)
        if mode == 0:
            source = indices * world_size + rank
            valid = source < num_tokens
        elif mode == 1:
            source = (indices % local_tokens) * world_size + indices // local_tokens
            valid = source < num_tokens
        else:
            source = (indices % world_size) * local_tokens + indices // world_size
            valid = indices < num_tokens
        output.zero_()
        output[valid] = x[source[valid]]
        return output
    width = math.prod(x.shape[1:])
    _interleaved_rows_kernel[(triton.cdiv(rows, 4), triton.cdiv(width, 1024))](
        x.contiguous(),
        output,
        num_tokens,
        rows,
        WIDTH=width,
        LOCAL_ROWS=local_tokens,
        WORLD_SIZE=world_size,
        RANK=rank,
        MODE=mode,
        BLOCK_ROWS=4,
        BLOCK_COLS=1024,
        num_warps=8,
    )
    return output


def hc_sp_scatter_interleaved(x, *, num_tokens, local_tokens, world_size, rank):
    return _transform(x, num_tokens, local_tokens, world_size, rank, 0)


def hc_sp_pack_interleaved(x, *, num_tokens, local_tokens, world_size):
    return _transform(x, num_tokens, local_tokens, world_size, 0, 1)


def hc_sp_unpack_interleaved(x, *, num_tokens, local_tokens, world_size):
    return _transform(x, num_tokens, local_tokens, world_size, 0, 2)
