"""Convert concatenated-K logits to request-local columns without index tensors."""

import triton
import triton.language as tl


@triton.jit
def _compact_logits_kernel(
    src,
    dst,
    starts,
    ends,
    src_stride_row,
    src_stride_col,
    dst_stride_row,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(starts + row).to(tl.int64)
    end = tl.load(ends + row).to(tl.int64)
    value = tl.load(
        src + row.to(tl.int64) * src_stride_row + (start + col) * src_stride_col,
        mask=(col < width) & (start + col < end),
        other=-float("inf"),
    ).to(tl.float32)
    tl.store(dst + row.to(tl.int64) * dst_stride_row + col, value, col < width)


def compact_logits_into(src, dst, starts, ends):
    """Copy each row's [start, end) interval, padding the destination with -inf."""
    if dst.numel() == 0:
        return
    _compact_logits_kernel[(dst.shape[0], triton.cdiv(dst.shape[1], 256))](
        src,
        dst,
        starts,
        ends,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.shape[1],
        BLOCK=256,
    )
