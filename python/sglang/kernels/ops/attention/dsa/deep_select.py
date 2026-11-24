from __future__ import annotations

import torch
import triton
import triton.language as tl

# The wheel documents K in 1..4096; anything wider is outside its contract.
MAX_TOPK = 4096


@triton.jit
def _pack_ragged_scores(
    scores,
    lengths,
    row_starts,
    packed,
    SCORE_ROW_STRIDE: tl.constexpr,
    SCORE_COL_STRIDE: tl.constexpr,
    PACKED_ROW_STRIDE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(lengths + row)
    # Indices-only DeepSelect does not read scores for rows shorter than topk.
    if length > TOPK:
        start = tl.load(row_starts + row) if HAS_STARTS else 0
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(
            scores + row * SCORE_ROW_STRIDE + (start + cols) * SCORE_COL_STRIDE,
            mask=cols < length,
            other=0,
        )
        tl.store(packed + row * PACKED_ROW_STRIDE + cols, values, mask=cols < length)



@triton.jit(do_not_specialize=["SCORE_ROW_STRIDE", "PACKED_ROW_STRIDE"])
def _pack_ragged_scores_runtime(
    scores,
    lengths,
    row_starts,
    packed,
    SCORE_ROW_STRIDE,
    SCORE_COL_STRIDE: tl.constexpr,
    PACKED_ROW_STRIDE,
    HAS_STARTS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    ALIGNED_SCORE_ROWS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths + row)
    if length > TOPK:
        start = tl.load(row_starts + row) if HAS_STARTS else 0
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        score_stride = SCORE_ROW_STRIDE
        if ALIGNED_SCORE_ROWS:
            score_stride = tl.multiple_of(score_stride, 512)
        packed_stride = tl.multiple_of(PACKED_ROW_STRIDE, 512)
        values = tl.load(
            scores + row * score_stride + (start + cols) * SCORE_COL_STRIDE,
            mask=cols < length,
            other=0,
        )
        tl.store(packed + row * packed_stride + cols, values, mask=cols < length)


_WARMED_RAGGED_PACKERS: set[tuple[torch.device, int]] = set()


def _warm_ragged_packers(scores: torch.Tensor, topk: int) -> None:
    key = (scores.device, topk)
    if key in _WARMED_RAGGED_PACKERS:
        return
    # Seed the finite generic variants before the first dynamic business shape.
    # Run normal warmup before CUDA Graph capture, as for the selector itself.
    source = scores.new_zeros((4, 8192))
    destination = torch.empty_like(source)
    lengths = torch.full((4,), 8192, device=scores.device, dtype=torch.int32)
    starts = torch.zeros((4,), device=scores.device, dtype=torch.int32)
    for block in (1024, 2048):
        for has_starts in (False, True):
            _pack_ragged_scores_runtime[(4, triton.cdiv(8192, block))](
                source,
                lengths,
                starts if has_starts else None,
                destination,
                8192,
                1,
                8192,
                has_starts,
                topk,
                block,
                True,
                num_warps=4,
            )
    _WARMED_RAGGED_PACKERS.add(key)


def topk_transform_ragged(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    out_offsets: torch.Tensor,
    row_starts: torch.Tensor | None = None,
    *,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """PPU DeepSelect over row windows, returning rebased int32 indices and -1 padding.

    The wheel supports end lengths but not begin offsets. Pack nontrivial
    windows into BF16 storage when needed; short rows need only their indices.
    No FP32 score matrix is materialized and the source scores are unchanged.
    ``max_seq_len`` is an optional host-known upper bound on per-row lengths,
    allowing packed storage to omit the other requests' columns.
    """
    import deep_select

    rows, cols = scores.shape
    if rows == 0:
        return scores.new_empty((0, topk), dtype=torch.int32)

    _warm_ragged_packers(scores, topk)
    lengths = lengths.contiguous()
    out_offsets = out_offsets.contiguous()
    alignment = deep_select.get_stride_requirement()[0]
    needs_pack = (
        (row_starts is not None and cols > topk)
        or scores.stride(1) != 1
        or scores.stride(0) * scores.element_size() % alignment != 0
        or scores.data_ptr() % 16 != 0
    )
    if needs_pack:
        packed_cols = cols if max_seq_len is None else min(cols, max_seq_len)
        aligned_cols = triton.cdiv(packed_cols, alignment // scores.element_size()) * (
            alignment // scores.element_size()
        )
        packed = scores.new_empty((rows, aligned_cols))
        if packed_cols > topk:
            starts = row_starts.contiguous() if row_starts is not None else None
            block = 2048 if max_seq_len is not None and packed_cols >= 4096 else 1024
            if packed.stride(0) % 512 == 0:
                _pack_ragged_scores_runtime[(rows, triton.cdiv(packed_cols, block))](
                    scores,
                    lengths,
                    starts,
                    packed,
                    scores.stride(0),
                    scores.stride(1),
                    packed.stride(0),
                    starts is not None,
                    topk,
                    block,
                    scores.stride(0) % 512 == 0,
                    num_warps=4,
                )
            else:
                # Keep the general layout path if a future wheel changes alignment.
                _pack_ragged_scores[(rows, triton.cdiv(packed_cols, block))](
                    scores,
                    lengths,
                    starts,
                    packed,
                    scores.stride(0),
                    scores.stride(1),
                    packed.stride(0),
                    starts is not None,
                    topk,
                    block,
                    num_warps=4,
                )
        scores = packed[:, :packed_cols]

    _, indices = deep_select.topk(
        scores,
        topk,
        end=lengths,
        indices_type=torch.int32,
        output_idx_offset=out_offsets,
        idx_oob_fill_value=-1,
        return_value=False,
    )
    return indices.contiguous()


@triton.jit
def _gather_page_ids(
    indices,
    page_tables,
    out,
    K: tl.constexpr,
    table_width,
    IDX_ROW_STRIDE: tl.constexpr,
    TABLE_ROW_STRIDE: tl.constexpr,
    OUT_ROW_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """page_ids[row, i] = page_tables[row, indices[row, i]] (-1 stays -1).

    One launch replaces the gather the fused dsv4 prefill kernel does inline, so
    the selected columns and the page lookup stay separate but still single-pass.
    """
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < K
    column = tl.load(indices + row * IDX_ROW_STRIDE + cols, mask=mask, other=-1)
    in_table = mask & (column >= 0) & (column < table_width)
    page = tl.load(
        page_tables + row * TABLE_ROW_STRIDE + column, mask=in_table, other=-1
    )
    tl.store(out + row * OUT_ROW_STRIDE + cols, page, mask=mask)


def _alignment_bytes() -> int:
    import deep_select

    return deep_select.get_stride_requirement()[0]


def paged_topk_skip_reason(scores: torch.Tensor, topk: int) -> str | None:
    """Static wheel-contract check for the PAGED/bf16 route. Performs no host read."""
    if scores.dim() != 2:
        return f"scores rank {scores.dim()} != 2"
    if scores.dtype != torch.bfloat16:
        return f"scores dtype {scores.dtype} != bfloat16"
    if not 0 < topk <= MAX_TOPK:
        return f"topk {topk} outside the deep_select contract (1..{MAX_TOPK})"
    if scores.stride(1) != 1:
        return "scores last dimension is not contiguous"
    stride_bytes = scores.stride(0) * scores.element_size()
    if stride_bytes % _alignment_bytes():
        return f"scores row stride {stride_bytes}B not aligned to {_alignment_bytes()}B"
    return None


# Work-shape band the PAGED route is worth serving, measured on ZW-M890P (card idle,
# reps=9 grid + a reps=21 replicate of the crossover cells; evidence
# bench_thr_20261119T074835Z/grid_start0.json and bench_thr_boundary_20261119T075131Z).
# Cost of this path over the fused dsv4 bf16 prefill kernel, per call:
#
#   rows x cols      fused   deep_select+gather   shipped(+one host read)
#     1 x  2048      0.016ms        0.031ms              0.075ms   (+373%)
#    16 x 32768      0.029ms        0.033ms              0.072ms   (+146%)
#    64 x 32768      0.035ms        0.038ms              0.077ms   (+124%)
#   256 x 32768      0.106ms        0.072ms              0.115ms    (+8.2%)
#   512 x 32768      0.179ms        0.125ms              0.167ms    (-6.2%)
#  1024 x  8192      0.150ms        0.124ms              0.166ms   (+10.6%)
#  1024 x 32768      0.336ms        0.222ms              0.264ms   (-21.4%)
#  7808 x 64000      8.263ms        1.771ms              1.814ms   (-78.0%)
#
# The selection only pays for the extra launches once every row has a long window to
# search (cols) and there are enough rows to hide them (rows); below the band the fused
# kernel wins, so small decode-shaped calls are kept there by the dispatch below. The
# shipped column includes one ~42us host read per call (see _constant_window_start),
# which is why the band's corners are judged on the shipped numbers, not the kernel-only
# ones: 256 x 32768 and 1024 x 8192 lose *with* that read even though the kernel alone
# would win. Both floors sit on the far side of the measured crossover with margin
# (2x above the losing 256-row cell, 4x above the losing 8192-column cell).
PAGED_SERVE_MIN_ROWS = 512
PAGED_SERVE_MIN_COLS = 32768


def paged_topk_shape_reason(scores: torch.Tensor) -> str | None:
    """Return why this call's shape stays on the fused kernel, else None (serve it).

    Reads only ``scores.shape`` -- host-side metadata, so unlike the window-start check
    below this costs no device synchronisation and cannot slow the decode steps it skips.
    """
    if scores.dim() != 2:
        return None  # the contract check owns the rank message
    rows, cols = scores.shape
    if rows < PAGED_SERVE_MIN_ROWS or cols < PAGED_SERVE_MIN_COLS:
        return (
            f"rows={rows} cols={cols} below the served band "
            f"(rows>={PAGED_SERVE_MIN_ROWS} and cols>={PAGED_SERVE_MIN_COLS})"
        )
    return None


def _constant_window_start(row_starts: torch.Tensor | None) -> int | None:
    """Single shared window start for all rows, or None when the rows differ.

    The wheel has no ``begin`` argument (it searches from column 0 of the buffer it
    is handed), so a PAGED window can only be served by slicing the score matrix when
    every row starts at the same column. The spread is folded into one tiny host read:
    the value is needed to build the slice, so it cannot be avoided on this route.

    Known cost, measured and accepted (ruling seq=101 condition 3): the read is a flat
    ~42us per call regardless of shape. Inside the served band above that is 0.1%..25%
    of the call -- 42us against 1.8ms at 7808 x 64000 -- so the band keeps the prefill
    win; calls outside it never reach this function, which is what the shape dispatch
    is for. Removing the read entirely needs the window start as a host-side mirror
    from the metadata producer, which is tracked as a follow-up, not part of this patch.
    """
    if row_starts is None or row_starts.numel() == 0:
        return 0
    base = row_starts[:1]
    # One D2H transfer for both the shared start and whether the rows agree.
    host = torch.cat((base, (row_starts - base).abs().max().reshape(1))).cpu()
    if int(host[1]) != 0:
        return None
    return int(host[0])


def topk_transform_paged(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor | None,
    page_tables: torch.Tensor,
) -> torch.Tensor | None:
    """PAGED DSA top-k on the PPU(sail) deep_select wheel, page_size = 1.

    Mirrors the semantics of the fused dsv4 bf16 prefill kernel it replaces: select
    the top ``topk`` scores of every row inside ``[row_start, row_start + length)``,
    map the chosen columns through that row's page table and pad with -1. DeepSelect
    only performs the selection; the page-table lookup stays a gather of the same
    (already per-query expanded) table. Returns None when the layout cannot be served.

    Index convention: the fused kernel looks the page table up with the position
    *inside the window* (column - row_start), not the absolute column, so the
    selection runs on a zero-copy column slice and its indices go to the gather
    unchanged. With row_start == 0 -- what every measured prefill of this case
    config produced -- the two conventions are the same numbers.

    Whether a shape is worth serving is the caller's decision, made by
    :func:`paged_topk_shape_reason` before it gets here; this function stays
    shape-agnostic so kernel-level tests and benchmarks can drive any layout.
    """
    import deep_select

    rows, cols = scores.shape
    if rows == 0:
        return scores.new_empty((0, topk), dtype=torch.int32)

    start = _constant_window_start(row_starts)
    if start is None:
        return None
    if start < 0 or start >= cols:
        return None
    view = scores if start == 0 else scores[:, start:cols]
    # The wheel documents a 16-byte aligned base address plus the stride rule above,
    # which a column slice keeps by construction.
    if view.data_ptr() % 16 or view.stride(1) != 1:
        return None

    # Lengths are already relative to row_start, which is where the view begins;
    # clamping keeps pathological padding inside the wheel's [0, size(1)] range.
    end = lengths.to(torch.int32).contiguous().clamp(max=view.shape[1])

    _, columns = deep_select.topk(
        view,
        topk,
        sorted=False,
        begin=None,
        end=end,
        indices_type=torch.int32,
        sorted_index=False,
        output_idx=None,
        output_idx_offset=None,
        idx_oob_fill_value=-1,
        # Never let a stray NaN abort the device context inside the server.
        abort_when_nan_found=False,
        return_value=False,
    )
    columns = columns.contiguous()
    if columns.shape[0] != page_tables.shape[0]:
        return None

    page_ids = columns.new_empty((rows, topk), dtype=torch.int32)
    _gather_page_ids[(rows, triton.cdiv(topk, 1024))](
        columns,
        page_tables,
        page_ids,
        topk,
        page_tables.shape[1],
        columns.stride(0),
        page_tables.stride(0),
        page_ids.stride(0),
        1024,
    )
    return page_ids
