from __future__ import annotations

from typing import Dict

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

    lengths = lengths.contiguous()
    out_offsets = out_offsets.contiguous()
    alignment = deep_select.get_stride_requirement()[0]
    needs_pack = (
        (row_starts is not None and cols > topk)
        or scores.stride(1) != 1
        or scores.stride(0) * scores.element_size() % alignment != 0
        or scores.data_ptr() % alignment != 0
    )
    if needs_pack:
        packed_cols = cols if max_seq_len is None else min(cols, max_seq_len)
        aligned_cols = triton.cdiv(packed_cols, alignment // scores.element_size()) * (
            alignment // scores.element_size()
        )
        packed = scores.new_empty((rows, aligned_cols))
        if packed_cols > topk:
            starts = row_starts.contiguous() if row_starts is not None else None
            block = 4096 if max_seq_len is not None and packed_cols >= 4096 else 1024
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


# ---------------------------------------------------------------------------
# kpool (index_kpool > 1) group-level top-k
# ---------------------------------------------------------------------------

# The pool-level buckets ``fast_kpool_topk_transform_fused`` can serve; the same
# list gates this route so a bucket the fused kernel handles differently never
# reaches here.
KPOOL_SUPPORTED_GROUP_TOPK = (128, 160, 192, 224, 256, 512)

# Initial served band for the pool-level route. Pool-level selection searches a
# row that is ``pool_size`` times narrower than the token-level one with k divided
# by the same factor, so the crossover sits elsewhere than the PAGED band above;
# these floors come from the phase-1 kernel measurements on the PPU device (whole-chain
# cast+select+expand+tail vs the fused kpool kernel: 2048x16384 0.94 -> 0.44 ms,
# 8192x16384 3.67 -> 1.43 ms, 32768x16384 14.63 -> 5.26 ms, while 128 rows and
# 16 rows lose at 0.68x and 0.18x). The first phase-3 smoke run logs the real
# (rows, cols) distribution and these values get tightened to the measured
# crossover before anything is claimed.
KPOOL_SERVE_MIN_ROWS = 2048
KPOOL_SERVE_MIN_COLS = 8192

# BF16 staging buffers, one per device, grown but never shrunk (see
# ``_kpool_pack_buffer`` for why caching is mandatory here).
_kpool_pack_buffers: Dict[int, torch.Tensor] = {}


def kpool_group_topk_shape_reason(rows: int, cols: int, group_topk: int) -> str | None:
    """Return why a kpool group-level call stays on the fused kernel, else None.

    Reads only host-side integers, so it costs no device synchronisation on the
    decode steps it declines.
    """
    if group_topk not in KPOOL_SUPPORTED_GROUP_TOPK:
        return (
            f"pool-level topk {group_topk} is not a served kpool bucket "
            f"{KPOOL_SUPPORTED_GROUP_TOPK}"
        )
    if not 0 < group_topk <= MAX_TOPK:
        return f"group_topk {group_topk} outside the deep_select contract (1..{MAX_TOPK})"
    if rows < KPOOL_SERVE_MIN_ROWS or cols < KPOOL_SERVE_MIN_COLS:
        return (
            f"rows={rows} cols={cols} below the kpool served band "
            f"(rows>={KPOOL_SERVE_MIN_ROWS} and cols>={KPOOL_SERVE_MIN_COLS})"
        )
    return None


@triton.jit
def _pack_kpool_group_scores(
    scores,
    lengths,
    row_starts,
    packed,
    SCORE_ROW_STRIDE: tl.constexpr,
    SCORE_COL_STRIDE: tl.constexpr,
    PACKED_ROW_STRIDE: tl.constexpr,
    HAS_STARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy each row's pooled score window into the BF16 staging buffer.

    The kpool producer hands over float32 logits while the wheel accepts BF16 only,
    and the wheel has no ``begin`` argument. One launch therefore does both jobs the
    replacement needs: convert the dtype, and move ``[start, start + length)`` to
    column 0. The window start is added back through ``output_idx_offset``, so the
    indices this route returns are absolute pooled columns -- exactly what the
    existing group-expansion helpers expect.

    Rows at or below ``topk`` still get their window copied; the incumbent fused
    kernel reads them too, and skipping the copy would make the two routes differ in
    which buffer bytes they touch.
    """
    row = tl.program_id(0)
    length = tl.load(lengths + row)
    start = tl.load(row_starts + row) if HAS_STARTS else 0
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(
        scores + row * SCORE_ROW_STRIDE + (start + cols) * SCORE_COL_STRIDE,
        mask=cols < length,
        other=0.0,
    )
    tl.store(
        packed + row * PACKED_ROW_STRIDE + cols, values.to(tl.bfloat16), mask=cols < length
    )


def _kpool_pack_buffer(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """Cached BF16 staging buffer shaped at least ``[rows, cols]``, unit column stride.

    Pool-level top-k runs once per DSA layer per step, so allocating per call would
    churn the allocator on the hot path, and a fresh address per CUDA graph replay is
    not valid. The block is reused and only grown; stride stays a whole multiple of
    the wheel's alignment requirement so the cached pointer stays usable.
    """
    key = device.index if device.index is not None else torch.cuda.current_device()
    alignment = _alignment_bytes() // torch.bfloat16.itemsize
    aligned_cols = triton.cdiv(cols, alignment) * alignment
    cached = _kpool_pack_buffers.get(key)
    if cached is not None and cached.shape[0] >= rows and cached.stride(0) >= aligned_cols:
        return cached[:rows, :cols]
    rows_alloc = max(rows, cached.shape[0] if cached is not None else 0, 1)
    cols_alloc = max(aligned_cols, cached.stride(0) if cached is not None else 0)
    allocated = torch.empty((rows_alloc, cols_alloc), dtype=torch.bfloat16, device=device)
    if allocated.data_ptr() % _alignment_bytes():
        # The wheel documents an aligned base address; a buffer that cannot honour
        # it must never be cached, or every later call would fail on it.
        raise RuntimeError(
            f"the kpool staging buffer is not {_alignment_bytes()}B aligned "
            f"(addr={allocated.data_ptr():#x})"
        )
    _kpool_pack_buffers[key] = allocated
    return allocated[:rows, :cols]


def topk_kpool_groups(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    group_topk: int,
    row_starts: torch.Tensor | None = None,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Pool-level top-k of every pooled-score row on the PPU DeepSelect kernel.

    Returns int32 ``[rows, group_topk]`` absolute pooled-column ids, valid entries
    first and ``-1`` past each row's length -- the same convention the fused kpool
    kernel's group selection produces. Only the selection happens here; expanding
    each chosen group into its ``pool_size`` tokens, appending the forced tail and
    the page-table or buffer-offset transform stay in the caller's existing
    ``expand_pooled_groups_to_topk`` / ``append_kpool_tail_to_topk`` helpers.

    Shape gating is the caller's job (see :func:`kpool_group_topk_shape_reason`), so
    kernel-level tests and benchmarks can drive any layout.
    """
    import deep_select

    rows, cols = scores.shape
    if rows == 0:
        return scores.new_empty((0, group_topk), dtype=torch.int32)

    lengths = lengths.to(torch.int32).contiguous()
    starts = None if row_starts is None else row_starts.to(torch.int32).contiguous()
    packed_cols = cols if max_seq_len is None else min(cols, max_seq_len)
    packed = _kpool_pack_buffer(rows, packed_cols, scores.device)
    block = 4096 if packed_cols >= 4096 else 1024
    _pack_kpool_group_scores[(rows, triton.cdiv(packed_cols, block))](
        scores,
        lengths,
        starts,
        packed,
        scores.stride(0),
        scores.stride(1),
        packed.stride(0),
        starts is not None,
        block,
    )
    # A packed window starts at column 0, so the wheel's per-row offset has to carry
    # the original window start back in to keep the columns absolute.
    offsets = torch.zeros(rows, dtype=torch.int32, device=scores.device)
    if starts is not None:
        offsets = starts
    _, groups = deep_select.topk(
        packed,
        group_topk,
        sorted=False,
        begin=None,
        end=lengths.clamp(max=packed_cols),
        indices_type=torch.int32,
        sorted_index=False,
        output_idx=None,
        output_idx_offset=offsets,
        idx_oob_fill_value=-1,
        # Never let a stray NaN abort the device context inside the server.
        abort_when_nan_found=False,
        return_value=False,
    )
    return groups.contiguous()
