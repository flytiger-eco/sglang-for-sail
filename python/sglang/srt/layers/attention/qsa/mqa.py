"""Weight-free QSA MQA scoring: TileLang, optional SAIL DeepGEMM FP8,
and torch fallback/reference implementations."""

import functools
import math
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_hip, print_info_once, print_warning_once

try:
    import flashinfer.comm  # noqa: F401
except ImportError:
    pass

try:
    import tilelang
    from tilelang import language as T

    HAS_TILELANG = True
except ImportError:
    tilelang = None
    T = None
    HAS_TILELANG = False


def _validate_q(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(f"QSA requires q [tokens, heads, head_dim], got {q.shape}")


def _validate_k(k: torch.Tensor) -> None:
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(f"QSA MQA requires k [tokens, 1, head_dim], got {k.shape}")


def torch_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for packed, variable-length prefill MQA."""

    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    logits = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= row_starts.to(q.device).reshape(-1, 1)) & (
        columns < row_ends.to(q.device).reshape(-1, 1)
    )
    return logits.masked_fill(~valid, -float("inf"))


def _validate_decode_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    _validate_q(q)
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "QSA decode cache must be [pages, page_size, 1, head_dim], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[-1] != q.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA decode page table must have one row per query")
    if context_lens.numel() != q.shape[0]:
        raise ValueError("QSA decode context lengths must have one entry per query")


def torch_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Torch reference for variable-length paged decode MQA."""

    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    gathered = k_cache[page_table.long().clamp_min(0).reshape(-1), :, 0].reshape(
        batch, total, q.shape[-1]
    )
    scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    scores = torch.relu(scores).sum(dim=-1) / (score_scale or math.sqrt(q.shape[-1]))
    positions = torch.arange(total, device=q.device).unsqueeze(0)
    scores.masked_fill_(
        positions >= context_lens.to(q.device).reshape(-1, 1), -float("inf")
    )
    logits = torch.full(
        (batch, max_model_len), -float("inf"), dtype=torch.float32, device=q.device
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        logits[:, :copy_len] = scores[:, :copy_len]
    return logits


_FP8_E4M3_FMAX = 448.0
# PPU avg-MQA supports logical head dimensions 64/128, not 32.
_FP8_AVG_HEAD_DIMS = (64, 128)


@functools.lru_cache(maxsize=1)
def _resolve_fp8_avg_mqa_logits():
    """SAIL DeepGEMM fp8_mqa_avg_logits kernel; None when unavailable."""
    from sglang.srt.utils import get_device_sm, is_ppu

    # Only enable the PPU0015 FP8 MMA implementation validated on 890P.
    # Other PPU generations and upstream CUDA DeepGEMM are separate targets.
    if not is_ppu() or get_device_sm() != 89:
        print_warning_once(
            "SGLANG_SAIL_QSA_INDEXER_FP8_AVG_LOGITS requires PPU SM89 (890P); "
            "falling back to the TileLang/torch QSA indexer scoring."
        )
        return None
    try:
        from deep_gemm import fp8_mqa_avg_logits
    except ImportError:
        print_warning_once(
            "SGLANG_SAIL_QSA_INDEXER_FP8_AVG_LOGITS is set but the installed "
            "deep_gemm has no fp8_mqa_avg_logits; falling back to the "
            "TileLang/torch QSA indexer scoring."
        )
        return None
    signature = fp8_mqa_avg_logits.__doc__ or ""
    if "max_seqlen_k" not in signature or "clean_logits" in signature:
        print_warning_once(
            "QSA requires the compressed-output DeepGEMM AVG API with mandatory "
            "max_seqlen_k and no clean_logits argument; falling back to "
            "TileLang/torch for an unrecognized or legacy API."
        )
        return None
    return fp8_mqa_avg_logits


def can_use_deep_gemm_fp8_avg_mqa(q: torch.Tensor) -> bool:
    """Whether QSA may use the PPU compressed-output scorer for these queries."""
    return (
        q.is_cuda
        and envs.SGLANG_SAIL_QSA_INDEXER_FP8_AVG_LOGITS.get()
        and q.shape[1] == 4
        and q.shape[-1] in _FP8_AVG_HEAD_DIMS
        and _resolve_fp8_avg_mqa_logits() is not None
    )


def _fp8_e4m3_per_row_quant(x: torch.Tensor):
    """[rows, dim] -> (fp8_e4m3 [rows, dim], fp32 dequant scale [rows]).

    On CUDA this reuses the in-tree JIT per-token quant kernel (single
    launch, warp per row); the torch fallback keeps the CPU reference path.
    """

    if x.is_cuda:
        from sglang.kernels.ops.quantization.per_token_quant_fp8 import (
            per_token_quant_fp8,
        )

        x = x.contiguous()
        x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty(x.shape[0], 1, dtype=torch.float32, device=x.device)
        per_token_quant_fp8(x, x_fp8, scale)
        return x_fp8, scale.squeeze(1)
    xf = x.float()
    scale = xf.abs().amax(dim=-1).clamp_min(1e-12) / _FP8_E4M3_FMAX
    return (xf / scale.unsqueeze(-1)).to(torch.float8_e4m3fn), scale


def deep_gemm_fp8_avg_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
    *,
    max_seqlen_k: int,
) -> torch.Tensor:
    """SAIL DeepGEMM avg-MQA logits for the QSA indexer prefill scoring.

    Computes sum_h(relu(q·k)) * q_scale * k_scale / sqrt(head_dim): q is
    quantized per query row and the compressed keys per token, with the
    dequant scales passed through the kernel's q_scale / k_scales arguments,
    so the result matches the BF16 weight-free formula up to FP8 rounding.
    The kernel hardcodes the 1/sqrt(head_dim) score scale.

    Returns FP32 [rows, max_seqlen_k] with possibly padded row stride.
    Each row's valid prefix has length row_end - row_start; column j refers
    to K[row_start + j]. All other entries are unspecified. The positive,
    host-known capacity must cover every valid window (in compressed K units).
    """

    fp8_mqa_avg_logits = _resolve_fp8_avg_mqa_logits()
    if fp8_mqa_avg_logits is None:
        raise RuntimeError("fp8_mqa_avg_logits is unavailable")
    if not isinstance(max_seqlen_k, int) or max_seqlen_k <= 0:
        raise ValueError("compressed AVG requires a positive integer max_seqlen_k")
    _validate_q(q)
    _validate_k(k)
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    head_dim = q.shape[-1]
    if q.shape[1] != 4 or head_dim not in _FP8_AVG_HEAD_DIMS:
        raise ValueError(
            "the validated fp8_mqa_avg_logits path requires 4 query heads "
            "and head_dim 64/128"
        )
    if score_scale is not None and not math.isclose(score_scale, math.sqrt(head_dim)):
        raise ValueError(
            "fp8_mqa_avg_logits hardcodes score_scale = sqrt(head_dim), "
            f"got {score_scale} for head_dim {head_dim}"
        )
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        return torch.empty((rows, max_seqlen_k), dtype=torch.float32, device=q.device)
    q_fp8, q_scale = _fp8_e4m3_per_row_quant(q.reshape(rows, -1))
    q_fp8 = q_fp8.view(rows, q.shape[1], head_dim)
    k_fp8, k_scales = _fp8_e4m3_per_row_quant(k[:, 0])
    print_info_once(
        "Using SAIL DeepGEMM fp8_mqa_avg_logits for QSA prefill scoring "
        "(FP8 Q/K, FP32 compressed logits, bounded strided top-k)."
    )
    return fp8_mqa_avg_logits(
        q_fp8.contiguous(),
        (k_fp8.contiguous(), k_scales.contiguous()),
        row_starts.to(device=q.device, dtype=torch.int32).contiguous(),
        row_ends.to(device=q.device, dtype=torch.int32).contiguous(),
        max_seqlen_k=max_seqlen_k,
        q_scale=q_scale.contiguous(),
        logits_dtype=torch.float32,
    )


if HAS_TILELANG:

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], T.bfloat16),  # type: ignore
            K: T.Tensor([keys, head_dim], T.bfloat16),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as bx:
                q_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))

                T.copy(Q[row_base * heads, 0], q_shared)
                for ni in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n), num_stages=num_stages
                ):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        Logits[row_base + qi, start_min + ni * block_n + n] = reduced[
                            n, qi
                        ]

        return kernel

    @tilelang.jit
    def _tilelang_qsa_mqa_mask_kernel(threads: int = 512, block_k: int = 4096):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(rows, threads=threads) as bx:
                tx = T.thread_binding(0, threads, thread="threadIdx.x")
                for block in T.Pipelined(T.ceildiv(keys, block_k)):
                    for item in T.serial(block_k // threads):
                        column = block * block_k + item * threads + tx
                        if column < Starts[bx] or column >= Ends[bx]:
                            Logits[bx, column] = -T.infinity(T.float32)

        return kernel

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _tilelang_qsa_mqa_decode_kernel(
        heads: int,
        head_dim: int,
        page_size: int = 64,
        groups_per_cta: int = 1,
        num_stages: int = 3,
        threads: int = 128,
    ):
        # The MMA layout needs 64 GEMM rows; a compressed page has full_page // ratio,
        # so pages are packed as sub-pages of one 64-row tile.
        GROUP = 64
        assert GROUP % page_size == 0, page_size
        sub_pages = GROUP // page_size
        batch = T.dynamic("batch")
        pages = T.dynamic("pages")
        max_pages = T.dynamic("max_pages")
        max_model_len = T.dynamic("max_model_len")

        @T.prim_func
        def kernel(
            Q: T.Tensor([batch, 1, heads, head_dim], T.bfloat16),  # type: ignore
            KCache: T.Tensor([pages, page_size, 1, head_dim], T.bfloat16),  # type: ignore
            PageTable: T.Tensor([batch, max_pages], T.int32),  # type: ignore
            ContextLens: T.Tensor([batch], T.int32),  # type: ignore
            Logits: T.Tensor([batch, max_model_len], T.float32),  # type: ignore
            Scale: T.float32,
        ):
            with T.Kernel(
                batch,
                T.ceildiv(T.ceildiv(max_pages, sub_pages), groups_per_cta),
                threads=threads,
            ) as (bx, group_block):
                q_shared = T.alloc_shared([heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([GROUP, head_dim], T.bfloat16)
                scores = T.alloc_fragment([GROUP, heads], T.float32)
                reduced = T.alloc_fragment([GROUP], T.float32)
                T.copy(Q[bx, 0, :, :], q_shared)
                context_len = ContextLens[bx]

                for gi in T.Pipelined(groups_per_cta, num_stages=num_stages):
                    group = group_block * groups_per_cta + gi
                    if group * GROUP < context_len:
                        # TileLang's pipeliner rejects dynamic loops around smem copies;
                        # unroll at the Python level instead.
                        for sp in range(sub_pages):
                            if (group * sub_pages + sp) * page_size < context_len:
                                T.copy(
                                    KCache[
                                        PageTable[bx, group * sub_pages + sp],
                                        :,
                                        0,
                                        :,
                                    ],
                                    k_shared[sp * page_size : (sp + 1) * page_size, :],
                                )
                        T.gemm(
                            k_shared,
                            q_shared,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for token, head in T.Parallel(GROUP, heads):
                            scores[token, head] = T.max(scores[token, head], 0.0)
                        T.reduce_sum(scores, reduced, dim=1, clear=True)
                        for token in T.Parallel(GROUP):
                            position = group * GROUP + token
                            if position < context_len:
                                Logits[bx, position] = reduced[token] / Scale

        return kernel


def tilelang_qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang packed prefill kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_q(q)
    _validate_k(k)
    rows, keys = q.shape[0], k.shape[0]
    if not rows or not keys:
        logits = torch.zeros((rows, keys), dtype=torch.float32, device=q.device)
        return logits.masked_fill_(
            torch.ones_like(logits, dtype=torch.bool), -float("inf")
        )
    heads, head_dim = q.shape[1:]
    block_q = max(1, 128 // heads)
    padding = (-rows) % block_q
    padded_rows = rows + padding
    # A torch.cat of the padding rows would copy the whole [rows, keys] fp32 matrix,
    # doubling the dominant prefill buffer; allocate pre-padded instead.
    logits = torch.zeros((padded_rows, keys), dtype=torch.float32, device=q.device)
    q_padded = q.to(torch.bfloat16).contiguous()
    starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    if padding:
        q_padded = torch.cat([q_padded, q_padded.new_zeros(padding, heads, head_dim)])
        starts = torch.cat([starts, starts[-1:].expand(padding)])
        ends = torch.cat([ends, ends[-1:].expand(padding)])

    _tilelang_qsa_mqa_prefill_kernel(heads=heads, head_dim=head_dim, block_q=block_q)(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(torch.bfloat16).contiguous(),
        logits,
        starts,
        ends,
    )
    # A leading-dimension slice that retains every column is already
    # contiguous, so do not copy this large matrix again when removing padding.
    logits = logits[:rows]
    logits.div_(score_scale or math.sqrt(head_dim))
    _tilelang_qsa_mqa_mask_kernel()(logits, starts[:rows], ends[:rows])
    return logits


def tilelang_qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Validated TileLang paged decode kernel with weights removed."""

    if not HAS_TILELANG:
        raise RuntimeError("TileLang is unavailable")
    _validate_decode_inputs(q, k_cache, page_table, context_lens)
    page_size = int(k_cache.shape[1])
    if page_size < 8 or 64 % page_size != 0:
        raise ValueError(
            "TileLang QSA decode requires a compressed page size of "
            f"8/16/32/64 (64-row GEMM sub-page packing), got {page_size}"
        )
    logits = torch.full(
        (q.shape[0], max_model_len),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    if not q.shape[0] or not max_model_len:
        return logits
    # CUDA MMA accepts an eight-wide N dimension, while ROCm MFMA requires
    # sixteen. Zero-padding preserves the weight-free head sum on both paths.
    query_heads, head_dim = q.shape[1:]
    head_alignment = 16 if is_hip() else 8
    kernel_heads = max(
        head_alignment,
        ((query_heads + head_alignment - 1) // head_alignment) * head_alignment,
    )
    q_kernel = q.to(torch.bfloat16)
    if kernel_heads != query_heads:
        q_kernel = torch.cat(
            [
                q_kernel,
                q_kernel.new_zeros(q.shape[0], kernel_heads - query_heads, head_dim),
            ],
            dim=1,
        )
    _tilelang_qsa_mqa_decode_kernel(
        heads=kernel_heads, head_dim=head_dim, page_size=page_size
    )(
        q_kernel.unsqueeze(1).contiguous(),
        k_cache.to(torch.bfloat16).contiguous(),
        page_table.to(device=q.device, dtype=torch.int32).contiguous(),
        context_lens.to(device=q.device, dtype=torch.int32).contiguous(),
        logits,
        float(score_scale or math.sqrt(head_dim)),
    )
    return logits


def qsa_mqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    """Global-column, masked baseline scorer (TileLang or torch reference)."""
    if q.is_cuda and HAS_TILELANG:
        print_info_once("Using TileLang for QSA prefill scoring.")
        return tilelang_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)
    if q.is_cuda:
        print_warning_once("Using torch fallback for QSA prefill scoring.")
    return torch_qsa_mqa_prefill(q, k, row_starts, row_ends, score_scale)


def qsa_mqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: Optional[float] = None,
) -> torch.Tensor:
    if q.is_cuda and HAS_TILELANG:
        return tilelang_qsa_mqa_decode(
            q, k_cache, page_table, context_lens, max_model_len, score_scale
        )
    return torch_qsa_mqa_decode(
        q, k_cache, page_table, context_lens, max_model_len, score_scale
    )


__all__ = [
    "HAS_TILELANG",
    "deep_gemm_fp8_avg_mqa_prefill",
    "qsa_mqa_decode",
    "qsa_mqa_prefill",
    "tilelang_qsa_mqa_decode",
    "tilelang_qsa_mqa_prefill",
    "torch_qsa_mqa_decode",
    "torch_qsa_mqa_prefill",
]
