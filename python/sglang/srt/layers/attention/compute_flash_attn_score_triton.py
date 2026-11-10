"""
Compute token importance scores from flash attention's softmax_lse output.

This module computes the column-sum of the attention probability matrix
WITHOUT materializing it, using flash attention's returned softmax_lse.

Algorithm:
    importance[k] = Σ_q softmax(qk^T × scale)[q, k]
    Uses p = exp(qk^T × scale - lse) to avoid O(N²) memory.
"""

import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 0 else 1


# =============================================================================
# PyTorch Fallback Implementations
# =============================================================================

def _compute_key_importance_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    batch, heads, seq_len, head_dim = q.shape

    qk = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    p = torch.exp(qk - lse.unsqueeze(-1))
    importance = p.sum(dim=-2)

    return importance


def _compute_key_importance_varlen_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    total_tokens, n_heads, head_dim = q.shape
    batch_size = cu_seqlens.numel() - 1

    importance = torch.zeros(total_tokens, dtype=torch.float32, device=q.device)

    for b in range(batch_size):
        start = cu_seqlens[b].item()
        end = cu_seqlens[b + 1].item()
        seq_len = end - start

        if seq_len == 0:
            continue

        q_seq = q[start:end].transpose(0, 1)
        k_seq = k[start:end].transpose(0, 1)
        lse_seq = lse[:, start:end]

        qk = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * sm_scale
        p = torch.exp(qk - lse_seq.unsqueeze(-1))
        seq_importance = p.sum(dim=-2)

        importance[start:end] = seq_importance.mean(dim=0)

    return importance


# =============================================================================
# Triton Kernels (only compiled if triton is available)
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _compute_key_importance_kernel(
        Q, K, LSE, Out_Score,
        stride_q_b, stride_q_h, stride_q_n, stride_q_d,
        stride_k_b, stride_k_h, stride_k_n, stride_k_d,
        stride_lse_b, stride_lse_h, stride_lse_n,
        stride_out_b, stride_out_h, stride_out_n,
        sm_scale,
        H, Q_LEN, K_LEN,
        ACTUAL_HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        cur_n_idx = tl.program_id(0)
        cur_head_idx = tl.program_id(1)
        cur_batch_idx = tl.program_id(2)

        k_start_ptr = K + (cur_batch_idx * stride_k_b + cur_head_idx * stride_k_h)

        offs_n = cur_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        k_mask = (offs_n[None, :] < K_LEN) & (offs_d[:, None] < ACTUAL_HEAD_DIM)
        k_ptrs = k_start_ptr + (offs_n[None, :] * stride_k_n + offs_d[:, None] * stride_k_d)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        acc_score = tl.zeros([BLOCK_N], dtype=tl.float32)

        q_start_ptr = Q + (cur_batch_idx * stride_q_b + cur_head_idx * stride_q_h)
        lse_start_ptr = LSE + (cur_batch_idx * stride_lse_b + cur_head_idx * stride_lse_h)

        for start_m in range(0, Q_LEN, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + tl.arange(0, BLOCK_M)

            q_mask = (offs_m[:, None] < Q_LEN) & (offs_d[None, :] < ACTUAL_HEAD_DIM)
            q_ptrs = q_start_ptr + (offs_m[:, None] * stride_q_n + offs_d[None, :] * stride_q_d)
            q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

            lse_ptrs = lse_start_ptr + offs_m * stride_lse_n
            lse_mask = offs_m < Q_LEN
            lse_block = tl.load(lse_ptrs, mask=lse_mask, other=0.0)

            qk = tl.dot(q_block, k_block)
            qk *= sm_scale

            p_block = tl.exp(qk - lse_block[:, None])

            q_seq_mask = offs_m[:, None] < Q_LEN
            p_block = tl.where(q_seq_mask, p_block, 0.0)

            acc_score += tl.sum(p_block, axis=0)

        out_start_ptr = Out_Score + (cur_batch_idx * stride_out_b + cur_head_idx * stride_out_h)
        out_ptrs = out_start_ptr + offs_n * stride_out_n

        tl.store(out_ptrs, acc_score, mask=offs_n < K_LEN)

    @triton.jit
    def _compute_key_importance_varlen_kernel(
        Q, K, LSE, Out_Score, CuSeqLens, n_heads,
        stride_q_tok, stride_q_h, stride_q_d,
        stride_k_tok, stride_k_h, stride_k_d,
        stride_lse_h, stride_lse_tok,
        stride_out_tok,
        sm_scale,
        ACTUAL_HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr
    ):
        pid_seq = tl.program_id(2)
        pid_head = tl.program_id(1)
        pid_n = tl.program_id(0)

        start_idx = tl.load(CuSeqLens + pid_seq)
        end_idx = tl.load(CuSeqLens + pid_seq + 1)
        seq_len = end_idx - start_idx

        if pid_n * BLOCK_N >= seq_len:
            return

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        k_mask = (offs_n[None, :] < seq_len) & (offs_d[:, None] < ACTUAL_HEAD_DIM)

        k_ptrs = K + ((start_idx + offs_n[None, :]) * stride_k_tok + pid_head * stride_k_h + offs_d[:, None] * stride_k_d)
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)

        acc_score = tl.zeros([BLOCK_N], dtype=tl.float32)

        lse_base_ptr = LSE + (pid_head * stride_lse_h + start_idx * stride_lse_tok)
        q_base_ptr = Q + (start_idx * stride_q_tok + pid_head * stride_q_h)

        for start_m in range(0, seq_len, BLOCK_M):
            offs_m = start_m + tl.arange(0, BLOCK_M)

            q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < ACTUAL_HEAD_DIM)
            q_ptrs = q_base_ptr + (offs_m[:, None] * stride_q_tok + offs_d[None, :] * stride_q_d)
            q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)

            lse_ptrs = lse_base_ptr + offs_m * stride_lse_tok
            lse_block = tl.load(lse_ptrs, mask=offs_m < seq_len, other=0.0)

            qk = tl.dot(q_block, k_block)
            qk *= sm_scale

            p_block = tl.exp(qk - lse_block[:, None])

            q_seq_mask = offs_m[:, None] < seq_len
            p_block = tl.where(q_seq_mask, p_block, 0.0)

            acc_score += tl.sum(p_block, axis=0)

        avg_score = acc_score / n_heads
        out_ptrs = Out_Score + (start_idx + offs_n)
        tl.atomic_add(out_ptrs, avg_score, mask=offs_n < seq_len)


# =============================================================================
# Unified Public API
# =============================================================================

def compute_flash_attn_score(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute key token importance scores from attention.

    This computes the column-sum of the attention probability matrix
    WITHOUT materializing the O(N²) matrix, using flash attention's
    returned softmax_lse (logsumexp) to reconstruct probabilities.

    Args:
        q: Query tensor.
           - Standard: [Batch, NumHeads, SeqLen, HeadSize]
           - Varlen: [TotalTokens, NumHeads, HeadSize]
        k: Key tensor (same shape as q).
        lse: Logsumexp from flash attention.
           - Standard: [Batch, NumHeads, SeqLen]
           - Varlen: [NumHeads, TotalTokens]
        cu_seqlens: Cumulative sequence lengths [Batch+1].
                    Required for varlen mode, None for standard.
        sm_scale: Softmax scale factor. Default: 1/sqrt(head_size).

    Returns:
        importance: Importance score per token.
           - Standard: [Batch, NumHeads, SeqLen]
           - Varlen: [TotalTokens]
    """
    is_varlen = cu_seqlens is not None

    if is_varlen:
        return _compute_varlen_importance(q, k, cu_seqlens, lse, sm_scale)
    else:
        return _compute_batched_importance(q, k, lse, sm_scale)


def _compute_batched_importance(
    q: torch.Tensor,
    k: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: Optional[float],
) -> torch.Tensor:
    assert q.shape[1] > 0, "n_heads must be positive for attention score computation"
    if q.dim() == 4:
        if q.shape[1] != lse.shape[1]:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
    else:
        raise ValueError(f"Expected 4D tensor for batched mode, got {q.dim()}D")

    batch, heads, seq_len, head_dim = q.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    if not TRITON_AVAILABLE:
        return _compute_key_importance_torch(q, k, lse, sm_scale)

    q = q.contiguous()
    k = k.contiguous()
    lse = lse.contiguous()

    out_score = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(seq_len, BLOCK_N), heads, batch)

    BLOCK_D = _next_power_of_2(head_dim)
    _compute_key_importance_kernel[grid](
        q, k, lse, out_score,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        out_score.stride(0), out_score.stride(1), out_score.stride(2),
        sm_scale,
        heads, seq_len, seq_len,
        ACTUAL_HEAD_DIM=head_dim, BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    return out_score


def _compute_varlen_importance(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: Optional[float],
) -> torch.Tensor:
    assert q.dim() == 3, f"Q should be [TotalTokens, Heads, HeadDim], got {q.dim()}D"

    total_tokens, n_heads, head_dim = q.shape
    assert n_heads > 0, "n_heads must be positive for attention score computation"

    batch_size = cu_seqlens.numel() - 1

    assert lse.dim() == 2, f"LSE should be [Heads, TotalTokens], got {lse.dim()}D"
    assert lse.shape[0] == n_heads, f"LSE heads mismatch: {lse.shape[0]} vs {n_heads}"
    assert lse.shape[1] == total_tokens, f"LSE tokens mismatch: {lse.shape[1]} vs {total_tokens}"

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    if not TRITON_AVAILABLE:
        return _compute_key_importance_varlen_torch(q, k, cu_seqlens, lse, sm_scale)

    q = q.contiguous()
    k = k.contiguous()
    lse = lse.contiguous()
    cu_seqlens = cu_seqlens.contiguous()

    out_score = torch.zeros((total_tokens,), dtype=torch.float32, device=q.device)

    BLOCK_M = 128
    BLOCK_N = 64

    grid = (triton.cdiv(max_seqlen, BLOCK_N), n_heads, batch_size)

    BLOCK_D = _next_power_of_2(head_dim)
    _compute_key_importance_varlen_kernel[grid](
        q, k, lse, out_score, cu_seqlens, n_heads,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        lse.stride(0), lse.stride(1),
        out_score.stride(0),
        sm_scale,
        ACTUAL_HEAD_DIM=head_dim, BLOCK_D=BLOCK_D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    return out_score
