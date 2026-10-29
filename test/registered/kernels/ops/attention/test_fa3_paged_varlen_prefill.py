# Regression UT for the FA3 paged-varlen prefill NaN seen on a TP2/PP0
# attention dump (Qwen3-8B layer 19): GQA, bf16, causal, multi-segment
# `cu_seqlens_q` with a trailing pad row, and a page table whose unused tail
# columns are zero-padded while block 0 holds NaN K/V.
#
# The original reproducer loaded fixed `.pt` dumps. This UT rebuilds the same
# geometry from `torch.randn` only (fixed seed), so it is self-contained in CI
# and reads no dumped tensor.
#
# Geometry reproduced from the dump:
#   cu_seqlens_q  = [0, 165, 485]   -> 2 segments, 1 pad row past 485
#   cache_seqlens = [677, 320]      -> segment 0 has a 512-token prefix
#   page_table    = 2 x 11 columns  -> seq 1 uses 5 pages (320 = 5 * 64) and
#                                      zero-pads the remaining 6 columns
#   block 0                         -> target of the zero-padded columns, and
#                                      the block holding NaN K/V in the dump
#
# FA3's entry point has no mask/attn-mask tensor (see
# `sglang.kernels.ops.attention.flash_attention.flash_attn_with_kvcache`; the
# `score_mod` argument is dropped for ver==3). The only mask the kernel sees is
# the segment/causal mask, so "NaN behind the mask" is expressed as NaN in
# cache slots that no covered query may read:
#   * block 0, reachable only through the zero-padded page-table tail.
#
# Correctness is checked against a hand-written fp32 attention: paged gather of
# K/V, explicit bottom-right aligned causal mask, plain softmax, no kernel.
# Inputs are built and the reference is evaluated on CPU, so a single fixed
# seed pins the exact tensors on every platform.

import sys
import unittest

import pytest
import torch

from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.test.ci.ci_register import register_cuda_ci

# Deliberately not `sglang.test.test_utils.CustomTestCase`: that module imports
# the benchmark stack (`datasets` -> `pyarrow`), which a kernel-only image does
# not need. `CustomTestCase` only adds `setUpClass` cleanup and the CI GPU-idle
# wait, neither of which this module uses.

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_SKIP_REASON = "FA3 requires CUDA on SM80/SM89/SM90"

_DEVICE = "cuda"

PAGE_SIZE = 64  # as in the dump
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2  # 8:1 GQA, as in the dump
HEAD_DIM = 64  # FA-friendly dim; the dump used 256
SOFTMAX_SCALE = 0.0625  # from the dump (1/16), not 1/sqrt(HEAD_DIM)

SEQ_LENS_Q = (165, 320)  # -> cu_seqlens_q [0, 165, 485]
CACHE_SEQLENS = (677, 320)  # -> segment 0 carries a 512-token prefix
PAD_BLOCK = 0  # block the zero-padded page-table tail points at
PAD_ROWS = 1  # q buffer rows past cu_seqlens_q[-1]

MAX_PAGES_PER_SEQ = (max(CACHE_SEQLENS) + PAGE_SIZE - 1) // PAGE_SIZE  # 11
_REAL_BLOCKS = 11 + 5  # pages actually referenced by the two segments
NUM_BLOCKS = 1 + _REAL_BLOCKS  # + PAD_BLOCK, i.e. 17 blocks as in the dump


def _fa3_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major in (8, 9)


skip_condition = not _fa3_available()


def _fa3_kernel():
    """FA3 paged-kvcache entry point for the current platform.

    On CUDA this is the repo dispatcher that
    `sglang.srt.layers.attention.flashattention_backend` imports directly. On
    PPU the vendor FA3 ops live in the PPU hardware backend and the backend
    reaches them through the `ppu_fa3_hooks` REPLACE hook; calling the wrapper
    itself keeps this UT from patching module globals.
    """
    from sglang.srt.utils import is_ppu

    if is_ppu():
        from sglang.srt.hardware_backend.ppu.attention.flash_attention import (
            flash_attn_with_kvcache as ppu_flash_attn_with_kvcache,
        )

        return ppu_flash_attn_with_kvcache
    return flash_attn_with_kvcache


def _make_case(seed: int, pad_block_fill: str = "random"):
    """Build the synthetic case on CPU from a fixed seed.

    The returned tensors stay on CPU so the fp32 reference can run there;
    `_to_device` uploads the copies the kernel needs.

    `pad_block_fill` controls block `PAD_BLOCK`, which only the zero-padded
    page-table tail can reach:
      - "random": finite noise, the baseline;
      - "nan": NaN K/V, the masked-NaN case.
    """
    gen = torch.Generator().manual_seed(seed)

    total_q = sum(SEQ_LENS_Q)
    q_buf = torch.randn(
        total_q + PAD_ROWS,
        NUM_Q_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        generator=gen,
    )

    k_cache = torch.randn(
        NUM_BLOCKS,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        generator=gen,
    )
    v_cache = torch.randn(
        NUM_BLOCKS,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        generator=gen,
    )
    if pad_block_fill == "nan":
        k_cache[PAD_BLOCK].fill_(float("nan"))
        v_cache[PAD_BLOCK].fill_(float("nan"))
    elif pad_block_fill != "random":
        raise ValueError(f"unknown pad_block_fill {pad_block_fill!r}")

    # Blocks 1..16 are a random permutation, so a kernel that assumes
    # page_table == arange, or that ignores the page table entirely, fails.
    blocks = (torch.randperm(_REAL_BLOCKS, generator=gen) + 1).tolist()

    page_table = [[PAD_BLOCK] * MAX_PAGES_PER_SEQ for _ in SEQ_LENS_Q]
    cursor = 0
    for s, n in enumerate(CACHE_SEQLENS):
        n_pages = (n + PAGE_SIZE - 1) // PAGE_SIZE
        page_table[s][:n_pages] = blocks[cursor : cursor + n_pages]
        cursor += n_pages

    cu_seqlens_q = [0]
    for n in SEQ_LENS_Q:
        cu_seqlens_q.append(cu_seqlens_q[-1] + n)

    return {
        # Kept on CPU: the fp32 reference runs here, the kernel gets copies.
        "q": q_buf,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_table": torch.tensor(page_table, dtype=torch.int32),
        "cache_seqlens": torch.tensor(CACHE_SEQLENS, dtype=torch.int32),
        "cu_seqlens_q": torch.tensor(cu_seqlens_q, dtype=torch.int32),
        "total_q": total_q,
        "heads_per_kv": NUM_Q_HEADS // NUM_KV_HEADS,
        "max_seqlen_q": max(SEQ_LENS_Q),
    }


def _to_device(case):
    return {
        key: (value.to(_DEVICE) if torch.is_tensor(value) else value)
        for key, value in case.items()
    }


def _fp32_reference(case):
    """Hand-written fp32 attention on CPU: paged gather + bottom-right causal."""
    q = case["q"].float()
    k_cache, v_cache = case["k_cache"].float(), case["v_cache"].float()
    page_table = case["page_table"]
    cu_seqlens_q = case["cu_seqlens_q"].tolist()
    cache_seqlens = case["cache_seqlens"].tolist()
    heads_per_kv = case["heads_per_kv"]

    outs = []
    for s, n in enumerate(cache_seqlens):
        qs, qe = cu_seqlens_q[s], cu_seqlens_q[s + 1]
        n_pages = (n + PAGE_SIZE - 1) // PAGE_SIZE
        pages = page_table[s, :n_pages].long()
        k = k_cache[pages].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:n]
        v = v_cache[pages].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:n]
        q_s = q[qs:qe]

        seqlen_q = qe - qs
        # Causal mask aligned to the bottom-right corner: query row i is at
        # absolute position (n - seqlen_q) + i.
        pos_q = torch.arange(seqlen_q) + (n - seqlen_q)
        causal_mask = torch.arange(n)[None, :] > pos_q[:, None]

        out = torch.empty(seqlen_q, NUM_Q_HEADS, HEAD_DIM)
        for g in range(NUM_KV_HEADS):
            q_g = q_s[:, g * heads_per_kv : (g + 1) * heads_per_kv]
            scores = torch.einsum("lhd,nd->lhn", q_g, k[:, g]) * SOFTMAX_SCALE
            scores = scores.masked_fill(causal_mask[:, None, :], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[:, g * heads_per_kv : (g + 1) * heads_per_kv] = torch.einsum(
                "lhn,nd->lhd", probs, v[:, g]
            )
        outs.append(out)
    return torch.cat(outs, dim=0)


def _run_kernel(case):
    dev = _to_device(case)
    total_q = case["total_q"]
    out = _fa3_kernel()(
        q=dev["q"][:total_q],
        k_cache=dev["k_cache"],
        v_cache=dev["v_cache"],
        page_table=dev["page_table"],
        cache_seqlens=dev["cache_seqlens"],
        cu_seqlens_q=dev["cu_seqlens_q"],
        max_seqlen_q=case["max_seqlen_q"],
        softmax_scale=SOFTMAX_SCALE,
        causal=True,
    )
    return out.reshape(total_q, -1)


def _assert_matches_reference(out, case, label):
    ref = _fp32_reference(case).reshape(case["total_q"], -1)

    nan_count = int(torch.isnan(out).sum())
    assert nan_count == 0, f"{label}: kernel output has {nan_count} NaN elements"

    diff = (out.float().cpu() - ref).abs()
    max_err, mean_err = diff.max().item(), diff.mean().item()
    print(f"{label}: max diff {max_err:.3e}  mean diff {mean_err:.3e}")
    # bf16 K/V with head_dim 64: rounding of V dominates, ~4e-3 per element.
    assert max_err <= 2e-2, f"{label}: max abs err {max_err:.3e} > 2e-2"
    assert mean_err <= 2e-3, f"{label}: mean abs err {mean_err:.3e} > 2e-3"


@pytest.mark.skipif(skip_condition, reason=_SKIP_REASON)
def test_paged_varlen_prefill_matches_fp32_reference():
    case = _make_case(seed=0)
    _assert_matches_reference(_run_kernel(case), case, "random")


@pytest.mark.skipif(skip_condition, reason=_SKIP_REASON)
def test_paged_varlen_prefill_nan_behind_mask_does_not_leak():
    """NaN in the zero-padded page-table tail must not reach any output row.

    The padded tail columns of segment 1 point at `PAD_BLOCK`, which holds NaN
    K/V here. No covered query may read those slots, so the kernel output must
    stay finite, match the fp32 reference, and be bit-identical to the same
    case built with finite data.
    """
    baseline = _run_kernel(_make_case(seed=0, pad_block_fill="random"))
    poisoned_case = _make_case(seed=0, pad_block_fill="nan")
    out = _run_kernel(poisoned_case)

    _assert_matches_reference(out, poisoned_case, "nan-behind-mask")
    differing = int((out.cpu() != baseline.cpu()).sum())
    assert differing == 0, (
        f"NaN behind the mask changed {differing} output elements; "
        "block PAD_BLOCK is reachable only through the zero-padded page-table tail"
    )


class TestFA3PagedVarlenPrefill(unittest.TestCase):
    """unittest wrapper so `python -m unittest` and direct invocation also work."""

    @unittest.skipIf(skip_condition, _SKIP_REASON)
    def test_paged_varlen_prefill_smoke(self):
        test_paged_varlen_prefill_matches_fp32_reference()

    @unittest.skipIf(skip_condition, _SKIP_REASON)
    def test_nan_behind_mask_smoke(self):
        test_paged_varlen_prefill_nan_behind_mask_does_not_leak()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
