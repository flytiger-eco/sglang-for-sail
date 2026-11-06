"""Regression test for the DeepSeek-V4.1 top-k v3 exact-fallback race.

`radix_topk_exact` narrows the threshold with four histogram passes.  Each pass
ends by subtracting `s_histogram[threshold + 1]` from a per-thread `remain`, and
the next pass resets that histogram.  Resetting without a barrier lets a fast
warp zero bins a slower warp has not read yet, so `remain` diverges between
threads: the crossing-bin search then stores a threshold that does not match the
histogram, the final scatter writes more than `topk` entries, and at serving
shapes that becomes an out-of-bounds shared-memory write (`invalid sreg
alignment` / illegal memory access inside `topk_512_transform_v3`).

The window is narrow: on an idle device a single process can run hundreds of
thousands of exact invocations without failing, while real serving load (or a
few concurrent processes, as here) reproduces it within a minute.  The test
therefore runs several worker processes concurrently and forces the exact path
with rows whose scores sit in one 8-bit radix bin (60% in [2.0, 2.5)) while the
values inside it stay distinct, so later passes subtract non-zero counts.

Each worker checks the selected indices against `torch.topk`: in bounds, unique,
and the same score multiset.  The fixed kernel must report zero bad rows; the
unfixed one reports tens to hundreds per 1e5 exact invocations.

`SGLANG_TEST_TOPK_RACE_{WORKERS,ITERS,SEQ,BATCH}` scale the worker count, the
per-worker iteration count, and the row length / batch size.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE_SIZE = 256
TOPK = 512


def _build_rows(batch: int, seq: int, device: torch.device) -> torch.Tensor:
    rows = torch.empty(batch, seq, device=device, dtype=torch.float32)
    for b in range(batch):
        n_hot = min(int(seq * 0.6) + (b % 5) * 97, seq)
        v = torch.rand(seq, device=device) * 2.0
        v[:n_hot] = 2.0 + 0.5 * torch.rand(n_hot, device=device)
        rows[b] = v[torch.randperm(seq, device=device)]
    return rows


def _bad_rows(out: torch.Tensor, scores: torch.Tensor, ref: torch.Tensor) -> int:
    idx = out.to(torch.int64)
    valid = (idx >= 0) & (idx < scores.shape[1])
    ok = valid.all(dim=1)
    uniq = idx.sort(dim=1).values
    ok &= ~(uniq[:, 1:] == uniq[:, :-1]).any(dim=1)
    gathered = scores.gather(1, idx.clamp(0, scores.shape[1] - 1))
    gathered = torch.where(valid, gathered, torch.full_like(gathered, -float("inf")))
    got = torch.sort(gathered, dim=-1, descending=True).values[:, :TOPK]
    ok &= (got == ref).all(dim=1)
    return int((~ok).sum())


def _worker() -> int:
    batch = envs.SGLANG_TEST_TOPK_RACE_BATCH.get()
    seq = envs.SGLANG_TEST_TOPK_RACE_SEQ.get()
    iters = envs.SGLANG_TEST_TOPK_RACE_ITERS.get()
    torch.manual_seed(7)
    dev = torch.device("cuda")
    scores = _build_rows(batch, seq, dev)
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=dev)
    npages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = (
        torch.arange(npages, dtype=torch.int32, device=dev)
        .unsqueeze(0)
        .repeat(batch, 1)
        .contiguous()
    )
    out = torch.full((batch, TOPK), -1, dtype=torch.int32, device=dev)
    ref = scores.topk(TOPK, dim=-1, sorted=True).values
    bad = 0
    for _ in range(iters):
        out.fill_(-1)
        topk_transform_paged(scores, seq_lens, page_table, out, PAGE_SIZE, None)
        torch.cuda.synchronize()
        bad += _bad_rows(out, scores, ref)
    print(f"worker bad_rows={bad}", flush=True)
    return 0 if bad == 0 else 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/PPU device required")
def test_topk_v3_exact_fallback_under_concurrency():
    workers = envs.SGLANG_TEST_TOPK_RACE_WORKERS.get()
    procs = [
        subprocess.Popen(
            [sys.executable, __file__, "--worker"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for _ in range(workers)
    ]
    outs = [p.communicate()[0] for p in procs]
    codes = [p.returncode for p in procs]
    assert codes == [0] * workers, (
        f"worker exit codes {codes}; selection mismatch in the exact fallback:\n"
        + "\n".join(outs)
    )


if __name__ == "__main__":
    if "--worker" in sys.argv:
        sys.exit(_worker())
    sys.exit(pytest.main([__file__]))
