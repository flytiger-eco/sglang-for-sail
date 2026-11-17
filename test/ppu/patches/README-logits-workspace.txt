PPU v4.1 resident prefill logits workspace

This change builds on avoid-fpad-full-copy.patch. It changes Python allocation
only: no SGLang GPU kernel, DeepGEMM GPU kernel, Torch/native library or precision
change. This patch contains no runtime diagnostic/probe hooks.

The actual PPU DeepGEMM package currently allocates logits internally. Before
starting new server processes, apply the embedded Python-only optional out API
inside the authorized work container:

python3 test/ppu/patches/apply_deep_gemm_logits_out.py --backup-dir /path/to/run/deep-gemm-backup

The helper supports exactly the inspected source hash and is idempotent. It fails
on unknown contents rather than modifying a different DeepGEMM version. The
original source is backed up. No native recompilation is needed. The embedded
deep_gemm_mqa_logits_out.patch is also reviewable/applicable in that package's
root directory. Without enabling the SGLang option, the original allocating API
continues to work.

The package's TOP-LEVEL fp8_fp4_mqa_logits is a pybind dispatcher and does not
expose out. SGLang calls deep_gemm.jit_kernels.attention.fp8_fp4_mqa_logits
explicitly only for the workspace path; disabled/startup paths retain the original
top-level API. The device regression compares the public original entry with
this Python attention entry using the same tensors and existing kernel source.

Enable the new SGLang path with:

export SGLANG_PPU_INDEXER_PREALLOCATE_LOGITS=1

Capacity follows SGLANG_SPARSE_INDEXER_MAX_LOGITS_MB (default4096MiB). Registration
is per worker/device across candidate/full-topk layers and target/draft models.
The scheduler allocates the buffer once after engine loading/warmups/graph
capture and before serving requests. Startup warmups retain the original path.
The allocation is outside capture and adds its full capacity to ready memory
usage, using the original non-static headroom; it does not shrink/recalculate the
already created KV pool. Budget this capacity explicitly for other deployments.

Every steady-state producer gets a flat-prefix view with original alignedrows4
and FP32 alignedcolumns256. DeepGEMM writes directly into that view. The complete
generator producer/consumer interval owns a lease; no previous tile may still be
consumed when it is overwritten. One worker/device allocation is shared rather
than replicated per model layer or request. Host leases serialize reentry and
stream changes wait on the previous stream. Nested same-thread use and capacity
overflow fail explicitly; there is no extra large allocation fallback after
serving starts. This prefill workspace does not support use as graph-capture
scratch. The current test configuration uses eager prefill and decode graphs.

Optionally enable SGLANG_PPU_INDEXER_PREALLOCATE_MASK=1 together with logits
preallocation. One bool arena (logits capacity / 4 bytes, default 1024 MiB) is
allocated before serving; torch.ge(..., out=mask_view) writes the causal mask
directly. Logits and mask coexist, so default total resident storage is 5 GiB
per worker/device. The logits lease protects both buffers through consumption.
The integer causal comparison and masked_fill_ are unchanged. Mask storage is
reused across requests/tiles without a full-size allocating comparison.
Candidate block-score reductions, KV gather and other temporaries are unchanged. Published candidate
IDs and other results retain their original storage/lifetimes. Preallocating
logits removes its varying-size runtime torch.empty calls; it is not a guarantee
against every model allocation OOM. Validate full original inference and record
any remaining allocation phases. CPU/PPU tests cover pointer reuse, actual stride,
lease lifetime, overflow, numeric equivalence, and direct writes without a
second logits allocation.
