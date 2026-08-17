# Bug Report: 3 regressions from PPU SDK v2.1.0→v2.1.1 / sgl-kernel 0.4.2.post2→0.4.3

## Summary

Upgrading from (SDK v2.1.0 + sgl-kernel 0.4.2.post2) to (SDK v2.1.1 + sgl-kernel 0.4.3) introduced three independent regressions in the SGLang PPU CI suite. All three tests pass on the older stack.

| # | Symptom | Component | Severity |
|---|---------|-----------|----------|
| 1 | `ppu-llc` SIGABRT during Triton kernel compile | ppu-llc (SDK) | compile-time crash |
| 2 | Marlin MoE emits **NaN** on 34% of output elements | sgl-kernel / marlin | **silent wrong results** |
| 3 | Triton attention D=80 precision exceeds tolerance | sgl-kernel / triton | numerical |

Issue #2 is the most serious: it produces NaN rather than failing loudly, so a production workload using Marlin-quantized MoE on PPU would emit garbage.

## Environment

| Component | Old (passing) | New (failing) |
|-----------|---------------|---------------|
| PPU SDK | v2.1.0 | **2.1.1-a5c56e** |
| sgl-kernel | 0.4.2.post2 | **0.4.3** |
| PyTorch | 2.10.0 | 2.11.0 |
| Triton | — | 3.6.0 |
| Python | 3.12 | 3.12.3 |
| ppu-llc | — | LLVM 13.0.1, ppu 2.1.1-a5c56e |
| Driver / VBIOS | — | 1.6.1-8deb0c / 1.6.12-af69f0 |
| Hardware | PPU 810E | PPU 810E |
| Image (new) | — | `pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.11.0-ubuntu24.04-cuda13.0-sglang0.5.13-py312` |

---

## Issue 1: ppu-llc SIGABRT on causal_conv1d batch_gather kernel (float32)

### Reproducer
```bash
cd test
python3 -m pytest "registered/layers/mamba/test_causal_conv1d.py::test_causal_conv1d_update_with_batch_gather[3-True-2064-3-1-False-False-itype0]" -v
```

### Error
```
RuntimeError: ppu-llc error: `/usr/local/PPU_SDK/bin/ppu-llc -lineinfo \
  --ppu-backend-options --ppu-blksync-schedule-boundary=false \
  --ppu-backend-options --ppu-enable-rewrite-partial-reg-uses=true \
  --ppu-backend-options --ppu-max-vreg-count=256 \
  --ppu-backend-options --max-analysis-recursion-depth=7 \
  --ppu-backend-options --enable-threadIdx-x-div32-always-uniform=true \
  -v --gpu-name=sm_80 /tmp/tmp9m62g8it.tix.trans -o /tmp/tmp9m62g8it.tix.trans.o` \
  failed with return code 134
```
Return code 134 = SIGABRT (internal assertion failure in ppu-llc).

### Scope
- **Only `itype0` (float32)** fails; `itype1` (fp16) and `itype2` (bf16) pass.
- **Only the `batch_gather` variant** fails; all 12 plain `test_causal_conv1d_update` cases pass.
- Suggests the crash is in a codegen path specific to wider registers + the gather addressing pattern.

### Debug hint
Set `TRITON_CACHE_DIR=/tmp/triton_debug` to preserve the `.tix.trans` intermediate IR for offline reproduction against ppu-llc directly.

---

## Issue 2: Marlin MoE produces NaN (most serious)

### Reproducer
```bash
cd test
python3 -m pytest registered/quant/test_marlin_moe.py::TestFusedMarlinMoe::test_fused_marlin_moe --tb=short
```

### Error
```
AssertionError: Tensor-likes are not close!
Mismatched elements: 86008 / 251904 (34.1%)
Greatest absolute difference: nan at index (12, 0) (up to 0.15 allowed)
Greatest relative difference: nan at index (12, 0) (up to 0 allowed)
```

### Context — two layered problems
Running the generated config matrix surfaces two distinct failures. `assert_close` reports the first, so the second was initially masked:

1. **Small config**: max abs diff `0.11041` vs `atol=0.05`, 69/31488 elements (0.2%). Deterministic — identical value across runs.
2. **Larger m**: **NaN** on 86008/251904 elements (34.1%).

Raising the tolerance to 0.15 clears (1) and exposes (2). (2) is a genuine correctness bug, not a tolerance question.

### Relevant PPU-specific code path
PPU forces `use_atomic_add = False` in `python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py:213`, because the atomic-add reduction path deadlocks the kernel (GPU spins at 100% until SIGKILL — a pre-existing, separately-documented PPU issue). So PPU exercises the **workspace-reduce** path exclusively. Both the precision drift and the NaN appear on that path.

Worth checking whether the workspace buffer is correctly zero-initialized / sized for larger `m` in 0.4.3.

---

## Issue 3: Triton attention D=80 precision regression

### Reproducer
```bash
cd test
python3 -m pytest registered/attention/test_triton_attention_kernels.py::TestTritonAttention::test_extend_attention_unified_vs_regular
```

### Error
Unified kernel vs 2-stage kernel exceeds `atol=0.15` at config `(B=8, N_CTX=256, H_Q=64, H_KV=8, D=80)` — non-power-of-2 head dim.

### History
This exact failure existed on sgl-kernel 0.4.1, was **fixed in 0.4.2.post2** (verified 7/7 pass), and has **regressed in 0.4.3**. D=128 configs still pass.

### Reproducibility note
3 consecutive runs: 2 failed / 1 failed / 1 failed (the count varies because a cold Triton cache takes a different codegen path on the first compile). Not flaky in the sense of nondeterminism — it fails consistently once the cache is warm.

---

## Current workarounds (SGLang PPU CI)

| Issue | Workaround |
|-------|------------|
| 1 | `register_ppu_ci(..., disabled="ppu-llc SIGABRT on causal_conv1d_update_with_batch_gather itype0 (float32); SDK 2.1.1 bug")` |
| 2 | `register_ppu_ci(..., disabled="Marlin MoE produces NaN on PPU (34% of elements) at larger m; workspace-reduce fallback path. sgl-kernel 0.4.3 + SDK 2.1.1")` |
| 3 | D=80 config removed from the config list when `is_ppu_platform()` |

All three should be reverted once the underlying issues are fixed.

## Ask

- **Issue 2 first** — NaN output is a correctness bug that would silently corrupt production inference, not just a test failure.
- Issues 1 and 3 block test coverage but fail loudly.
- If a v2.1.0-based comparison run is useful, that stack is `reg.docker.alibaba-inc.com/aisw/llm:v2.1.0-pytorch2.10.0-ubuntu24.04-cuda13.0-sglang0.5.10-py312`.
