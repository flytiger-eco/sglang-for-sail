# PPU CI port to v0.5.13 — remaining manual work

The bulk port from `sglang-for-test@v0.5.12_dev` is done and mechanical. What is
left needs a human decision, so it is listed here rather than guessed at.

Ported automatically by `scripts/ci/ppu/port_ppu_registrations.py` and
`scripts/ci/ppu/port_ppu_skip_guards.py` (both idempotent — safe to re-run).

## What landed

| | v0.5.12_dev (source) | v0.5.13 (here) |
|---|---|---|
| registered PPU tests | 382 | **365** |
| active | 252 | 243 |
| skip_guard | 109 | 102 |
| disabled | 21 | 20 |

The 17-file gap is entirely files that do not exist in v0.5.13 (below). 31 further
registrations landed at a **moved** path — v0.5.13 reorganized `test/registered/`
(e.g. `distributed/` → `disaggregation/`, `models/` → `models_e2e/`).

## 1. Blocking — must be resolved before the first run

### 1.1 PPU wheel versions in `scripts/ci/ppu/ppu_install_dependency.sh`

The script was copied verbatim and still pins the **2.1.0** SDK generation:

```
sglang_kernel==0.4.2.post2+v0.1.0.ppu2.1.0
flashinfer_python==0.6.8.post1+v0.1.0.ppu2.1.0
triton==3.6.0+v0.1.0.ppu2.1.0
sglang==0.5.12+v0.1.0.ppu2.1.0        # <- also still 0.5.12
...
```

The new base image is `v2.1.1-...-sglang0.5.13`, i.e. PPU SDK **2.1.1**. Every
`+v0.1.0.ppu2.1.0` suffix and the `sglang==0.5.12` pin likely need bumping. These
were deliberately **not** guessed: a wrong version string fails deep in the job with
a confusing resolver error. Confirm the correct strings against the 2.1.1 pip index.

### 1.2 Self-hosted runner registration

Both workflows use `runs-on: ppu`. Runner labels are per-repository, so ppu1/ppu2
must be registered as runners on `sglang-for-sail` as well — an existing
registration on `sglang-for-test` does not carry over. Until then jobs queue forever.

### 1.3 Repository secrets

Both workflows pass `secrets.PPU_ARTIFACTORY_USER` / `secrets.PPU_ARTIFACTORY_PASSWORD`
into the container; `ppu_install_dependency.sh` writes them to `~/.netrc` to reach
`art-pub.eng.t-head.cn`. Secrets are per-repository — add both to `sglang-for-sail`
settings. If missing, the install step silently skips writing `.netrc` and every PPU
wheel install then fails on auth.

### 1.4 NAS mounts

`nightly-test-ppu.yml` mounts `/nas_aisw/cache/sglang-jit-ci` (JIT cache) and expects
model checkpoints under `/nas_aisw/datasets/checkpoints/LLM` (`SGLANG_NAS_MODEL_BASE`).
Verify both are present on the runners used here.

## 2. Files absent from v0.5.13 (17) — decide per file

These carried a PPU registration in v0.5.12 but have no counterpart here. Some were
split into variants, some appear removed outright.

**Split into variants — pick which to register:**

| v0.5.12 file | v0.5.13 candidates |
|---|---|
| `4-gpu-models/test_gpt_oss_4gpu.py` | `models_e2e/test_gpt_oss_4gpu_bf16.py`, `models_e2e/test_gpt_oss_4gpu_mxfp4.py`, `perf/test_gpt_oss_4gpu_perf.py` |
| `radix_cache/test_unified_radix_cache_kl.py` | `radix_cache/unified_radix_tree/test_unified_radix_cache_kl_{full,swa,mamba,cp}.py` |
| `radix_cache/test_unified_radix_cache_kl_hicache.py` | same directory, hicache variants |
| `quant/test_deepseek_v3_fp4_4gpu.py` | `quant/test_deepseek_v3_fp4_4gpu_extra.py` (only the `_extra` variant remains) |

**No counterpart found — confirm removed upstream, then drop from the port:**

- `4-gpu-models/test_qwen35_fp4_mtp_v2.py`
- `4-gpu-models/test_qwen35_models.py`
- `8-gpu-models/test_mimo_models.py`
- `core/test_gemma4_moe_deterministic.py`
- `kernels/test_nsa_indexer.py`
- `language/test_srt_backend.py`
- `quant/test_deepseek_v32_fp4_mtp_4gpu.py`
- `spec/eagle/test_eagle_infer_a.py`, `test_eagle_infer_b.py`, `test_eagle_infer_beta.py`
  (only `test_eagle_infer_beta_dp_attention*.py` remain here)
- `unit/ci/test_ci_utils.py`
- `unit/managers/test_dp_budget.py`
- `unit/mem_cache/test_nsa_pool_host_unit.py`

## 3. Skip guards whose class was renamed (7) — add by hand

The file matched but the class did not, so the guard was not applied. Without it
these fail instead of skipping when the model is missing.

| File | Class in v0.5.12 |
|---|---|
| `models_e2e/test_nvidia_nemotron_3_super_bf16.py` | `TestNvidiaNemotron3SuperBF16MTP` |
| `disaggregation/test_disaggregation_hybrid_attention.py` | `TestDisaggregationHybridAttentionMambaDPDecode` |
| `pp/test_pp_single_node.py` | `TestQwenPPAccuracy`, `TestQwenVLPPAccuracy`, `TestQwenPPTieWeightsAccuracy`, `TestQwenMoePPAccuracy`, `TestQwen35PPAccuracy` |

## 4. New in v0.5.13 (620) — no PPU coverage yet

620 target files carry a CI registration but no PPU one, because they did not exist
in v0.5.12. They are simply not in any PPU suite today; nothing is broken. To extend
coverage, run them on hardware and register the ones that pass. Full list:

```bash
python3 scripts/ci/ppu/port_ppu_registrations.py \
  --source <path-to-sglang-for-test> --target . --report /tmp/report.txt
```

## 5. Not ported: `results.json` (`--results-json` / `--capture-output`)

This repo's `test/run_suite.py` predates that feature, so the workflows were taken
from the pre-`results.json` baseline to keep the CLI in sync. Port
`sglang-for-test` PR #55 first if the machine-readable results artifact is wanted.

## 6. Untested here

The first real run is the actual verification. Locally confirmed: AST collection
(365 registrations across the 5 suites), `validate_all_suites` passes, workflow YAML
parses, isort/black clean at the repo's pinned versions. **No test has been executed
on PPU hardware against v0.5.13** — expect the v0.5.13 code changes themselves to
surface new failures independent of this port.
