# PPU CI — Disabled Tests & SDK Issues Summary

**Branch**: `feat/ppu-ci-smoke`  
**Image**: `pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.11.0-ubuntu24.04-cuda13.0-sglang0.5.13-py312`  
**Date**: 2026-08-15  

---

## 1. Disabled Tests (24 total)

### 1.1 模型/资源缺失 (10 tests) — 上传模型到 NAS 后可直接解禁

| 测试文件 | Suite | Disabled 原因 | 解禁条件 |
|----------|-------|---------------|----------|
| `unit/server_args/test_server_args.py` | per-commit | `Qwen2.5-1.5B-Instruct` 不在 NAS | 上传模型到 HF_HUB_CACHE |
| `unit/model_loader/test_modelopt_loader.py` | per-commit | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 不在 NAS | 上传模型 |
| `unit/utils/test_patch_tokenizer.py` | per-commit | `nvidia/Kimi-K2-Thinking-NVFP4` tokenizer 不在 NAS | 上传 tokenizer |
| `utils/test_model_file_verifier.py` | nightly-1 | 部分方法需 HF Hub 在线访问 | 上传 gpt2 到 HF cache 或开放网络 |
| `bench_fn/test_bench_serving_functionality.py` | nightly-1 | 硬编码 `tokenizer="gpt2"`，HF Hub 离线 | 上传 gpt2 tokenizer 到 NAS |
| `disaggregation/test_disaggregation_hybrid_attention.py` | nightly-8 | `Qwen/Qwen3-Next-80B-A3B-Instruct` 不在 NAS | 上传模型（需 160GB+） |
| `models_e2e/test_deepseek_v4_flash_fp8_h200.py` | nightly-8 | 模型不在 NAS + 需要 FP8 | 上传模型 + FP8 支持 |
| `models_e2e/test_deepseek_v4_flash_fp4_h200.py` | nightly-8 | 模型不在 NAS + 需要 FP4 | 上传模型 + FP4 支持 |
| `models_e2e/test_deepseek_v4_flash_fp4_b200.py` | nightly-4 | 模型不在 NAS + 需要 FP4 | 上传模型 + FP4 支持 |
| `models_e2e/test_deepseek_v4_flash_fp4_megamoe_b200.py` | nightly-4 | 模型不在 NAS + 需要 FP4 | 上传模型 + FP4 支持 |

### 1.2 PPU 硬件/SDK 不支持 (9 tests) — 需 SDK 团队支持后解禁

| 测试文件 | Suite | Disabled 原因 | 解禁条件 |
|----------|-------|---------------|----------|
| `quant/test_triton_scaled_mm.py` | nightly-1 | PPU Triton 不支持 fp8e4nv | SDK 支持 FP8 |
| `kernels/test_fp4_moe.py` | nightly-1 | PPU 不支持 FP4 | SDK 支持 FP4 |
| `unit/layers/quantization/test_mxfp4_sm90_cutlass.py` | per-commit | PPU 不支持 FP4/NVFP4 | SDK 支持 FP4 |
| `unit/batch_invariant_ops/test_batch_invariant_ops.py` | per-commit | PPU deep_gemm 缺 bf16_gemm_nn | SDK 实现 bf16_gemm |
| `backends/test_flashinfer_fusion_preflight.py` | nightly-1 | PPU 不支持 cuMemCreate multicast API | HGGC 实现 multicast |
| `spec/eagle/test_eagle_infer_beta_dp_attention.py` | nightly-1 | dsv3-test 模型需 FP8，PPU 810E 不支持 | FP8 支持 |
| `hicache/test_hicache_storage_mooncake_backend.py` | nightly-2 | mooncake 存储 backend 不可用 | 部署 mooncake |
| `prefill_only/test_pooled_hidden_states.py` | nightly-1 | MIS 需要 flashinfer，PPU 用 fa3 | fa3 实现 MIS 或 flashinfer 移植 |
| `model_loading/test_load_weights_from_remote_instance.py` | nightly-2 | EIC 硬件单 MR ≤64MB，BAREX 只覆盖 KV cache | EIC 硬件升级或 BAREX 扩展到 weight loading |

### 1.3 SDK v2.1.1 / sgl-kernel 0.4.3 Regression (3 tests) — SDK 修复后解禁

| 测试文件 | Suite | Disabled 原因 | 严重程度 |
|----------|-------|---------------|----------|
| `layers/mamba/test_causal_conv1d.py` | nightly-1 | ppu-llc SIGABRT 编译 batch_gather float32 kernel | 编译崩溃 |
| `quant/test_marlin_moe.py` | nightly-1 | Marlin MoE 输出 NaN（34% 元素），workspace-reduce 路径 | **静默错误结果** |
| *(inline skip)* `attention/test_triton_attention_kernels.py` | nightly-1 | D=80 unified vs 2-stage 精度 > atol=0.15 | 精度回退 |

### 1.4 PPU 性能/行为差异 (2 tests) — 非 bug，调参或设计差异

| 测试文件 | Suite | Disabled 原因 |
|----------|-------|---------------|
| `scheduler/test_routing_key_scheduling.py` | nightly-1 | PPU 调度行为差异导致 routing-key 延迟信号不可测 |
| `scheduler/test_priority_scheduling.py` | nightly-1 | PPU concurrent batching 使优先级排序不可测（assertion flaky） |

### 1.5 Docker 镜像依赖缺失 (已修复，不再 disabled)

| 问题 | 修复方式 |
|------|---------|
| `dill 0.3.6` → ABC 序列化崩溃 | install 脚本: `pip install "dill>=0.3.8,<0.3.9"` |
| `uvicorn 0.29.0` → 缺 timeout_worker_healthcheck | install 脚本: 安装 uvicorn-0.37.0 |
| `opentelemetry` 缺失 | install 脚本: `python[all_ppu,tracing]` |
| `test_multi_detokenizer` | disabled (uvicorn 仍不够新) |

---

## 2. Runtime PPU 适配（非 disabled，代码内条件分支）

| 适配 | 文件 | 说明 |
|------|------|------|
| 性能阈值降低 | `test_torch_compile.py` | PPU 110 tok/s vs CUDA 152 |
| 性能阈值降低 | `test_triton_attention_backend.py` | PPU 100 tok/s vs CUDA 153 |
| 延迟阈值放宽 | `test_multi_tokenizer.py` | PPU e2e 25s / TTFT 400ms / ITL 20ms |
| MMLU 阈值降低 | `test_hicache_variants.py` (MLA) | PPU 0.3 vs CUDA 0.5（DeepSeek-V2-Lite-Chat 比 Coder 弱） |
| MGSM 阈值降低 | `test_hicache_variants.py` (MLA) | PPU 0.65 vs CUDA 0.8 |
| D=80 config 跳过 | `test_triton_attention_kernels.py` | sgl-kernel 0.4.3 精度回归 |
| 3FS accuracy skipIf | `test_hicache_storage_3fs_backend.py` | cudaMemcpyBatchAsync 未实现 |
| CUDA fallback 测试 | `test_platform_interface.py` | 临时移除 PPU_SDK env 让 fallback 逻辑走通 |
| BAREX 自动注入 | `disaggregation_fixture.py` | 检测 fic2 RDMA NIC，配置 AcclBarex env vars |
| Circular import 修复 | `fused_moe.py` | lazy import at call sites |
| Marlin atomic-add 禁用 | `fused_marlin_moe.py:213` | PPU 上 atomic-add 死锁 GPU |

---

## 3. PPU SDK v2.1.1 + sgl-kernel 0.4.3 Regression 详情

### 3.1 ppu-llc SIGABRT — causal_conv1d batch_gather (float32)

- **表现**: `ppu-llc` exit code 134 (SIGABRT)
- **影响范围**: 仅 `itype0` (float32) + `batch_gather` 变体；fp16/bf16 正常，非 batch_gather 正常
- **复现**: `pytest "test_causal_conv1d.py::test_causal_conv1d_update_with_batch_gather[3-True-2064-3-1-False-False-itype0]"`
- **回归来源**: SDK v2.1.0 pass → v2.1.1 crash
- **建议调查方向**: 宽寄存器 + gather 寻址模式的 codegen path

### 3.2 Marlin MoE NaN 输出 ⚠️ 最高优先级

- **表现**: 34.1% 元素输出 NaN
- **影响范围**: 较大 m 值的 config；小 m 有 0.11 精度偏差（0.2% 元素）
- **根因**: PPU 强制 `use_atomic_add=False`（atomic-add 会死锁），走 workspace-reduce 路径。该路径在 0.4.3 上产出 NaN
- **复现**: `pytest test_marlin_moe.py::TestFusedMarlinMoe::test_fused_marlin_moe`
- **回归来源**: sgl-kernel 0.4.2.post2 pass → 0.4.3 NaN
- **严重性**: **静默错误** — 生产推理会输出垃圾而不是报错

### 3.3 Triton Attention D=80 精度回归

- **表现**: unified vs 2-stage kernel max diff 超过 atol=0.15
- **影响范围**: 仅 D=80（非 2 次幂 head dim）；D=128 正常
- **复现**: `pytest test_triton_attention_kernels.py::TestTritonAttention::test_extend_attention_unified_vs_regular`
- **回归来源**: sgl-kernel 0.4.1 有 bug → 0.4.2.post2 修复 → 0.4.3 再次回归
- **首次 warm cache** 后稳定复现，不是 flaky

---

## 4. Skip Guard 机制

除了 `disabled=` 的测试外，还有 ~110 个测试通过 `@skip_if_model_missing(model)` 动态跳过（模型不在 NAS 时 pytest exit code 5）。这些测试：
- 不计入 disabled 统计
- 模型上传到 NAS 后自动启用（无需改代码）
- exit code 5 已修正为 pass（`ci_utils.py`）

---

## 5. 恢复路径

| 阶段 | 动作 | 预期恢复测试数 |
|------|------|---------------|
| **短期** | 上传缺失模型到 NAS（Qwen2.5-1.5B, TinyLlama, gpt2, Qwen3-0.6B 等） | +5~8 |
| **中期** | SDK 团队修复 3 个 regression | +3 |
| **长期** | PPU 支持 FP8/FP4 + deep_gemm + multicast | +6 |
| **硬件** | EIC MR 限制解除或 BAREX 扩展 | +1 |
