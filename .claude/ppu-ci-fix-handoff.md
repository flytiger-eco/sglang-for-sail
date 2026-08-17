# PPU CI Fix Handoff — 新 session 起手 prompt

请直接把下面这段作为新 session 的第一条消息发给我：

---

请加载本项目和全局记忆。

我正在 `/Users/shiyunzhi.syz/Documents/infer_client/sglang-for-sail` 仓库（`feat/ppu-ci-smoke` 分支）修复 PPU CI 的 26 个失败测试。目标是**全部 pass，拿到基线**。

## 背景

- 仓库：`flytiger-eco/sglang-for-sail`，分支 `feat/ppu-ci-smoke`，base `v0.5.13`
- 首次全量跑结果：309/335 passed（92.2%），26 个失败
- 源仓库（已全绿的参考配置）：`/Users/shiyunzhi.syz/Documents/infer_client/sglang_open`，分支 `v0.5.12_dev`
- 源仓库的完整 RCA 文档：`/Users/shiyunzhi.syz/Documents/infer_client/sglang_open/docs/ppu_ci_test_status.md`
- moss 的失败分析：`/Users/shiyunzhi.syz/Downloads/SGLang PPU CI 测试报告.md`
- PPU runner：ppu1（SSH host `ppu1`）和 ppu2（SSH host `ppu2`）
- Docker 镜像：`pkg.flytiger-eco.com/docker_release/llm:v2.1.1-pytorch2.11.0-ubuntu24.04-cuda13.0-sglang0.5.13-py312`
- Remote 必须用 SSH（`git@github.com:flytiger-eco/sglang-for-sail.git`）

## 修复清单（按优先级）

| # | 测试数 | 问题 | 修复方式 | 参考 |
|---|---|---|---|---|
| 1 | 3 | Circular import (test_fused_moe, test_block_int8, test_triton_moe_channel_fp8_kernel) | `fused_moe.py` 的 PPU import 改 lazy import（移到函数内部） | 源仓库 commit `ad74ca1d8` |
| 2 | 4 | 性能阈值 (test_torch_compile, test_triton_attention_backend, test_triton_attention_kernels, test_multi_tokenizer) | 加 `is_in_ppu_ci()` 分支降低阈值 | 源仓库 PR #42 里的改法 |
| 3 | 5 | Disaggregation BAREX fixture 缺失 (test_disaggregation_basic, _decode_offload, _decode_radix_cache, _pp, test_tracing_disaggregation) | 移植 `_PPU_BAREX_ENV` + `apply_ppu_barex_env()` + `MC_LOCAL_HOSTNAME` 到 `disaggregation_fixture.py` | 源仓库 `python/sglang/test/` 目录下的 disaggregation fixture |
| 4 | 6 | 模型缺失 (test_server_args, test_modelopt_loader, test_patch_tokenizer, test_pp_single_node, 2x test_deepseek_v4_flash) | `disabled="model not available on NAS"` 或 `@skip_if_model_missing` | — |
| 5 | 2 | HF Hub offline (test_bench_serving_functionality, test_model_file_verifier) | `disabled="requires HF Hub online access"` | 源仓库里已 disabled |
| 6 | 1 | HiCache MLA 精度 (test_hicache_variants) | 查源仓库用什么模型、阈值是多少；可能需要换模型或调阈值 | — |
| 7 | 1 | test_dumper — 模型缺失不是 EIC 问题 | `@skip_if_model_missing("Qwen/Qwen3-0.6B")` | 源仓库 PR #37 |
| 8 | 1 | test_load_weights_from_remote_instance — EIC 单 MR 限制，BAREX 方案也不够 | `disabled="EIC weight block >64MB MR limit; BAREX only covers KV cache"` | 源仓库里也 disabled 了 |
| 9 | 1 | nightly-1a timeout — job 110 分钟跑不完 | 提高 timeout 或优化 partition | 源仓库用 120min + 2 partition |

## 工作流程

1. 先读源仓库 `docs/ppu_ci_test_status.md` 对应章节，确认修复方式
2. 在 `sglang-for-sail` 仓库上改代码
3. 每改一批，在 ppu1 上 preflight 验证（`docker run` + `run_suite.py`）
4. 全部改完后 commit + push + 触发 nightly 验证
5. **按 `.claude/rules/pre-push-review.md` 规则，push 前必须 spawn review agent**

## 关键注意事项

- v0.5.13 用 **plugin hook 机制** 注入 PPU ops（不是直接改 flashattention_backend.py），hook 在 `load_plugins()` 调用后生效
- `ppu_skip_utils.py` 的 guard 已 gate 在 `PPU_SDK` 环境变量上（非 PPU runner 不会 skip）
- install 脚本**不重装镜像自带的 PPU 栈**（flashinfer 0.6.12、sgl-kernel 0.4.3 等已在镜像里）
- `bench_one_batch.py` 已加了 `load_plugins()`（本 session 修的）
- 实习生 moss 在同一仓库的 `feat/ppu-pr-test` 分支并行开发，注意不要冲突

开始吧。请先读源仓库文档确认每个修复的具体做法，然后按优先级逐个修。
