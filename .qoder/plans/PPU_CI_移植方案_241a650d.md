# PPU CI 从 v0.5.12 移植到 v0.5.13 方案

## 背景与现状

| 维度 | v0.5.12 (sglang-for-test) | v0.5.13 (sglang-for-sail) |
|------|--------------------------|--------------------------|
| PPU 平台层 | 无，is_ppu() 散落在各文件 | 已有 hardware_backend/ppu/、platforms/ppu.py |
| register_ppu_ci 测试文件 | 382 个 | 7 个 (仅 feat/ppu-ci-smoke 分支) |
| ppu_skip 测试文件 | 146 个 | 0 个 |
| CI workflow | pr-test-ppu.yml + nightly-test-ppu.yml | 仅 pr-test-ppu.yml (smoke) |
| ppu_skip_utils.py | 有 | 无 |
| test/registered/ppu/ | 4 个测试文件 | 1 个 (test_ppu_basic.py) |

**关键差异**：v0.5.13 的 run_suite.py 命名从 `stage-a-test` 改为 `base-a-test`，新增了 XPU/MUSA 后端，PPU nightly suites 为空 `[]`。v0.5.13 已有 PPU 平台抽象层，部分 v0.5.12 的源码补丁可能已被平台层覆盖。

---

## 阶段 1：CI 基础设施（可直接复制，低风险）

**目标**：搭建 PPU CI 的 workflow 和脚本骨架

### 1.1 复制 GitHub Actions workflow 文件
- 源：`sglang-for-test/.github/workflows/nightly-test-ppu.yml` → 目标：`sglang-for-sail/.github/workflows/nightly-test-ppu.yml`
- 源：`sglang-for-test/.github/workflows/pr-test-ppu.yml` → 覆盖目标同名文件（feat/ppu-ci-smoke 的 smoke 版替换为完整版）
- **注意**：更新 workflow 中的分支名（`ppu_latest` → 适配 v0.5.13 的分支名）和触发条件
- 检查 `PPU_BASE_IMAGE` 镜像版本是否需要更新

### 1.2 复制 PPU 安装脚本
- 源：`sglang-for-test/scripts/ci/ppu/ppu_install_dependency.sh` → 覆盖目标同名文件

### 1.3 复制 Claude 规则
- 源：`sglang-for-test/.claude/rules/pre-push-review.md` → 目标

### 1.4 复制文档
- `docs/ppu_ci_status_summary.md`
- `docs/ppu_ci_test_status.md`
- `docs/ppu_marlin_hang_sdk_report.md`

**验收**：workflow 文件语法正确，`scripts/ci/ppu/` 目录完整

---

## 阶段 2：CI 测试框架适配（需适配，中风险）

**目标**：让 `run_suite.py --hw ppu` 能正常调度 PPU 测试套件

### 2.1 新增 ppu_skip_utils.py
- 源：`sglang-for-test/python/sglang/test/ci/ppu_skip_utils.py` → 目标同路径
- 这是新文件，可直接复制，但需检查导入路径是否与 v0.5.13 兼容

### 2.2 适配 ci_register.py
- v0.5.13 的 ci_register.py 已支持 `register_ppu_ci`（HWBackend.PPU 已存在）
- **无需改动**，仅验证 `register_ppu_ci` 函数和 `REGISTRY_FUNCS` 字典中 PPU 条目完整

### 2.3 适配 ci_utils.py
- 对比两个版本的 diff，将 v0.5.12 中 PPU 相关的工具函数移植到 v0.5.13
- 重点关注：PPU 环境检测、模型路径处理

### 2.4 修改 run_suite.py
- 将 PPU nightly suites 从空 `[]` 改为：
  ```python
  HWBackend.PPU: [
      "nightly-1-ppu",
      "nightly-2-ppu",
      "nightly-4-ppu",
      "nightly-8-ppu",
  ],
  ```
- 在 `PER_COMMIT_SUITES` 中确认 `per-commit-1-ppu` 和 `stage-a-test-1-gpu-ppu` 存在
- 注意 v0.5.13 命名约定变化（`stage-a-test` → `base-a-test`），PPU 套件名保持原样还是跟随新约定需确认

### 2.5 适配 server_fixtures
- 检查 `sglang-for-test/python/sglang/test/ci/` 下的 server_fixtures 相关文件
- 如有 PPU 专用 fixture，移植到 v0.5.13

**验收**：`python3 test/run_suite.py --hw ppu --suite per-commit-1-ppu --dry-run` 能列出测试文件

---

## 阶段 3：PPU 专用测试文件（新增文件，低风险）

**目标**：移植 PPU 专属测试用例

### 3.1 移植 test/registered/ppu/ 测试文件
- `test_ppu_constraints.py`（新）
- `test_ppu_fa3_eval.py`（新）
- `test_ppu_mla_eval.py`（新）
- `test_ppu_basic.py`（已存在，需对比是否需要更新）
- `__init__.py`（新）

### 3.2 移植 JIT kernel 测试
- 源：`sglang-for-test/python/sglang/jit_kernel/tests/test_moe_wna16_marlin.py` → 目标同路径
- 检查导入路径与 v0.5.13 的 jit_kernel 模块结构兼容

**验收**：新文件能被 `run_suite.py` 正确发现和注册

---

## 阶段 4：源码补丁评估与移植（高风险，需逐文件分析）

**目标**：移植 PPU 特定的源码 workaround

**关键前提**：v0.5.13 已有 `platforms/ppu.py` (PPUDeviceMixin) 和 `hardware_backend/ppu/`，需先确认每个补丁是否已被平台层覆盖。

### 4.1 fused_marlin_moe.py — Marlin atomic_add 死锁修复
- **v0.5.12 改动**：强制 PPU 上 `use_atomic_add = False`（PPU SM 8.0a 在小 M 时 atomic_add 死锁）
- **v0.5.13 状态**：该文件已重构（diff 84 行），但未包含 PPU workaround
- **操作**：在 v0.5.13 的 `fused_marlin_moe.py` 中 `use_atomic_add` 赋值后添加：
  ```python
  if is_ppu():
      use_atomic_add = False
  ```
- **导入**：v0.5.13 中 `is_ppu` 在 `sglang.srt.platforms.device_mixin`

### 4.2 lora_moe_runner_marlin.py — 同一死锁修复
- **v0.5.12 改动**：硬编码 `use_atomic_add=True` 需改为 PPU 感知
- **v0.5.13 状态**：两版本文件完全相同（0 diff），可直接套用相同改动
- **操作**：在 `use_atomic_add=True` 处加 PPU 判断

### 4.3 marlin_utils.py — dense Marlin atomic_add 风险
- **v0.5.12 改动**：`should_use_atomic_add_reduce` 中加 PPU 判断
- **v0.5.13 状态**：diff 28 行，需对比
- **操作**：将 PPU 判断逻辑移植到 v0.5.13 版本

### 4.4 deepep.py — DeepEP dispatch 逻辑
- **v0.5.12 改动**：PPU 专用 dispatch_dtype 逻辑（11 处 PPU 相关行）
- **v0.5.13 状态**：diff 316 行，文件有较大变化，且 v0.5.13 已有 `hardware_backend/ppu/` 可能覆盖部分逻辑
- **操作**：需逐行对比，确认哪些 PPU 逻辑已被平台层覆盖，哪些仍需保留

### 4.5 fused_moe.py — MoE triton utils 循环依赖修复
- **v0.5.12 改动**：lazy import 解决循环依赖（12 处 PPU 相关行）
- **v0.5.13 状态**：diff 24 行，变化较小
- **操作**：直接移植 PPU 相关改动

### 4.6 flashattention_backend.py — PPU FA 后端
- **v0.5.12 改动**：2 处 PPU 相关行
- **v0.5.13 状态**：diff 1280 行，文件有大幅重构。v0.5.13 已有 `hardware_backend/ppu/attention/` 可能已覆盖
- **操作**：**低优先级**，先确认 v0.5.13 的 PPU attention hooks 是否已处理

### 4.7 pyproject_other.toml — PPU 依赖
- **v0.5.12 改动**：4 处 PPU 相关行
- **v0.5.13 状态**：diff 27 行
- **操作**：合并 PPU 依赖配置

**验收**：源码改动不破坏 v0.5.13 已有的 PPU 平台层功能

---

## 阶段 5：测试文件批量注册（大量文件，中风险）

**目标**：为 v0.5.13 的测试文件添加 `register_ppu_ci` 调用和 PPU skip 逻辑

### 5.1 策略：文件级对比而非直接复制
- v0.5.13 有 2692 个文件变更，测试文件本身内容已不同，**不能直接覆盖**
- 需对每个测试文件：取 v0.5.13 版本为基准 → 将 v0.5.12 的 PPU 注册信息追加

### 5.2 register_ppu_ci 注册（~375 个文件）
- 对比 v0.5.12 和 v0.5.13 的同名测试文件
- 将 v0.5.12 中的 `register_ppu_ci(...)` 调用移植到 v0.5.13 版本
- **注意**：部分测试文件在 v0.5.13 中可能已删除或重命名，需跳过或映射

### 5.3 ppu_skip / skip_if_ppu 逻辑（~146 个文件）
- 移植 PPU skip 装饰器和条件跳过逻辑
- 检查 `ppu_skip_utils.py` 中的工具函数是否被正确引用

### 5.4 自动化辅助
- 可编写脚本：从 v0.5.12 测试文件中提取 `register_ppu_ci` 行和 `ppu_skip` 行，自动插入到 v0.5.13 对应文件的正确位置
- 人工 review 每个自动插入的结果

### 5.5 处理 v0.5.13 新增/删除的测试文件
- v0.5.13 删除了 63 个文件中的部分测试 → 对应的 PPU 注册也要删除
- v0.5.13 新增的测试文件 → 评估是否需要添加 PPU 注册

**验收**：`run_suite.py --hw ppu --suite per-commit-1-ppu --dry-run` 列出与 v0.5.12 相同数量的测试文件

---

## 阶段 6：验证与冒烟测试

### 6.1 本地验证
- `python3 test/run_suite.py --hw ppu --suite per-commit-1-ppu --dry-run` 检查测试列表
- `python3 -c "from sglang.srt.platforms import is_ppu; print(is_ppu())"` 验证平台检测
- 语法检查所有修改的 Python 文件

### 6.2 CI 冒烟测试
- Push 到 `feat/ppu-ci-smoke` 分支
- 触发 `pr-test-ppu.yml` workflow
- 验证 per-commit-1-ppu suite 能在 PPU runner 上运行

### 6.3 回归验证
- 确认 v0.5.13 已有的 CUDA/AMD/NPU CI 不受影响
- 确认 PPU 平台层功能（FA3、FlashMLA、DSA）正常

### 6.4 逐步启用 nightly suites
- 先启用 `nightly-1-ppu`
- 再启用 `nightly-2-ppu`、`nightly-4-ppu`、`nightly-8-ppu`

---

## 风险矩阵

| 阶段 | 风险 | 缓解措施 |
|------|------|---------|
| 1 | 低 — 新增文件 | 直接复制 |
| 2 | 中 — 框架适配 | 参照 v0.5.13 已有结构 |
| 3 | 低 — 新增文件 | 检查导入兼容性 |
| 4 | **高** — 源码冲突 | 逐文件评估，先确认平台层是否已覆盖 |
| 5 | 中 — 大量文件 | 脚本辅助 + 人工 review |
| 6 | 低 — 验证 | 分步启用 |

## 建议执行顺序

1 → 2 → 3 → 6(冒烟) → 4 → 5 → 6(完整)

先搭建 CI 骨架并跑通 smoke test，再逐步移植源码补丁和批量注册测试。