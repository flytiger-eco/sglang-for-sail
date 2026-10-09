# CI 注册完整性门禁修复设计

## 目标

修复 `test_qsa_prefill_compressed_pack.py` 缺少直接执行入口导致 PPU nightly 全量失败的问题，并将同类错误前置到不占用 PPU 的 PR 静态校验阶段。补全人工审批后的事件触发链，并通过 GitHub required checks 阻止未经有效 CI 验证的代码合入 `v0.5.18`。

## 根因

1. `test/registered/kernels/ops/attention/test_qsa_prefill_compressed_pack.py` 注册了 CUDA CI，但没有 `if __name__ == "__main__":` 执行入口。
2. `run_suite.py` 在筛选硬件和 suite 前扫描全部 `test/registered/**/*.py`，因此该 CUDA 测试的结构错误会阻断 PPU Perf、Answer、Accuracy 等所有 suite。
3. 现有 `scripts/lint/check_registered_tests.py` 只对包含 `unittest.TestCase` 的文件检查入口，未覆盖 pytest 函数式测试。
4. `.github/workflows/ci.yaml` 的 Smoke Test 只是占位 `echo`，没有运行注册校验。
5. CI 仅监听 PR 创建、同步、重开和 ready-for-review；组织级 Human Review Gate 超时后，后续 `pull_request_review` approval 不会重新触发流水线。
6. `v0.5.18` 没有要求 lint、注册校验及 Full CI 成功的 required status checks。

## 方案

### 测试文件修复

在问题测试文件末尾增加：

```python
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

保持 pytest 测试发现行为不变，同时满足 CI 通过 `python3 file.py -f` 直接执行测试文件的约束。

### 通用静态注册校验

扩展 `scripts/lint/check_registered_tests.py`：

- 对所有至少包含一个启用 registry 的文件检查 `ut_parse_one_file()` 返回的 `has_main_entry`。
- 不再仅限于 `unittest.TestCase` 文件；pytest 函数式测试同样必须有入口。
- 对缺失入口输出与 `collect_tests()` 一致的明确错误信息。
- 保留既有 registry 缺失、CUDA legacy suite 和非 dispatchable suite 检查。

增加独立单元测试，使用临时仓库目录验证：

- pytest 函数测试有 registry、无 main 时失败；
- pytest 函数测试补充 `pytest.main()` 后通过；
- disabled-only registry 不要求入口，与 `collect_tests()` 保持一致；
- 既有 `unittest.TestCase` 缺失入口检查继续有效。

### PR 流水线前置门禁

修改 `.github/workflows/ci.yaml`：

- 将占位 Smoke Test 替换为 `Registration Validation`。
- 在 `ubuntu-latest` 上执行 `python3 scripts/lint/check_registered_tests.py`，不安装或导入 SGLang，不占用 PPU。
- Human Review Gate 同时依赖注册校验与 AI review；注册校验失败时不进入人工审批和 Full CI。
- 增加 `pull_request_review: types: [submitted]` 触发。
- 对 review 事件仅接受真人 `approved`，避免 COMMENTED、CHANGES_REQUESTED 或机器人 review 启动完整链路。
- 保持现有 PR 更新事件和每 PR concurrency group。

审批事件触发后的流程为：Pre-check → Registration Validation/AI Review → Human Review（立即识别已有 approval）→ Full CI。

### GitHub 合入规则

代码 PR 验证后，通过 GitHub ruleset 为 `v0.5.18` 增加 required status checks：

- `lint`
- `Registration Validation`
- `Full CI / pr-test-ppu-finish`

该配置属于仓库外部状态，不写入源码；执行前读取现有 ruleset，采用新增仓库级规则或最小范围更新，避免修改组织级共享规则。修改后通过 GitHub API 回读确认。

## 错误处理与资源控制

- 注册校验失败应在 CPU runner 上终止，不派发任何 PPU Pod。
- Human Review Gate 超时仍保持失败；之后的真人 approval 通过事件重新启动流水线。
- Full CI 的最终汇总 Job 继续将任一实际测试失败传播为失败状态。
- 不降低 `collect_tests()` 的全局严格检查，nightly 保留第二道防线。

## 验证

1. TDD 红阶段：新增 pytest 函数式测试缺少 main 的夹具，确认现有 checker 错误通过。
2. 绿阶段：修改 checker 后确认新增测试和 `scripts/lint` 全部单测通过。
3. 执行 `python3 scripts/lint/check_registered_tests.py`，确认目标分支全部注册文件通过。
4. 验证问题文件可被 `ut_parse_one_file()` 判定为 `has_main_entry=True`。
5. 使用 YAML 解析和 workflow 定向检查确认 `ci.yaml` 语法与依赖关系。
6. 推送独立分支后，通过 GitHub API 回读远端 workflow 内容。
7. 创建以 `v0.5.18` 为 base 的 PR，确认 Registration Validation 在 CPU 阶段执行。
8. 经真人 approval 后确认新 run 自动产生且 Full CI 启动。
9. Full CI 验证完成后再更新 required checks，并通过 API 回读规则。

## 边界

- 不直接修改或推送 `v0.5.18`。
- 不修改组织级 `flytiger-eco/.github` reusable workflow。
- 不在本次修复中调整 PPU 测试内容、模型配置或 nightly 调度策略。
