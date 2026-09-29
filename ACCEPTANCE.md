# 验收记录

验收日期：2026-09-29。需求来源为 `docs/CODEX_TASK.md`。用户确认未提供额外的 MODEL_SPEC.md、ACCEPTANCE.md 和 user_confirmed_units.jsonl；本项目据主文档补充模型规范、验收记录与两条示例金标，没有使用未提供的附件或外部预训练权重。

## 交付与结论

已实现本地 React/TypeScript/Vite + FastAPI/PyTorch 工作台，包括预训练和后训练两页、真实独立训练进程、检查点恢复、SSE 指标、数据版本、人工审核、推理与纠错闭环。运行方法见 [README](README.md)，结构与训练公式见 [MODEL_SPEC](MODEL_SPEC.md)。

**CPU 调试模型的端到端流程通过；100M 正式模型的 GPU 训练尚未验收。** 本机只检测到 Intel 核显，PyTorch 为 2.8.0+cpu，CUDA 不可用。界面明确诊断，禁止以 CPU 启动正式规模模型。默认正式配置为 14 层、768 隐藏维、12 注意力头、3072 FFN；示例词表下实际参数量为 99,644,184，参数量随训练词表变化。

## 自动验证

最新汇总见 [verification.json](docs/verification.json)，所有记录均来自实际命令输出。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 后端测试 | 22 通过 | [原始输出](docs/verification/backend-tests.txt) |
| Python 格式检查 | 通过 | [原始输出](docs/verification/python-format.txt) |
| 前端测试 | 4 通过 | [原始输出](docs/verification/frontend-tests.txt) |
| TypeScript 与生产构建 | 通过 | [原始输出](docs/verification/frontend-build.txt) |
| CPU 端到端 smoke | 通过 | [原始输出](docs/verification/cpu-smoke.txt) / [结构化结果](docs/smoke-results.json) |

测试覆盖原文拼接一致性、Unicode 码点和跨度、两条金标、重复与标点保留、冲突审核、跨集合泄漏、不可变版本、未加载记录保留、模型初始化、检查点完整性、暂停恢复后的权重一致性、控制状态竞争，以及认证与 API 行为。后端输出有一条上游 Starlette/AnyIO 弃用警告，不影响当前测试通过。

CPU smoke 实际运行预训练 8 步、后训练 12 步，调用两种推理，验证暂停保存、继续、停止保存、并发启动返回 409、SSE 重放和安全权重 ZIP 导出。最新运行暂停于 step 1、停止于 step 2，重放 13 个事件。独立验证集未提供，验证指标保持空值，微调显式使用教学模式；这些数字不代表模型泛化能力。

## 浏览器实测

通过应用内浏览器完成以下操作：

- 保存原始语料、创建 CPU 调试模型、提交预训练，并观察实际训练指标与检查点。
- 使用预训练检查点启动后训练，读取实际单位预测、跨度和边界概率。
- 刷新页面后选择持久化历史任务、恢复检查点测试。
- 将错误预测送回标注编辑器，保存为草稿版本；冲突样本可见且不会自动成为训练金标。
- 在 390 px 窄视口检查三栏纵向排列，无横向溢出。
- 嵌入初始化调整后，再次完成 8 步预训练，并通过新页面从检查点 `134a8afb` 返回真实 MLM 候选概率。
- 从该检查点建立后训练分支 `7d5b58ed`，完成 12 步微调并返回原文可完整拼接的实际单位预测。

截图：[预训练](docs/screenshots/pretrain.jpg)、[后训练](docs/screenshots/finetune.jpg)、[窄屏布局](docs/screenshots/mobile.jpg)。截图展示调试运行，不是正式训练结果。浏览器旧标签在长时间热更新后出现点击状态未更新，重新打开页面后同一流程正常完成。

## 使用边界

- 尚未在 CUDA 上实测 AMP、显存占用、OOM 恢复和正式 100M 两阶段训练，需要兼容 NVIDIA GPU 与对应 PyTorch build 后运行 GPU smoke。
- 两条示例用于规则和流程验证，不足以训练可用语言模型；日期、英文、emoji 等语义单位需另行人工标注并版本化。
- 导入上限 64 MB；近重复泄漏检测是启发式，仍需维护文档和模板来源分组。
- 安全导出包提供推理权重、词表和配置；不提供任意外部训练状态导入。本地续训使用工作台自身完整检查点。
- 本版本不实现远程或分布式训练、知识网络、长期记忆和自动审核。
