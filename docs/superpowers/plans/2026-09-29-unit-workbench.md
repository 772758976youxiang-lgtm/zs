# 变量单位训练工作台 Implementation Plan

> **For agentic workers:** Use subagent-driven-development for training implementation and independent reviews. User approved execution of CODEX_TASK.md; no additional design gate.

**Goal:** 本地可运行的真实字符级预训练、边界微调工作台。
**Architecture:** React + TypeScript + Vite 独立前端；FastAPI + SQLite 任务持久化；独立 Python worker 执行 PyTorch 训练。所有文件保存在 workspace，模型与数据快照不可变。
**Tech Stack:** Python 3.11, PyTorch, FastAPI, Pydantic, SQLite, React, TypeScript.

## 已确认的设计
主交接文档为唯一收到的规格。用户确认不存在其他附件。补充模型：14 层、768 hidden、12 heads、3072 FFN、512 位置，字符词表与 MLM 权重共享；约 99.6M 加词表参数（实际后端计算）。CPU debug 为 2 层、64 hidden、4 heads、128 FFN。默认正式模型只能 CUDA 训练，CPU 小模型需显式选择。单位跨度采用 Unicode codepoint，空文本返回空单位。

## 任务与验证
- [x] 数据契约：backend/app/data.py 和 tests/test_data.py；原文覆盖、span/gap、冲突、审核、分组划分、词表仅训练集。pytest 验证两条金标、emoji、空白、重复及冲突。
- [x] 训练核心：backend/training/{model,engine}.py 和 tests/test_training.py；MLM、边界 BCE、安全原子检查点、精确续训、有限步数 CPU 真训练。先写失败测试，再实现并验证损失、恢复、还原约束。
- [x] 任务与 API：backend/app/{store,schemas,main,worker}.py 和 tests/test_api.py；SQLite 互斥预约、独立 subprocess、事件持久化、恢复标记、来源与本地 token 验证。测试生命周期、409、恶意来源、超长输入、检查点推理。
- [x] 前端：frontend/src/{App,api}.tsx 与 style.css；两页三栏，模型/数据/任务/检查点/MLM/切分全部接真实 API；npm build 与浏览器完整流程测试。
- [x] 交付：scripts/{start,smoke,check_environment}、README、MODEL_SPEC、ACCEPTANCE、依赖锁定、测试输出和两页截图。CPU 验证通过后保留本地运行服务，GPU 未执行据实标注。

## 训练与 API 内部契约
engine.run_training(spec, on_event, should_stop) -> checkpoint metadata；spec 包含 run_id, task ('pretrain'|'finetune'), config, model_config, records (train/val split), vocab (字符数组), output_dir, resume_path, parent_path, dataset_hash, rule_version。每次完整更新调用 should_stop() 返回 None/'pause'/'stop'。on_event 接收 metrics/checkpoint 事件。engine.infer_mlm(path,text,mask_positions,top_k=5)、infer_units(path,text,threshold=.5) 返回 JSON。engine.inspect_model(config,vocab_size) 返回 parameter_count/layers。
model_config: hidden_size,num_layers,num_heads,intermediate_size,max_length,dropout。config: learning_rate,micro_batch,gradient_accumulation,max_steps,save_every,eval_every,mask_ratio,precision,gradient_clip,activation_checkpointing,device,seed,sequence_length。
checkpoint 事件字段 type='checkpoint', path, global_step, task, metrics；指标事件 type='metrics', global_step, train_loss, val_loss, learning_rate, gradient_norm, chars_per_second, elapsed_seconds 等。回调不会承担训练计算。

## 验收原则
所有结果可追溯到实际检查点；独立验证数据缺失显示未评估；CPU smoke 不能称为 GPU 或泛化验收；不调用外部模型。缺少的配套规格明确作为本次工程选择，不冒称原附件。
