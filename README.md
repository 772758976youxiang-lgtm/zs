# Unit · 变量单位训练工作台

本地 React + FastAPI + PyTorch 全栈项目。先用字符掩码学习中文上下文，再从预训练检查点训练单位边界；两阶段均执行真实训练，不调用外部语言模型。预训练 `/pretrain`，后训练 `/finetune`，均为中文白色三栏界面。

**当前机器只有 Intel UHD Graphics 770，未检测到 CUDA GPU。已完成 CPU 小模型开发验证，尚未执行 100M GPU 验收，也没有充足语料或独立测试集来证明中文泛化效果。** 默认两条样例是用户确认金标，仅用于教学。

## 启动

在本项目根目录打开 PowerShell。已安装的本地环境可直接运行：

```powershell
.\scripts\start.ps1
```

打开 http://127.0.0.1:5173/pretrain 。后端为 http://127.0.0.1:8000 ，API 文档 `/docs`。前后端是两个进程，训练是另外的独立进程。服务日志在 `workspace/backend*.log` 和 `workspace/frontend*.log`，每个训练任务有自己的 `workspace/runs/<id>/worker.log`。

手动前后端分开启动：

```powershell
# 终端 1，项目根目录
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
# 终端 2，项目根目录
npm --prefix frontend run dev
```

首次安装或在另一台 Windows 机器安装（Python 3.11+、Node 22+）：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
npm --prefix frontend ci
.\.venv\Scripts\python.exe scripts/check_environment.py
```

`requirements-lock.txt` 记录本机实际 CPU 开发环境；`requirements.txt` 列出直接依赖。使用 NVIDIA 机器时先查看 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 与 [历史版本安装矩阵](https://pytorch.org/get-started/previous-versions/)，根据实际驱动选择兼容的 CUDA PyTorch build，再记录新的环境锁定。不要照抄一条通用驱动安装命令，不要在无 CUDA 时把正式模型改成 CPU。

## 完整操作流程

1. 预训练左栏粘贴语料并保存，或流式导入 UTF-8 TXT / JSONL（上限 64 MiB，BOM 单独记录，正文不裁剪）。较大文件保存在后端并生成记录偏移索引，浏览器不加载整份语料。
2. 选择数据集，点“新建模型”。正式默认 14 层 × 768，约 100M；本机请选择明确标记的 CPU debug（2 层 × 64）。实际参数量和字符覆盖率由后端计算。新建只创建配置及词表。
3. 配置有限训练步数，点“开始训练”。观察真实 loss、step、学习率、梯度范数、速度、epoch、耗时和显存指标。高级指标在“更多真实训练指标”展开。保存完成后右栏选择检查点，输入文本并点击字符遮盖，再预测候选字符。
4. 转到后训练，确认两条教学样例或导入自己的金标。点击字符间隙切开/合并；新增或纠错样本默认草稿，勾选人工审核才进入标注集。修改边界自动取消当前审核状态。冲突草稿可保存且会显示标记；两份冲突答案不能同时确认为金标，需先解决边界分歧。
5. 选择预训练父检查点和标注集，创建全参数微调分支。没有独立验证集时必须明确勾选教学过拟合模式；验证成绩仍显示“未评估”。
6. 右栏测试单位提取，可查看每个间隙概率、码点跨度、JSON、阈值、实际设备和检查点信息。纠错会进入左栏草稿，不自动进入当前训练快照。

训练任务和事件保存于 SQLite；刷新后从历史任务选择即可重连。暂停先进入 pausing，完成一致性保存才显示 paused；继续沿用同一任务，配置与数据不可偷偷改变。停止保留最后完整检查点。退出后台服务不等于暂停训练，请使用界面暂停/停止；失去 worker 的任务会标为 interrupted。

## 数据格式与划分

```json
{"text":"我养了一只猫叫魔丸","units":["我","养","了","一","只","猫","叫","魔丸"],"split":"train","status":"confirmed","source":"人工标注","group_id":"family-001"}
```

预训练只需 text，可附 source/group_id/split。JSONL 每行一个对象，正文换行用 JSON 转义。split 为 train/val/test；同一文档、模板、同句改写应采用相同 group_id。后端检查精确重复、三字符片段候选的高相似度跨集合泄漏；这是启发式检测，不能证明不存在语义泄漏，分组来源仍需人工维护。词表只由已确认训练记录建立。

对所有单位都要求 `''.join(units)==text`。后端索引是 Unicode codepoint，前端使用 Array.from；不使用 UTF-16 长度解释跨度。标点、助词、重复内容均保留。输入超出模型最大位置容量明确报错，不静默截断。

保存的数据集版本和训练快照不可变并带 SHA-256。标注编辑每页加载 50 条；翻页先保存当前修改，按记录 ID 与原版本合并，未加载记录不会丢失。全量导入导出支持 JSONL。

## 测试、构建与接口类型

```powershell
.\.venv\Scripts\python.exe -m pytest backend/tests -q
npm --prefix frontend test
npm --prefix frontend run build
.\.venv\Scripts\python.exe scripts/smoke.py --device cpu
# 只有检测到兼容 CUDA GPU 才运行，首版保持有限步数
.\.venv\Scripts\python.exe scripts/smoke.py --device cuda
# 后端 schema 改动后重新生成前端类型
.\.venv\Scripts\python.exe scripts/export_openapi.py
npm --prefix frontend run types
# 一次执行全部核心验收并保存真实输出
.\.venv\Scripts\python.exe scripts/verify.py
# 仅停止前后端服务；训练任务须在界面单独暂停/停止
.\.venv\Scripts\python.exe scripts/stop_services.py
```

smoke 验证实际 subprocess 预训练→后训练→推理→暂停→继续→停止、409 并发保护、事件补发及安全权重 ZIP。结果默认写到 `docs/smoke-results.json`；测试数据写到隔离的 `workspace/smoke`。CPU smoke 不代表正式模型训练验收。详细实测与限制见 `ACCEPTANCE.md`、模型公式见 `MODEL_SPEC.md`。

## 本地安全与恢复

默认仅 loopback，CORS 精确允许 localhost/127.0.0.1:5173，API 与 SSE 采用同一本地会话 token，拒绝不可信 Origin/跨站来源。路径由服务器生成并限制在 workspace 内；用户文件名不决定磁盘路径；启动进程使用参数数组、不拼 shell。单个 workspace 使用 SQLite 事务预约和文件锁限制一个训练 worker，同时通过进程代次避免旧回调污染恢复任务。

本系统只恢复自身生成且校验完整的四文件检查点：`model.safetensors`、`training.pt`、`metadata.json`、`checksums.json`。权重使用 safetensors，训练状态使用 [torch.load(weights_only=True)](https://docs.pytorch.org/docs/stable/generated/torch.load.html)，不提供任意外部 pickle 导入。校验和用于检测损坏，不替代来源信任。请勿手动将不可信文件放入 workspace 冒充本地检查点。

导出 ZIP 只有安全推理权重与配置/词表，不包含 optimizer、训练 pickle 状态；这是供外部集成的推理包，不能直接作为本工作台的续训导入包。本工作台测试使用完整的内部检查点。不要手动移动运行中的 workspace。需要备份时先暂停并等待保存完成，再复制整个 workspace。

## 常见问题

- **没有 CUDA**：本机是 Intel 核显，当前只允许显式 CPU debug。准备兼容 NVIDIA GPU 后按官方说明安装匹配 build，重新检测设备。
- **没有验证集**：页面显示未评估，教学微调需显式 demo 模式。不要把训练集 F1 当泛化成绩。
- **草稿不能训练**：标注完成后勾选审核并保存新版本，再选择该版本启动。
- **409 已有任务**：在历史任务中暂停/停止现有任务，完成保存后再启动。
- **OOM / 非有限数**：任务失败并保留最近完整检查点，错误在任务和日志可见。调整微批次、长度、精度或激活重计算时建立新任务，不假装精确续训。
- **端口已占用**：确认是否已有本工作台服务。不要重复启动；访问现有页面或停止自己之前启动的服务。

首版不包含知识网络、实体合并、概念归一化、长期记忆、远程训练、自动标注、大规模分布式数据加载或通用分词规则。负数、日期、英文、emoji 等未提供金标的语义边界由人工定义和版本化。
