import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRight,
  ArrowRight,
  BookOpen,
  Box,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Database,
  Download,
  FileText,
  FlaskConical,
  Layers3,
  LoaderCircle,
  MessageSquare,
  Pause,
  Play,
  Plus,
  RefreshCw,
  Save,
  Settings2,
  ShieldCheck,
  Square,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  session,
  url,
  type Dataset,
  type Model,
  type Run,
  type Checkpoint,
  type Row,
} from "./api";
import { unitsFromGaps, gapsFromUnits, commitRow } from "./units";
import type { components } from "./generated/api";
import Metrics from "./Metrics";

const GOLD: Row[] = [
  {
    text: "我养了一只猫叫魔丸",
    units: ["我", "养", "了", "一", "只", "猫", "叫", "魔丸"],
    source: "user_confirmed",
    group_id: "teaching-1",
    split: "train",
    status: "confirmed",
  },
  {
    text: "我养了十二只猫，它们都很乖很可爱",
    units: [
      "我",
      "养",
      "了",
      "十二",
      "只",
      "猫",
      "，",
      "它",
      "们",
      "都",
      "很",
      "乖",
      "很",
      "可爱",
    ],
    source: "user_confirmed",
    group_id: "teaching-2",
    split: "train",
    status: "confirmed",
  },
];
const STATUS: Record<string, string> = {
  queued: "等待启动",
  running: "训练中",
  pausing: "正在保存并暂停",
  paused: "已暂停",
  stopping: "正在保存并停止",
  cancelled: "已停止",
  completed: "已完成",
  failed: "失败",
  interrupted: "已中断",
};
const fmt = (n: any, d = 3) =>
  typeof n === "number"
    ? n.toLocaleString("zh-CN", { maximumFractionDigits: d })
    : "—";
const short = (s?: string) => (s ? s.slice(0, 8) : "—");
const active = (s?: string) =>
  ["queued", "running", "pausing", "stopping"].includes(s || "");
type Config = Required<components["schemas"]["TrainConfig"]>;
const DEFAULT: Config = {
  learning_rate: 0.0003,
  micro_batch: 2,
  gradient_accumulation: 1,
  max_steps: 100,
  save_every: 25,
  eval_every: 10,
  mask_ratio: 0.15,
  precision: "fp32",
  gradient_clip: 1,
  activation_checkpointing: false,
  device: "cuda",
  seed: 42,
  sequence_length: 128,
};

function Chart({ events }: { events: any[] }) {
  const points = events.filter(
    (e) => e.type === "metrics" && typeof e.train_loss === "number",
  );
  if (!points.length)
    return (
      <div className="chart-empty">
        <div className="chart-grid" />
        <span>
          <Layers3 size={22} />
          训练开始后，曲线将在这里生长
        </span>
        <small>仅展示真实训练事件，无模拟数据</small>
      </div>
    );
  const max =
    Math.max(
      ...points.map((e) => Math.max(e.train_loss, e.val_loss || 0)),
      0.001,
    ) * 1.15;
  const maxStep = Math.max(...points.map((e) => e.global_step), 1);
  const coords = (key: string) =>
    points
      .filter((e) => typeof e[key] === "number")
      .map(
        (e) =>
          `${38 + (e.global_step / maxStep) * 412},${142 - (e[key] / max) * 124}`,
      )
      .join(" ");
  return (
    <div className="real-chart">
      <svg viewBox="0 0 480 168" role="img" aria-label="真实训练与验证损失曲线">
        {[0, 1, 2, 3].map((i) => (
          <g key={i}>
            <line
              x1="38"
              x2="450"
              y1={18 + i * 41}
              y2={18 + i * 41}
              stroke="#e8ece7"
              strokeDasharray="3 4"
            />
            <text x="0" y={22 + i * 41}>
              {(max * (1 - i / 3)).toFixed(1)}
            </text>
          </g>
        ))}
        <polyline
          points={coords("train_loss")}
          fill="none"
          stroke="#5c795e"
          strokeWidth="2.5"
        />
        <polyline
          points={coords("val_loss")}
          fill="none"
          stroke="#b69b69"
          strokeWidth="2"
        />
        {points.length === 1 && (
          <circle
            cx={450}
            cy={142 - (points[0].train_loss / max) * 124}
            r="3"
            fill="#5c795e"
          />
        )}
        <text x="38" y="164">
          0
        </text>
        <text x="400" y="164">
          step {maxStep}
        </text>
      </svg>
    </div>
  );
}

export default function App() {
  const [page, setPage] = useState<"pretrain" | "finetune">(
    location.pathname === "/finetune" ? "finetune" : "pretrain",
  );
  const [ready, setReady] = useState(false),
    [system, setSystem] = useState<any>(null),
    [datasets, setDatasets] = useState<Dataset[]>([]),
    [models, setModels] = useState<Model[]>([]),
    [runs, setRuns] = useState<Run[]>([]),
    [cps, setCps] = useState<Checkpoint[]>([]);
  const [datasetId, setDatasetId] = useState(""),
    [modelId, setModelId] = useState(""),
    [runId, setRunId] = useState(""),
    [cpId, setCpId] = useState(""),
    [parentId, setParentId] = useState("");
  const [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [busy, setBusy] = useState(""),
    [events, setEvents] = useState<any[]>([]),
    [config, setConfig] = useState<Config>(DEFAULT);
  const [text, setText] = useState(""),
    [datasetName, setDatasetName] = useState(""),
    [split, setSplit] = useState("train"),
    [group, setGroup] = useState(""),
    [rule, setRule] = useState("unit-v1");
  const [annotations, setAnnotations] = useState<Row[]>(GOLD),
    [annotationIndex, setAnnotationIndex] = useState(0),
    [annotationText, setAnnotationText] = useState(GOLD[0].text),
    [gaps, setGaps] = useState<boolean[]>(gapsFromUnits(GOLD[0].units!)),
    [confirmed, setConfirmed] = useState(true);
  const [annotationOffset, setAnnotationOffset] = useState(0),
    [annotationTotal, setAnnotationTotal] = useState(2);
  const [showModel, setShowModel] = useState(false),
    [modelName, setModelName] = useState(""),
    [modelDescription, setModelDescription] = useState(""),
    [preset, setPreset] = useState("100m"),
    [demo, setDemo] = useState(false);
  const [testText, setTestText] = useState(""),
    [mask, setMask] = useState<number[]>([]),
    [threshold, setThreshold] = useState(0.5),
    [result, setResult] = useState<any>(null),
    [testSent, setTestSent] = useState("");
  const upload = useRef<HTMLInputElement>(null);
  const currentPage = useRef(page);
  currentPage.current = page;
  const navState = useRef({
    pretrain: {
      datasetId: "",
      runId: "",
      cpId: "",
      split: "train",
      group: "",
      datasetName: "",
    },
    finetune: {
      datasetId: "",
      runId: "",
      cpId: "",
      split: "train",
      group: "",
      datasetName: "",
    },
  });
  const run = runs.find((r) => r.id === runId),
    model = models.find((m) => m.id === modelId),
    dataset = datasets.find((d) => d.id === datasetId);
  const stageDatasets = datasets.filter((d) => d.kind === page),
    stageRuns = runs.filter((r) => r.task === page),
    stageCps = cps.filter((c) => c.task === page);
  const units = unitsFromGaps(annotationText, gaps),
    metrics = run?.metrics || {};
  async function refresh() {
    const [d, m, r, c] = await Promise.all([
      api<Dataset[]>("/api/datasets"),
      api<Model[]>("/api/models"),
      api<Run[]>("/api/runs"),
      api<Checkpoint[]>("/api/checkpoints"),
    ]);
    setDatasets(d);
    setModels(m);
    setRuns(r);
    setCps(c);
  }
  async function act(key: string, fn: () => Promise<void>) {
    setBusy(key);
    setError("");
    setNotice("");
    try {
      await fn();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy("");
    }
  }
  useEffect(() => {
    let live = true;
    session()
      .then(async () => {
        await refresh();
        const s = await api("/api/system");
        if (live) {
          setSystem(s);
          setReady(true);
        }
      })
      .catch((e) => setError("无法连接本地后端：" + e.message));
    return () => {
      live = false;
    };
  }, []);
  useEffect(() => {
    if (!ready) return;
    const timer = setInterval(() => refresh().catch(() => {}), 3000);
    return () => clearInterval(timer);
  }, [ready]);
  useEffect(() => {
    setEvents([]);
    if (!runId || !ready) return;
    const stream = new EventSource(url(`/api/runs/${runId}/events`));
    stream.onmessage = (e) => {
      const event = JSON.parse(e.data);
      setEvents((prev) =>
        prev.some((p) => p.id === event.id)
          ? prev
          : [...prev, event].slice(-1000),
      );
      if (
        event.type === "state" ||
        event.type === "checkpoint" ||
        event.type === "error"
      )
        void refresh();
    };
    return () => stream.close();
  }, [runId, ready]);
  useEffect(() => {
    if (!cpId) {
      const latest = cps.find((c) => c.task === page && c.global_step > 0);
      if (latest) setCpId(latest.id);
    }
  }, [cps, page, cpId]);
  useEffect(() => {
    const pop = () =>
      navigate(
        location.pathname === "/finetune" ? "finetune" : "pretrain",
        false,
      );
    addEventListener("popstate", pop);
    return () => removeEventListener("popstate", pop);
  }, [
    page,
    datasetId,
    runId,
    cpId,
    split,
    group,
    datasetName,
    annotations,
    annotationText,
    gaps,
    confirmed,
  ]);
  function navigate(next: "pretrain" | "finetune", push = true) {
    if (next === page) return;
    navState.current[page] = {
      datasetId,
      runId,
      cpId,
      split,
      group,
      datasetName,
    };
    if (page === "finetune") setAnnotations(editedRows());
    const state = navState.current[next];
    setPage(next);
    if (push) history.pushState({}, "", `/${next}`);
    setDatasetId(state.datasetId);
    setRunId(state.runId);
    setCpId(state.cpId);
    setSplit(state.split);
    setGroup(state.group);
    setDatasetName(state.datasetName);
    setResult(null);
    setTestSent("");
    setError("");
    setNotice("");
  }
  function editedRows() {
    return commitRow(annotations, annotationIndex, {
      text: annotationText,
      units,
      split,
      group_id: group || annotations[annotationIndex]?.group_id,
      status: confirmed ? "confirmed" : "draft",
      spans: undefined,
      gap_labels: undefined,
    }) as Row[];
  }
  function showAnnotation(row: Row, index: number) {
    setAnnotationIndex(index);
    setAnnotationText(row.text);
    setGaps(gapsFromUnits(row.units || [row.text]));
    setConfirmed(row.status === "confirmed");
    setSplit(row.split || "train");
    setGroup(row.group_id || "");
  }
  function showAnnotationPage(
    data: { records: Row[]; total: number },
    offset = 0,
    index = 0,
  ) {
    setAnnotations(data.records);
    setAnnotationOffset(offset);
    setAnnotationTotal(data.total);
    showAnnotation(
      data.records[Math.min(index, data.records.length - 1)],
      Math.min(index, data.records.length - 1),
    );
  }
  function selectAnnotation(index: number) {
    const rows = editedRows();
    setAnnotations(rows);
    showAnnotation(rows[index], index);
  }
  function updateAnnotationText(value: string) {
    setAnnotationText(value);
    setGaps(Array(Math.max(0, Array.from(value).length - 1)).fill(false));
    setConfirmed(false);
    setNotice("原文已修改：边界已重置，当前样本转为草稿；保存时创建新版本。");
  }
  async function saveData(nextOffset = annotationOffset) {
    await act("dataset", async () => {
      const records: Row[] =
        page === "pretrain"
          ? [
              {
                text,
                split,
                source: "manual",
                ...(group ? { group_id: group } : {}),
              },
            ]
          : editedRows();
      const ds = await api<Dataset>("/api/datasets", {
        name:
          datasetName ||
          `${page === "pretrain" ? "中文语料" : "标注集"} ${new Date().toLocaleTimeString()}`,
        kind: page,
        rule_version: rule,
        records,
        base_dataset_id: page === "finetune" && datasetId ? datasetId : null,
      });
      setDatasetId(ds.id);
      if (page === "finetune") {
        const offset = Math.min(
          nextOffset,
          Math.floor((ds.count - 1) / 50) * 50,
        );
        const data = await api(
          `/api/datasets/${ds.id}/records?offset=${offset}`,
        );
        showAnnotationPage(
          data,
          offset,
          offset === annotationOffset ? annotationIndex : 0,
        );
      }
      setNotice(
        "已保存不可变数据版本 · " + short(ds.hash) + "；未加载的记录完整保留。",
      );
    });
  }
  async function importFile(file: File) {
    await act("import", async () => {
      const form = new FormData();
      form.append("file", file);
      form.append("name", datasetName || file.name);
      form.append("kind", page);
      form.append("split", split);
      form.append("rule_version", rule);
      const ds = await api<Dataset>("/api/datasets/import", form);
      setDatasetId(ds.id);
      if (page === "finetune") {
        const data = await api(`/api/datasets/${ds.id}/records`);
        showAnnotationPage(data);
      }
      setNotice(
        "导入完成，正文空白已保留。标注集按每页 50 条加载，翻页保存当前修改，完整保留其余记录。",
      );
    });
  }
  async function loadDataset(id: string) {
    setDatasetId(id);
    if (id && page === "finetune")
      await act("load", async () => {
        const data = await api(`/api/datasets/${id}/records`);
        showAnnotationPage(data);
        setRule(datasets.find((d) => d.id === id)?.rule_version || "unit-v1");
        if (data.total > 50)
          setNotice(
            "每页加载 50 条；翻页会先保存当前修改，未加载的记录完整保留。",
          );
      });
    else if (page === "finetune") {
      showAnnotationPage({ records: GOLD, total: 2 });
    }
  }
  async function createModel() {
    await act("model", async () => {
      const m = await api<Model>("/api/models", {
        name: modelName || "未命名模型",
        description: modelDescription,
        dataset_id: datasetId,
        preset,
        seed: config.seed,
      });
      setModelId(m.id);
      if (preset === "debug")
        setConfig((c) => ({ ...c, device: "cpu", sequence_length: 64 }));
      setShowModel(false);
      setNotice("已创建模型配置和训练集词表，尚未训练。");
    });
  }
  async function startRun() {
    await act("start", async () => {
      const r = await api<Run>("/api/runs", {
        model_id: modelId,
        dataset_id: datasetId,
        task: page,
        config,
        parent_checkpoint_id: page === "finetune" ? parentId : null,
        demo_overfit: demo,
      });
      setRunId(r.id);
      setNotice("任务已提交到独立训练进程。");
    });
  }
  async function test() {
    await act("test", async () => {
      const response = await api(
        page === "pretrain" ? "/api/inference/mlm" : "/api/inference/units",
        page === "pretrain"
          ? { checkpoint_id: cpId, text: testText, mask_positions: mask }
          : { checkpoint_id: cpId, text: testText, threshold },
      );
      if (currentPage.current === page) {
        setResult(response);
        setTestSent(testText);
      }
    });
  }
  function selectRun(id: string) {
    setRunId(id);
    const selected = runs.find((r) => r.id === id);
    if (selected) {
      setModelId(selected.model_id);
      if (typeof selected.parent_checkpoint_id === "string")
        setParentId(selected.parent_checkpoint_id);
      if (page === "pretrain") setDatasetId(selected.dataset_id);
      setConfig(selected.config as Config);
      if (selected.last_checkpoint_id) setCpId(selected.last_checkpoint_id);
    }
  }
  function appendAnnotation(row: Row) {
    const rows = editedRows();
    setAnnotations([...rows, row]);
    showAnnotation(row, rows.length);
  }
  function correct() {
    appendAnnotation({
      text: testSent,
      units: result.units,
      status: "draft",
      split: "train",
      source: "model_correction",
    });
    setNotice("预测已复制到左栏草稿，请纠正边界并保存到待审核标注。");
  }
  function selectParent(id: string) {
    setParentId(id);
    const cp = cps.find((c) => c.id === id);
    if (cp) setModelId(cp.model_id);
  }
  const field = (key: keyof Config, label: string, step = 1) => (
    <label className="field" key={key}>
      {label}
      <input
        type="number"
        step={step}
        value={config[key] as number}
        onChange={(e) =>
          setConfig((c) => ({ ...c, [key]: Number(e.target.value) }))
        }
      />
    </label>
  );
  return (
    <div className="app-shell">
      <header className="topbar">
        <a
          className="brand"
          href="/pretrain"
          onClick={(e) => {
            e.preventDefault();
            navigate("pretrain");
          }}
        >
          <span className="brand-mark">
            <Box size={22} />
          </span>
          <strong>
            Unit<span>变量单位工作台</span>
          </strong>
          <span className="local-pill">LOCAL</span>
        </a>
        <nav aria-label="训练阶段">
          <button
            className={page === "pretrain" ? "selected" : ""}
            onClick={() => navigate("pretrain")}
          >
            <span>01</span>预训练
          </button>
          <button
            className={page === "finetune" ? "selected" : ""}
            onClick={() => navigate("finetune")}
          >
            <span>02</span>后训练
          </button>
        </nav>
        <div className="top-status">
          <span className={`dot ${ready ? "green" : ""}`} />
          {ready ? "本地已连接" : "连接中"}
          <i />
          <span className="device-name">
            {system?.cuda_available
              ? system.devices[0]?.name
              : "CPU · 调试环境"}
          </span>
        </div>
      </header>
      <div className="page-intro">
        <div>
          <div className="eyebrow">
            THE LANGUAGE LAB /{" "}
            {page === "pretrain" ? "PRETRAINING" : "BOUNDARY LEARNING"}
          </div>
          <h1>{page === "pretrain" ? "先理解上下文。" : "让每个单位清晰。"}</h1>
          <p>
            {page === "pretrain"
              ? "从原始中文出发，训练属于你的字符级上下文模型。"
              : "保留每一个字符，学习你定义的分界方式。"}
          </p>
        </div>
        <div className="intro-note">
          <ShieldCheck size={17} />
          <span>
            数据与模型，留在本机
            <small>无云端调用 · 无遥测 · 可追溯检查点</small>
          </span>
        </div>
      </div>
      {(error || notice) && (
        <div
          className={`toast ${error ? "error" : ""}`}
          role={error ? "alert" : "status"}
        >
          <span>{error || notice}</span>
          <button
            aria-label="关闭提示"
            onClick={() => {
              setError("");
              setNotice("");
            }}
          >
            <X size={16} />
          </button>
        </div>
      )}
      <main className="workbench">
        <section className="panel data-panel">
          <div className="panel-heading">
            <div className="heading-icon">
              <Database size={18} />
            </div>
            <div>
              <h2>{page === "pretrain" ? "预训练内容" : "标注工作区"}</h2>
              <p>
                {page === "pretrain"
                  ? "为模型提供学习的起点"
                  : "只编辑边界，不丢失原文"}
              </p>
            </div>
            <span className="step-label">01 / DATA</span>
          </div>
          <div className="panel-body">
            <label className="field">
              数据集
              <select
                aria-label="选择数据集"
                value={datasetId}
                onChange={(e) => void loadDataset(e.target.value)}
              >
                <option value="">选择已有数据集，或创建新版本</option>
                {stageDatasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} · {d.count} 条
                  </option>
                ))}
              </select>
            </label>
            <div className="editor-top">
              <span>
                <FileText size={14} />
                {page === "pretrain" ? "原始文本" : "原文与单位边界"}
              </span>
              <span>UTF-8</span>
            </div>
            {page === "pretrain" ? (
              <>
                <textarea
                  className="corpus-editor"
                  aria-label="预训练内容"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder={
                    "在这里粘贴中文语料…\n\n可以是一段叙述、一篇文章，\n或你希望模型理解的领域知识。\n\n正文空白与标点将完整保留。"
                  }
                />
                <div className="editor-footer">
                  <span>{Array.from(text).length.toLocaleString()} 个字符</span>
                  <span>Unicode 码点</span>
                </div>
              </>
            ) : (
              <>
                <div className="sample-tabs">
                  {annotations.map((r, i) => (
                    <button
                      key={i}
                      className={i === annotationIndex ? "selected" : ""}
                      onClick={() => selectAnnotation(i)}
                      title={r.text}
                    >
                      样本 {annotationOffset + i + 1}
                      {r.conflict ? " · 冲突" : ""}
                    </button>
                  ))}
                  <button
                    aria-label="新增标注样本"
                    onClick={() =>
                      appendAnnotation({
                        text: "",
                        units: [],
                        status: "draft",
                        split: "train",
                        source: "manual",
                      })
                    }
                  >
                    <Plus size={14} />
                  </button>
                </div>
                <div className="annotation-pager">
                  <span>
                    {annotationOffset + 1}–
                    {annotationOffset + annotations.length} / {annotationTotal}{" "}
                    条
                  </span>
                  <button
                    disabled={
                      !!busy || annotationOffset === 0 || !annotationText
                    }
                    onClick={() => void saveData(annotationOffset - 50)}
                  >
                    保存并上一页
                  </button>
                  <button
                    disabled={
                      !!busy ||
                      annotationOffset + 50 >= annotationTotal ||
                      !annotationText
                    }
                    onClick={() => void saveData(annotationOffset + 50)}
                  >
                    保存并下一页
                  </button>
                </div>
                <textarea
                  aria-label="标注原文"
                  className="annotation-editor"
                  value={annotationText}
                  onChange={(e) => updateAnnotationText(e.target.value)}
                />
                <p className="help">
                  点击字符间隙切开 / 合并。绿色竖线表示分界。
                </p>
                <div className="gap-editor">
                  {Array.from(annotationText).map((char, i) => (
                    <span className="gap-part" key={i}>
                      <span className="char">
                        {char === " " ? "␣" : char === "\n" ? "↵" : char}
                      </span>
                      {i < Array.from(annotationText).length - 1 && (
                        <button
                          aria-label={`间隙 ${i + 1}${gaps[i] ? " 合并" : " 切开"}`}
                          className={gaps[i] ? "cut" : ""}
                          onClick={() => {
                            setGaps((g) => g.map((v, j) => (j === i ? !v : v)));
                            setConfirmed(false);
                          }}
                        >
                          {gaps[i] ? "│" : "·"}
                        </button>
                      )}
                    </span>
                  ))}
                </div>
                <div className="unit-flow">
                  {units.map((u, i) => (
                    <span key={i}>{u}</span>
                  ))}
                </div>
                <div className="valid-line">
                  <Check size={13} /> 原文还原一致 · {units.length} 个单位
                </div>
                <details className="small-details">
                  <summary>查看 units / spans / gap labels</summary>
                  <pre>
                    {JSON.stringify(
                      {
                        units,
                        spans: units.map((_, i) => [
                          Array.from(units.slice(0, i).join("")).length,
                          Array.from(units.slice(0, i + 1).join("")).length,
                        ]),
                        gap_labels: gaps.map(Number),
                      },
                      null,
                      2,
                    )}
                  </pre>
                </details>
                <label className="checkline">
                  <input
                    type="checkbox"
                    checked={confirmed}
                    onChange={(e) => setConfirmed(e.target.checked)}
                  />
                  我已审核当前样本边界，确认作为金标
                </label>
                <div className="subtle-note">
                  两条默认样例仅用于教学，不是充足训练集。新样本默认保存为草稿。
                </div>
              </>
            )}
            <div className="dataset-fields">
              <label className="field">
                版本名称
                <input
                  placeholder="例如：中文语料 v1"
                  value={datasetName}
                  onChange={(e) => setDatasetName(e.target.value)}
                />
              </label>
              <div className="form-grid">
                <label className="field">
                  数据划分
                  <select
                    value={split}
                    onChange={(e) => setSplit(e.target.value)}
                  >
                    <option value="train">训练集</option>
                    <option value="val">验证集</option>
                    <option value="test">测试集</option>
                  </select>
                </label>
                <label className="field">
                  来源分组
                  <input
                    value={group}
                    placeholder="文档 / 模板 ID"
                    onChange={(e) => setGroup(e.target.value)}
                  />
                </label>
              </div>
              {page === "finetune" && (
                <label className="field">
                  人工规范版本
                  <input
                    value={rule}
                    onChange={(e) => setRule(e.target.value)}
                  />
                </label>
              )}
            </div>
            <button
              className="button primary full"
              disabled={
                !!busy ||
                !ready ||
                (page === "pretrain" ? !text : !annotationText)
              }
              onClick={() => void saveData()}
            >
              <Save size={15} />
              {page === "pretrain"
                ? "保存为数据集"
                : confirmed
                  ? "保存已审核版本"
                  : "保存到待审核标注"}
            </button>
            <button
              className="button dashed full"
              disabled={!!busy || !ready}
              onClick={() => upload.current?.click()}
            >
              <Upload size={15} />
              导入 {page === "pretrain" ? "TXT / JSONL" : "标注 JSONL"}
              <span>≤ 64 MB</span>
            </button>
            <input
              ref={upload}
              type="file"
              hidden
              accept={page === "pretrain" ? ".txt,.jsonl" : ".jsonl"}
              onChange={(e) => {
                if (e.target.files?.[0]) void importFile(e.target.files[0]);
                e.target.value = "";
              }}
            />
            {dataset && (
              <div className="dataset-summary">
                <div>
                  <strong>{dataset.name}</strong>
                  <a
                    href={url(`/api/datasets/${dataset.id}/export`)}
                    aria-label="导出数据集"
                  >
                    <Download size={15} />
                  </a>
                </div>
                <div className="stats-three">
                  {[
                    ["训练", dataset.counts.train],
                    ["验证", dataset.counts.val],
                    ["已审核", dataset.confirmed_count],
                  ].map(([a, b]) => (
                    <span key={a}>
                      <b>{b}</b>
                      {a}
                    </span>
                  ))}
                </div>
                <small>
                  快照 {short(dataset.hash)} · {dataset.rule_version}
                </small>
                {dataset.warning && (
                  <p className="environment-note">{dataset.warning}</p>
                )}
              </div>
            )}
          </div>
          <div className="panel-foot">
            <ShieldCheck size={13} />
            原文完整覆盖，不裁剪、不去重
          </div>
        </section>
        <section className="panel training-panel">
          <div className="panel-heading">
            <div className="heading-icon">
              <Layers3 size={18} />
            </div>
            <div>
              <h2>{page === "pretrain" ? "模型与训练" : "边界学习"}</h2>
              <p>
                {page === "pretrain"
                  ? "从随机初始化，到上下文理解"
                  : "在预训练检查点上创建新分支"}
              </p>
            </div>
            <span className="step-label">02 / TRAIN</span>
          </div>
          <div className="panel-body">
            <div className="model-row">
              <label className="field grow">
                {page === "pretrain" ? "当前模型" : "预训练父检查点"}
                {page === "pretrain" ? (
                  <select
                    aria-label="选择模型"
                    value={modelId}
                    onChange={(e) => setModelId(e.target.value)}
                  >
                    <option value="">选择一个模型</option>
                    {models.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} · {fmt(m.parameter_count / 1e6, 2)}M
                      </option>
                    ))}
                  </select>
                ) : (
                  <select
                    aria-label="预训练父检查点"
                    value={parentId}
                    onChange={(e) => selectParent(e.target.value)}
                  >
                    <option value="">选择预训练检查点</option>
                    {cps
                      .filter((c) => c.task === "pretrain")
                      .map((c) => (
                        <option key={c.id} value={c.id}>
                          {short(c.id)} · step {c.global_step}
                        </option>
                      ))}
                  </select>
                )}
              </label>
              {page === "pretrain" && (
                <button
                  className="button compact"
                  disabled={!datasetId || !!busy}
                  onClick={() => setShowModel(true)}
                >
                  <Plus size={15} />
                  新建模型
                </button>
              )}
            </div>
            <div className="model-card">
              <div className="model-card-top">
                <span className="model-symbol">
                  <Box size={21} />
                </span>
                <div>
                  <strong>{model?.name || "等待新建训练模型"}</strong>
                  <span>
                    {model
                      ? `${model.config.num_layers} 层双向编码器 · 字符级 Transformer`
                      : "字符级双向编码器 + 掩码预测 + 边界分类"}
                  </span>
                </div>
                <span className="tag">
                  {model?.preset === "debug" ? "DEBUG" : "FROM SCRATCH"}
                </span>
              </div>
              <div className="model-specs">
                <div>
                  <b>
                    {model ? fmt(model.parameter_count / 1e6, 2) : "~100"}
                    <small>M</small>
                  </b>
                  <span>{model ? "实际参数量" : "默认参数规模"}</span>
                </div>
                <div>
                  <b>{model?.config.hidden_size || 768}</b>
                  <span>隐藏维度</span>
                </div>
                <div>
                  <b>{model?.vocab.length ?? "—"}</b>
                  <span>训练集字符词表</span>
                </div>
              </div>
              {model && (
                <details className="small-details">
                  <summary>结构与参数分布</summary>
                  {model.layers.map((l) => (
                    <div className="layer-line" key={l.name}>
                      <span>{l.name}</span>
                      <b>{fmt(l.parameters, 0)}</b>
                    </div>
                  ))}
                  <small>
                    验证集词表覆盖率：
                    {model.validation_coverage == null
                      ? "未评估"
                      : fmt(model.validation_coverage * 100) + "%"}
                  </small>
                </details>
              )}
            </div>
            <div className="section-title">
              <h3>训练观察</h3>
              <span
                className={`status-pill ${active(run?.status) ? "live" : ""}`}
              >
                <span className="dot" />
                {run ? STATUS[run.status] : "尚未开始"}
              </span>
            </div>
            <div className="chart-card">
              <div className="chart-heading">
                <strong>
                  {page === "pretrain" ? "MLM loss" : "边界 BCE loss"}
                </strong>
                <div>
                  <span className="legend train" />
                  训练
                  <span className="legend val" />
                  验证
                </div>
              </div>
              <Chart events={events} />
              <div className="metric-strip">
                <div>
                  <span>STEP</span>
                  <b>
                    {run?.global_step || 0}
                    <small>
                      {" "}
                      / {run?.config.max_steps || config.max_steps}
                    </small>
                  </b>
                </div>
                <div>
                  <span>TRAIN LOSS</span>
                  <b>{fmt(metrics.train_loss)}</b>
                </div>
                <div>
                  <span>VAL LOSS</span>
                  <b>
                    {metrics.val_loss == null
                      ? "未评估"
                      : fmt(metrics.val_loss)}
                  </b>
                </div>
              </div>
            </div>
            <div className="minor-metrics">
              <span>
                梯度范数 <b>{fmt(metrics.gradient_norm)}</b>
              </span>
              <span>
                字符 / 秒 <b>{fmt(metrics.chars_per_second, 0)}</b>
              </span>
              <span>
                耗时 <b>{fmt(metrics.elapsed_seconds, 1)} s</b>
              </span>
            </div>
            <details className="small-details">
              <summary>更多真实训练指标</summary>
              <Metrics metrics={metrics} stage={page} />
              <pre>
                {Object.keys(metrics).length
                  ? JSON.stringify(metrics, null, 2)
                  : "尚无训练事件。GPU 利用率不可用时不会用显存占用代替。"}
              </pre>
            </details>
            {run && (
              <small>
                任务规范 {run.rule_version} · 数据快照 {short(run.dataset_hash)}
              </small>
            )}
            <div className="section-title">
              <h3>
                <Settings2 size={15} />
                训练设置
              </h3>
              <span className="muted">以 optimizer update 计步</span>
            </div>
            <div className="form-grid">
              {field("learning_rate", "学习率", 0.0001)}
              {field("max_steps", "训练步数")}
              {field("sequence_length", "序列长度")}
              {field("micro_batch", "微批次大小")}
              <label className="field">
                训练设备
                <select
                  value={config.device}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      device: e.target.value as Config["device"],
                    }))
                  }
                >
                  <option value="cuda">CUDA GPU · 正式训练</option>
                  <option value="cpu">CPU · 仅调试小模型</option>
                </select>
              </label>
              {field("gradient_accumulation", "梯度累积")}
            </div>
            {system?.diagnostic && (
              <p className="environment-note">
                <CircleHelp size={14} />
                {system.diagnostic}
              </p>
            )}
            <details className="advanced">
              <summary>
                <ChevronDown size={14} />
                高级设置<span>精度、检查点、遮盖</span>
              </summary>
              <div className="form-grid">
                {field("save_every", "保存间隔")}
                {field("eval_every", "验证间隔")}
                {field("mask_ratio", "遮盖比例", 0.05)}
                {field("gradient_clip", "梯度裁剪", 0.1)}
                {field("seed", "随机种子")}
                <label className="field">
                  混合精度
                  <select
                    value={config.precision}
                    onChange={(e) =>
                      setConfig((c) => ({
                        ...c,
                        precision: e.target.value as Config["precision"],
                      }))
                    }
                  >
                    <option value="fp32">FP32</option>
                    <option value="fp16">FP16 · CUDA</option>
                    <option value="bf16">BF16 · 支持的 CUDA</option>
                  </select>
                </label>
              </div>
              <label className="checkline">
                <input
                  type="checkbox"
                  checked={config.activation_checkpointing}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      activation_checkpointing: e.target.checked,
                    }))
                  }
                />
                激活重计算，以计算时间换取显存
              </label>
            </details>
            {page === "finetune" && (
              <label className="checkline demo-check">
                <input
                  type="checkbox"
                  checked={demo}
                  onChange={(e) => setDemo(e.target.checked)}
                />
                教学过拟合模式（没有独立验证集，成绩不代表泛化）
              </label>
            )}
            <div className="training-actions">
              <button
                className="button primary"
                disabled={
                  !!busy ||
                  !modelId ||
                  !datasetId ||
                  runs.some((r) => active(r.status)) ||
                  (page === "finetune" && !parentId)
                }
                onClick={() => void startRun()}
              >
                {busy === "start" ? (
                  <LoaderCircle size={16} className="spin" />
                ) : (
                  <Play size={15} />
                )}
                开始训练
              </button>
              <button
                className="button"
                disabled={
                  !!busy ||
                  !run ||
                  !["running", "paused", "interrupted"].includes(run.status)
                }
                onClick={() =>
                  void act("control", async () => {
                    await api(
                      `/api/runs/${runId}/${run?.status === "running" ? "pause" : "resume"}`,
                      {},
                    );
                  })
                }
              >
                {run?.status === "paused" || run?.status === "interrupted" ? (
                  <Play size={14} />
                ) : (
                  <Pause size={14} />
                )}{" "}
                {run?.status === "paused" || run?.status === "interrupted"
                  ? "继续训练"
                  : "暂停并保存"}
              </button>
              <button
                className="button icon-only"
                aria-label="停止并保存"
                title="停止并保存"
                disabled={
                  !!busy ||
                  !run ||
                  !["running", "pausing", "paused", "interrupted"].includes(
                    run.status,
                  )
                }
                onClick={() =>
                  void act("stop", async () => {
                    await api(`/api/runs/${runId}/stop`, {});
                  })
                }
              >
                <Square size={14} />
              </button>
            </div>
            {run?.error && <p className="error-inline">{run.error}</p>}
            <div className="section-title">
              <h3>任务与检查点</h3>
              <span className="muted">持久化保存</span>
            </div>
            <select
              aria-label="历史训练任务"
              value={runId}
              onChange={(e) => selectRun(e.target.value)}
            >
              <option value="">选择历史训练任务</option>
              {stageRuns.map((r) => (
                <option key={r.id} value={r.id}>
                  {short(r.id)} · {STATUS[r.status]} · step {r.global_step}
                </option>
              ))}
            </select>
            <div className="checkpoint-list">
              {cps
                .filter((c) => c.run_id === runId)
                .slice(0, 8)
                .map((c) => (
                  <div className="checkpoint" key={c.id}>
                    <Save size={15} />
                    <div>
                      <strong>step {c.global_step}</strong>
                      <small>
                        {short(c.id)}{" "}
                        {run?.last_checkpoint_id === c.id ? "· LAST" : ""}{" "}
                        {run?.best_checkpoint_id === c.id ? "· BEST" : ""}
                      </small>
                    </div>
                    <button title="在右侧测试" onClick={() => setCpId(c.id)}>
                      <FlaskConical size={14} />
                    </button>
                    <a
                      title="导出安全推理权重"
                      href={url(`/api/checkpoints/${c.id}/export`)}
                    >
                      <Download size={14} />
                    </a>
                    {page === "pretrain" && (
                      <button
                        title="转到后训练"
                        onClick={() => {
                          selectParent(c.id);
                          navigate("finetune");
                        }}
                      >
                        <ArrowUpRight size={15} />
                      </button>
                    )}
                  </div>
                ))}
              {!cps.some((c) => c.run_id === runId) && (
                <div className="empty-checkpoint">
                  <Save size={16} />
                  完成一次保存后，检查点会出现在这里。
                </div>
              )}
            </div>
          </div>
          <div className="panel-foot">
            <span className="dot green" />
            新建模型与开始训练是独立操作
          </div>
        </section>
        <section className="panel test-panel">
          <div className="panel-heading">
            <div className="heading-icon">
              <MessageSquare size={18} />
            </div>
            <div>
              <h2>{page === "pretrain" ? "上下文填空" : "单位提取测试"}</h2>
              <p>
                {page === "pretrain"
                  ? "看看模型学到了什么"
                  : "用真实检查点检验你的边界"}
              </p>
            </div>
            <span className="step-label">03 / TEST</span>
          </div>
          <div className="panel-body test-body">
            <label className="field">
              测试检查点
              <select
                aria-label="测试检查点"
                value={cpId}
                onChange={(e) => {
                  setCpId(e.target.value);
                  setResult(null);
                }}
              >
                <option value="">选择已落盘的检查点</option>
                {stageCps.map((c) => (
                  <option key={c.id} value={c.id}>
                    step {c.global_step} · {short(c.id)}
                  </option>
                ))}
              </select>
            </label>
            <div className="inference-note">
              <span className="dot" />
              CPU 推理 · 与训练进程隔离
            </div>
            <div className="conversation" aria-live="polite">
              {!result ? (
                <div className="test-welcome">
                  <div className="welcome-symbol">
                    <FlaskConical size={26} />
                  </div>
                  <h3>
                    {page === "pretrain"
                      ? "给上下文留一个空。"
                      : "一句话，多个独立单位。"}
                  </h3>
                  <p>
                    {page === "pretrain"
                      ? "输入一句话，点击想要遮住的字符。\n模型会根据双向上下文预测候选字符。"
                      : "输入整句，模型将为每个间隙预测边界。\n所有字符都会按原顺序完整保留。"}
                  </p>
                  <div className="tiny-rule" />
                  <small>
                    {cpId
                      ? "已选择检查点，可以开始测试"
                      : "训练并保存检查点后，即可开始测试"}
                  </small>
                </div>
              ) : (
                <>
                  <div className="user-bubble">{testSent}</div>
                  <div className="assistant-label">
                    <span className="brand-mark small">
                      <Box size={12} />
                    </span>
                    UNIT MODEL<span>CPU</span>
                  </div>
                  <div className="model-bubble">
                    {page === "pretrain" ? (
                      result.predictions?.map((p: any) => (
                        <div className="prediction" key={p.position}>
                          <small>
                            位置 {p.position} · 原字符「
                            {Array.from(testSent)[p.position]}」
                          </small>
                          {p.candidates.map((c: any, i: number) => (
                            <div key={i}>
                              <b>{c.character}</b>
                              <span className="probability">
                                <i
                                  style={{ width: `${c.probability * 100}%` }}
                                />
                              </span>
                              <span>{fmt(c.probability * 100, 1)}%</span>
                            </div>
                          ))}
                        </div>
                      ))
                    ) : (
                      <>
                        <div className="unit-flow result-units">
                          {result.units?.map((u: string, i: number) => (
                            <span key={i}>{u}</span>
                          ))}
                        </div>
                        <div className="valid-line">
                          <Check size={13} />
                          原文还原：
                          {result.units?.join("") === testSent
                            ? "一致"
                            : "不一致"}
                        </div>
                        <button className="text-button" onClick={correct}>
                          纠正边界并加入草稿 <ArrowRight size={13} />
                        </button>
                      </>
                    )}
                    {result.warning && (
                      <p className="environment-note">{result.warning}</p>
                    )}
                    <div className="result-source">
                      检查点 {short(result.checkpoint_id)} · step{" "}
                      {result.global_step}
                      <br />
                      {fmt(result.latency_ms, 1)} ms · {result.device}
                    </div>
                    <details className="small-details">
                      <summary>查看完整 JSON 与概率</summary>
                      <pre>{JSON.stringify(result, null, 2)}</pre>
                    </details>
                  </div>
                </>
              )}
            </div>
            <div className="test-composer">
              <textarea
                aria-label="测试文本"
                placeholder={
                  page === "pretrain"
                    ? "输入用于填空测试的句子…"
                    : "输入需要提取单位的句子…"
                }
                value={testText}
                onChange={(e) => {
                  setTestText(e.target.value);
                  setMask([]);
                }}
              />
              {page === "pretrain" && testText && (
                <div className="mask-picker">
                  {Array.from(testText).map((c, i) => (
                    <button
                      title={`遮盖第 ${i + 1} 个字符`}
                      className={mask.includes(i) ? "masked" : ""}
                      key={i}
                      onClick={() =>
                        setMask((m) =>
                          m.includes(i) ? m.filter((p) => p !== i) : [...m, i],
                        )
                      }
                    >
                      {mask.includes(i) ? "□" : c === " " ? "␣" : c}
                    </button>
                  ))}
                </div>
              )}
              {page === "finetune" && (
                <label className="threshold">
                  分界阈值{" "}
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step=".05"
                    value={threshold}
                    onChange={(e) => setThreshold(Number(e.target.value))}
                  />
                  <b>{threshold.toFixed(2)}</b>
                </label>
              )}
              <div className="composer-foot">
                <span>
                  {page === "pretrain"
                    ? `已遮盖 ${mask.length} 个字符`
                    : "原文不会去重或删改"}
                </span>
                <button
                  className="send-button"
                  aria-label={
                    page === "pretrain" ? "预测候选字符" : "提取变量单位"
                  }
                  disabled={
                    !cpId ||
                    !testText ||
                    !!busy ||
                    (page === "pretrain" && !mask.length)
                  }
                  onClick={() => void test()}
                >
                  {busy === "test" ? (
                    <LoaderCircle size={17} className="spin" />
                  ) : (
                    <ArrowRight size={18} />
                  )}
                </button>
              </div>
            </div>
            <p className="test-disclaimer">
              {page === "pretrain"
                ? "这是上下文填空，不是开放聊天或下一字续写。"
                : "预测不是金标。纠错样本经人工审核后才能进入训练。"}
            </p>
          </div>
          <div className="panel-foot">
            <BookOpen size={13} />
            {page === "pretrain"
              ? "掩码位置按 Unicode 码点传递"
              : "保留标点、助词与重复单位"}
          </div>
        </section>
      </main>
      <footer className="app-footer">
        <span>
          UNIT WORKBENCH <i /> unit-v1
        </span>
        <span>每个字符都在，每次学习可见。</span>
        <button
          onClick={() =>
            void act("refresh", async () => {
              setSystem(await api("/api/system"));
            })
          }
        >
          <RefreshCw size={12} />
          刷新设备状态
        </button>
      </footer>
      {showModel && (
        <div className="modal-backdrop">
          <section
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="model-title"
          >
            <div className="section-title">
              <h2 id="model-title">新建训练模型</h2>
              <button
                aria-label="关闭新建模型"
                onClick={() => setShowModel(false)}
              >
                <X size={18} />
              </button>
            </div>
            <p>从当前数据集的训练部分构建字符词表；不会自动开始训练。</p>
            <label className="field">
              模型名称
              <input
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="例如：中文上下文 v1"
              />
            </label>
            <label className="field">
              说明
              <input
                value={modelDescription}
                onChange={(e) => setModelDescription(e.target.value)}
              />
            </label>
            <label className="field">
              模型规格
              <select
                aria-label="模型规格"
                value={preset}
                onChange={(e) => setPreset(e.target.value)}
              >
                <option value="100m">
                  约 100M · 14 层 × 768 · 正式 CUDA 训练
                </option>
                <option value="debug">
                  CPU 调试 · 2 层 × 64 · 不代表正式模型
                </option>
              </select>
            </label>
            <div className="subtle-note">
              实际参数量由后端计算，随训练集字符词表大小变化。随机种子：
              {config.seed}
            </div>
            <button
              className="button primary full"
              disabled={!!busy || !modelName}
              onClick={() => void createModel()}
            >
              <Plus size={15} />
              创建模型配置
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
