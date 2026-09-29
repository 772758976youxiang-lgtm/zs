import asyncio
from contextlib import asynccontextmanager
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile

from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from filelock import FileLock

from .data import prepare_records, build_vocab, digest, now
from .schemas import (
    DatasetCreate,
    ModelCreate,
    ModelConfig,
    RunCreate,
    MLMRequest,
    UnitsRequest,
    DatasetInfo,
    ModelInfo,
    RunInfo,
    CheckpointInfo,
)
from .store import Store, Conflict
from .system import system_info
from backend.training.engine import (
    inspect_model,
    infer_mlm,
    infer_units,
    read_checkpoint,
)

ROOT = Path(__file__).resolve().parents[2]
ORIGINS = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]


def create_app(workspace=None):
    store = Store(workspace or os.environ.get("UNIT_WORKSPACE") or ROOT / "workspace")

    @asynccontextmanager
    async def lifespan(app):
        store.reconcile()
        yield

    app = FastAPI(title="变量单位训练工作台", version="1.0.0", lifespan=lifespan)
    app.state.store = store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Local-Token", "Last-Event-ID"],
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )

    @app.middleware("http")
    async def local_session(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in ORIGINS:
            return JSONResponse(
                {"error": {"code": "origin_denied", "message": "仅允许本地工作台来源"}},
                status_code=403,
            )
        if request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse(
                {"error": {"code": "origin_denied", "message": "禁止跨站请求"}},
                status_code=403,
            )
        if request.method != "OPTIONS" and request.url.path not in (
            "/api/session",
            "/api/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ):
            token = request.headers.get("x-local-token") or request.query_params.get(
                "token", ""
            )
            if not secrets.compare_digest(token, store.token):
                return JSONResponse(
                    {
                        "error": {
                            "code": "local_session_required",
                            "message": "本地会话已失效，请刷新页面",
                        }
                    },
                    status_code=403,
                )
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 65 * 1024 * 1024:
            return JSONResponse(
                {"error": {"code": "file_too_large", "message": "导入上限 64 MiB"}},
                status_code=413,
            )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def schema_error(request, exc):
        return JSONResponse(
            {"error": {"code": "validation_error", "message": str(exc)}},
            status_code=422,
        )

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse(
            {
                "error": {
                    "code": "conflict"
                    if isinstance(exc, Conflict)
                    else "validation_error",
                    "message": str(exc),
                }
            },
            status_code=409 if isinstance(exc, Conflict) else 422,
        )

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse(
            {"error": {"code": "not_found", "message": str(exc)}}, status_code=404
        )

    def records_for(dataset):
        path = store.path(dataset["path"])
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if digest(records) != dataset["hash"]:
            raise ValueError("数据快照校验失败")
        return records

    def save_dataset(payload, imported=None):
        raw_records = payload.records
        if payload.base_dataset_id:
            base = store.get("datasets", payload.base_dataset_id)
            if base["kind"] != payload.kind:
                raise ValueError("数据版本类型不兼容")
            previous = records_for(base)
            replacements = {r["id"]: r for r in raw_records if r.get("id")}
            existing = {r["id"] for r in previous}
            if set(replacements) - existing:
                raise ValueError("编辑样本不属于原始数据版本")
            raw_records = [replacements.get(r["id"], r) for r in previous] + [
                r for r in raw_records if not r.get("id")
            ]
        records = prepare_records(raw_records, payload.kind, payload.rule_version)
        id = uuid.uuid4().hex
        path = store.path(f"datasets/{id}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        offsets = []
        with path.with_suffix(".tmp").open("wb") as f:
            for record in records:
                offsets.append(f.tell())
                f.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
        path.with_suffix(".tmp").replace(path)
        path.with_suffix(".index.json").write_text(
            json.dumps(offsets), encoding="utf-8"
        )
        vocab = build_vocab(records)
        return store.put(
            "datasets",
            {
                "id": id,
                "name": payload.name,
                "kind": payload.kind,
                "rule_version": payload.rule_version,
                "path": str(path.relative_to(store.root)),
                "hash": digest(records),
                "created_at": now(),
                "count": len(records),
                "counts": {
                    split: sum(r["split"] == split for r in records)
                    for split in ("train", "val", "test")
                },
                "confirmed_count": sum(r["status"] == "confirmed" for r in records),
                "characters": sum(len(r["text"]) for r in records),
                "vocab_size": len(vocab),
                "import": imported,
                "warning": (
                    "含冲突草稿：请审核并解决边界分歧；草稿不会进入训练。"
                    if any(r.get("conflict") for r in records)
                    else (
                        "仅教学/演示数据，不代表足够训练集"
                        if len(records) <= 2
                        else None
                    )
                ),
            },
        )

    def checkpoint(id):
        cp = store.get("checkpoints", id)
        path = store.path(cp["path"])
        try:
            read_checkpoint(path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "检查点文件不完整或元数据损坏，请使用最后一个校验完整的检查点"
            ) from exc
        return cp, path

    def launch(run):
        log_path = store.path(f"runs/{run['id']}/worker.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log:
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "backend.app.worker",
                        "--workspace",
                        str(store.root),
                        "--run-id",
                        run["id"],
                        "--generation",
                        run["generation"],
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    env=dict(os.environ, PYTHONUTF8="1"),
                )
            except OSError as exc:
                store.update_run(
                    run["id"], status="failed", error=str(exc), exit_code=-1
                )
                raise ValueError(f"训练进程启动失败：{exc}")

        def reap():
            code = process.wait()
            current = store.get("runs", run["id"])
            if (
                current["status"] in ("queued", "running", "pausing", "stopping")
                and current.get("generation") == run["generation"]
            ):
                try:
                    store.update_run(
                        run["id"],
                        expected={
                            "generation": run["generation"],
                            "status": current["status"],
                        },
                        status="failed",
                        error="训练进程异常退出；检查 worker.log",
                        exit_code=code,
                    )
                    store.event(
                        run["id"],
                        {
                            "type": "error",
                            "error": "训练进程异常退出",
                            "exit_code": code,
                        },
                    )
                except Conflict:
                    pass

        threading.Thread(target=reap, daemon=True).start()

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": "unit-v1"}

    @app.get("/api/session")
    def session():
        return {"token": store.token}

    @app.get("/api/system")
    def system():
        return system_info()

    @app.get("/api/datasets", response_model=list[DatasetInfo])
    def datasets():
        return store.list("datasets")

    @app.post("/api/datasets", response_model=DatasetInfo)
    def create_dataset(payload: DatasetCreate):
        return save_dataset(payload)

    @app.post("/api/datasets/validate")
    def validate_dataset(payload: DatasetCreate):
        return {
            "valid": True,
            "records": prepare_records(
                payload.records, payload.kind, payload.rule_version
            ),
        }

    @app.post("/api/datasets/import", response_model=DatasetInfo)
    async def import_dataset(
        file: UploadFile = File(...),
        name: str = Form(...),
        kind: str = Form("pretrain"),
        rule_version: str = Form("unit-v1"),
        split: str = Form("train"),
    ):
        suffix = Path(file.filename or "").suffix.lower()
        if (
            suffix not in (".txt", ".jsonl")
            or kind == "finetune"
            and suffix != ".jsonl"
        ):
            raise ValueError("只接受 UTF-8 TXT / JSONL；标注数据必须为 JSONL")
        total = 0
        with tempfile.TemporaryFile() as temp:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 64 * 1024 * 1024:
                    raise ValueError("导入上限 64 MiB，请按文档分片")
                temp.write(chunk)
            temp.seek(0)
            bom = temp.read(3) == b"\xef\xbb\xbf"
            temp.seek(0)
            text_file = io.TextIOWrapper(temp, encoding="utf-8-sig", newline="")
            try:
                if suffix == ".txt":
                    records = [
                        {
                            "text": text_file.read(),
                            "source": file.filename,
                            "split": split,
                        }
                    ]
                else:
                    records = []
                    for line in text_file:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("JSONL 每行必须为对象")
                        record.setdefault("source", file.filename)
                        record.setdefault("split", split)
                        records.append(record)
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"UTF-8 / JSONL 解析失败：{exc}") from exc
            finally:
                text_file.detach()
        return save_dataset(
            DatasetCreate(
                name=name, kind=kind, rule_version=rule_version, records=records
            ),
            {"filename": file.filename, "bom_removed": bom, "bytes": total},
        )

    @app.get("/api/datasets/{id}/records")
    def dataset_records(
        id: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)
    ):
        ds = store.get("datasets", id)
        path = store.path(ds["path"])
        offsets = json.loads(path.with_suffix(".index.json").read_text())
        rows = []
        with path.open("rb") as f:
            for pos in offsets[offset : offset + limit]:
                f.seek(pos)
                rows.append(json.loads(f.readline()))
        return {"records": rows, "total": ds["count"]}

    @app.get("/api/datasets/{id}/export")
    def dataset_export(id: str):
        ds = store.get("datasets", id)
        return FileResponse(
            store.path(ds["path"]),
            filename=f"{id}.jsonl",
            media_type="application/x-ndjson",
        )

    @app.post("/api/models", response_model=ModelInfo)
    def create_model(payload: ModelCreate):
        ds = store.get("datasets", payload.dataset_id)
        records = records_for(ds)
        vocab = build_vocab(records)
        if not vocab:
            raise ValueError("请先提供已确认的训练集，再建立词表")
        config = payload.config or (
            ModelConfig(
                hidden_size=64,
                num_layers=2,
                num_heads=4,
                intermediate_size=128,
                max_length=256,
            )
            if payload.preset == "debug"
            else ModelConfig()
        )
        info = inspect_model(config.model_dump(), len(vocab))
        val_text = "".join(r["text"] for r in records if r["split"] == "val")
        return store.put(
            "models",
            dict(
                payload.model_dump(),
                id=uuid.uuid4().hex,
                config=config.model_dump(),
                vocab=vocab,
                vocab_hash=digest(vocab),
                created_at=now(),
                **info,
                validation_coverage=sum(c in set(vocab) for c in val_text)
                / len(val_text)
                if val_text
                else None,
            ),
        )

    @app.get("/api/models", response_model=list[ModelInfo])
    def models():
        return store.list("models")

    @app.post("/api/runs", response_model=RunInfo)
    def create_run(payload: RunCreate):
        store.reconcile()
        model = store.get("models", payload.model_id)
        ds = store.get("datasets", payload.dataset_id)
        if payload.task != ds["kind"]:
            raise ValueError("数据集类型与训练阶段不符")
        if payload.config.device == "cpu" and (
            model["preset"] != "debug" or model["parameter_count"] > 5000000
        ):
            raise ValueError("正式模型需要 CUDA GPU；CPU 仅允许显式 debug 小模型")
        system = system_info()
        if payload.config.device == "cuda" and not system["cuda_available"]:
            raise ValueError(system["diagnostic"])
        if payload.config.device == "cpu" and payload.config.precision != "fp32":
            raise ValueError("CPU 调试只支持 fp32")
        if not system["precision"][payload.config.precision]:
            raise ValueError("当前设备不支持所选精度")
        if payload.config.sequence_length > model["config"]["max_length"]:
            raise ValueError("序列长度超过模型位置容量")
        records = [r for r in records_for(ds) if r["status"] == "confirmed"]
        if not any(r["split"] == "train" for r in records):
            raise ValueError("没有已确认的训练样本；草稿不可用于训练")
        parent_path = None
        if payload.task == "finetune":
            if not payload.parent_checkpoint_id:
                raise ValueError("后训练必须选择预训练检查点")
            cp, path = checkpoint(payload.parent_checkpoint_id)
            if cp["model_id"] != model["id"] or cp["task"] != "pretrain":
                raise ValueError("父检查点必须是此模型的预训练检查点")
            parent_path = str(path)
            if (
                not any(r["split"] == "val" for r in records)
                and not payload.demo_overfit
            ):
                raise ValueError("无独立验证集；仅可显式开启 demo overfit 教学模式")
            long = next(
                (
                    len(r["text"])
                    for r in records
                    if len(r["text"]) > payload.config.sequence_length
                ),
                None,
            )
            if long:
                raise ValueError(
                    f"标注样本长度 {long} 超过 {payload.config.sequence_length}；请按已知单位边界分块，分块会损失上下文"
                )
        elif payload.parent_checkpoint_id:
            raise ValueError("预训练新任务不接受后训练父检查点")
        if payload.task == "pretrain" and build_vocab(records) != model["vocab"]:
            raise ValueError("数据集训练字符与模型词表不一致，请新建模型")
        id = uuid.uuid4().hex
        spec = dict(
            run_id=id,
            task=payload.task,
            config=payload.config.model_dump(),
            model_config=model["config"],
            records=records,
            vocab=model["vocab"],
            output_dir=str(store.path(f"runs/{id}/checkpoints")),
            resume_path=None,
            parent_path=parent_path,
            dataset_hash=digest(records),
            rule_version=ds["rule_version"],
        )
        spec_path = store.path(f"runs/{id}/spec.json")
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        run = store.create_run(
            dict(
                payload.model_dump(),
                id=id,
                status="queued",
                global_step=0,
                metrics={},
                error=None,
                spec_path=str(spec_path.relative_to(store.root)),
                spec_hash=digest(spec),
                rule_version=ds["rule_version"],
                dataset_hash=spec["dataset_hash"],
                control=None,
                generation=uuid.uuid4().hex,
            )
        )
        store.event(id, {"type": "state", "status": "queued", "stage": payload.task})
        launch(run)
        return run

    @app.get("/api/runs", response_model=list[RunInfo])
    def runs():
        store.reconcile()
        return store.list("runs")

    @app.get("/api/runs/{id}", response_model=RunInfo)
    def get_run(id: str):
        store.reconcile()
        return store.get("runs", id)

    @app.post("/api/runs/{id}/pause")
    def pause(id: str):
        run = store.get("runs", id)
        if run["status"] in ("paused", "pausing"):
            return run
        return store.update_run(
            id, allowed=["running"], status="pausing", control="pause"
        )

    @app.post("/api/runs/{id}/stop")
    def stop(id: str):
        run = store.get("runs", id)
        if run["status"] in ("cancelled", "completed", "stopping"):
            return run
        if run["status"] in ("paused", "interrupted"):
            return store.update_run(
                id,
                expected={"generation": run.get("generation"), "status": run["status"]},
                status="cancelled",
                control=None,
            )
        return store.update_run(
            id, allowed=["running", "pausing"], status="stopping", control="stop"
        )

    @app.post("/api/runs/{id}/resume")
    def resume(id: str):
        run = store.get("runs", id)
        if not run.get("last_checkpoint_id"):
            raise ValueError("没有完整检查点可用于继续训练")
        checkpoint(run["last_checkpoint_id"])
        launch(store.claim_resume(id))
        return store.get("runs", id)

    @app.get("/api/runs/{id}/events")
    async def events(
        id: str, request: Request, after: int = Query(0, ge=0), stream: bool = True
    ):
        store.get("runs", id)
        after = max(after, int(request.headers.get("last-event-id", "0")))
        if not stream:
            return store.events(id, after)

        async def generate():
            cursor = after
            while not await request.is_disconnected():
                batch = store.events(id, cursor)
                for event in batch:
                    cursor = event["id"]
                    yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                if not batch:
                    yield ": keepalive\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/runs/{id}/checkpoints", response_model=list[CheckpointInfo])
    def checkpoints(id: str):
        store.get("runs", id)
        return [c for c in store.list("checkpoints") if c["run_id"] == id]

    @app.get("/api/checkpoints", response_model=list[CheckpointInfo])
    def all_checkpoints():
        return store.list("checkpoints")

    @app.get("/api/checkpoints/{id}/export")
    def export(id: str):
        cp, path = checkpoint(id)
        target = store.path(f"exports/{id}.zip")
        target.parent.mkdir(exist_ok=True)
        with FileLock(str(target) + ".lock"):
            if not target.exists():
                temporary = target.with_suffix(".tmp")
                with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
                    for name in ("model.safetensors", "metadata.json"):
                        z.write(path / name, name)
                    z.writestr(
                        "README.txt",
                        "Inference-only export: safe tensor weights and model/vocabulary metadata. Training state intentionally excluded. This application only resumes its own verified local checkpoints.",
                    )
                temporary.replace(target)
        return FileResponse(
            target, filename=f"{id}-inference.zip", media_type="application/zip"
        )

    @app.post("/api/inference/mlm")
    def mlm(payload: MLMRequest):
        _, path = checkpoint(payload.checkpoint_id)
        return dict(
            infer_mlm(path, payload.text, payload.mask_positions, payload.top_k),
            checkpoint_id=payload.checkpoint_id,
        )

    @app.post("/api/inference/units")
    def units(payload: UnitsRequest):
        _, path = checkpoint(payload.checkpoint_id)
        return dict(
            infer_units(path, payload.text, payload.threshold),
            checkpoint_id=payload.checkpoint_id,
        )

    return app


app = create_app()
