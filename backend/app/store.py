"""SQLite transactions coordinate all API processes and persistent worker events."""

import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
import psutil
from .data import now


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "metadata.sqlite"
        with self.connect() as con:
            con.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS entities(kind TEXT,id TEXT,body TEXT,PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,status TEXT,body TEXT);
                CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,body TEXT);
                CREATE INDEX IF NOT EXISTS events_run ON events(run_id,id);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT);
            """)
            con.execute(
                "INSERT OR IGNORE INTO settings VALUES (?,?)",
                ("token", secrets.token_urlsafe(32)),
            )
            self.token = con.execute(
                "SELECT value FROM settings WHERE key=?", ("token",)
            ).fetchone()[0]

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db, timeout=30)
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def put(self, kind, obj):
        with self.connect() as con:
            con.execute(
                "INSERT INTO entities VALUES (?,?,?)",
                (kind, obj["id"], json.dumps(obj, ensure_ascii=False)),
            )
        return obj

    def get(self, kind, id):
        with self.connect() as con:
            row = (
                con.execute("SELECT body FROM runs WHERE id=?", (id,)).fetchone()
                if kind == "runs"
                else con.execute(
                    "SELECT body FROM entities WHERE kind=? AND id=?", (kind, id)
                ).fetchone()
            )
        if not row:
            raise KeyError(f"{kind}/{id} 不存在")
        return json.loads(row[0])

    def list(self, kind):
        with self.connect() as con:
            rows = (
                con.execute("SELECT body FROM runs ORDER BY rowid DESC").fetchall()
                if kind == "runs"
                else con.execute(
                    "SELECT body FROM entities WHERE kind=? ORDER BY rowid DESC",
                    (kind,),
                ).fetchall()
            )
        return [json.loads(r[0]) for r in rows]

    @staticmethod
    def _free(con):
        if con.execute(
            "SELECT 1 FROM runs WHERE status IN ('queued','running','pausing','stopping') LIMIT 1"
        ).fetchone():
            raise Conflict("已有训练任务占用设备；请先暂停或停止")

    def create_run(self, obj):
        obj = dict(obj, created_at=now(), updated_at=now(), queued_at=time.time())
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            self._free(con)
            con.execute(
                "INSERT INTO runs VALUES (?,?,?)",
                (obj["id"], obj["status"], json.dumps(obj, ensure_ascii=False)),
            )
        return obj

    def update_run(self, id, allowed=None, expected=None, **changes):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT body FROM runs WHERE id=?", (id,)).fetchone()
            if not row:
                raise KeyError(id)
            obj = json.loads(row[0])
            if expected and any(obj.get(k) != v for k, v in expected.items()):
                raise Conflict("训练进程代次或状态已变化，拒绝过期更新")
            if allowed is not None and obj["status"] not in allowed:
                raise Conflict(f"当前状态 {obj['status']} 不允许此操作")
            obj.update(changes, updated_at=now())
            con.execute(
                "UPDATE runs SET status=?,body=? WHERE id=?",
                (obj["status"], json.dumps(obj, ensure_ascii=False), id),
            )
        return obj

    def claim_resume(self, id):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            self._free(con)
            row = con.execute("SELECT body FROM runs WHERE id=?", (id,)).fetchone()
            if not row:
                raise KeyError(id)
            obj = json.loads(row[0])
            if obj["status"] not in ("paused", "interrupted"):
                raise Conflict("只有暂停或中断任务可以继续")
            obj.update(
                status="queued",
                control=None,
                queued_at=time.time(),
                updated_at=now(),
                pid=None,
                pid_created=None,
                generation=uuid.uuid4().hex,
            )
            con.execute(
                "UPDATE runs SET status=?,body=? WHERE id=?",
                ("queued", json.dumps(obj, ensure_ascii=False), id),
            )
        return obj

    def finish_worker(self, id, generation, status, metrics):
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT body FROM runs WHERE id=?", (id,)).fetchone()
            if not row:
                raise KeyError(id)
            obj = json.loads(row[0])
            if obj.get("generation") != generation:
                raise Conflict("过期训练进程不能结束新任务")
            if obj.get("control") == "stop":
                status = "cancelled"
            obj.update(
                status=status,
                metrics=metrics,
                control=None,
                exit_code=0,
                updated_at=now(),
            )
            con.execute(
                "UPDATE runs SET status=?,body=? WHERE id=?",
                (status, json.dumps(obj, ensure_ascii=False), id),
            )
        return obj

    def event(self, run_id, event):
        obj = dict(event, run_id=run_id, timestamp=now())
        with self.connect() as con:
            cur = con.execute(
                "INSERT INTO events(run_id,body) VALUES (?,?)",
                (run_id, json.dumps(obj, ensure_ascii=False, allow_nan=False)),
            )
            obj["id"] = cur.lastrowid
        return obj

    def events(self, run_id, after=0):
        with self.connect() as con:
            rows = con.execute(
                "SELECT id,body FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT 1000",
                (run_id, after),
            ).fetchall()
        return [dict(json.loads(body), id=id) for id, body in rows]

    def reconcile(self):
        for run in self.list("runs"):
            if run["status"] not in ("queued", "running", "pausing", "stopping"):
                continue
            if run["status"] == "queued" and time.time() - run.get("queued_at", 0) < 60:
                continue
            alive = False
            try:
                proc = psutil.Process(run.get("pid") or -1)
                alive = (
                    abs(proc.create_time() - run.get("pid_created", 0)) < 1
                    and proc.is_running()
                )
            except (psutil.Error, ValueError, TypeError):
                pass
            if not alive:
                try:
                    self.update_run(
                        run["id"],
                        expected={
                            k: run.get(k) for k in ("generation", "pid", "status")
                        },
                        status="interrupted",
                        error="训练进程已退出；可从最后完整检查点继续",
                    )
                    self.event(run["id"], {"type": "state", "status": "interrupted"})
                except Conflict:
                    pass

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("禁止访问 workspace 之外的路径")
        return path
