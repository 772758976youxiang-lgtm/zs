"""Executed as an independent subprocess, never within an HTTP request loop."""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import psutil
from .store import Store, Conflict
from .data import digest
from backend.training.engine import run_training
from filelock import FileLock
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generation", required=True)
    args = parser.parse_args()
    store = Store(args.workspace)
    run = store.get("runs", args.run_id)
    try:
        store.update_run(
            run["id"],
            allowed=["queued"],
            expected={"generation": args.generation},
            status="running",
            pid=os.getpid(),
            pid_created=psutil.Process().create_time(),
        )
        store.event(
            run["id"], {"type": "state", "status": "running", "stage": run["task"]}
        )
        spec = json.loads(store.path(run["spec_path"]).read_text(encoding="utf-8"))
        if digest(spec) != run["spec_hash"]:
            raise ValueError("任务快照校验失败，不能继续训练")
        if run.get("last_checkpoint_id"):
            checkpoint = store.get("checkpoints", run["last_checkpoint_id"])
            spec["resume_path"] = str(store.path(checkpoint["path"]))

        def on_event(event):
            event = dict(event, stage=run["task"])
            if event["type"] == "checkpoint":
                path = Path(event["path"]).resolve()
                relative = str(path.relative_to(store.root))
                id = event.get("checkpoint_id") or f"{run['id']}-{event['global_step']}"
                item = dict(
                    event,
                    id=id,
                    run_id=run["id"],
                    path=relative,
                    model_id=run["model_id"],
                    dataset_id=run["dataset_id"],
                )
                try:
                    store.put("checkpoints", item)
                except Exception:
                    # A repeated callback must refer to the very same immutable checkpoint.
                    if store.get("checkpoints", id)["path"] != relative:
                        raise
                updates = {
                    "last_checkpoint_id": id,
                    "global_step": event["global_step"],
                }
                if event.get("is_best"):
                    updates["best_checkpoint_id"] = id
                store.update_run(
                    run["id"], expected={"generation": args.generation}, **updates
                )
                event["checkpoint_id"] = id
                event.pop("path", None)
            if event["type"] == "metrics":
                store.update_run(
                    run["id"],
                    expected={"generation": args.generation},
                    metrics=event,
                    global_step=event["global_step"],
                )
            store.event(run["id"], event)

        def control():
            return store.get("runs", run["id"]).get("control")

        with FileLock(str(store.root / "training.lock"), timeout=30):
            result = run_training(spec, on_event, control)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        final = store.finish_worker(
            run["id"], args.generation, result["status"], result.get("metrics", {})
        )
        store.event(
            run["id"],
            {
                "type": "state",
                "status": final["status"],
                "global_step": result["global_step"],
            },
        )
    except BaseException as exc:
        diagnostic = str(exc)
        if "out of memory" in diagnostic.lower():
            diagnostic += "；显存不足，请新建分支降低微批次/序列长度或开启激活重计算。"
        try:
            store.update_run(
                run["id"],
                expected={"generation": args.generation},
                status="failed",
                error=diagnostic,
                exit_code=1,
                control=None,
            )
            store.event(run["id"], {"type": "error", "error": diagnostic})
        except Conflict:
            pass
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
