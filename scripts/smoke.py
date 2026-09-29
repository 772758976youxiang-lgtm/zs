"""Real API -> subprocess -> checkpoint -> inference smoke, finite and local."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from backend.app.main import create_app

GOLD = [
    {
        "text": "我养了一只猫叫魔丸",
        "units": ["我", "养", "了", "一", "只", "猫", "叫", "魔丸"],
        "group_id": "gold-1",
        "status": "confirmed",
    },
    {
        "text": "我养了十二只猫，它们都很乖很可爱",
        "units": [
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
        "group_id": "gold-2",
        "status": "confirmed",
    },
]


def smoke(root, device="cpu"):
    with TestClient(create_app(root)) as c:
        c.headers["X-Local-Token"] = c.get("/api/session").json()["token"]

        def post(path, body):
            r = c.post(path, json=body)
            if r.status_code >= 400:
                raise AssertionError((path, r.status_code, r.text))
            return r.json()

        def wait(id, status="completed", timeout=180):
            end = time.time() + timeout
            while time.time() < end:
                r = c.get("/api/runs/" + id).json()
                if r["status"] == status:
                    return r
                if r["status"] in ("failed", "interrupted", "cancelled"):
                    raise AssertionError(r)
                time.sleep(0.1)
            raise AssertionError("timeout " + str(r))

        system = c.get("/api/system").json()
        if device == "cuda" and not system["cuda_available"]:
            raise SystemExit("GPU smoke 未执行：没有可用 CUDA GPU。")
        ds = post(
            "/api/datasets",
            {
                "name": "CPU smoke 中文语料",
                "records": [
                    {"text": r["text"], "group_id": r["group_id"]} for r in GOLD
                ],
            },
        )
        model = post(
            "/api/models",
            {
                "name": f"{device} smoke",
                "dataset_id": ds["id"],
                "preset": "100m" if device == "cuda" else "debug",
            },
        )
        pretrain_steps = 2 if device == "cuda" else 8
        finetune_steps = 2 if device == "cuda" else 12
        config = {
            "device": device,
            "max_steps": pretrain_steps,
            "save_every": 4,
            "eval_every": 4,
            "micro_batch": 1 if device == "cuda" else 2,
            "sequence_length": 32,
            "learning_rate": 0.001,
        }
        run = post(
            "/api/runs",
            {
                "model_id": model["id"],
                "dataset_id": ds["id"],
                "task": "pretrain",
                "config": config,
            },
        )
        run = wait(run["id"])
        cp = run["last_checkpoint_id"]
        mlm = post(
            "/api/inference/mlm",
            {"checkpoint_id": cp, "text": GOLD[0]["text"], "mask_positions": [5]},
        )
        assert mlm["predictions"] and mlm["global_step"] == pretrain_steps
        anno = post(
            "/api/datasets",
            {"name": "两条金标教学", "kind": "finetune", "records": GOLD},
        )
        fine = post(
            "/api/runs",
            {
                "model_id": model["id"],
                "dataset_id": anno["id"],
                "task": "finetune",
                "parent_checkpoint_id": cp,
                "demo_overfit": True,
                "config": dict(config, max_steps=finetune_steps),
            },
        )
        fine = wait(fine["id"])
        units = post(
            "/api/inference/units",
            {"checkpoint_id": fine["last_checkpoint_id"], "text": GOLD[1]["text"]},
        )
        assert "".join(units["units"]) == GOLD[1]["text"]
        assert len(units["boundary_probabilities"]) == len(GOLD[1]["text"]) - 1
        longrun = post(
            "/api/runs",
            {
                "model_id": model["id"],
                "dataset_id": ds["id"],
                "task": "pretrain",
                "config": dict(config, max_steps=1000),
            },
        )
        wait(longrun["id"], "running")
        conflict = c.post(
            "/api/runs",
            json={
                "model_id": model["id"],
                "dataset_id": ds["id"],
                "task": "pretrain",
                "config": config,
            },
        )
        assert conflict.status_code == 409
        post(f"/api/runs/{longrun['id']}/pause", {})
        paused = wait(longrun["id"], "paused")
        assert paused["last_checkpoint_id"]
        post(f"/api/runs/{longrun['id']}/resume", {})
        wait(longrun["id"], "running")
        post(f"/api/runs/{longrun['id']}/stop", {})
        stopped = wait(longrun["id"], "cancelled")
        events = c.get(f"/api/runs/{run['id']}/events?stream=false").json()
        replay = c.get(
            f"/api/runs/{run['id']}/events?stream=false&after={events[0]['id']}"
        ).json()
        assert len(replay) == len(events) - 1
        exported = c.get(f"/api/checkpoints/{cp}/export")
        assert exported.status_code == 200 and exported.content[:2] == b"PK"
        return {
            "system": system,
            "pretrain": run,
            "finetune": fine,
            "mlm": mlm,
            "units": units,
            "pause_step": paused["global_step"],
            "stop_step": stopped["global_step"],
            "concurrent_start_http": conflict.status_code,
            "sse_replayed": len(replay),
            "export_bytes": len(exported.content),
            "note": "CPU debug smoke only; no independent validation dataset; no GPU or generalization claim.",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--workspace", default="workspace/smoke")
    parser.add_argument("--output", default="docs/smoke-results.json")
    args = parser.parse_args()
    result = smoke(Path(args.workspace).resolve(), args.device)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output": args.output,
                "pretrain_step": result["pretrain"]["global_step"],
                "finetune_step": result["finetune"]["global_step"],
                "pause_step": result["pause_step"],
                "stop_step": result["stop_step"],
            },
            ensure_ascii=False,
        )
    )
