"""Reproducible local checks; writes real outputs rather than a hand-written pass list."""

import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
node = Path(shutil.which("node") or "")
npm = node.parent / "node_modules/npm/bin/npm-cli.js"
if not npm.exists():
    raise SystemExit(
        "找不到 Node 的 npm-cli.js，请先手动运行 README 中的前端测试命令。"
    )
commands = [
    ("backend-tests", [sys.executable, "-m", "pytest", "backend/tests", "-q"]),
    (
        "python-format",
        [sys.executable, "-m", "ruff", "format", "--check", "backend", "scripts"],
    ),
    ("frontend-tests", [str(node), str(npm), "--prefix", "frontend", "test"]),
    ("frontend-build", [str(node), str(npm), "--prefix", "frontend", "run", "build"]),
    ("cpu-smoke", [sys.executable, "scripts/smoke.py", "--device", "cpu"]),
]
results = []
(root / "docs/verification").mkdir(parents=True, exist_ok=True)
for name, command in commands:
    process = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ, PYTHONUTF8="1"),
        timeout=240,
    )
    output = process.stdout + process.stderr
    (root / f"docs/verification/{name}.txt").write_text(output, encoding="utf-8")
    results.append(
        {
            "name": name,
            "exit_code": process.returncode,
            "output": f"docs/verification/{name}.txt",
        }
    )
    print(f"{name}: {'PASS' if process.returncode == 0 else 'FAIL'}", flush=True)
    if process.returncode:
        print(output, flush=True)
report = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "checks": results,
    "gpu": "not executed: no compatible NVIDIA CUDA GPU detected",
    "success": all(r["exit_code"] == 0 for r in results),
}
(root / "docs/verification.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
sys.exit(0 if report["success"] else 1)
