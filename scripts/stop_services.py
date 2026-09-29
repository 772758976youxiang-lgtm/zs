"""Stop only this project's UI/API processes; independent training workers survive."""

import json
from pathlib import Path
import psutil

root = Path(__file__).resolve().parents[1]
registry = root / "workspace/services.json"
if not registry.exists():
    raise SystemExit("没有找到本项目启动脚本的服务记录。")
targets = []
for pid in json.loads(registry.read_text(encoding="utf-8-sig")).values():
    try:
        process = psutil.Process(pid)
        for candidate in process.children(recursive=True) + [process]:
            args = candidate.cmdline()
            cwd = Path(candidate.cwd()).resolve()
            service = ("uvicorn" in args and "backend.app.main:app" in args) or any(
                "vite/bin/vite.js" in arg.replace("\\", "/") for arg in args
            )
            if service and cwd in (root, root / "frontend"):
                targets.append(candidate)
    except psutil.Error:
        pass
for process in targets:
    try:
        process.terminate()
        print(f"Stopped UI/API service PID {process.pid}")
    except psutil.Error:
        pass
print("仅停止前后端服务。训练任务请在界面中暂停或停止。")
