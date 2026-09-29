import platform
import shutil
import subprocess
import torch
import psutil


def system_info():
    cuda = torch.cuda.is_available()
    devices = []
    if cuda:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            devices.append(
                {"name": p.name, "total_bytes": total, "free_bytes": free, "index": i}
            )
    driver = None
    command = shutil.which("nvidia-smi")
    if command:
        try:
            driver = subprocess.run(
                [command, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "devices": devices,
        "driver": driver,
        "ram_available_bytes": psutil.virtual_memory().available,
        "precision": {
            "fp32": True,
            "fp16": cuda,
            "bf16": cuda and torch.cuda.is_bf16_supported(),
        },
        "diagnostic": None
        if cuda
        else "未检测到可用 CUDA GPU。正式模型禁止在 CPU 启动；可显式选择 CPU 调试小模型。",
        "inference_device": "cpu",
    }
