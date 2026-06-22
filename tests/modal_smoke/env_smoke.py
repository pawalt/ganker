from __future__ import annotations

import platform
import sys
from typing import Any

from .common import package_version, run_command


def collect_env() -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "mode": "env",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "ganker": package_version("ganker"),
            "torch": package_version("torch"),
            "megatron-core": package_version("megatron-core"),
            "megatron-bridge": package_version("megatron-bridge"),
            "torchmonarch": package_version("torchmonarch"),
        },
        "nvidia_smi": run_command(["nvidia-smi"]),
    }

    try:
        import torch
    except Exception as exc:
        result["torch"] = {"imported": False, "error": repr(exc)}
        return result

    cuda_available = bool(torch.cuda.is_available())
    result["torch"] = {
        "imported": True,
        "version": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
    }

    result["imports"] = {}
    for module in (
        "megatron.core",
        "megatron.core.pipeline_parallel.schedules",
        "megatron.bridge",
    ):
        try:
            __import__(module)
        except Exception as exc:
            result["imports"][module] = {"ok": False, "error": repr(exc)}
        else:
            result["imports"][module] = {"ok": True}
    return result
