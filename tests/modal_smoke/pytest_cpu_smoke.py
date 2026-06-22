from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any


def run_pytest_cpu() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-m",
            "not megatron",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "ok": completed.returncode == 0,
        "mode": "pytest-cpu",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

