"""Modal entrypoint for Ganker/Megatron smoke tests.

Usage:

    source ~/.codex/modal.env
    modal run modal_apps/megatron_smoke.py --mode env
    modal run modal_apps/megatron_smoke.py --mode pytest-cpu
    modal run modal_apps/megatron_smoke.py --mode megatron
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/ganker")
PYTHON_VERSION = os.getenv("GANKER_MODAL_PYTHON", "3.12")
GPU = os.getenv("GANKER_MODAL_GPU", "L40S")
BASE_IMAGE = os.getenv("GANKER_MODAL_BASE_IMAGE", "")


def _base_image():
    if BASE_IMAGE:
        return modal.Image.from_registry(BASE_IMAGE, add_python=PYTHON_VERSION)
    return modal.Image.debian_slim(python_version=PYTHON_VERSION)


def _common_image(*, include_bridge: bool = False):
    packages = [
        "torch<3",
        "megatron-core",
        "torchmonarch>=0.5.0",
        "pytest>=8.0",
        "pytest-asyncio>=0.23",
    ]
    if include_bridge:
        packages.append("megatron-bridge")

    return (
        _base_image()
        .apt_install("git", "curl")
        .uv_pip_install(*packages)
        .add_local_dir(
            PROJECT_ROOT,
            remote_path=str(REMOTE_ROOT),
            ignore=[
                ".git",
                ".jj",
                ".venv",
                ".pytest_cache",
                ".ruff_cache",
                "__pycache__",
                ".local_artifacts",
            ],
        )
        .env(
            {
                "PYTHONPATH": str(REMOTE_ROOT / "src"),
                "GANKER_ARTIFACT_ROOT": "/tmp/ganker-artifacts",
            }
        )
    )


core_image = _common_image(include_bridge=False)
bridge_image = _common_image(include_bridge=True)
app = modal.App("ganker-megatron-smoke")


def _run_remote_script(args: list[str]) -> dict[str, Any]:
    command = [
        "python",
        str(REMOTE_ROOT / "scripts" / "megatron_bridge_smoke.py"),
        *args,
    ]
    completed = subprocess.run(
        command,
        cwd=str(REMOTE_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {
            "ok": False,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "command": command,
        }
    import json

    return json.loads(completed.stdout)


@app.function(gpu=GPU, image=core_image, timeout=60 * 60)
def run_core_remote(mode: str, script_args: list[str]) -> dict[str, Any]:
    return _run_remote_script(["--mode", mode, *script_args])


@app.function(gpu=GPU, image=bridge_image, timeout=60 * 60)
def run_bridge_remote(mode: str, script_args: list[str]) -> dict[str, Any]:
    return _run_remote_script(["--mode", mode, *script_args])


@app.local_entrypoint()
def main(
    mode: str = "env",
    num_steps: int = 1,
    sequence_length: int = 16,
    micro_batch_size: int = 1,
    hidden_size: int = 32,
    num_layers: int = 2,
    num_attention_heads: int = 4,
    vocab_size: int = 128,
    allow_cpu: bool = False,
):
    script_args = [
        "--num-steps",
        str(num_steps),
        "--sequence-length",
        str(sequence_length),
        "--micro-batch-size",
        str(micro_batch_size),
        "--hidden-size",
        str(hidden_size),
        "--num-layers",
        str(num_layers),
        "--num-attention-heads",
        str(num_attention_heads),
        "--vocab-size",
        str(vocab_size),
    ]
    if allow_cpu:
        script_args.append("--allow-cpu")

    if mode == "ganker":
        result = run_bridge_remote.remote(mode, script_args)
    elif mode in {"env", "pytest-cpu", "megatron"}:
        result = run_core_remote.remote(mode, script_args)
    else:
        raise ValueError(f"unknown mode: {mode}")

    print(result)
