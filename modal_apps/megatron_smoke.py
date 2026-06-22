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
import sys
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


def _common_image():
    packages = [
        "torch<3",
        "megatron-core",
        "torchmonarch>=0.5.0",
        "pytest>=8.0",
        "pytest-asyncio>=0.23",
    ]

    return (
        _base_image()
        .apt_install("git", "curl")
        .uv_pip_install(*packages)
        .env(
            {
                "PYTHONPATH": str(REMOTE_ROOT / "src"),
                "GANKER_ARTIFACT_ROOT": "/tmp/ganker-artifacts",
            }
        )
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
    )


core_image = _common_image()
app = modal.App("ganker-megatron-smoke")


def _add_remote_import_paths() -> None:
    for path in (REMOTE_ROOT / "tests", REMOTE_ROOT / "src", REMOTE_ROOT):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _run_remote_smoke(args: list[str]) -> dict[str, Any]:
    _add_remote_import_paths()
    import json

    from modal_smoke.common import result_to_json
    from modal_smoke.cli import run_from_argv

    return json.loads(result_to_json(run_from_argv(args)))


@app.function(gpu=GPU, image=core_image, timeout=60 * 60)
def run_core_remote(mode: str, script_args: list[str]) -> dict[str, Any]:
    return _run_remote_smoke(["--mode", mode, *script_args])


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

    if mode in {"env", "pytest-cpu", "megatron", "ganker"}:
        result = run_core_remote.remote(mode, script_args)
    else:
        raise ValueError(f"unknown mode: {mode}")

    print(result)
