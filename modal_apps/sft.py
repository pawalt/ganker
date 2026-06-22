"""Modal entrypoint for toy SFT over the public Ganker API.

Usage:

    source ~/.codex/modal.env
    modal run modal_apps/sft.py --mode env
    modal run modal_apps/sft.py --mode toy-sft
"""

from __future__ import annotations

import json
import importlib
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


image = _common_image()
app = modal.App("ganker-sft")


def _add_remote_import_paths() -> None:
    for path in (REMOTE_ROOT, REMOTE_ROOT / "tests", REMOTE_ROOT / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True))


def _run_env() -> dict[str, Any]:
    _add_remote_import_paths()
    collect_env = importlib.import_module("modal_smoke.env_smoke").collect_env

    return _json_safe(collect_env())


def _run_toy_sft(
    *,
    dataset_path: str,
    artifact_root: str,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    vocab_size: int,
    seed: int,
) -> dict[str, Any]:
    _add_remote_import_paths()

    from examples.sft import SFTDataConfig, ToyTokenizer, load_jsonl_sft_batches, run_sft
    from ganker import ServiceClient

    tokenizer = ToyTokenizer(vocab_size=vocab_size)
    batches = load_jsonl_sft_batches(
        dataset_path,
        tokenizer=tokenizer,
        config=SFTDataConfig(
            sequence_length=sequence_length,
            batch_size=micro_batch_size,
            shuffle=True,
            seed=seed,
        ),
    )

    client = None
    try:
        client = ServiceClient.local(
            Path(artifact_root),
            training_backend="megatron",
            training_backend_config={
                "runtime_kind": "core",
                "tensor_device": "cuda",
                "micro_batch_size": micro_batch_size,
                "global_batch_size": micro_batch_size,
                "sequence_length": sequence_length,
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_attention_heads": num_attention_heads,
                "seed": seed,
                "load_weights": False,
            },
            timeout=60,
        )
        summary = run_sft(
            client,
            base_model="local/tiny-config",
            dataset=batches,
            tuning="full",
            lora_rank=0,
            learning_rate=learning_rate,
            max_steps=max_steps,
            save_every=save_every,
        )
        payload = {
            "ok": True,
            "mode": "toy-sft",
            "runtime_kind": "core",
            "dataset_path": dataset_path,
            "batch_count": len(batches),
            **summary.to_dict(),
        }
        payload["artifact_exists"] = Path(summary.artifact_path).exists()
        return _json_safe(payload)
    finally:
        if client is not None:
            client.close()


@app.function(gpu=GPU, image=image, timeout=60 * 60)
def run_remote(
    mode: str,
    dataset_path: str,
    artifact_root: str,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    vocab_size: int,
    seed: int,
) -> dict[str, Any]:
    if mode == "env":
        return _run_env()
    if mode == "toy-sft":
        return _run_toy_sft(
            dataset_path=dataset_path,
            artifact_root=artifact_root,
            max_steps=max_steps,
            save_every=save_every,
            learning_rate=learning_rate,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            vocab_size=vocab_size,
            seed=seed,
        )
    raise ValueError(f"unknown mode: {mode}")


@app.local_entrypoint()
def main(
    mode: str = "toy-sft",
    dataset_path: str = str(REMOTE_ROOT / "examples" / "tiny_sft.jsonl"),
    artifact_root: str = "/tmp/ganker-sft",
    max_steps: int = 4,
    save_every: int = 0,
    learning_rate: float = 1e-4,
    sequence_length: int = 64,
    micro_batch_size: int = 1,
    hidden_size: int = 32,
    num_layers: int = 2,
    num_attention_heads: int = 4,
    vocab_size: int = 128,
    seed: int = 1234,
):
    result = run_remote.remote(
        mode,
        dataset_path,
        artifact_root,
        max_steps,
        save_every,
        learning_rate,
        sequence_length,
        micro_batch_size,
        hidden_size,
        num_layers,
        num_attention_heads,
        vocab_size,
        seed,
    )
    print(result)
