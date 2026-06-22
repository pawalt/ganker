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
BRIDGE_BASE_IMAGE = os.getenv("GANKER_MODAL_BRIDGE_BASE_IMAGE", "nvcr.io/nvidia/pytorch:26.02-py3")
BRIDGE_REPO = os.getenv(
    "GANKER_MEGATRON_BRIDGE_REPO",
    "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
)
BRIDGE_REF = os.getenv("GANKER_MEGATRON_BRIDGE_REF", "v0.4.2")
BRIDGE_UV_VERSION = os.getenv("GANKER_MEGATRON_BRIDGE_UV_VERSION", "0.7.2")
TORCHMONARCH_VERSION = os.getenv("GANKER_MODAL_TORCHMONARCH_VERSION", "0.5.0")


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


def _bridge_image():
    return (
        modal.Image.from_registry(BRIDGE_BASE_IMAGE)
        .apt_install("git", "curl")
        .run_commands(
            f"curl -LsSf https://astral.sh/uv/{BRIDGE_UV_VERSION}/install.sh | sh",
            "rm -rf /opt/Megatron-Bridge /opt/venv",
            (
                "git clone --depth 1 --branch "
                f"{BRIDGE_REF} --recurse-submodules --shallow-submodules "
                f"{BRIDGE_REPO} /opt/Megatron-Bridge"
            ),
            "/root/.local/bin/uv venv /opt/venv --system-site-packages",
            (
                "cd /opt/Megatron-Bridge && "
                "UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy "
                "/root/.local/bin/uv sync --frozen --only-group build"
            ),
            (
                "cd /opt/Megatron-Bridge && "
                "UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy "
                "MAX_JOBS=4 NVTE_BUILD_NUM_PHILOX_ROUNDS=3 "
                "/root/.local/bin/uv sync --link-mode copy --frozen --no-dev "
                "--no-install-package transformer-engine"
            ),
            (
                "UV_PROJECT_ENVIRONMENT=/opt/venv "
                f"/root/.local/bin/uv pip install --python /opt/venv/bin/python "
                f"torchmonarch=={TORCHMONARCH_VERSION}"
            ),
        )
        .env(
            {
                "PATH": "/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "VIRTUAL_ENV": "/opt/venv",
                "UV_PROJECT_ENVIRONMENT": "/opt/venv",
                "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}:/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM",
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
bridge_image = _bridge_image()
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


def _run_hf_small_sft(
    *,
    dataset_path: str,
    artifact_root: str,
    base_model: str,
    tuning: str,
    lora_rank: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    _add_remote_import_paths()
    if tuning not in ("full", "lora"):
        raise ValueError("tuning must be 'full' or 'lora'")
    if tuning == "lora" and lora_rank <= 0:
        raise ValueError("lora_rank must be positive for LoRA")

    from examples.sft import HFAutoTokenizerAdapter, SFTDataConfig, load_jsonl_sft_batches, run_sft
    from ganker import ServiceClient

    tokenizer = HFAutoTokenizerAdapter.from_pretrained(base_model)
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
                "runtime_kind": "bridge",
                "tensor_device": "cuda",
                "micro_batch_size": micro_batch_size,
                "global_batch_size": micro_batch_size,
                "sequence_length": sequence_length,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "seed": seed,
                "trust_remote_code": True,
                "load_weights": True,
            },
            timeout=60,
        )
        summary = run_sft(
            client,
            base_model=base_model,
            dataset=batches,
            tuning=tuning,
            lora_rank=lora_rank if tuning == "lora" else 0,
            learning_rate=learning_rate,
            max_steps=max_steps,
            save_every=save_every,
        )
        payload = {
            "ok": True,
            "mode": "hf-small-sft",
            "runtime_kind": "bridge",
            "bridge_base_image": BRIDGE_BASE_IMAGE,
            "bridge_ref": BRIDGE_REF,
            "tuning": tuning,
            "lora_rank": lora_rank if tuning == "lora" else 0,
            "dataset_path": dataset_path,
            "batch_count": len(batches),
            **summary.to_dict(),
        }
        payload["artifact_exists"] = Path(summary.artifact_path).exists()
        if payload["artifact_exists"]:
            artifact_payload = json.loads(Path(summary.artifact_path).read_text())
            payload["artifact_format"] = artifact_payload.get("artifact_format")
            for key in (
                "hf_checkpoint_path",
                "hf_adapter_path",
                "hf_weights_path",
                "hf_weights_index_path",
                "hf_adapter_config_path",
                "hf_adapter_weights_path",
                "hf_checkpoint_bytes",
                "hf_weight_count",
                "hf_weight_format",
            ):
                if key in artifact_payload:
                    payload[key] = artifact_payload[key]
            for key in (
                "hf_checkpoint_path",
                "hf_adapter_path",
                "hf_weights_path",
                "hf_weights_index_path",
                "hf_adapter_config_path",
                "hf_adapter_weights_path",
            ):
                if key in payload:
                    payload[f"{key}_exists"] = Path(payload[key]).exists()
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


@app.function(gpu=GPU, image=bridge_image, timeout=60 * 60)
def run_bridge_remote(
    mode: str,
    dataset_path: str,
    artifact_root: str,
    base_model: str,
    tuning: str,
    lora_rank: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    if mode == "hf-small-sft":
        return _run_hf_small_sft(
            dataset_path=dataset_path,
            artifact_root=artifact_root,
            base_model=base_model,
            tuning=tuning,
            lora_rank=lora_rank,
            max_steps=max_steps,
            save_every=save_every,
            learning_rate=learning_rate,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            seed=seed,
        )
    raise ValueError(f"unknown bridge mode: {mode}")


@app.local_entrypoint()
def main(
    mode: str = "toy-sft",
    dataset_path: str = str(REMOTE_ROOT / "examples" / "tiny_sft.jsonl"),
    artifact_root: str = "/tmp/ganker-sft",
    base_model: str = "Qwen/Qwen3-0.6B",
    tuning: str = "full",
    lora_rank: int = 8,
    max_steps: int = 1,
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
    if mode == "hf-small-sft":
        result = run_bridge_remote.remote(
            mode,
            dataset_path,
            artifact_root,
            base_model,
            tuning,
            lora_rank,
            max_steps,
            save_every,
            learning_rate,
            sequence_length,
            micro_batch_size,
            seed,
        )
    else:
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
