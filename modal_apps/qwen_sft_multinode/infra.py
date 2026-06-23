"""Modal clustered infra for Qwen LoRA SFT.

Deploy:

    source ~/.codex/modal.env
    GANKER_QWEN_SFT_MULTINODE_NODES=2 uv run modal deploy modal_apps/qwen_sft_multinode/infra.py
"""

from __future__ import annotations

import json
import importlib
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import modal
import modal.experimental

from ganker.distributed.torchrun import (
    DistributedTrainingConfig,
    TorchrunLaunchConfig,
    entrypoint_args_from_mapping,
    parse_gpu_count,
    read_json,
    training_config_from_mapping,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = Path("/workspace/ganker")
MODEL = os.getenv("GANKER_QWEN_SFT_MULTINODE_MODEL", "Qwen/Qwen3-0.6B")
APP_NAME = os.getenv("GANKER_QWEN_SFT_MULTINODE_APP", "ganker-qwen-sft-multinode")
REGION = os.getenv("GANKER_MODAL_REGION", "us-east-1")
GPU = os.getenv("GANKER_QWEN_SFT_MULTINODE_GPU", "H100:8")
SINGLE_NODE_BASELINE_GPU = os.getenv(
    "GANKER_QWEN_SFT_SINGLE_NODE_GPU",
    os.getenv("GANKER_MODAL_GPU", "H100"),
)
BRIDGE_BASE_IMAGE = os.getenv("GANKER_MODAL_BRIDGE_BASE_IMAGE", "nvcr.io/nvidia/pytorch:26.02-py3")
BRIDGE_REPO = os.getenv(
    "GANKER_MEGATRON_BRIDGE_REPO",
    "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
)
BRIDGE_REF = os.getenv("GANKER_MEGATRON_BRIDGE_REF", "v0.4.2")
BRIDGE_UV_VERSION = os.getenv("GANKER_MEGATRON_BRIDGE_UV_VERSION", "0.7.2")
CLUSTER_SIZE = int(os.getenv("GANKER_QWEN_SFT_MULTINODE_NODES", "2"))
GPUS_PER_NODE = int(os.getenv("GANKER_QWEN_SFT_MULTINODE_GPUS_PER_NODE", str(parse_gpu_count(GPU))))
MASTER_PORT = int(os.getenv("GANKER_QWEN_SFT_MULTINODE_MASTER_PORT", "29500"))
RDMA_ENABLED = os.getenv("GANKER_QWEN_SFT_MULTINODE_RDMA", "1") != "0"
EFA_ENABLED = os.getenv("GANKER_QWEN_SFT_MULTINODE_EFA", "1") != "0"
ARTIFACT_VOLUME_NAME = os.getenv("GANKER_QWEN_SFT_ARTIFACT_VOLUME", "ganker-qwen-sft-artifacts")
HF_CACHE_VOLUME_NAME = os.getenv("GANKER_HF_CACHE_VOLUME", "huggingface-cache")
ARTIFACT_ROOT = Path(os.getenv("GANKER_QWEN_SFT_ARTIFACT_ROOT", "/vol/ganker-artifacts"))
ARTIFACT_MOUNT = str(ARTIFACT_ROOT)
DEFAULT_DATASET_PATH = REMOTE_ROOT / "examples" / "tiny_sft.jsonl"


def _repo_ignore() -> list[str]:
    return [
        ".git",
        ".jj",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".local_artifacts",
    ]


def _hf_secrets() -> list[modal.Secret]:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        return []
    return [
        modal.Secret.from_dict(
            {
                "HF_TOKEN": hf_token,
                "HUGGING_FACE_HUB_TOKEN": hf_token,
            }
        )
    ]


def _training_image() -> modal.Image:
    return (
        modal.Image.from_registry(BRIDGE_BASE_IMAGE)
        .apt_install(
            "git",
            "curl",
            "libibverbs-dev",
            "libibverbs1",
            "libhwloc15",
            "libnl-route-3-200",
        )
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
        )
        .env(
            {
                "PATH": "/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "VIRTUAL_ENV": "/opt/venv",
                "UV_PROJECT_ENVIRONMENT": "/opt/venv",
                "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}:/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM",
                "GANKER_ARTIFACT_ROOT": ARTIFACT_MOUNT,
                "HF_HUB_CACHE": "/root/.cache/huggingface",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
        .add_local_dir(PROJECT_ROOT, remote_path=str(REMOTE_ROOT), ignore=_repo_ignore())
    )


training_image = _training_image()
app = modal.App(APP_NAME)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


def add_remote_import_paths() -> None:
    import sys

    for path in (REMOTE_ROOT, REMOTE_ROOT / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def distributed_training_config(
    *,
    n_nodes: int = CLUSTER_SIZE,
    gpus_per_node: int = GPUS_PER_NODE,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    micro_batch_size: int = 1,
    global_batch_size: int | None = None,
) -> DistributedTrainingConfig:
    return DistributedTrainingConfig(
        n_nodes=n_nodes,
        gpus_per_node=gpus_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
    )


def job_config(
    *,
    mode: str,
    comparison_id: str,
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    artifact_root: str = str(ARTIFACT_ROOT),
    base_model: str = MODEL,
    n_nodes: int = CLUSTER_SIZE,
    gpus_per_node: int = GPUS_PER_NODE,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    micro_batch_size: int = 1,
    global_batch_size: int | None = None,
    sequence_length: int = 32,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = 1,
    save_every: int = 0,
    seed: int = 1234,
    master_port: int = MASTER_PORT,
) -> dict[str, Any]:
    distributed = distributed_training_config(
        n_nodes=n_nodes,
        gpus_per_node=gpus_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
    )
    return {
        "mode": mode,
        "comparison_id": comparison_id,
        "dataset_path": dataset_path,
        "artifact_root": artifact_root,
        "base_model": base_model,
        "n_nodes": distributed.n_nodes,
        "gpus_per_node": distributed.gpus_per_node,
        "world_size": distributed.world_size,
        "data_parallel_size": distributed.data_parallel_size,
        "tensor_model_parallel_size": distributed.tensor_model_parallel_size,
        "pipeline_model_parallel_size": distributed.pipeline_model_parallel_size,
        "micro_batch_size": distributed.micro_batch_size,
        "global_batch_size": distributed.effective_global_batch_size,
        "grad_accum_steps": distributed.grad_accum_steps,
        "sequence_length": sequence_length,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "save_every": save_every,
        "seed": seed,
        "master_port": master_port,
    }


def _validate_cluster_config(
    config: dict[str, Any],
    *,
    actual_cluster_size: int | None = None,
) -> DistributedTrainingConfig:
    distributed = training_config_from_mapping(config)
    expected_cluster_size = actual_cluster_size if actual_cluster_size is not None else CLUSTER_SIZE
    if distributed.n_nodes != expected_cluster_size:
        raise ValueError(
            "config n_nodes must match GANKER_QWEN_SFT_MULTINODE_NODES because Modal "
            "cluster size is fixed at import time: "
            f"config={distributed.n_nodes} actual={expected_cluster_size}"
        )
    if distributed.gpus_per_node != GPUS_PER_NODE:
        raise ValueError(
            "config gpus_per_node must match the Modal GPU request: "
            f"config={distributed.gpus_per_node} gpu={GPU!r} parsed={GPUS_PER_NODE}"
        )
    if config["mode"] == "qwen-lora-sft":
        distributed.require_supported_model_parallel(allow_pipeline_parallel=True)
    elif config["mode"] == "hf-ddp-baseline":
        distributed.require_dp_only()
    return distributed


def _result_path(config: dict[str, Any]) -> str:
    comparison_id = str(config["comparison_id"]).replace("/", "_")
    mode = str(config["mode"]).replace("/", "_")
    return f"/tmp/ganker-qwen-sft-multinode/{comparison_id}-{mode}.json"


def _namespace_from_config(config: dict[str, Any], *, result_path: str) -> Namespace:
    return Namespace(
        mode=str(config["mode"]),
        result_path=result_path,
        dataset_path=str(config["dataset_path"]),
        artifact_root=str(config["artifact_root"]),
        base_model=str(config["base_model"]),
        lora_rank=int(config["lora_rank"]),
        learning_rate=float(config["learning_rate"]),
        max_steps=int(config["max_steps"]),
        save_every=int(config.get("save_every", 0)),
        sequence_length=int(config["sequence_length"]),
        micro_batch_size=int(config["micro_batch_size"]),
        global_batch_size=int(config["global_batch_size"]),
        tensor_model_parallel_size=int(config.get("tensor_model_parallel_size", 1)),
        pipeline_model_parallel_size=int(config.get("pipeline_model_parallel_size", 1)),
        seed=int(config["seed"]),
        comparison_id=str(config["comparison_id"]),
    )


@app.function(
    image=training_image,
    gpu=GPU,
    timeout=60 * 60 * 24,
    region=REGION,
    volumes={
        ARTIFACT_MOUNT: artifact_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=_hf_secrets(),
    experimental_options={"efa_enabled": EFA_ENABLED},
)
@modal.experimental.clustered(size=CLUSTER_SIZE, rdma=RDMA_ENABLED)  # type: ignore[reportCallIssue,reportOptionalCall]
def run_clustered_trainer(config: dict[str, Any]) -> dict[str, Any]:
    add_remote_import_paths()
    artifact_volume.reload()
    cluster_info = modal.experimental.get_cluster_info()
    container_ips = [str(value) for value in cluster_info.container_ips]
    distributed = _validate_cluster_config(config, actual_cluster_size=len(container_ips))
    result_path = _result_path(config)
    launch = TorchrunLaunchConfig(
        nnodes=distributed.n_nodes,
        nproc_per_node=distributed.gpus_per_node,
        node_rank=int(cluster_info.rank),
        master_addr=container_ips[0],
        master_port=int(config.get("master_port", MASTER_PORT)),
    )
    entrypoint = REMOTE_ROOT / "modal_apps" / "qwen_sft_multinode" / "train_entry.py"
    args = launch.distributed_run_args(
        str(entrypoint),
        entrypoint_args_from_mapping(config, result_path=result_path),
    )
    print(
        json.dumps(
            {
                "event": "launch_torchrun",
                "cluster_rank": cluster_info.rank,
                "cluster_id": cluster_info.cluster_id,
                "container_ips": container_ips,
                "argv": args,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    distributed_run = importlib.import_module("torch.distributed.run")
    distributed_run.run(distributed_run.parse_args(args))

    if int(cluster_info.rank) != 0:
        return {
            "ok": True,
            "mode": str(config["mode"]),
            "cluster_rank": int(cluster_info.rank),
            "returned": False,
        }

    artifact_volume.commit()
    payload = read_json(result_path)
    payload["modal_cluster"] = {
        "cluster_id": cluster_info.cluster_id,
        "cluster_size": len(container_ips),
        "cluster_rank": int(cluster_info.rank),
        "gpu": GPU,
        "region": REGION,
        "rdma_enabled": RDMA_ENABLED,
        "efa_enabled": EFA_ENABLED,
        "container_ips": container_ips,
    }
    return payload


@app.function(
    image=training_image,
    gpu=SINGLE_NODE_BASELINE_GPU,
    timeout=60 * 60 * 24,
    region=REGION,
    volumes={
        ARTIFACT_MOUNT: artifact_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=_hf_secrets(),
)
def run_single_node_ganker_baseline(config: dict[str, Any]) -> dict[str, Any]:
    add_remote_import_paths()
    artifact_volume.reload()
    result_path = _result_path(config)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(int(config.get("master_port", MASTER_PORT)) + 17))

    from modal_apps.qwen_sft_multinode import train_entry

    train_entry.run_qwen_lora_sft(_namespace_from_config(config, result_path=result_path))
    artifact_volume.commit()
    payload = read_json(result_path)
    payload["modal_single_node"] = {
        "gpu": SINGLE_NODE_BASELINE_GPU,
        "region": REGION,
    }
    return payload


@app.function(
    image=training_image,
    timeout=60 * 60,
    region=REGION,
    volumes={
        ARTIFACT_MOUNT: artifact_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=_hf_secrets(),
)
def prepare_real_sft_dataset(
    dataset_path: str,
    dataset_name: str,
    dataset_split: str,
    dataset_format: str,
    dataset_size: int,
    seed: int,
) -> dict[str, Any]:
    add_remote_import_paths()
    from examples.sft import materialize_hf_sft_jsonl

    payload = materialize_hf_sft_jsonl(
        dataset_path,
        dataset_name=dataset_name,
        split=dataset_split,
        dataset_format=dataset_format,
        max_examples=dataset_size,
        seed=seed,
    )
    artifact_volume.commit()
    return payload
