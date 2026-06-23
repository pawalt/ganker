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
from modal_apps.qwen_sft import infra as single_node_infra


PROJECT_ROOT = single_node_infra.PROJECT_ROOT
REMOTE_ROOT = single_node_infra.REMOTE_ROOT
MODEL = os.getenv("GANKER_QWEN_SFT_MULTINODE_MODEL", single_node_infra.MODEL)
APP_NAME = os.getenv("GANKER_QWEN_SFT_MULTINODE_APP", "ganker-qwen-sft-multinode")
REGION = os.getenv("GANKER_MODAL_REGION", single_node_infra.REGION)
GPU = os.getenv("GANKER_QWEN_SFT_MULTINODE_GPU", "H100:8")
SINGLE_NODE_BASELINE_GPU = os.getenv(
    "GANKER_QWEN_SFT_SINGLE_NODE_GPU",
    os.getenv("GANKER_MODAL_GPU", "H100"),
)
CLUSTER_SIZE = int(os.getenv("GANKER_QWEN_SFT_MULTINODE_NODES", "2"))
GPUS_PER_NODE = int(os.getenv("GANKER_QWEN_SFT_MULTINODE_GPUS_PER_NODE", str(parse_gpu_count(GPU))))
MASTER_PORT = int(os.getenv("GANKER_QWEN_SFT_MULTINODE_MASTER_PORT", "29500"))
RDMA_ENABLED = os.getenv("GANKER_QWEN_SFT_MULTINODE_RDMA", "1") != "0"
EFA_ENABLED = os.getenv("GANKER_QWEN_SFT_MULTINODE_EFA", "1") != "0"
ARTIFACT_ROOT = single_node_infra.ARTIFACT_ROOT
ARTIFACT_MOUNT = single_node_infra.ARTIFACT_MOUNT
DEFAULT_DATASET_PATH = REMOTE_ROOT / "examples" / "tiny_sft.jsonl"


def _training_image() -> modal.Image:
    return single_node_infra.bridge_image.run_commands(
        (
            "UV_PROJECT_ENVIRONMENT=/opt/venv "
            "/root/.local/bin/uv pip install --python /opt/venv/bin/python "
            "'datasets>=2.20' 'accelerate>=0.33' 'peft>=0.13' "
            "'transformers>=4.51,<5' 'safetensors>=0.4'"
        )
    )


training_image = _training_image()
app = modal.App(APP_NAME)
artifact_volume = single_node_infra.artifact_volume
hf_cache_volume = single_node_infra.hf_cache_volume


def add_remote_import_paths() -> None:
    single_node_infra.add_remote_import_paths()


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


def _validate_cluster_config(config: dict[str, Any]) -> DistributedTrainingConfig:
    distributed = training_config_from_mapping(config)
    if distributed.n_nodes != CLUSTER_SIZE:
        raise ValueError(
            "config n_nodes must match GANKER_QWEN_SFT_MULTINODE_NODES because Modal "
            f"cluster size is fixed at import time: config={distributed.n_nodes} env={CLUSTER_SIZE}"
        )
    if distributed.gpus_per_node != GPUS_PER_NODE:
        raise ValueError(
            "config gpus_per_node must match the Modal GPU request: "
            f"config={distributed.gpus_per_node} gpu={GPU!r} parsed={GPUS_PER_NODE}"
        )
    if config["mode"] in {"qwen-lora-sft", "hf-ddp-baseline"}:
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
    secrets=single_node_infra._hf_secrets(),
    experimental_options={"efa_enabled": EFA_ENABLED},
)
@modal.experimental.clustered(size=CLUSTER_SIZE, rdma=RDMA_ENABLED)  # type: ignore[reportCallIssue,reportOptionalCall]
def run_clustered_trainer(config: dict[str, Any]) -> dict[str, Any]:
    add_remote_import_paths()
    artifact_volume.reload()
    distributed = _validate_cluster_config(config)
    cluster_info = modal.experimental.get_cluster_info()
    result_path = _result_path(config)
    launch = TorchrunLaunchConfig(
        nnodes=distributed.n_nodes,
        nproc_per_node=distributed.gpus_per_node,
        node_rank=int(cluster_info.rank),
        master_addr=str(cluster_info.container_ips[0]),
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
                "container_ips": cluster_info.container_ips,
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
        "cluster_size": CLUSTER_SIZE,
        "cluster_rank": int(cluster_info.rank),
        "gpu": GPU,
        "region": REGION,
        "rdma_enabled": RDMA_ENABLED,
        "efa_enabled": EFA_ENABLED,
        "container_ips": list(cluster_info.container_ips),
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
    secrets=single_node_infra._hf_secrets(),
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
    secrets=single_node_infra._hf_secrets(),
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
