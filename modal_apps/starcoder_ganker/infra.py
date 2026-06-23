"""Modal infra for a StarCoderData-style code SFT example using Ganker.

This mirrors the Modal multinode StarCoder guide at the workflow level:
materialize code data, run a multinode trainer, then sample from a checkpoint.
Training itself goes through Ganker's Megatron Bridge backend instead of TRL/FSDP.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
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
from modal_apps.qwen_sft_multinode import infra as qwen_infra
from modal_apps.starcoder_ganker import common


APP_NAME = common.APP_NAME
REGION = os.getenv("GANKER_MODAL_REGION", qwen_infra.REGION)
GPU = os.getenv("GANKER_STARCODER_GPU", qwen_infra.GPU)
SINGLE_NODE_GPU = os.getenv("GANKER_STARCODER_SINGLE_NODE_GPU", qwen_infra.SINGLE_NODE_BASELINE_GPU)
CLUSTER_SIZE = int(os.getenv("GANKER_STARCODER_NODES", str(qwen_infra.CLUSTER_SIZE)))
GPUS_PER_NODE = int(os.getenv("GANKER_STARCODER_GPUS_PER_NODE", str(parse_gpu_count(GPU))))
MASTER_PORT = int(os.getenv("GANKER_STARCODER_MASTER_PORT", str(qwen_infra.MASTER_PORT)))
RDMA_ENABLED = os.getenv("GANKER_STARCODER_RDMA", "1") != "0"
EFA_ENABLED = os.getenv("GANKER_STARCODER_EFA", "1") != "0"

REMOTE_ROOT = qwen_infra.REMOTE_ROOT
PROJECT_ROOT = qwen_infra.PROJECT_ROOT
ARTIFACT_ROOT = qwen_infra.ARTIFACT_ROOT
ARTIFACT_MOUNT = qwen_infra.ARTIFACT_MOUNT
DATASET_MOUNT = str(common.DATASET_MOUNT)
DEFAULT_DATASET_PATH = common.DEFAULT_DATASET_PATH


app = modal.App(APP_NAME)
dataset_volume = modal.Volume.from_name(common.DATASET_VOLUME_NAME, create_if_missing=True)
artifact_volume = qwen_infra.artifact_volume
hf_cache_volume = qwen_infra.hf_cache_volume
training_image = qwen_infra.training_image


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


def _dataset_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.11")
        .uv_pip_install(
            "datasets>=2.20,<4",
            "huggingface_hub>=0.24",
            "hf_transfer>=0.1.8",
        )
        .env(
            {
                "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_XET_HIGH_PERFORMANCE": "1",
            }
        )
        .add_local_dir(PROJECT_ROOT, remote_path=str(REMOTE_ROOT), ignore=_repo_ignore())
    )


def _sglang_image() -> modal.Image:
    return (
        modal.Image.from_registry(common.SGLANG_IMAGE)
        .entrypoint([])
        .apt_install("git", "curl")
        .uv_pip_install(
            "grpcio>=1.81.1",
            "protobuf>=6.33.6",
            "torchmonarch>=0.5.0",
            "typing_extensions>=4.13",
        )
        .env(
            {
                "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}",
                "GANKER_ARTIFACT_ROOT": ARTIFACT_MOUNT,
                "HF_HUB_CACHE": "/root/.cache/huggingface",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
        .add_local_dir(PROJECT_ROOT, remote_path=str(REMOTE_ROOT), ignore=_repo_ignore())
    )


dataset_image = _dataset_image()
sglang_image = _sglang_image()


def add_remote_import_paths() -> None:
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
    comparison_id: str,
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    artifact_root: str = str(ARTIFACT_ROOT),
    base_model: str = common.MODEL,
    n_nodes: int = CLUSTER_SIZE,
    gpus_per_node: int = GPUS_PER_NODE,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    micro_batch_size: int = 1,
    global_batch_size: int | None = None,
    sequence_length: int = 512,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = 10,
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
        "mode": "qwen-lora-sft",
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
        "example": "starcoder-ganker",
    }


def _validate_cluster_config(config: dict[str, Any]) -> DistributedTrainingConfig:
    distributed = training_config_from_mapping(config)
    if distributed.n_nodes != CLUSTER_SIZE:
        raise ValueError(
            "config n_nodes must match GANKER_STARCODER_NODES because Modal cluster "
            f"size is fixed at import time: config={distributed.n_nodes} env={CLUSTER_SIZE}"
        )
    if distributed.gpus_per_node != GPUS_PER_NODE:
        raise ValueError(
            "config gpus_per_node must match the Modal GPU request: "
            f"config={distributed.gpus_per_node} gpu={GPU!r} parsed={GPUS_PER_NODE}"
        )
    distributed.require_dp_only()
    return distributed


def _result_path(config: dict[str, Any]) -> str:
    comparison_id = str(config["comparison_id"]).replace("/", "_")
    return f"/tmp/ganker-starcoder-code-sft/{comparison_id}.json"


def _entrypoint_args(config: dict[str, Any], *, result_path: str) -> list[str]:
    return entrypoint_args_from_mapping(config, result_path=result_path)


def _single_node_config(config: dict[str, Any]) -> dict[str, Any]:
    single = dict(config)
    single["n_nodes"] = 1
    single["gpus_per_node"] = 1
    single["world_size"] = 1
    single["data_parallel_size"] = 1
    single["global_batch_size"] = int(single["micro_batch_size"])
    single["grad_accum_steps"] = 1
    return single


@app.function(
    image=dataset_image,
    timeout=60 * 60 * 12,
    region=REGION,
    volumes={DATASET_MOUNT: dataset_volume},
    secrets=_hf_secrets(),
)
def prepare_starcoder_dataset(
    *,
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    dataset_id: str = common.DATASET_ID,
    languages: list[str] | None = None,
    max_files_per_language: int = 1,
    max_examples: int = 256,
    min_chars: int = 16,
    max_chars: int = 12_000,
    seed: int = 1234,
    content_column: str = "content",
    prompt_column: str = "prompt",
    completion_column: str = "completion",
    shuffle_buffer: int = 10_000,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    add_remote_import_paths()
    dataset_volume.reload()

    from examples.sft import materialize_hf_code_sft_jsonl, select_starcoder_parquet_files
    from examples.sft.code_data import STARCODER_DATASET

    selected_languages = languages or list(common.DEFAULT_LANGUAGES)
    data_files: list[str] = []
    if dataset_id == STARCODER_DATASET:
        if not selected_languages:
            selected_languages = list(common.DEFAULT_STARCODER_LANGUAGES)
        from huggingface_hub import HfApi

        all_paths = HfApi().list_repo_files(repo_id=dataset_id, repo_type="dataset")
        data_files = select_starcoder_parquet_files(
            all_paths,
            languages=selected_languages,
            max_files_per_language=max_files_per_language,
        )
    payload = materialize_hf_code_sft_jsonl(
        dataset_path,
        dataset_name=dataset_id,
        data_files=data_files or None,
        split="train",
        language=",".join(selected_languages),
        allowed_languages=selected_languages,
        content_column=content_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        max_examples=max_examples,
        min_chars=min_chars,
        max_chars=max_chars,
        seed=seed,
        shuffle_buffer=shuffle_buffer,
        trust_remote_code=trust_remote_code,
    )
    payload["selected_languages"] = selected_languages
    payload["selected_file_count"] = len(data_files)
    payload["starcoder_shard_mode"] = dataset_id == STARCODER_DATASET
    dataset_volume.commit()
    return payload


@app.function(
    image=dataset_image,
    timeout=60 * 10,
    region=REGION,
    volumes={DATASET_MOUNT: dataset_volume},
)
def clear_dataset_volume() -> dict[str, Any]:
    dataset_volume.reload()
    mount = Path(DATASET_MOUNT)
    removed = 0
    if mount.exists():
        for item in mount.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
            removed += 1
    dataset_volume.commit()
    return {"ok": True, "removed": removed, "dataset_mount": DATASET_MOUNT}


@app.function(
    image=training_image,
    gpu=GPU,
    timeout=60 * 60 * 24,
    region=REGION,
    volumes={
        DATASET_MOUNT: dataset_volume,
        ARTIFACT_MOUNT: artifact_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=_hf_secrets(),
    experimental_options={"efa_enabled": EFA_ENABLED},
)
@modal.experimental.clustered(size=CLUSTER_SIZE, rdma=RDMA_ENABLED)  # type: ignore[reportCallIssue,reportOptionalCall]
def run_clustered_code_sft(config: dict[str, Any]) -> dict[str, Any]:
    add_remote_import_paths()
    dataset_volume.reload()
    artifact_volume.reload()
    dataset_path = Path(str(config["dataset_path"]))
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path is missing in the dataset volume: {dataset_path}")

    distributed = _validate_cluster_config(config)
    cluster_info = modal.experimental.get_cluster_info()
    result_path = _result_path(config)
    container_ips = [str(value) for value in cluster_info.container_ips]
    launch = TorchrunLaunchConfig(
        nnodes=distributed.n_nodes,
        nproc_per_node=distributed.gpus_per_node,
        node_rank=int(cluster_info.rank),
        master_addr=container_ips[0],
        master_port=int(config.get("master_port", MASTER_PORT)),
    )
    entrypoint = REMOTE_ROOT / "modal_apps" / "qwen_sft_multinode" / "train_entry.py"
    args = launch.distributed_run_args(str(entrypoint), _entrypoint_args(config, result_path=result_path))
    print(
        json.dumps(
            {
                "event": "launch_starcoder_ganker_torchrun",
                "cluster_rank": int(cluster_info.rank),
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
            "mode": "starcoder-ganker-code-sft",
            "cluster_rank": int(cluster_info.rank),
            "returned": False,
        }

    artifact_volume.commit()
    payload = read_json(result_path)
    payload["mode"] = "starcoder-ganker-code-sft"
    payload["example"] = "starcoder-ganker"
    payload["modal_cluster"] = {
        "cluster_id": cluster_info.cluster_id,
        "cluster_size": CLUSTER_SIZE,
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
    gpu=SINGLE_NODE_GPU,
    timeout=60 * 60 * 24,
    region=REGION,
    volumes={
        DATASET_MOUNT: dataset_volume,
        ARTIFACT_MOUNT: artifact_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=_hf_secrets(),
)
def run_single_node_code_sft(config: dict[str, Any]) -> dict[str, Any]:
    add_remote_import_paths()
    dataset_volume.reload()
    artifact_volume.reload()
    config = _single_node_config(config)

    dataset_path = Path(str(config["dataset_path"]))
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path is missing in the dataset volume: {dataset_path}")

    result_path = _result_path(config)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(int(config.get("master_port", MASTER_PORT)) + 17))

    from modal_apps.qwen_sft_multinode import train_entry

    namespace = qwen_infra._namespace_from_config(config, result_path=result_path)
    train_entry.run_qwen_lora_sft(namespace)
    artifact_volume.commit()
    payload = read_json(result_path)
    payload["mode"] = "starcoder-ganker-code-sft-single-node"
    payload["example"] = "starcoder-ganker"
    payload["modal_single_node"] = {
        "gpu": SINGLE_NODE_GPU,
        "region": REGION,
    }
    return payload


@app.function(
    image=sglang_image,
    gpu=common.SGLANG_GPU,
    timeout=60 * 60,
    region=REGION,
    volumes={
        ARTIFACT_MOUNT: artifact_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=_hf_secrets(),
)
def run_sglang_eval(
    *,
    run_id: str,
    prompts: list[str] | None = None,
    checkpoint_version: int = -1,
    base_model: str = common.MODEL,
    sglang_base_url: str = "",
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    port: int = 30000,
    startup_timeout: int = 900,
    context_length: int = 4096,
    mem_fraction_static: float = 0.75,
) -> dict[str, Any]:
    if not run_id:
        raise ValueError("run_id is required")
    add_remote_import_paths()
    artifact_volume.reload()

    from ganker.artifacts import FilesystemArtifactStore
    from ganker.backends.sglang import SGLangInferenceBackend
    from ganker.config import SGLangBackendConfig
    from ganker.contracts import ArtifactKind, ModelInput, SamplingParams, WeightArtifact

    store = FilesystemArtifactStore(Path(ARTIFACT_ROOT))
    if checkpoint_version >= 0:
        manifest_path = Path(ARTIFACT_ROOT) / "weights" / run_id / f"checkpoint-{checkpoint_version}.manifest.json"
        payload_path = Path(ARTIFACT_ROOT) / "weights" / run_id / f"checkpoint-{checkpoint_version}.payload.json"
        artifact = WeightArtifact(
            run_id=run_id,
            checkpoint_version=checkpoint_version,
            kind=ArtifactKind.DELTA,
            manifest_path=str(manifest_path),
            payload_path=str(payload_path),
        )
    else:
        artifact = store.latest(run_id)

    backend = SGLangInferenceBackend(
        store,
        config=SGLangBackendConfig(
            base_url=sglang_base_url,
            model_path=base_model,
            launch_server=not bool(sglang_base_url),
            host="127.0.0.1",
            port=port,
            request_timeout=120,
            startup_timeout=float(startup_timeout),
            return_logprobs=True,
            enable_lora=True,
            max_lora_rank=256,
            extra_server_args={
                "trust-remote-code": True,
                "context-length": context_length,
                "mem-fraction-static": mem_fraction_static,
                "chunked-prefill-size": min(1024, context_length),
            },
        ),
    )
    eval_prompts = prompts or common.DEFAULT_EVAL_PROMPTS
    try:
        loaded = backend.refresh_weights(run_id=run_id, artifact=artifact)
        generations = []
        for prompt in eval_prompts:
            sample = backend.sample(
                run_id=run_id,
                prompt=ModelInput.from_text(prompt),
                sampling_params=SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ),
                num_samples=1,
            )
            sequence = sample.sequences[0]
            generations.append(
                {
                    "prompt": prompt,
                    "text": sequence.text,
                    "tokens": sequence.tokens,
                    "stop_reason": sequence.stop_reason,
                    "usage": {
                        "input_tokens": sample.usage.input_tokens,
                        "output_tokens": sample.usage.output_tokens,
                    },
                }
            )
        return {
            "ok": True,
            "mode": "starcoder-ganker-sglang-eval",
            "run_id": run_id,
            "checkpoint_version": loaded.checkpoint_version,
            "artifact_payload_path": loaded.payload_path,
            "base_model": base_model,
            "sglang_base_url": sglang_base_url,
            "prompt_count": len(eval_prompts),
            "generations": generations,
        }
    finally:
        backend.close()
