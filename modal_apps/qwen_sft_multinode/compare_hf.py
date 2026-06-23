"""Compare clustered Ganker SFT against HF Trainer DDP and single-node Ganker.

Run:

    source ~/.codex/modal.env
    GANKER_QWEN_SFT_MULTINODE_NODES=2 uv run modal run modal_apps/qwen_sft_multinode/compare_hf.py \
      --dataset-size 256 \
      --max-steps 20 \
      --sequence-length 256
"""

from __future__ import annotations

import json
import math
from typing import Any
import uuid

from examples.sft import (
    DEFAULT_REAL_DATASET,
    DEFAULT_REAL_DATASET_FORMAT,
    DEFAULT_REAL_DATASET_SPLIT,
)
from modal_apps.qwen_sft_multinode import infra


app = infra.app
DEFAULT_DATASET_PATH = infra.ARTIFACT_ROOT / "datasets" / "alpaca-qwen-sft-multinode.jsonl"


def compare_loss_curves(left: list[float], right: list[float]) -> dict[str, Any]:
    pair_count = min(len(left), len(right))
    if pair_count == 0:
        return {"ok": False, "reason": "one or both loss curves are empty"}
    pairs = list(zip(left[:pair_count], right[:pair_count], strict=True))
    abs_diffs = [abs(a - b) for a, b in pairs]
    left_delta = left[pair_count - 1] - left[0]
    right_delta = right[pair_count - 1] - right[0]
    return {
        "ok": True,
        "paired_steps": pair_count,
        "all_losses_finite": all(math.isfinite(value) for pair in pairs for value in pair),
        "left_initial_loss": float(left[0]),
        "left_final_loss": float(left[pair_count - 1]),
        "left_loss_delta": float(left_delta),
        "right_initial_loss": float(right[0]),
        "right_final_loss": float(right[pair_count - 1]),
        "right_loss_delta": float(right_delta),
        "final_loss_abs_diff": float(abs(left[pair_count - 1] - right[pair_count - 1])),
        "mean_abs_loss_diff": float(sum(abs_diffs) / len(abs_diffs)),
        "direction_agrees": (left_delta <= 0 and right_delta <= 0)
        or (left_delta >= 0 and right_delta >= 0),
    }


def _shared_config(
    *,
    comparison_id: str,
    dataset_path: str,
    artifact_root: str,
    n_nodes: int,
    gpus_per_node: int,
    micro_batch_size: int,
    global_batch_size: int,
    sequence_length: int,
    lora_rank: int,
    learning_rate: float,
    max_steps: int,
    save_every: int,
    seed: int,
    master_port: int,
) -> dict[str, Any]:
    return {
        "comparison_id": comparison_id,
        "dataset_path": dataset_path,
        "artifact_root": artifact_root,
        "base_model": infra.MODEL,
        "n_nodes": n_nodes,
        "gpus_per_node": gpus_per_node,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "micro_batch_size": micro_batch_size,
        "global_batch_size": global_batch_size,
        "sequence_length": sequence_length,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "save_every": save_every,
        "seed": seed,
        "master_port": master_port,
    }


def _single_node_config(shared: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "qwen-lora-sft",
        "base_model": shared["base_model"],
        "dataset_path": shared["dataset_path"],
        "artifact_root": shared["artifact_root"],
        "comparison_id": f"{shared['comparison_id']}-single-ganker",
        "n_nodes": 1,
        "gpus_per_node": 1,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "lora_rank": shared["lora_rank"],
        "max_steps": shared["max_steps"],
        "save_every": shared["save_every"],
        "learning_rate": shared["learning_rate"],
        "sequence_length": shared["sequence_length"],
        "micro_batch_size": shared["micro_batch_size"],
        "global_batch_size": shared["micro_batch_size"],
        "seed": shared["seed"],
        "master_port": shared["master_port"],
    }


@app.local_entrypoint()
def main(
    dataset_name: str = DEFAULT_REAL_DATASET,
    dataset_split: str = DEFAULT_REAL_DATASET_SPLIT,
    dataset_format: str = DEFAULT_REAL_DATASET_FORMAT,
    dataset_size: int = 256,
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    artifact_root: str = str(infra.ARTIFACT_ROOT),
    comparison_id: str = "",
    n_nodes: int = infra.CLUSTER_SIZE,
    gpus_per_node: int = infra.GPUS_PER_NODE,
    micro_batch_size: int = 1,
    global_batch_size: int = 0,
    sequence_length: int = 256,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = 20,
    save_every: int = 0,
    seed: int = 1234,
    master_port: int = infra.MASTER_PORT,
    run_single_node_ganker: bool = True,
) -> None:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    run_id = comparison_id or f"qwen-multinode-compare-{uuid.uuid4().hex[:8]}"
    effective_global_batch_size = global_batch_size or (n_nodes * gpus_per_node * micro_batch_size)
    dataset_info = infra.prepare_real_sft_dataset.remote(
        dataset_path,
        dataset_name,
        dataset_split,
        dataset_format,
        dataset_size,
        seed,
    )
    shared = _shared_config(
        comparison_id=run_id,
        dataset_path=dataset_path,
        artifact_root=artifact_root,
        n_nodes=n_nodes,
        gpus_per_node=gpus_per_node,
        micro_batch_size=micro_batch_size,
        global_batch_size=effective_global_batch_size,
        sequence_length=sequence_length,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_every=save_every,
        seed=seed,
        master_port=master_port,
    )
    multinode_ganker = infra.run_clustered_trainer.remote(
        {
            **shared,
            "mode": "qwen-lora-sft",
        }
    )
    hf_ddp = infra.run_clustered_trainer.remote(
        {
            **shared,
            "mode": "hf-ddp-baseline",
        }
    )

    single_node_ganker = None
    if run_single_node_ganker:
        single_node_ganker = infra.run_single_node_ganker_baseline.remote(_single_node_config(shared))

    report = {
        "ok": True,
        "mode": "qwen-multinode-loss-comparison",
        "comparison_id": run_id,
        "dataset": dataset_info,
        "config": shared,
        "comparisons": {
            "multinode_ganker_vs_hf_ddp": compare_loss_curves(
                multinode_ganker["losses"],
                hf_ddp["losses"],
            ),
            "multinode_ganker_vs_single_node_ganker": (
                compare_loss_curves(multinode_ganker["losses"], single_node_ganker["losses"])
                if single_node_ganker is not None
                else {"ok": False, "reason": "single-node Ganker baseline disabled"}
            ),
        },
        "multinode_ganker": multinode_ganker,
        "huggingface_ddp": hf_ddp,
        "single_node_ganker": single_node_ganker,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
