"""Run a clean pipeline-parallel Qwen LoRA SFT smoke on Modal.

Run:

    source ~/.codex/modal.env
    GANKER_QWEN_SFT_MULTINODE_NODES=1 \
    GANKER_QWEN_SFT_MULTINODE_GPU=H100:8 \
    uv run modal run modal_apps/qwen_pipeline_parallel_sft/sft.py
"""

from __future__ import annotations

import json
import uuid

from modal_apps.qwen_sft_multinode import infra


app = infra.app


@app.local_entrypoint()
def main(
    dataset_path: str = str(infra.DEFAULT_DATASET_PATH),
    artifact_root: str = str(infra.ARTIFACT_ROOT),
    base_model: str = infra.MODEL,
    comparison_id: str = "",
    n_nodes: int = infra.CLUSTER_SIZE,
    gpus_per_node: int = infra.GPUS_PER_NODE,
    tensor_model_parallel_size: int = 2,
    pipeline_model_parallel_size: int = 2,
    micro_batch_size: int = 1,
    global_batch_size: int = 0,
    sequence_length: int = 32,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = 1,
    seed: int = 1234,
    master_port: int = infra.MASTER_PORT,
) -> None:
    if tensor_model_parallel_size <= 0:
        raise ValueError("tensor_model_parallel_size must be positive")
    if pipeline_model_parallel_size <= 1:
        raise ValueError("pipeline_model_parallel_size must be greater than 1 for this example")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive")

    world_size = n_nodes * gpus_per_node
    model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size
    if world_size % model_parallel_size:
        raise ValueError(
            "world size must be divisible by tensor_model_parallel_size * "
            "pipeline_model_parallel_size"
        )
    data_parallel_size = world_size // model_parallel_size
    effective_global_batch_size = (
        global_batch_size
        or micro_batch_size * data_parallel_size * pipeline_model_parallel_size
    )

    run_id = comparison_id or f"qwen-pp-sft-{uuid.uuid4().hex[:8]}"
    config = infra.job_config(
        mode="qwen-lora-sft",
        comparison_id=run_id,
        dataset_path=dataset_path,
        artifact_root=artifact_root,
        base_model=base_model,
        n_nodes=n_nodes,
        gpus_per_node=gpus_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        micro_batch_size=micro_batch_size,
        global_batch_size=effective_global_batch_size,
        sequence_length=sequence_length,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_every=0,
        seed=seed,
        master_port=master_port,
    )
    result = infra.run_clustered_trainer.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
