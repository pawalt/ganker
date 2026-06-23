"""Run clustered Qwen LoRA SFT on Modal.

Run:

    source ~/.codex/modal.env
    GANKER_QWEN_SFT_MULTINODE_NODES=2 uv run modal run modal_apps/qwen_sft_multinode/sft.py \
      --mode qwen-lora-sft
"""

from __future__ import annotations

import json
import uuid

from modal_apps.qwen_sft_multinode import infra


app = infra.app


@app.local_entrypoint()
def main(
    mode: str = "qwen-lora-sft",
    dataset_path: str = str(infra.DEFAULT_DATASET_PATH),
    artifact_root: str = str(infra.ARTIFACT_ROOT),
    base_model: str = infra.MODEL,
    comparison_id: str = "",
    n_nodes: int = infra.CLUSTER_SIZE,
    gpus_per_node: int = infra.GPUS_PER_NODE,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    micro_batch_size: int = 1,
    global_batch_size: int = 0,
    sequence_length: int = 32,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = 1,
    save_every: int = 0,
    seed: int = 1234,
    master_port: int = infra.MASTER_PORT,
) -> None:
    if mode not in {"torchrun-env", "nccl-smoke", "qwen-lora-sft", "hf-ddp-baseline"}:
        raise ValueError(f"unsupported mode: {mode}")
    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    run_id = comparison_id or f"qwen-multinode-{uuid.uuid4().hex[:8]}"
    config = infra.job_config(
        mode=mode,
        comparison_id=run_id,
        dataset_path=dataset_path,
        artifact_root=artifact_root,
        base_model=base_model,
        n_nodes=n_nodes,
        gpus_per_node=gpus_per_node,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size or None,
        sequence_length=sequence_length,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        max_steps=max_steps,
        save_every=save_every,
        seed=seed,
        master_port=master_port,
    )
    result = infra.run_clustered_trainer.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
