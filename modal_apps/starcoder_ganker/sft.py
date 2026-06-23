"""Run Ganker LoRA SFT on a StarCoderData-style code dataset.

Run:

    source ~/.codex/modal.env
    GANKER_STARCODER_NODES=2 uv run modal run modal_apps/starcoder_ganker/sft.py
"""

from __future__ import annotations

import json
import uuid

from modal_apps.starcoder_ganker import common, infra


app = infra.app


@app.local_entrypoint()
def main(
    dataset_path: str = str(common.DEFAULT_DATASET_PATH),
    artifact_root: str = str(infra.ARTIFACT_ROOT),
    base_model: str = common.MODEL,
    run_id: str = "",
    n_nodes: int = infra.CLUSTER_SIZE,
    gpus_per_node: int = infra.GPUS_PER_NODE,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    micro_batch_size: int = 1,
    global_batch_size: int = 0,
    sequence_length: int = 512,
    lora_rank: int = 8,
    learning_rate: float = 1e-4,
    max_steps: int = 10,
    save_every: int = 0,
    seed: int = 1234,
    master_port: int = infra.MASTER_PORT,
    single_node: bool = False,
    prepare_dataset: bool = False,
    dataset_id: str = common.DATASET_ID,
    languages: str = ",".join(common.DEFAULT_LANGUAGES),
    dataset_examples: int = 256,
) -> None:
    if prepare_dataset:
        dataset_payload = infra.prepare_starcoder_dataset.remote(
            dataset_path=dataset_path,
            dataset_id=dataset_id,
            languages=common.csv_list(languages),
            max_examples=dataset_examples,
            seed=seed,
        )
        print(json.dumps({"prepared_dataset": dataset_payload}, indent=2, sort_keys=True))

    comparison_id = run_id or f"{common.RUN_PREFIX}-{uuid.uuid4().hex[:8]}"
    config = infra.job_config(
        comparison_id=comparison_id,
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
    if single_node:
        result = infra.run_single_node_code_sft.remote(config)
    else:
        result = infra.run_clustered_code_sft.remote(config)
    print(json.dumps(result, indent=2, sort_keys=True))
