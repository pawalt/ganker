from pathlib import Path

import pytest

from ganker.distributed.torchrun import (
    DistributedTrainingConfig,
    TorchrunLaunchConfig,
    entrypoint_args_from_mapping,
    parse_gpu_count,
    rank_result_path,
    read_json,
    select_data_parallel_item,
    training_config_from_mapping,
    write_json,
)


def test_distributed_training_config_derives_dp_and_grad_accum():
    config = DistributedTrainingConfig(
        n_nodes=2,
        gpus_per_node=8,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        micro_batch_size=1,
        global_batch_size=16,
    )

    assert config.world_size == 16
    assert config.data_parallel_size == 16
    assert config.grad_accum_steps == 1
    assert config.as_dict()["world_size"] == 16


def test_distributed_training_config_rejects_invalid_global_batch():
    with pytest.raises(ValueError, match="global_batch_size"):
        DistributedTrainingConfig(
            n_nodes=2,
            gpus_per_node=8,
            micro_batch_size=1,
            global_batch_size=15,
        )


def test_distributed_training_config_first_milestone_is_dp_only():
    config = DistributedTrainingConfig(
        n_nodes=2,
        gpus_per_node=8,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        micro_batch_size=1,
        global_batch_size=8,
    )

    with pytest.raises(ValueError, match="DP-only"):
        config.require_dp_only()


def test_torchrun_launch_args_match_modal_cluster_shape():
    launch = TorchrunLaunchConfig(
        nnodes=2,
        nproc_per_node=8,
        node_rank=1,
        master_addr="fd00::1",
        master_port=29501,
    )

    assert launch.distributed_run_args("/workspace/train.py", ["--mode", "env"]) == [
        "--nnodes=2",
        "--nproc-per-node=8",
        "--node-rank=1",
        "--master-addr=fd00::1",
        "--master-port=29501",
        "/workspace/train.py",
        "--mode",
        "env",
    ]
    assert launch.torchrun_argv("/workspace/train.py")[0] == "torchrun"
    assert launch.env()["NODE_RANK"] == "1"


def test_parse_gpu_count_handles_modal_gpu_specs():
    assert parse_gpu_count("H100:8") == 8
    assert parse_gpu_count("A100") == 1
    assert parse_gpu_count("", default=4) == 4
    with pytest.raises(ValueError, match="invalid Modal GPU spec"):
        parse_gpu_count("H100:zero")


def test_training_config_from_mapping_and_entrypoint_args():
    values = {
        "mode": "qwen-lora-sft",
        "comparison_id": "cmp",
        "dataset_path": "/data/train.jsonl",
        "artifact_root": "/vol/artifacts",
        "base_model": "Qwen/Qwen3-0.6B",
        "n_nodes": 2,
        "gpus_per_node": 8,
        "micro_batch_size": 1,
        "global_batch_size": 16,
        "sequence_length": 128,
        "lora_rank": 8,
        "learning_rate": 1e-4,
        "max_steps": 2,
        "seed": 1234,
    }

    config = training_config_from_mapping(values)
    args = entrypoint_args_from_mapping(values, result_path="/tmp/result.json")

    assert config.world_size == 16
    assert args[:4] == ["--mode", "qwen-lora-sft", "--result-path", "/tmp/result.json"]
    assert "--base-model" in args
    assert "Qwen/Qwen3-0.6B" in args


def test_select_data_parallel_item_round_robins_by_step_and_rank():
    items = ["a", "b", "c", "d"]

    assert select_data_parallel_item(items, step=0, data_parallel_rank=0, data_parallel_size=2) == "a"
    assert select_data_parallel_item(items, step=0, data_parallel_rank=1, data_parallel_size=2) == "b"
    assert select_data_parallel_item(items, step=1, data_parallel_rank=0, data_parallel_size=2) == "c"
    assert select_data_parallel_item(items, step=2, data_parallel_rank=1, data_parallel_size=2) == "b"


def test_json_helpers_round_trip_rank_result(tmp_path: Path):
    result_path = tmp_path / "result.json"
    per_rank = rank_result_path(result_path, 7)

    write_json(per_rank, {"ok": True, "rank": 7})

    assert per_rank.name == "result.rank-00007.json"
    assert read_json(per_rank) == {"ok": True, "rank": 7}
