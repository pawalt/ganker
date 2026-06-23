"""Torchrun child entrypoint for clustered Qwen SFT jobs."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.megatron import MegatronTrainingBackend
from ganker.config import MegatronBackendConfig
from ganker.contracts import AdamParams, ArtifactKind, TuningMode
from ganker.distributed.torchrun import (
    DistributedTrainingConfig,
    MegatronRankInfo,
    rank_info_from_global_rank,
    select_data_parallel_items,
    write_json,
)


def main() -> None:
    args = parse_args()
    if args.mode == "torchrun-env":
        run_torchrun_env(args)
    elif args.mode == "nccl-smoke":
        run_nccl_smoke(args)
    elif args.mode == "qwen-lora-sft":
        run_qwen_lora_sft(args)
    elif args.mode == "hf-ddp-baseline":
        run_hf_ddp_baseline(args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--tensor-model-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--comparison-id", required=True)
    return parser.parse_args()


def run_torchrun_env(args: argparse.Namespace) -> None:
    torch = importlib.import_module("torch")
    dist = importlib.import_module("torch.distributed")
    dist.init_process_group("gloo")
    rank = _rank()
    payload = {
        "rank": rank,
        "world_size": _world_size(),
        "local_rank": _local_rank(),
        "master_addr": os.environ.get("MASTER_ADDR", ""),
        "master_port": os.environ.get("MASTER_PORT", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "hostname": importlib.import_module("socket").gethostname(),
    }
    gathered = _gather_objects(payload, dst=0)
    if rank == 0:
        write_json(
            args.result_path,
            {
                "ok": True,
                "mode": "torchrun-env",
                "ranks": gathered,
                "world_size": _world_size(),
            },
        )
    dist.barrier()
    dist.destroy_process_group()


def run_nccl_smoke(args: argparse.Namespace) -> None:
    torch = importlib.import_module("torch")
    dist = importlib.import_module("torch.distributed")
    if not torch.cuda.is_available():
        raise RuntimeError("nccl-smoke requires CUDA")
    torch.cuda.set_device(_local_rank())
    dist.init_process_group("nccl")
    rank = _rank()
    device = torch.device("cuda", _local_rank())
    value = torch.tensor([rank + 1], dtype=torch.float32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = _world_size() * (_world_size() + 1) / 2
    if rank == 0:
        write_json(
            args.result_path,
            {
                "ok": bool(math.isclose(float(value.item()), expected)),
                "mode": "nccl-smoke",
                "world_size": _world_size(),
                "all_reduce_sum": float(value.item()),
                "expected_sum": float(expected),
            },
        )
    dist.barrier()
    dist.destroy_process_group()


def run_qwen_lora_sft(args: argparse.Namespace) -> None:
    distributed = _distributed_config(args)
    distributed.require_supported_model_parallel()
    dist = importlib.import_module("torch.distributed")
    rank = _rank()
    shared_artifact_root = Path(args.artifact_root)
    artifact_root = (
        shared_artifact_root
        if rank == 0
        else Path("/tmp/ganker-rank-artifacts") / f"rank-{rank:05d}"
    )
    os.environ["GANKER_ARTIFACT_ROOT"] = str(artifact_root)
    os.environ["GANKER_MEGATRON_CHECKPOINT_ROOT"] = str(shared_artifact_root)

    config = MegatronBackendConfig(
        runtime_kind="bridge",
        tensor_model_parallel_size=distributed.tensor_model_parallel_size,
        pipeline_model_parallel_size=distributed.pipeline_model_parallel_size,
        micro_batch_size=distributed.micro_batch_size,
        global_batch_size=distributed.effective_global_batch_size,
        sequence_length=int(args.sequence_length),
        trust_remote_code=True,
        load_weights=True,
        tensor_device="cuda",
        seed=int(args.seed),
    )
    backend = MegatronTrainingBackend(
        FilesystemArtifactStore(artifact_root),
        config=config,
    )
    run = backend.create_training_run(
        base_model=str(args.base_model),
        tuning_mode=TuningMode.LORA,
        lora_rank=int(args.lora_rank),
    )
    rank_info = _megatron_rank_info(distributed)

    tokenizer, batches = _load_batches(args)
    _ = tokenizer

    losses: list[float] = []
    optimizer_step = 0
    checkpoint_version = 0
    for step in range(int(args.max_steps)):
        local_microbatches = select_data_parallel_items(
            batches,
            step=step,
            data_parallel_rank=rank_info.data_parallel_rank,
            data_parallel_size=rank_info.data_parallel_size,
            grad_accum_steps=distributed.grad_accum_steps,
        )
        batch = [datum for microbatch in local_microbatches for datum in microbatch]
        fb = backend.forward_backward(
            run_id=run.run_id,
            data=batch,
            loss_fn="cross_entropy",
            loss_fn_config={},
        )
        step_result = backend.optim_step(
            run_id=run.run_id,
            params=AdamParams(learning_rate=float(args.learning_rate)),
        )
        optimizer_step = int(step_result.optimizer_step)
        checkpoint_version = int(step_result.checkpoint_version)
        losses.append(_average_scalar(float(fb.output.loss)))
        if rank == 0:
            print(
                json.dumps(
                    {
                        "event": "qwen_lora_sft_step",
                        "step": step + 1,
                        "loss": losses[-1],
                        "optimizer_step": optimizer_step,
                        "checkpoint_version": checkpoint_version,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(json.dumps({"event": "qwen_lora_sft_save_start"}, sort_keys=True), flush=True)
    saved = backend.save_weights(run_id=run.run_id, kind=ArtifactKind.DELTA)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(json.dumps({"event": "qwen_lora_sft_save_done"}, sort_keys=True), flush=True)

    if rank == 0:
        artifact_payload = _artifact_payload(saved.payload_path)
        write_json(
            args.result_path,
            {
                "ok": True,
                "mode": "qwen-lora-sft-multinode",
                "framework": "ganker-megatron-bridge",
                "model": str(args.base_model),
                "comparison_id": str(args.comparison_id),
                "run_id": run.run_id,
                "dataset_path": str(args.dataset_path),
                "batch_count": len(batches),
                "steps": len(losses),
                "losses": losses,
                "loss_curve": _loss_curve_records(losses),
                "final_loss": losses[-1] if losses else None,
                "optimizer_step": optimizer_step,
                "checkpoint_version": checkpoint_version,
                "artifact_path": saved.payload_path if saved is not None else "",
                "manifest_path": saved.manifest_path if saved is not None else "",
                "artifact": artifact_payload,
                "distributed": distributed.as_dict(),
                "rank": rank_info.as_dict(),
            },
        )

    backend.close()


def run_hf_ddp_baseline(args: argparse.Namespace) -> None:
    distributed = _distributed_config(args)
    distributed.require_dp_only()
    if int(args.max_steps) <= 0:
        raise ValueError("max_steps must be positive")

    torch = importlib.import_module("torch")
    torch_nn_functional = importlib.import_module("torch.nn.functional")
    torch_utils_data = importlib.import_module("torch.utils.data")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")

    if torch.cuda.is_available():
        torch.cuda.set_device(_local_rank())
    seed = int(args.seed)
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from examples.sft import HFAutoTokenizerAdapter, SFTDataConfig, encode_sft_example, load_jsonl_examples

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.base_model), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_adapter = HFAutoTokenizerAdapter(tokenizer)
    examples = load_jsonl_examples(str(args.dataset_path))
    random_module = importlib.import_module("random")
    random_module.Random(seed).shuffle(examples)

    data_config = SFTDataConfig(
        sequence_length=int(args.sequence_length),
        batch_size=int(args.micro_batch_size),
        shuffle=False,
        seed=seed,
    )
    features: list[dict[str, Any]] = []
    for example in examples:
        datum = encode_sft_example(example, tokenizer=tokenizer_adapter, config=data_config)
        if datum is None:
            continue
        features.append(
            {
                "input_ids": list(datum.model_input.token_ids),
                "target_tokens": [int(value) for value in datum.loss_fn_inputs["target_tokens"].tolist()],
                "weights": [float(value) for value in datum.loss_fn_inputs["weights"].tolist()],
            }
        )
    if not features:
        raise ValueError("dataset produced no HF Trainer features")

    class ListDataset(torch_utils_data.Dataset):
        def __len__(self) -> int:
            return len(features)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return features[index]

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "input_ids": torch.tensor([row["input_ids"] for row in batch], dtype=torch.long),
            "target_tokens": torch.tensor([row["target_tokens"] for row in batch], dtype=torch.long),
            "weights": torch.tensor([row["weights"] for row in batch], dtype=torch.float32),
        }

    class ComparableTrainer(transformers.Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(
                    [param for param in self.model.parameters() if param.requires_grad],
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                    weight_decay=self.args.weight_decay,
                )
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            _ = kwargs
            target_tokens = inputs.pop("target_tokens")
            weights = inputs.pop("weights")
            outputs = model(**inputs)
            logits = outputs.logits.float()
            losses = torch_nn_functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_tokens.reshape(-1),
                reduction="none",
            )
            flat_weights = weights.reshape(-1).float()
            loss = (losses * flat_weights).sum() / flat_weights.sum().clamp_min(1.0)
            return (loss, outputs) if return_outputs else loss

    class LossRecorder(transformers.TrainerCallback):
        def __init__(self) -> None:
            self.losses: list[dict[str, float | int]] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = args, control, kwargs
            if logs is None or "loss" not in logs:
                return
            if getattr(state, "is_world_process_zero", True):
                self.losses.append({"step": int(state.global_step), "loss": float(logs["loss"])})

    output_dir = Path(str(args.artifact_root)) / "hf-ddp" / str(args.comparison_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    lora_rank = int(args.lora_rank)
    lora_config = peft.LoraConfig(
        r=lora_rank,
        lora_alpha=2 * lora_rank,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = peft.get_peft_model(model, lora_config)

    recorder = LossRecorder()
    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(args.micro_batch_size),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=bool(torch.cuda.is_available()),
        fp16=False,
        remove_unused_columns=False,
        lr_scheduler_type="constant",
        warmup_steps=0,
        seed=seed,
        data_seed=seed,
        ddp_find_unused_parameters=False,
    )
    trainer = ComparableTrainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(),
        data_collator=collate,
        callbacks=[recorder],
    )
    train_result = trainer.train()
    losses = [float(point["loss"]) for point in recorder.losses]
    if trainer.is_world_process_zero():
        adapter_dir = output_dir / "adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir / "tokenizer")
        if not losses:
            raise ValueError("HF Trainer did not emit any step losses")
        write_json(
            args.result_path,
            {
                "ok": True,
                "mode": "hf-ddp-baseline",
                "framework": "huggingface-trainer-peft-ddp",
                "model": str(args.base_model),
                "comparison_id": str(args.comparison_id),
                "dataset_path": str(args.dataset_path),
                "feature_count": len(features),
                "steps": len(losses),
                "losses": losses,
                "loss_curve": recorder.losses,
                "final_loss": losses[-1],
                "output_dir": str(output_dir),
                "adapter_dir": str(adapter_dir),
                "train_metrics": {key: float(value) for key, value in train_result.metrics.items()},
                "distributed": distributed.as_dict(),
            },
        )
    _barrier_and_destroy()


def _distributed_config(args: argparse.Namespace) -> DistributedTrainingConfig:
    world_size = _world_size()
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if gpus_per_node <= 0:
        gpus_per_node = 1
    n_nodes = max(world_size // gpus_per_node, 1)
    return DistributedTrainingConfig(
        n_nodes=n_nodes,
        gpus_per_node=gpus_per_node,
        tensor_model_parallel_size=int(args.tensor_model_parallel_size),
        pipeline_model_parallel_size=int(args.pipeline_model_parallel_size),
        micro_batch_size=int(args.micro_batch_size),
        global_batch_size=int(args.global_batch_size),
    )


def _megatron_rank_info(distributed: DistributedTrainingConfig) -> MegatronRankInfo:
    try:
        parallel_state = importlib.import_module("megatron.core.parallel_state")
    except ImportError:
        return rank_info_from_global_rank(distributed, global_rank=_rank())
    if not parallel_state.model_parallel_is_initialized():
        return rank_info_from_global_rank(distributed, global_rank=_rank())
    return MegatronRankInfo(
        global_rank=_rank(),
        world_size=_world_size(),
        data_parallel_rank=int(parallel_state.get_data_parallel_rank()),
        data_parallel_size=int(parallel_state.get_data_parallel_world_size()),
        tensor_model_parallel_rank=int(parallel_state.get_tensor_model_parallel_rank()),
        tensor_model_parallel_size=distributed.tensor_model_parallel_size,
        pipeline_model_parallel_rank=int(parallel_state.get_pipeline_model_parallel_rank()),
        pipeline_model_parallel_size=distributed.pipeline_model_parallel_size,
    )


def _load_batches(args: argparse.Namespace):
    from examples.sft import HFAutoTokenizerAdapter, SFTDataConfig, load_jsonl_sft_batches

    tokenizer = HFAutoTokenizerAdapter.from_pretrained(str(args.base_model))
    batches = load_jsonl_sft_batches(
        str(args.dataset_path),
        tokenizer=tokenizer,
        config=SFTDataConfig(
            sequence_length=int(args.sequence_length),
            batch_size=int(args.micro_batch_size),
            shuffle=True,
            seed=int(args.seed),
        ),
    )
    return tokenizer, batches


def _average_scalar(value: float) -> float:
    torch = importlib.import_module("torch")
    dist = importlib.import_module("torch.distributed")
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    device = torch.device("cuda", _local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def _gather_objects(payload: dict[str, Any], *, dst: int) -> list[dict[str, Any]]:
    dist = importlib.import_module("torch.distributed")
    rank = _rank()
    gathered = [None for _ in range(_world_size())] if rank == dst else None
    dist.gather_object(payload, object_gather_list=gathered, dst=dst)
    if rank == dst:
        if gathered is None:
            return []
        return [item for item in gathered if item is not None]
    return []


def _barrier_and_destroy() -> None:
    dist = importlib.import_module("torch.distributed")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _loss_curve_records(losses: list[float]) -> list[dict[str, float | int]]:
    return [{"step": index + 1, "loss": float(loss)} for index, loss in enumerate(losses)]


def _artifact_payload(path: str) -> dict[str, Any]:
    if not path:
        return {}
    payload_path = Path(path)
    if not payload_path.exists():
        return {"payload_path": str(payload_path), "payload_exists": False}
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["payload_path"] = str(payload_path)
    payload["payload_exists"] = True
    return payload


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


if __name__ == "__main__":
    main()
