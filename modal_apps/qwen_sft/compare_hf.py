"""Compare Qwen LoRA SFT losses against a Hugging Face Trainer baseline.

Run:

    source ~/.codex/modal.env
    GANKER_MODAL_GPU=A100 uv run modal run modal_apps/qwen_sft/compare_hf.py \
      --startup-timeout 900 \
      --dataset-size 256 \
      --max-steps 20 \
      --sequence-length 256
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any
import uuid

import modal

from ganker.client import ServiceClient

from examples.sft import (
    DEFAULT_REAL_DATASET,
    DEFAULT_REAL_DATASET_FORMAT,
    DEFAULT_REAL_DATASET_SPLIT,
)
from modal_apps.qwen_sft import infra


app = infra.app
JOB_MODULE = "modal_apps.qwen_sft.compare_hf"
GANKER_JOB_FUNCTION = "run_qwen_lora_sft_loss_curve"
DEFAULT_DATASET_PATH = infra.ARTIFACT_ROOT / "datasets" / "alpaca-qwen-sft.jsonl"


def _hf_trainer_image() -> modal.Image:
    return (
        modal.Image.from_registry(infra.BRIDGE_BASE_IMAGE)
        .apt_install("git", "curl")
        .run_commands(
            f"curl -LsSf https://astral.sh/uv/{infra.BRIDGE_UV_VERSION}/install.sh | sh",
            "rm -rf /opt/venv",
            "/root/.local/bin/uv venv /opt/venv",
            (
                "UV_PROJECT_ENVIRONMENT=/opt/venv "
                "/root/.local/bin/uv pip install --python /opt/venv/bin/python "
                "'datasets>=2.20' 'accelerate>=0.33' 'peft>=0.13' "
                "'transformers>=4.51,<5' 'safetensors>=0.4' "
                "'grpcio>=1.81.1' 'protobuf>=6.33.6' "
                f"'torchmonarch=={infra.TORCHMONARCH_VERSION}'"
            ),
        )
        .env(
            {
                "PATH": "/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "VIRTUAL_ENV": "/opt/venv",
                "UV_PROJECT_ENVIRONMENT": "/opt/venv",
                "PYTHONPATH": f"{infra.REMOTE_ROOT}:{infra.REMOTE_ROOT / 'src'}",
                "GANKER_ARTIFACT_ROOT": infra.ARTIFACT_MOUNT,
                "HF_HUB_CACHE": "/root/.cache/huggingface",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
        .add_local_dir(infra.PROJECT_ROOT, remote_path=str(infra.REMOTE_ROOT), ignore=infra._repo_ignore())
    )


hf_trainer_image = _hf_trainer_image()


def run_qwen_lora_sft_loss_curve(
    client: ServiceClient,
    context: infra.JobContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the Ganker/Megatron Bridge side of the loss-curve comparison."""

    from examples.sft import HFAutoTokenizerAdapter, SFTDataConfig, load_jsonl_sft_batches, run_sft

    tokenizer = HFAutoTokenizerAdapter.from_pretrained(infra.MODEL)
    batches = load_jsonl_sft_batches(
        str(config["dataset_path"]),
        tokenizer=tokenizer,
        config=SFTDataConfig(
            sequence_length=int(config["sequence_length"]),
            batch_size=int(config["micro_batch_size"]),
            shuffle=True,
            seed=int(config["seed"]),
        ),
    )
    summary = run_sft(
        client,
        base_model=infra.MODEL,
        dataset=batches,
        tuning="lora",
        lora_rank=int(config["lora_rank"]),
        learning_rate=float(config["learning_rate"]),
        max_steps=int(config["max_steps"]),
        save_every=int(config["save_every"]),
    )
    telemetry = client.get_telemetry_summary(summary.run_id, request_id="qwen-loss-curve-telemetry")
    context.reload_artifacts()

    return {
        "ok": True,
        "framework": "ganker-megatron-bridge",
        "model": infra.MODEL,
        "dataset_path": str(config["dataset_path"]),
        "batch_count": len(batches),
        "loss_curve": loss_curve_records(summary.losses),
        "telemetry_events": telemetry.summary.event_count,
        "telemetry_input_tokens": telemetry.summary.total.input_tokens,
        "telemetry_training_steps": telemetry.summary.total.training_steps,
        **context.base_payload(),
        **summary.to_dict(),
    }


@app.function(
    image=hf_trainer_image,
    timeout=60 * 60,
    region=infra.REGION,
    volumes={
        infra.ARTIFACT_MOUNT: infra.artifact_volume,
        "/root/.cache/huggingface": infra.hf_cache_volume,
    },
    secrets=infra._hf_secrets(),
)
def prepare_real_sft_dataset(
    dataset_path: str,
    dataset_name: str,
    dataset_split: str,
    dataset_format: str,
    dataset_size: int,
    seed: int,
) -> dict[str, Any]:
    infra.add_remote_import_paths()
    from examples.sft import materialize_hf_sft_jsonl

    payload = materialize_hf_sft_jsonl(
        dataset_path,
        dataset_name=dataset_name,
        split=dataset_split,
        dataset_format=dataset_format,
        max_examples=dataset_size,
        seed=seed,
    )
    infra.artifact_volume.commit()
    return payload


@app.function(
    image=hf_trainer_image,
    gpu=infra.GPU,
    timeout=60 * 60,
    region=infra.REGION,
    volumes={
        infra.ARTIFACT_MOUNT: infra.artifact_volume,
        "/root/.cache/huggingface": infra.hf_cache_volume,
    },
    secrets=infra._hf_secrets(),
)
def run_hf_trainer_baseline(config: dict[str, Any]) -> dict[str, Any]:
    infra.add_remote_import_paths()
    infra.artifact_volume.reload()

    torch = importlib.import_module("torch")
    torch_nn_functional = importlib.import_module("torch.nn.functional")
    torch_utils_data = importlib.import_module("torch.utils.data")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")

    from examples.sft import HFAutoTokenizerAdapter, SFTDataConfig, encode_sft_example, load_jsonl_examples

    seed = int(config["seed"])
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = transformers.AutoTokenizer.from_pretrained(infra.MODEL, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer_adapter = HFAutoTokenizerAdapter(tokenizer)

    examples = load_jsonl_examples(str(config["dataset_path"]))
    random_module = importlib.import_module("random")
    random_module.Random(seed).shuffle(examples)

    data_config = SFTDataConfig(
        sequence_length=int(config["sequence_length"]),
        batch_size=int(config["micro_batch_size"]),
        shuffle=False,
        seed=seed,
    )
    features = []
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

    class GankerComparableTrainer(transformers.Trainer):
        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer requires a train_dataset")
            return torch_utils_data.DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                shuffle=False,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

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
            self.losses.append({"step": int(state.global_step), "loss": float(logs["loss"])})

    output_dir = Path(str(config["artifact_root"])) / "hf-trainer" / str(config["comparison_id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        infra.MODEL,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    lora_rank = int(config["lora_rank"])
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
        per_device_train_batch_size=int(config["micro_batch_size"]),
        max_steps=int(config["max_steps"]),
        learning_rate=float(config["learning_rate"]),
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
    )
    trainer = GankerComparableTrainer(
        model=model,
        args=training_args,
        train_dataset=ListDataset(),
        data_collator=collate,
        callbacks=[recorder],
    )
    train_result = trainer.train()
    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    infra.artifact_volume.commit()

    losses = [float(point["loss"]) for point in recorder.losses]
    if not losses:
        raise ValueError("HF Trainer did not emit any step losses")
    return {
        "ok": True,
        "framework": "huggingface-trainer-peft",
        "model": infra.MODEL,
        "dataset_path": str(config["dataset_path"]),
        "feature_count": len(features),
        "steps": len(losses),
        "losses": losses,
        "loss_curve": recorder.losses,
        "final_loss": losses[-1],
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "train_metrics": {key: float(value) for key, value in train_result.metrics.items()},
    }


def loss_curve_records(losses: list[float]) -> list[dict[str, float | int]]:
    return [{"step": index + 1, "loss": float(loss)} for index, loss in enumerate(losses)]


def compare_loss_curves(ganker_losses: list[float], hf_losses: list[float]) -> dict[str, Any]:
    pair_count = min(len(ganker_losses), len(hf_losses))
    if pair_count == 0:
        return {"ok": False, "reason": "one or both loss curves are empty"}
    pairs = list(zip(ganker_losses[:pair_count], hf_losses[:pair_count], strict=True))
    abs_diffs = [abs(left - right) for left, right in pairs]
    ganker_delta = ganker_losses[pair_count - 1] - ganker_losses[0]
    hf_delta = hf_losses[pair_count - 1] - hf_losses[0]
    return {
        "ok": True,
        "paired_steps": pair_count,
        "all_losses_finite": all(math.isfinite(value) for pair in pairs for value in pair),
        "ganker_initial_loss": float(ganker_losses[0]),
        "ganker_final_loss": float(ganker_losses[pair_count - 1]),
        "ganker_loss_delta": float(ganker_delta),
        "hf_initial_loss": float(hf_losses[0]),
        "hf_final_loss": float(hf_losses[pair_count - 1]),
        "hf_loss_delta": float(hf_delta),
        "final_loss_abs_diff": float(abs(ganker_losses[pair_count - 1] - hf_losses[pair_count - 1])),
        "mean_abs_loss_diff": float(sum(abs_diffs) / len(abs_diffs)),
        "direction_agrees": (ganker_delta <= 0 and hf_delta <= 0) or (ganker_delta >= 0 and hf_delta >= 0),
    }


def _job_config(
    *,
    dataset_path: str,
    artifact_root: str,
    comparison_id: str,
    lora_rank: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "dataset_path": dataset_path,
        "artifact_root": artifact_root,
        "comparison_id": comparison_id,
        "lora_rank": lora_rank,
        "max_steps": max_steps,
        "save_every": save_every,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "micro_batch_size": micro_batch_size,
        "seed": seed,
    }


@app.local_entrypoint()
def main(
    dataset_name: str = DEFAULT_REAL_DATASET,
    dataset_split: str = DEFAULT_REAL_DATASET_SPLIT,
    dataset_format: str = DEFAULT_REAL_DATASET_FORMAT,
    dataset_size: int = 256,
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    artifact_root: str = str(infra.ARTIFACT_ROOT),
    lora_rank: int = 8,
    max_steps: int = 20,
    save_every: int = 0,
    learning_rate: float = 1e-4,
    sequence_length: int = 256,
    micro_batch_size: int = 1,
    seed: int = 1234,
    port: int = infra.MONARCH_PORT,
    controller_port: int = infra.CONTROLLER_PORT,
    startup_timeout: int = 900,
    deployment_id: str = "",
    run_id: str = "run-000001",
) -> None:
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    comparison_id = deployment_id or f"qwen-loss-compare-{uuid.uuid4().hex[:8]}"
    dataset_info = prepare_real_sft_dataset.remote(
        dataset_path,
        dataset_name,
        dataset_split,
        dataset_format,
        dataset_size,
        seed,
    )
    shared_config = _job_config(
        dataset_path=dataset_path,
        artifact_root=artifact_root,
        comparison_id=comparison_id,
        lora_rank=lora_rank,
        max_steps=max_steps,
        save_every=save_every,
        learning_rate=learning_rate,
        sequence_length=sequence_length,
        micro_batch_size=micro_batch_size,
        seed=seed,
    )
    ganker_result = infra.run_training_job.remote(
        comparison_id,
        run_id,
        artifact_root,
        port,
        controller_port,
        startup_timeout,
        JOB_MODULE,
        GANKER_JOB_FUNCTION,
        shared_config,
    )
    hf_result = run_hf_trainer_baseline.remote(shared_config)
    report = {
        "ok": True,
        "mode": "qwen-loss-curve-comparison",
        "comparison_id": comparison_id,
        "dataset": dataset_info,
        "config": shared_config,
        "comparison": compare_loss_curves(ganker_result["losses"], hf_result["losses"]),
        "ganker": ganker_result,
        "huggingface": hf_result,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
