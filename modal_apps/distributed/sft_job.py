"""Tinker-style SFT jobs that run against the distributed Modal infra.

The infra is deployable independently:

    uv run modal deploy modal_apps/distributed/infra.py

Run the Qwen/Megatron Bridge + SGLang rollout job:

    source ~/.codex/modal.env
    GANKER_MODAL_GPU=A100 uv run modal run modal_apps/distributed/sft_job.py \
      --mode qwen-bridge-sglang-distributed \
      --startup-timeout 900 \
      --sglang-startup-timeout 900
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast
import uuid

from ganker.client import SamplingClient, ServiceClient, TrainingClient
from ganker.contracts import (
    ArtifactKind,
    ModelInput,
    SamplingParams,
    TrainingRun,
    TuningMode,
    WeightArtifact,
)

from modal_apps.distributed import infra


app = infra.app
JOB_MODULE = "modal_apps.distributed.sft_job"
DEFAULT_PROMPT = "Answer in one short sentence: what is 2+2?"


def _require_tuning(tuning: str, lora_rank: int) -> Literal["full", "lora"]:
    if tuning not in {"full", "lora"}:
        raise ValueError("tuning must be 'full' or 'lora'")
    if tuning == "lora" and lora_rank <= 0:
        raise ValueError("lora_rank must be positive for LoRA")
    return cast(Literal["full", "lora"], tuning)


def _artifact_for_summary(
    *,
    run_id: str,
    checkpoint_version: int,
    artifact_path: str,
    manifest_path: str,
    tuning: str,
) -> WeightArtifact:
    return WeightArtifact(
        run_id=run_id,
        checkpoint_version=checkpoint_version,
        kind=ArtifactKind.DELTA if tuning == "lora" else ArtifactKind.FULL,
        manifest_path=manifest_path,
        payload_path=artifact_path,
    )


def _training_run_for_summary(
    *,
    run_id: str,
    base_model: str,
    checkpoint_version: int,
    tuning: str,
    lora_rank: int,
) -> TrainingRun:
    return TrainingRun(
        run_id=run_id,
        base_model=base_model,
        tuning_mode=TuningMode.LORA if tuning == "lora" else TuningMode.FULL,
        lora_rank=lora_rank if tuning == "lora" else 0,
        checkpoint_version=checkpoint_version,
    )


def _artifact_details(artifact_path: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_exists": Path(artifact_path).exists(),
    }
    if not payload["artifact_exists"]:
        return payload

    artifact_payload = json.loads(Path(artifact_path).read_text())
    payload["artifact_format"] = artifact_payload.get("artifact_format")
    for key in (
        "hf_checkpoint_path",
        "hf_adapter_path",
        "hf_weights_path",
        "hf_weights_index_path",
        "hf_adapter_config_path",
        "hf_adapter_weights_path",
        "hf_checkpoint_bytes",
        "hf_weight_count",
        "hf_weight_format",
    ):
        if key in artifact_payload:
            payload[key] = artifact_payload[key]
    for key in (
        "hf_checkpoint_path",
        "hf_adapter_path",
        "hf_weights_path",
        "hf_weights_index_path",
        "hf_adapter_config_path",
        "hf_adapter_weights_path",
    ):
        if key in payload:
            payload[f"{key}_exists"] = Path(payload[key]).exists()
    return payload


def run_toy_sft(
    client: ServiceClient,
    context: infra.DistributedJobContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    from examples.sft import SFTDataConfig, ToyTokenizer, load_jsonl_sft_batches, run_sft

    base_model = str(config["base_model"])
    tuning = str(config["tuning"])
    lora_rank = int(config["lora_rank"])
    tuning_literal = _require_tuning(tuning, lora_rank)
    tokenizer = ToyTokenizer(vocab_size=int(config["vocab_size"]))
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
        base_model=base_model,
        dataset=batches,
        tuning=tuning_literal,
        lora_rank=lora_rank if tuning == "lora" else 0,
        learning_rate=float(config["learning_rate"]),
        max_steps=int(config["max_steps"]),
        save_every=int(config["save_every"]),
    )
    artifact = _artifact_for_summary(
        run_id=summary.run_id,
        checkpoint_version=summary.checkpoint_version,
        artifact_path=summary.artifact_path,
        manifest_path=summary.manifest_path,
        tuning=tuning,
    )
    training_run = _training_run_for_summary(
        run_id=summary.run_id,
        base_model=base_model,
        checkpoint_version=summary.checkpoint_version,
        tuning=tuning,
        lora_rank=lora_rank,
    )
    training = TrainingClient(service=client, run=training_run)
    refreshed = training.refresh_weights(
        artifact,
        request_id="modal-distributed-sft-refresh",
    )
    sampler = SamplingClient(service=client, run=training_run, artifact=refreshed.artifact)
    sample = sampler.sample(
        ModelInput.from_ints([7, 8]),
        SamplingParams(max_tokens=4, temperature=0.7, top_p=0.9),
        request_id="modal-distributed-sft-sample",
    )
    telemetry = sampler.get_telemetry_summary(
        request_id="modal-distributed-sft-telemetry",
    )
    context.reload_artifacts()

    payload = {
        "ok": True,
        "mode": "sft-distributed",
        "dataset_path": str(config["dataset_path"]),
        "batch_count": len(batches),
        "sample_tokens": sample.sequences[0].tokens,
        "sample_checkpoint_version": sample.artifact.checkpoint_version,
        "telemetry_events": telemetry.summary.event_count,
        "telemetry_input_tokens": telemetry.summary.total.input_tokens,
        "telemetry_output_tokens": telemetry.summary.total.output_tokens,
        "telemetry_training_steps": telemetry.summary.total.training_steps,
        "telemetry_samples": telemetry.summary.total.samples,
        **context.base_payload(),
        **summary.to_dict(),
    }
    payload["manifest_exists"] = Path(summary.manifest_path).exists()
    payload.update(_artifact_details(summary.artifact_path))
    return payload


def run_qwen_sft(
    client: ServiceClient,
    context: infra.DistributedJobContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    from examples.sft import HFAutoTokenizerAdapter, SFTDataConfig, load_jsonl_sft_batches, run_sft

    base_model = str(config["base_model"])
    tuning = str(config["tuning"])
    lora_rank = int(config["lora_rank"])
    sample_max_tokens = int(config["sample_max_tokens"])
    if sample_max_tokens <= 0:
        raise ValueError("sample_max_tokens must be positive")
    tuning_literal = _require_tuning(tuning, lora_rank)

    tokenizer = HFAutoTokenizerAdapter.from_pretrained(base_model)
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
        base_model=base_model,
        dataset=batches,
        tuning=tuning_literal,
        lora_rank=lora_rank if tuning == "lora" else 0,
        learning_rate=float(config["learning_rate"]),
        max_steps=int(config["max_steps"]),
        save_every=int(config["save_every"]),
    )
    artifact = _artifact_for_summary(
        run_id=summary.run_id,
        checkpoint_version=summary.checkpoint_version,
        artifact_path=summary.artifact_path,
        manifest_path=summary.manifest_path,
        tuning=tuning,
    )
    training_run = _training_run_for_summary(
        run_id=summary.run_id,
        base_model=base_model,
        checkpoint_version=summary.checkpoint_version,
        tuning=tuning,
        lora_rank=lora_rank,
    )
    training = TrainingClient(service=client, run=training_run)
    refreshed = training.refresh_weights(
        artifact,
        request_id=f"modal-distributed-qwen-{config['rollout_backend']}-refresh",
    )
    sampler = SamplingClient(service=client, run=training_run, artifact=refreshed.artifact)
    if config["rollout_backend"] == "sglang":
        sample = sampler.sample_text(
            str(config["prompt"]),
            SamplingParams(max_tokens=sample_max_tokens, temperature=0.0, top_p=1.0),
            request_id="modal-distributed-qwen-sglang-sample",
        )
    else:
        sample = sampler.sample(
            ModelInput.from_ints([7, 8]),
            SamplingParams(max_tokens=4, temperature=0.7, top_p=0.9),
            request_id="modal-distributed-qwen-fake-sample",
        )
    telemetry = sampler.get_telemetry_summary(
        request_id=f"modal-distributed-qwen-{config['rollout_backend']}-telemetry",
    )
    context.reload_artifacts()

    mode = (
        "qwen-bridge-sglang-distributed"
        if config["rollout_backend"] == "sglang"
        else "qwen-bridge-sft-distributed"
    )
    payload = {
        "ok": True,
        "mode": mode,
        "runtime_kind": "bridge",
        "inference_backend": str(config["rollout_backend"]),
        "bridge_base_image": infra.BRIDGE_BASE_IMAGE,
        "bridge_ref": infra.BRIDGE_REF,
        "dataset_path": str(config["dataset_path"]),
        "batch_count": len(batches),
        "tuning": tuning,
        "lora_rank": lora_rank if tuning == "lora" else 0,
        "sample_tokens": sample.sequences[0].tokens,
        "sample_stop_reason": sample.sequences[0].stop_reason,
        "sample_checkpoint_version": sample.artifact.checkpoint_version,
        "telemetry_events": telemetry.summary.event_count,
        "telemetry_input_tokens": telemetry.summary.total.input_tokens,
        "telemetry_output_tokens": telemetry.summary.total.output_tokens,
        "telemetry_training_steps": telemetry.summary.total.training_steps,
        "telemetry_samples": telemetry.summary.total.samples,
        **context.base_payload(),
        **summary.to_dict(),
    }
    if config["rollout_backend"] == "sglang":
        payload.update(
            {
                "sglang_image": infra.SGLANG_IMAGE,
                "sglang_port": int(config["sglang_port"]),
                "sglang_context_length": int(config["sglang_context_length"]),
                "sglang_mem_fraction_static": float(config["sglang_mem_fraction_static"]),
                "prompt": str(config["prompt"]),
                "sample_text": sample.sequences[0].text,
            }
        )
    payload["manifest_exists"] = Path(summary.manifest_path).exists()
    payload.update(_artifact_details(summary.artifact_path))
    return payload


def _qwen_training_backend_config(
    *,
    micro_batch_size: int,
    sequence_length: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "runtime_kind": "bridge",
        "tensor_device": "cuda",
        "micro_batch_size": micro_batch_size,
        "global_batch_size": micro_batch_size,
        "sequence_length": sequence_length,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "seed": seed,
        "trust_remote_code": True,
        "load_weights": True,
    }


def _job_config(
    *,
    dataset_path: str,
    base_model: str,
    tuning: str,
    lora_rank: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    vocab_size: int,
    seed: int,
    rollout_backend: str,
    prompt: str,
    sample_max_tokens: int,
    sglang_port: int,
    sglang_context_length: int,
    sglang_mem_fraction_static: float,
) -> dict[str, Any]:
    return {
        "dataset_path": dataset_path,
        "base_model": base_model,
        "tuning": tuning,
        "lora_rank": lora_rank,
        "max_steps": max_steps,
        "save_every": save_every,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "micro_batch_size": micro_batch_size,
        "vocab_size": vocab_size,
        "seed": seed,
        "rollout_backend": rollout_backend,
        "prompt": prompt,
        "sample_max_tokens": sample_max_tokens,
        "sglang_port": sglang_port,
        "sglang_context_length": sglang_context_length,
        "sglang_mem_fraction_static": sglang_mem_fraction_static,
    }


@app.local_entrypoint()
def main(
    mode: str = "qwen-bridge-sglang-distributed",
    dataset_path: str = str(infra.REMOTE_ROOT / "examples" / "tiny_sft.jsonl"),
    artifact_root: str = str(infra.ARTIFACT_VOLUME_ROOT),
    base_model: str = "Qwen/Qwen3-0.6B",
    tuning: str = "lora",
    lora_rank: int = 8,
    max_steps: int = 1,
    save_every: int = 0,
    learning_rate: float = 1e-4,
    sequence_length: int = 32,
    micro_batch_size: int = 1,
    vocab_size: int = 128,
    seed: int = 1234,
    prompt: str = DEFAULT_PROMPT,
    sample_max_tokens: int = 12,
    sglang_port: int = infra.SGLANG_PORT,
    sglang_startup_timeout: int = 900,
    sglang_context_length: int = infra.SGLANG_CONTEXT_LENGTH,
    sglang_mem_fraction_static: float = infra.SGLANG_MEM_FRACTION_STATIC,
    port: int = infra.MONARCH_PORT,
    controller_port: int = infra.CONTROLLER_PORT,
    startup_timeout: int = 120,
    deployment_id: str = "",
    run_id: str = "run-000001",
) -> None:
    if mode not in {
        "fake-distributed",
        "qwen-bridge-sft-distributed",
        "qwen-bridge-sglang-distributed",
        "sft-distributed",
        "tcp-smoke",
    }:
        raise ValueError(f"unknown mode: {mode}")

    deployment = deployment_id or f"dev-{uuid.uuid4().hex[:8]}"
    if mode == "tcp-smoke":
        result = infra.run_tcp_smoke.remote(
            deployment,
            run_id,
            port or infra.TCP_SMOKE_PORT,
            startup_timeout,
        )
    elif mode == "fake-distributed":
        result = infra.run_fake_distributed.remote(
            deployment,
            run_id,
            artifact_root,
            port,
            controller_port,
            startup_timeout,
        )
    elif mode == "sft-distributed":
        result = infra.run_cpu_distributed_job.remote(
            deployment,
            run_id,
            artifact_root,
            port,
            controller_port,
            startup_timeout,
            "fake",
            None,
            "fake",
            None,
            "cpu",
            "cpu",
            JOB_MODULE,
            "run_toy_sft",
            _job_config(
                dataset_path=dataset_path,
                base_model=base_model,
                tuning=tuning,
                lora_rank=lora_rank,
                max_steps=max_steps,
                save_every=save_every,
                learning_rate=learning_rate,
                sequence_length=sequence_length,
                micro_batch_size=micro_batch_size,
                vocab_size=vocab_size,
                seed=seed,
                rollout_backend="fake",
                prompt=prompt,
                sample_max_tokens=sample_max_tokens,
                sglang_port=sglang_port,
                sglang_context_length=sglang_context_length,
                sglang_mem_fraction_static=sglang_mem_fraction_static,
            ),
            60,
        )
    elif mode == "qwen-bridge-sft-distributed":
        result = infra.run_bridge_distributed_job.remote(
            deployment,
            run_id,
            artifact_root,
            port,
            controller_port,
            startup_timeout,
            "megatron",
            _qwen_training_backend_config(
                micro_batch_size=micro_batch_size,
                sequence_length=sequence_length,
                seed=seed,
            ),
            "fake",
            None,
            "bridge",
            "cpu",
            JOB_MODULE,
            "run_qwen_sft",
            _job_config(
                dataset_path=dataset_path,
                base_model=base_model,
                tuning=tuning,
                lora_rank=lora_rank,
                max_steps=max_steps,
                save_every=save_every,
                learning_rate=learning_rate,
                sequence_length=sequence_length,
                micro_batch_size=micro_batch_size,
                vocab_size=vocab_size,
                seed=seed,
                rollout_backend="fake",
                prompt=prompt,
                sample_max_tokens=sample_max_tokens,
                sglang_port=sglang_port,
                sglang_context_length=sglang_context_length,
                sglang_mem_fraction_static=sglang_mem_fraction_static,
            ),
            120,
        )
    else:
        result = infra.run_bridge_distributed_job.remote(
            deployment,
            run_id,
            artifact_root,
            port,
            controller_port,
            startup_timeout,
            "megatron",
            _qwen_training_backend_config(
                micro_batch_size=micro_batch_size,
                sequence_length=sequence_length,
                seed=seed,
            ),
            "sglang",
            infra.sglang_backend_config(
                model=base_model,
                port=sglang_port,
                startup_timeout=sglang_startup_timeout,
                context_length=sglang_context_length,
                mem_fraction_static=sglang_mem_fraction_static,
                max_lora_rank=max(lora_rank, 1),
            ),
            "bridge",
            "sglang",
            JOB_MODULE,
            "run_qwen_sft",
            _job_config(
                dataset_path=dataset_path,
                base_model=base_model,
                tuning=tuning,
                lora_rank=lora_rank,
                max_steps=max_steps,
                save_every=save_every,
                learning_rate=learning_rate,
                sequence_length=sequence_length,
                micro_batch_size=micro_batch_size,
                vocab_size=vocab_size,
                seed=seed,
                rollout_backend="sglang",
                prompt=prompt,
                sample_max_tokens=sample_max_tokens,
                sglang_port=sglang_port,
                sglang_context_length=sglang_context_length,
                sglang_mem_fraction_static=sglang_mem_fraction_static,
            ),
            max(float(startup_timeout), float(sglang_startup_timeout)) + 300,
        )
    print(result)
