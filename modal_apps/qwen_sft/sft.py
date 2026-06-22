"""Clean Qwen LoRA SFT example on the dedicated Modal infra.

Run:

    source ~/.codex/modal.env
    GANKER_MODAL_GPU=A100 uv run modal run modal_apps/qwen_sft/sft.py \
      --startup-timeout 900 \
      --sglang-startup-timeout 900
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import uuid

from ganker.client import SamplingClient, ServiceClient, TrainingClient
from ganker.contracts import (
    ArtifactKind,
    SamplingParams,
    TrainingRun,
    TuningMode,
    WeightArtifact,
)

from modal_apps.qwen_sft import infra


app = infra.app
JOB_MODULE = "modal_apps.qwen_sft.sft"
JOB_FUNCTION = "run_qwen_lora_sft"
DEFAULT_PROMPT = "Answer in one short sentence: what is 2+2?"


def _artifact_details(artifact_path: str) -> dict[str, Any]:
    details: dict[str, Any] = {"artifact_exists": Path(artifact_path).exists()}
    if not details["artifact_exists"]:
        return details

    payload = json.loads(Path(artifact_path).read_text())
    details["artifact_format"] = payload.get("artifact_format")
    for key in (
        "hf_adapter_path",
        "hf_adapter_config_path",
        "hf_adapter_weights_path",
        "hf_checkpoint_bytes",
        "hf_weight_count",
        "hf_weight_format",
    ):
        if key in payload:
            details[key] = payload[key]
    for key in ("hf_adapter_path", "hf_adapter_config_path", "hf_adapter_weights_path"):
        if key in details:
            details[f"{key}_exists"] = Path(details[key]).exists()
    return details


def run_qwen_lora_sft(
    client: ServiceClient,
    context: infra.JobContext,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train one Qwen LoRA checkpoint, load it into SGLang, and sample text."""

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

    artifact = WeightArtifact(
        run_id=summary.run_id,
        checkpoint_version=summary.checkpoint_version,
        kind=ArtifactKind.DELTA,
        manifest_path=summary.manifest_path,
        payload_path=summary.artifact_path,
    )
    run = TrainingRun(
        run_id=summary.run_id,
        base_model=infra.MODEL,
        tuning_mode=TuningMode.LORA,
        lora_rank=int(config["lora_rank"]),
        checkpoint_version=summary.checkpoint_version,
    )

    training = TrainingClient(service=client, run=run)
    refreshed = training.refresh_weights(
        artifact,
        request_id="qwen-sft-refresh-sglang",
    )
    sampler = SamplingClient(service=client, run=run, artifact=refreshed.artifact)
    sample = sampler.sample_text(
        str(config["prompt"]),
        SamplingParams(max_tokens=int(config["sample_max_tokens"]), temperature=0.0, top_p=1.0),
        request_id="qwen-sft-sglang-sample",
    )
    telemetry = sampler.get_telemetry_summary(request_id="qwen-sft-telemetry")
    context.reload_artifacts()

    payload = {
        "ok": True,
        "mode": "qwen-lora-sft-sglang",
        "model": infra.MODEL,
        "dataset_path": str(config["dataset_path"]),
        "batch_count": len(batches),
        "prompt": str(config["prompt"]),
        "sample_text": sample.sequences[0].text,
        "sample_tokens": sample.sequences[0].tokens,
        "sample_stop_reason": sample.sequences[0].stop_reason,
        "sample_checkpoint_version": sample.artifact.checkpoint_version,
        "telemetry_events": telemetry.summary.event_count,
        "telemetry_input_tokens": telemetry.summary.total.input_tokens,
        "telemetry_output_tokens": telemetry.summary.total.output_tokens,
        "telemetry_training_steps": telemetry.summary.total.training_steps,
        "telemetry_samples": telemetry.summary.total.samples,
        "sglang_image": infra.SGLANG_IMAGE,
        "sglang_port": int(config["sglang_port"]),
        "sglang_context_length": int(config["sglang_context_length"]),
        "sglang_mem_fraction_static": float(config["sglang_mem_fraction_static"]),
        **context.base_payload(),
        **summary.to_dict(),
    }
    payload["manifest_exists"] = Path(summary.manifest_path).exists()
    payload.update(_artifact_details(summary.artifact_path))
    return payload


def _job_config(
    *,
    dataset_path: str,
    lora_rank: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    seed: int,
    prompt: str,
    sample_max_tokens: int,
    sglang_port: int,
    sglang_context_length: int,
    sglang_mem_fraction_static: float,
) -> dict[str, Any]:
    return {
        "dataset_path": dataset_path,
        "lora_rank": lora_rank,
        "max_steps": max_steps,
        "save_every": save_every,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "micro_batch_size": micro_batch_size,
        "seed": seed,
        "prompt": prompt,
        "sample_max_tokens": sample_max_tokens,
        "sglang_port": sglang_port,
        "sglang_context_length": sglang_context_length,
        "sglang_mem_fraction_static": sglang_mem_fraction_static,
    }


@app.local_entrypoint()
def main(
    dataset_path: str = str(infra.REMOTE_ROOT / "examples" / "tiny_sft.jsonl"),
    artifact_root: str = str(infra.ARTIFACT_ROOT),
    lora_rank: int = 8,
    max_steps: int = 1,
    save_every: int = 0,
    learning_rate: float = 1e-4,
    sequence_length: int = 32,
    micro_batch_size: int = 1,
    seed: int = 1234,
    prompt: str = DEFAULT_PROMPT,
    sample_max_tokens: int = 12,
    sglang_port: int = infra.SGLANG_PORT,
    sglang_startup_timeout: int = 900,
    sglang_context_length: int = infra.SGLANG_CONTEXT_LENGTH,
    sglang_mem_fraction_static: float = infra.SGLANG_MEM_FRACTION_STATIC,
    port: int = infra.MONARCH_PORT,
    controller_port: int = infra.CONTROLLER_PORT,
    startup_timeout: int = 900,
    deployment_id: str = "",
    run_id: str = "run-000001",
) -> None:
    if lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if sample_max_tokens <= 0:
        raise ValueError("sample_max_tokens must be positive")

    result = infra.run_sft_job.remote(
        deployment_id or f"qwen-sft-{uuid.uuid4().hex[:8]}",
        run_id,
        artifact_root,
        port,
        controller_port,
        startup_timeout,
        sglang_startup_timeout,
        JOB_MODULE,
        JOB_FUNCTION,
        _job_config(
            dataset_path=dataset_path,
            lora_rank=lora_rank,
            max_steps=max_steps,
            save_every=save_every,
            learning_rate=learning_rate,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            seed=seed,
            prompt=prompt,
            sample_max_tokens=sample_max_tokens,
            sglang_port=sglang_port,
            sglang_context_length=sglang_context_length,
            sglang_mem_fraction_static=sglang_mem_fraction_static,
        ),
    )
    print(result)
