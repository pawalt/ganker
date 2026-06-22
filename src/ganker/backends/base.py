"""Backend interfaces for training and inference engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ganker.contracts import (
    AdamParams,
    Datum,
    ForwardBackwardOutput,
    ModelInput,
    SampledSequence,
    SamplingParams,
    TrainingRun,
    Usage,
    WeightArtifact,
)


@dataclass(frozen=True)
class ForwardBackwardResult:
    run_id: str
    output: ForwardBackwardOutput
    gradient_version: int
    usage: Usage


@dataclass(frozen=True)
class OptimStepResult:
    run_id: str
    optimizer_step: int
    checkpoint_version: int
    usage: Usage


@dataclass(frozen=True)
class SampleResult:
    run_id: str
    sequences: list[SampledSequence]
    artifact: WeightArtifact
    usage: Usage


class TrainingBackend(Protocol):
    """Backend seam for local fake training and future Megatron execution."""

    def create_training_run(
        self,
        *,
        base_model: str,
        tuning_mode: int,
        lora_rank: int,
    ) -> TrainingRun:
        ...

    def forward_backward(
        self,
        *,
        run_id: str,
        data: list[Datum],
        loss_fn: str,
        loss_fn_config: dict[str, float],
    ) -> ForwardBackwardResult:
        ...

    def optim_step(
        self,
        *,
        run_id: str,
        params: AdamParams,
    ) -> OptimStepResult:
        ...

    def save_weights(self, *, run_id: str, kind: int) -> WeightArtifact:
        ...


class InferenceBackend(Protocol):
    """Backend seam for local fake rollout and future sglang execution."""

    def refresh_weights(
        self,
        *,
        run_id: str,
        artifact: WeightArtifact | None,
    ) -> WeightArtifact:
        ...

    def sample(
        self,
        *,
        run_id: str,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int,
    ) -> SampleResult:
        ...
