"""Typed contracts shared by components and Monarch actor endpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable


class TuningMode(IntEnum):
    LORA = 1
    FULL = 2


class ArtifactKind(IntEnum):
    FULL = 1
    DELTA = 2


@dataclass(frozen=True)
class RequestContext:
    request_id: str = ""


@dataclass(frozen=True)
class ModelInput:
    """Tokenized model input for a single datum or sampling prompt."""

    token_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_ints(cls, tokens: Iterable[int]) -> "ModelInput":
        return cls(token_ids=[int(token) for token in tokens])


@dataclass(frozen=True)
class TensorData:
    """Small local stand-in for Tinker tensor payloads."""

    values: list[int | float] = field(default_factory=list)

    @classmethod
    def from_ints(cls, values: Iterable[int]) -> "TensorData":
        return cls(values=[int(value) for value in values])

    @classmethod
    def from_floats(cls, values: Iterable[float]) -> "TensorData":
        return cls(values=[float(value) for value in values])

    @classmethod
    def from_torch(cls, tensor) -> "TensorData":
        return cls(values=list(tensor.tolist()))

    def tolist(self) -> list[int | float]:
        return list(self.values)


@dataclass(frozen=True)
class Datum:
    """A single training example."""

    model_input: ModelInput
    loss_fn_inputs: dict[str, TensorData] = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 1.0
    stop: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SampledSequence:
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    stop_reason: str = "length"


@dataclass(frozen=True)
class ForwardBackwardOutput:
    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    loss_fn_outputs: list[dict[str, TensorData]] = field(default_factory=list)


@dataclass(frozen=True)
class AdamParams:
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.0


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    training_steps: int = 0
    samples: int = 0

    def has_activity(self) -> bool:
        return any(
            (
                self.input_tokens,
                self.output_tokens,
                self.training_steps,
                self.samples,
            )
        )

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.training_steps += other.training_steps
        self.samples += other.samples


@dataclass(frozen=True)
class UsageEvent:
    request_id: str
    run_id: str
    event_source: str
    usage: Usage


@dataclass
class UsageBySource:
    event_source: str
    usage: Usage = field(default_factory=Usage)
    event_count: int = 0


@dataclass
class TelemetrySummary:
    run_id: str
    total: Usage = field(default_factory=Usage)
    event_count: int = 0
    by_source: list[UsageBySource] = field(default_factory=list)


@dataclass(frozen=True)
class WeightArtifact:
    run_id: str
    checkpoint_version: int
    kind: ArtifactKind
    manifest_path: str
    payload_path: str


@dataclass(frozen=True)
class TrainingRun:
    run_id: str
    base_model: str
    tuning_mode: TuningMode
    lora_rank: int
    checkpoint_version: int


@dataclass(frozen=True)
class CreateTrainingRunRequest:
    context: RequestContext
    base_model: str
    tuning_mode: TuningMode
    lora_rank: int


@dataclass(frozen=True)
class CreateTrainingRunResponse:
    request_id: str
    run: TrainingRun


@dataclass(frozen=True)
class ForwardBackwardRequest:
    context: RequestContext
    run_id: str
    data: list[Datum]
    loss_fn: str = "cross_entropy"
    loss_fn_config: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardBackwardResponse:
    request_id: str
    run_id: str
    output: ForwardBackwardOutput
    gradient_version: int
    usage: Usage

    @property
    def loss(self) -> float:
        return self.output.loss


@dataclass(frozen=True)
class OptimStepRequest:
    context: RequestContext
    run_id: str
    optimizer: AdamParams


@dataclass(frozen=True)
class OptimStepResponse:
    request_id: str
    run_id: str
    optimizer_step: int
    checkpoint_version: int
    usage: Usage


@dataclass(frozen=True)
class SaveWeightsRequest:
    context: RequestContext
    run_id: str
    kind: ArtifactKind


@dataclass(frozen=True)
class SaveWeightsResponse:
    request_id: str
    artifact: WeightArtifact


@dataclass(frozen=True)
class RefreshWeightsRequest:
    context: RequestContext
    run_id: str
    artifact: WeightArtifact | None = None


@dataclass(frozen=True)
class RefreshWeightsResponse:
    request_id: str
    artifact: WeightArtifact


@dataclass(frozen=True)
class SampleRequest:
    context: RequestContext
    run_id: str
    prompt: ModelInput
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    num_samples: int = 1


@dataclass(frozen=True)
class SampleResponse:
    request_id: str
    run_id: str
    sequences: list[SampledSequence]
    artifact: WeightArtifact
    usage: Usage


@dataclass(frozen=True)
class RecordTelemetryRequest:
    context: RequestContext
    event: UsageEvent


@dataclass(frozen=True)
class RecordTelemetryResponse:
    request_id: str
    event_count: int


@dataclass(frozen=True)
class GetTelemetrySummaryRequest:
    context: RequestContext
    run_id: str


@dataclass(frozen=True)
class GetTelemetrySummaryResponse:
    request_id: str
    summary: TelemetrySummary
