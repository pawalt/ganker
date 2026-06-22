from collections.abc import MutableMapping, MutableSequence
from typing import Any

from google.protobuf.message import Message

TUNING_MODE_UNSPECIFIED: int
TUNING_MODE_LORA: int
TUNING_MODE_FULL: int
ARTIFACT_KIND_UNSPECIFIED: int
ARTIFACT_KIND_FULL: int
ARTIFACT_KIND_DELTA: int
ARTIFACT_FILE_KIND_UNSPECIFIED: int
ARTIFACT_FILE_KIND_MANIFEST: int
ARTIFACT_FILE_KIND_PAYLOAD: int


class RequestContext(Message):
    request_id: str
    def __init__(self, *, request_id: str = ...) -> None: ...


class ModelInput(Message):
    token_ids: MutableSequence[int]
    text: str
    def __init__(
        self,
        *,
        token_ids: MutableSequence[int] | None = ...,
        text: str = ...,
    ) -> None: ...


class TensorData(Message):
    values: MutableSequence[float]
    def __init__(self, *, values: MutableSequence[float] | None = ...) -> None: ...


class Datum(Message):
    model_input: ModelInput
    loss_fn_inputs: MutableMapping[str, TensorData]
    def __init__(
        self,
        *,
        model_input: ModelInput | None = ...,
        loss_fn_inputs: MutableMapping[str, TensorData] | None = ...,
    ) -> None: ...


class SamplingParams(Message):
    max_tokens: int
    temperature: float
    stop: MutableSequence[str]
    top_p: float
    def __init__(
        self,
        *,
        max_tokens: int = ...,
        temperature: float = ...,
        stop: MutableSequence[str] | None = ...,
        top_p: float = ...,
    ) -> None: ...


class SampledSequence(Message):
    tokens: MutableSequence[int]
    logprobs: MutableSequence[float]
    stop_reason: str
    text: str
    def __init__(
        self,
        *,
        tokens: MutableSequence[int] | None = ...,
        logprobs: MutableSequence[float] | None = ...,
        stop_reason: str = ...,
        text: str = ...,
    ) -> None: ...


class LossFnOutput(Message):
    values: MutableMapping[str, TensorData]
    def __init__(self, *, values: MutableMapping[str, TensorData] | None = ...) -> None: ...


class ForwardBackwardOutput(Message):
    loss: float
    metrics: MutableMapping[str, float]
    loss_fn_outputs: MutableSequence[LossFnOutput]
    def __init__(
        self,
        *,
        loss: float = ...,
        metrics: MutableMapping[str, float] | None = ...,
        loss_fn_outputs: MutableSequence[LossFnOutput] | None = ...,
    ) -> None: ...


class AdamParams(Message):
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    def __init__(
        self,
        *,
        learning_rate: float = ...,
        beta1: float = ...,
        beta2: float = ...,
        eps: float = ...,
        weight_decay: float = ...,
    ) -> None: ...


class Usage(Message):
    input_tokens: int
    output_tokens: int
    training_steps: int
    samples: int
    def __init__(
        self,
        *,
        input_tokens: int = ...,
        output_tokens: int = ...,
        training_steps: int = ...,
        samples: int = ...,
    ) -> None: ...


class UsageEvent(Message):
    request_id: str
    run_id: str
    event_source: str
    usage: Usage


class UsageBySource(Message):
    event_source: str
    usage: Usage
    event_count: int
    def __init__(
        self,
        *,
        event_source: str = ...,
        usage: Usage | None = ...,
        event_count: int = ...,
    ) -> None: ...


class TelemetrySummary(Message):
    run_id: str
    total: Usage
    event_count: int
    by_source: MutableSequence[UsageBySource]
    def __init__(
        self,
        *,
        run_id: str = ...,
        total: Usage | None = ...,
        event_count: int = ...,
        by_source: MutableSequence[UsageBySource] | None = ...,
    ) -> None: ...


class WeightArtifact(Message):
    run_id: str
    checkpoint_version: int
    kind: int
    manifest_path: str
    payload_path: str
    def __init__(
        self,
        *,
        run_id: str = ...,
        checkpoint_version: int = ...,
        kind: int = ...,
        manifest_path: str = ...,
        payload_path: str = ...,
    ) -> None: ...


class TrainingRun(Message):
    run_id: str
    base_model: str
    tuning_mode: int
    lora_rank: int
    checkpoint_version: int
    def __init__(
        self,
        *,
        run_id: str = ...,
        base_model: str = ...,
        tuning_mode: int = ...,
        lora_rank: int = ...,
        checkpoint_version: int = ...,
    ) -> None: ...


class CreateTrainingRunRequest(Message):
    context: RequestContext
    base_model: str
    tuning_mode: int
    lora_rank: int
    def __init__(self, **kwargs: Any) -> None: ...


class CreateTrainingRunResponse(Message):
    request_id: str
    run: TrainingRun
    def __init__(self, **kwargs: Any) -> None: ...


class ForwardBackwardRequest(Message):
    context: RequestContext
    run_id: str
    data: MutableSequence[Datum]
    loss_fn: str
    loss_fn_config: MutableMapping[str, float]
    def __init__(self, **kwargs: Any) -> None: ...


class ForwardBackwardResponse(Message):
    request_id: str
    run_id: str
    output: ForwardBackwardOutput
    gradient_version: int
    usage: Usage
    def __init__(self, **kwargs: Any) -> None: ...


class OptimStepRequest(Message):
    context: RequestContext
    run_id: str
    optimizer: AdamParams
    def __init__(self, **kwargs: Any) -> None: ...


class OptimStepResponse(Message):
    request_id: str
    run_id: str
    optimizer_step: int
    checkpoint_version: int
    usage: Usage
    def __init__(self, **kwargs: Any) -> None: ...


class SaveWeightsRequest(Message):
    context: RequestContext
    run_id: str
    kind: int
    def __init__(self, **kwargs: Any) -> None: ...


class SaveWeightsResponse(Message):
    request_id: str
    artifact: WeightArtifact
    def __init__(self, **kwargs: Any) -> None: ...


class RefreshWeightsRequest(Message):
    context: RequestContext
    run_id: str
    artifact: WeightArtifact
    def __init__(self, **kwargs: Any) -> None: ...


class RefreshWeightsResponse(Message):
    request_id: str
    artifact: WeightArtifact
    def __init__(self, **kwargs: Any) -> None: ...


class SampleRequest(Message):
    context: RequestContext
    run_id: str
    prompt: ModelInput
    sampling_params: SamplingParams
    num_samples: int
    def __init__(self, **kwargs: Any) -> None: ...


class SampleResponse(Message):
    request_id: str
    run_id: str
    sequences: MutableSequence[SampledSequence]
    artifact: WeightArtifact
    usage: Usage
    def __init__(self, **kwargs: Any) -> None: ...


class GetTelemetrySummaryRequest(Message):
    context: RequestContext
    run_id: str
    def __init__(self, **kwargs: Any) -> None: ...


class GetTelemetrySummaryResponse(Message):
    request_id: str
    summary: TelemetrySummary
    def __init__(self, **kwargs: Any) -> None: ...


class DownloadArtifactFileRequest(Message):
    context: RequestContext
    artifact: WeightArtifact
    file_kind: int
    def __init__(self, **kwargs: Any) -> None: ...


class DownloadArtifactFileResponse(Message):
    request_id: str
    artifact: WeightArtifact
    file_kind: int
    path: str
    contents: bytes
    def __init__(self, **kwargs: Any) -> None: ...
