"""Conversions between public dataclass contracts and protobuf messages."""

from __future__ import annotations

from ganker.contracts import (
    AdamParams,
    ArtifactFileKind,
    ArtifactKind,
    CreateTrainingRunRequest,
    CreateTrainingRunResponse,
    Datum,
    DownloadArtifactFileRequest,
    DownloadArtifactFileResponse,
    ForwardBackwardOutput,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    GetTelemetrySummaryRequest,
    GetTelemetrySummaryResponse,
    ModelInput,
    OptimStepRequest,
    OptimStepResponse,
    RefreshWeightsRequest,
    RefreshWeightsResponse,
    RequestContext,
    SampleRequest,
    SampleResponse,
    SampledSequence,
    SamplingParams,
    SaveWeightsRequest,
    SaveWeightsResponse,
    TelemetrySummary,
    TensorData,
    TrainingRun,
    TuningMode,
    Usage,
    UsageBySource,
    WeightArtifact,
)
from ganker.errors import InvalidRequestError
from ganker.rpc.v1 import proxy_pb2 as pb


def _tuning_mode_to_proto(value: TuningMode) -> int:
    if value == TuningMode.LORA:
        return pb.TUNING_MODE_LORA
    if value == TuningMode.FULL:
        return pb.TUNING_MODE_FULL
    raise InvalidRequestError(f"unsupported tuning mode: {value}")


def _tuning_mode_from_proto(value: int) -> TuningMode:
    if value == pb.TUNING_MODE_LORA:
        return TuningMode.LORA
    if value == pb.TUNING_MODE_FULL:
        return TuningMode.FULL
    raise InvalidRequestError(f"unsupported tuning mode: {value}")


def _artifact_kind_to_proto(value: ArtifactKind) -> int:
    if value == ArtifactKind.FULL:
        return pb.ARTIFACT_KIND_FULL
    if value == ArtifactKind.DELTA:
        return pb.ARTIFACT_KIND_DELTA
    raise InvalidRequestError(f"unsupported artifact kind: {value}")


def _artifact_kind_from_proto(value: int) -> ArtifactKind:
    if value == pb.ARTIFACT_KIND_FULL:
        return ArtifactKind.FULL
    if value == pb.ARTIFACT_KIND_DELTA:
        return ArtifactKind.DELTA
    raise InvalidRequestError(f"unsupported artifact kind: {value}")


def _artifact_file_kind_to_proto(value: ArtifactFileKind) -> int:
    if value == ArtifactFileKind.MANIFEST:
        return pb.ARTIFACT_FILE_KIND_MANIFEST
    if value == ArtifactFileKind.PAYLOAD:
        return pb.ARTIFACT_FILE_KIND_PAYLOAD
    raise InvalidRequestError(f"unsupported artifact file kind: {value}")


def _artifact_file_kind_from_proto(value: int) -> ArtifactFileKind:
    if value == pb.ARTIFACT_FILE_KIND_MANIFEST:
        return ArtifactFileKind.MANIFEST
    if value == pb.ARTIFACT_FILE_KIND_PAYLOAD:
        return ArtifactFileKind.PAYLOAD
    raise InvalidRequestError(f"unsupported artifact file kind: {value}")


def context_to_proto(value: RequestContext) -> pb.RequestContext:
    return pb.RequestContext(request_id=value.request_id)


def context_from_proto(value: pb.RequestContext) -> RequestContext:
    return RequestContext(request_id=value.request_id)


def model_input_to_proto(value: ModelInput) -> pb.ModelInput:
    return pb.ModelInput(token_ids=value.token_ids, text=value.text)


def model_input_from_proto(value: pb.ModelInput) -> ModelInput:
    return ModelInput(
        token_ids=[int(token) for token in value.token_ids],
        text=value.text,
    )


def tensor_data_to_proto(value: TensorData) -> pb.TensorData:
    return pb.TensorData(values=[float(item) for item in value.values])


def tensor_data_from_proto(value: pb.TensorData) -> TensorData:
    return TensorData(values=list(value.values))


def datum_to_proto(value: Datum) -> pb.Datum:
    return pb.Datum(
        model_input=model_input_to_proto(value.model_input),
        loss_fn_inputs={
            key: tensor_data_to_proto(tensor)
            for key, tensor in value.loss_fn_inputs.items()
        },
    )


def datum_from_proto(value: pb.Datum) -> Datum:
    return Datum(
        model_input=model_input_from_proto(value.model_input),
        loss_fn_inputs={
            key: tensor_data_from_proto(tensor)
            for key, tensor in value.loss_fn_inputs.items()
        },
    )


def sampling_params_to_proto(value: SamplingParams) -> pb.SamplingParams:
    return pb.SamplingParams(
        max_tokens=value.max_tokens,
        temperature=value.temperature,
        stop=value.stop,
        top_p=value.top_p,
    )


def sampling_params_from_proto(value: pb.SamplingParams) -> SamplingParams:
    return SamplingParams(
        max_tokens=int(value.max_tokens),
        temperature=float(value.temperature),
        top_p=float(value.top_p) if value.top_p else 1.0,
        stop=list(value.stop),
    )


def sampled_sequence_to_proto(value: SampledSequence) -> pb.SampledSequence:
    return pb.SampledSequence(
        tokens=value.tokens,
        logprobs=value.logprobs,
        stop_reason=value.stop_reason,
        text=value.text,
    )


def sampled_sequence_from_proto(value: pb.SampledSequence) -> SampledSequence:
    return SampledSequence(
        text=value.text,
        tokens=[int(token) for token in value.tokens],
        logprobs=list(value.logprobs),
        stop_reason=value.stop_reason,
    )


def forward_backward_output_to_proto(value: ForwardBackwardOutput) -> pb.ForwardBackwardOutput:
    return pb.ForwardBackwardOutput(
        loss=value.loss,
        metrics=value.metrics,
        loss_fn_outputs=[
            pb.LossFnOutput(
                values={
                    key: tensor_data_to_proto(tensor)
                    for key, tensor in item.items()
                }
            )
            for item in value.loss_fn_outputs
        ],
    )


def forward_backward_output_from_proto(value: pb.ForwardBackwardOutput) -> ForwardBackwardOutput:
    return ForwardBackwardOutput(
        loss=float(value.loss),
        metrics={key: float(metric) for key, metric in value.metrics.items()},
        loss_fn_outputs=[
            {
                key: tensor_data_from_proto(tensor)
                for key, tensor in item.values.items()
            }
            for item in value.loss_fn_outputs
        ],
    )


def adam_params_to_proto(value: AdamParams) -> pb.AdamParams:
    return pb.AdamParams(
        learning_rate=value.learning_rate,
        beta1=value.beta1,
        beta2=value.beta2,
        eps=value.eps,
        weight_decay=value.weight_decay,
    )


def adam_params_from_proto(value: pb.AdamParams) -> AdamParams:
    return AdamParams(
        learning_rate=float(value.learning_rate),
        beta1=float(value.beta1),
        beta2=float(value.beta2),
        eps=float(value.eps),
        weight_decay=float(value.weight_decay),
    )


def usage_to_proto(value: Usage) -> pb.Usage:
    return pb.Usage(
        input_tokens=value.input_tokens,
        output_tokens=value.output_tokens,
        training_steps=value.training_steps,
        samples=value.samples,
    )


def usage_from_proto(value: pb.Usage) -> Usage:
    return Usage(
        input_tokens=int(value.input_tokens),
        output_tokens=int(value.output_tokens),
        training_steps=int(value.training_steps),
        samples=int(value.samples),
    )


def usage_by_source_to_proto(value: UsageBySource) -> pb.UsageBySource:
    return pb.UsageBySource(
        event_source=value.event_source,
        usage=usage_to_proto(value.usage),
        event_count=value.event_count,
    )


def usage_by_source_from_proto(value: pb.UsageBySource) -> UsageBySource:
    return UsageBySource(
        event_source=value.event_source,
        usage=usage_from_proto(value.usage),
        event_count=int(value.event_count),
    )


def telemetry_summary_to_proto(value: TelemetrySummary) -> pb.TelemetrySummary:
    return pb.TelemetrySummary(
        run_id=value.run_id,
        total=usage_to_proto(value.total),
        event_count=value.event_count,
        by_source=[usage_by_source_to_proto(item) for item in value.by_source],
    )


def telemetry_summary_from_proto(value: pb.TelemetrySummary) -> TelemetrySummary:
    return TelemetrySummary(
        run_id=value.run_id,
        total=usage_from_proto(value.total),
        event_count=int(value.event_count),
        by_source=[usage_by_source_from_proto(item) for item in value.by_source],
    )


def weight_artifact_to_proto(value: WeightArtifact) -> pb.WeightArtifact:
    return pb.WeightArtifact(
        run_id=value.run_id,
        checkpoint_version=value.checkpoint_version,
        kind=_artifact_kind_to_proto(value.kind),
        manifest_path=value.manifest_path,
        payload_path=value.payload_path,
    )


def weight_artifact_from_proto(value: pb.WeightArtifact) -> WeightArtifact:
    return WeightArtifact(
        run_id=value.run_id,
        checkpoint_version=int(value.checkpoint_version),
        kind=_artifact_kind_from_proto(value.kind),
        manifest_path=value.manifest_path,
        payload_path=value.payload_path,
    )


def training_run_to_proto(value: TrainingRun) -> pb.TrainingRun:
    return pb.TrainingRun(
        run_id=value.run_id,
        base_model=value.base_model,
        tuning_mode=_tuning_mode_to_proto(value.tuning_mode),
        lora_rank=value.lora_rank,
        checkpoint_version=value.checkpoint_version,
    )


def training_run_from_proto(value: pb.TrainingRun) -> TrainingRun:
    return TrainingRun(
        run_id=value.run_id,
        base_model=value.base_model,
        tuning_mode=_tuning_mode_from_proto(value.tuning_mode),
        lora_rank=int(value.lora_rank),
        checkpoint_version=int(value.checkpoint_version),
    )


def create_training_run_request_to_proto(
    value: CreateTrainingRunRequest,
) -> pb.CreateTrainingRunRequest:
    return pb.CreateTrainingRunRequest(
        context=context_to_proto(value.context),
        base_model=value.base_model,
        tuning_mode=_tuning_mode_to_proto(value.tuning_mode),
        lora_rank=value.lora_rank,
    )


def create_training_run_request_from_proto(
    value: pb.CreateTrainingRunRequest,
) -> CreateTrainingRunRequest:
    return CreateTrainingRunRequest(
        context=context_from_proto(value.context),
        base_model=value.base_model,
        tuning_mode=_tuning_mode_from_proto(value.tuning_mode),
        lora_rank=int(value.lora_rank),
    )


def create_training_run_response_to_proto(
    value: CreateTrainingRunResponse,
) -> pb.CreateTrainingRunResponse:
    return pb.CreateTrainingRunResponse(
        request_id=value.request_id,
        run=training_run_to_proto(value.run),
    )


def create_training_run_response_from_proto(
    value: pb.CreateTrainingRunResponse,
) -> CreateTrainingRunResponse:
    return CreateTrainingRunResponse(
        request_id=value.request_id,
        run=training_run_from_proto(value.run),
    )


def forward_backward_request_to_proto(value: ForwardBackwardRequest) -> pb.ForwardBackwardRequest:
    return pb.ForwardBackwardRequest(
        context=context_to_proto(value.context),
        run_id=value.run_id,
        data=[datum_to_proto(item) for item in value.data],
        loss_fn=value.loss_fn,
        loss_fn_config=value.loss_fn_config,
    )


def forward_backward_request_from_proto(value: pb.ForwardBackwardRequest) -> ForwardBackwardRequest:
    return ForwardBackwardRequest(
        context=context_from_proto(value.context),
        run_id=value.run_id,
        data=[datum_from_proto(item) for item in value.data],
        loss_fn=value.loss_fn,
        loss_fn_config={key: float(item) for key, item in value.loss_fn_config.items()},
    )


def forward_backward_response_to_proto(
    value: ForwardBackwardResponse,
) -> pb.ForwardBackwardResponse:
    return pb.ForwardBackwardResponse(
        request_id=value.request_id,
        run_id=value.run_id,
        output=forward_backward_output_to_proto(value.output),
        gradient_version=value.gradient_version,
        usage=usage_to_proto(value.usage),
    )


def forward_backward_response_from_proto(
    value: pb.ForwardBackwardResponse,
) -> ForwardBackwardResponse:
    return ForwardBackwardResponse(
        request_id=value.request_id,
        run_id=value.run_id,
        output=forward_backward_output_from_proto(value.output),
        gradient_version=int(value.gradient_version),
        usage=usage_from_proto(value.usage),
    )


def optim_step_request_to_proto(value: OptimStepRequest) -> pb.OptimStepRequest:
    return pb.OptimStepRequest(
        context=context_to_proto(value.context),
        run_id=value.run_id,
        optimizer=adam_params_to_proto(value.optimizer),
    )


def optim_step_request_from_proto(value: pb.OptimStepRequest) -> OptimStepRequest:
    return OptimStepRequest(
        context=context_from_proto(value.context),
        run_id=value.run_id,
        optimizer=adam_params_from_proto(value.optimizer),
    )


def optim_step_response_to_proto(value: OptimStepResponse) -> pb.OptimStepResponse:
    return pb.OptimStepResponse(
        request_id=value.request_id,
        run_id=value.run_id,
        optimizer_step=value.optimizer_step,
        checkpoint_version=value.checkpoint_version,
        usage=usage_to_proto(value.usage),
    )


def optim_step_response_from_proto(value: pb.OptimStepResponse) -> OptimStepResponse:
    return OptimStepResponse(
        request_id=value.request_id,
        run_id=value.run_id,
        optimizer_step=int(value.optimizer_step),
        checkpoint_version=int(value.checkpoint_version),
        usage=usage_from_proto(value.usage),
    )


def save_weights_request_to_proto(value: SaveWeightsRequest) -> pb.SaveWeightsRequest:
    return pb.SaveWeightsRequest(
        context=context_to_proto(value.context),
        run_id=value.run_id,
        kind=_artifact_kind_to_proto(value.kind),
    )


def save_weights_request_from_proto(value: pb.SaveWeightsRequest) -> SaveWeightsRequest:
    return SaveWeightsRequest(
        context=context_from_proto(value.context),
        run_id=value.run_id,
        kind=_artifact_kind_from_proto(value.kind),
    )


def save_weights_response_to_proto(value: SaveWeightsResponse) -> pb.SaveWeightsResponse:
    return pb.SaveWeightsResponse(
        request_id=value.request_id,
        artifact=weight_artifact_to_proto(value.artifact),
    )


def save_weights_response_from_proto(value: pb.SaveWeightsResponse) -> SaveWeightsResponse:
    return SaveWeightsResponse(
        request_id=value.request_id,
        artifact=weight_artifact_from_proto(value.artifact),
    )


def refresh_weights_request_to_proto(value: RefreshWeightsRequest) -> pb.RefreshWeightsRequest:
    request = pb.RefreshWeightsRequest(
        context=context_to_proto(value.context),
        run_id=value.run_id,
    )
    if value.artifact is not None:
        request.artifact.CopyFrom(weight_artifact_to_proto(value.artifact))
    return request


def refresh_weights_request_from_proto(value: pb.RefreshWeightsRequest) -> RefreshWeightsRequest:
    return RefreshWeightsRequest(
        context=context_from_proto(value.context),
        run_id=value.run_id,
        artifact=weight_artifact_from_proto(value.artifact)
        if value.HasField("artifact")
        else None,
    )


def refresh_weights_response_to_proto(value: RefreshWeightsResponse) -> pb.RefreshWeightsResponse:
    return pb.RefreshWeightsResponse(
        request_id=value.request_id,
        artifact=weight_artifact_to_proto(value.artifact),
    )


def refresh_weights_response_from_proto(value: pb.RefreshWeightsResponse) -> RefreshWeightsResponse:
    return RefreshWeightsResponse(
        request_id=value.request_id,
        artifact=weight_artifact_from_proto(value.artifact),
    )


def sample_request_to_proto(value: SampleRequest) -> pb.SampleRequest:
    return pb.SampleRequest(
        context=context_to_proto(value.context),
        run_id=value.run_id,
        prompt=model_input_to_proto(value.prompt),
        sampling_params=sampling_params_to_proto(value.sampling_params),
        num_samples=value.num_samples,
    )


def sample_request_from_proto(value: pb.SampleRequest) -> SampleRequest:
    return SampleRequest(
        context=context_from_proto(value.context),
        run_id=value.run_id,
        prompt=model_input_from_proto(value.prompt),
        sampling_params=sampling_params_from_proto(value.sampling_params),
        num_samples=int(value.num_samples),
    )


def sample_response_to_proto(value: SampleResponse) -> pb.SampleResponse:
    return pb.SampleResponse(
        request_id=value.request_id,
        run_id=value.run_id,
        sequences=[sampled_sequence_to_proto(item) for item in value.sequences],
        artifact=weight_artifact_to_proto(value.artifact),
        usage=usage_to_proto(value.usage),
    )


def sample_response_from_proto(value: pb.SampleResponse) -> SampleResponse:
    return SampleResponse(
        request_id=value.request_id,
        run_id=value.run_id,
        sequences=[sampled_sequence_from_proto(item) for item in value.sequences],
        artifact=weight_artifact_from_proto(value.artifact),
        usage=usage_from_proto(value.usage),
    )


def get_telemetry_summary_request_to_proto(
    value: GetTelemetrySummaryRequest,
) -> pb.GetTelemetrySummaryRequest:
    return pb.GetTelemetrySummaryRequest(
        context=context_to_proto(value.context),
        run_id=value.run_id,
    )


def get_telemetry_summary_request_from_proto(
    value: pb.GetTelemetrySummaryRequest,
) -> GetTelemetrySummaryRequest:
    return GetTelemetrySummaryRequest(
        context=context_from_proto(value.context),
        run_id=value.run_id,
    )


def get_telemetry_summary_response_to_proto(
    value: GetTelemetrySummaryResponse,
) -> pb.GetTelemetrySummaryResponse:
    return pb.GetTelemetrySummaryResponse(
        request_id=value.request_id,
        summary=telemetry_summary_to_proto(value.summary),
    )


def get_telemetry_summary_response_from_proto(
    value: pb.GetTelemetrySummaryResponse,
) -> GetTelemetrySummaryResponse:
    return GetTelemetrySummaryResponse(
        request_id=value.request_id,
        summary=telemetry_summary_from_proto(value.summary),
    )


def download_artifact_file_request_to_proto(
    value: DownloadArtifactFileRequest,
) -> pb.DownloadArtifactFileRequest:
    return pb.DownloadArtifactFileRequest(
        context=context_to_proto(value.context),
        artifact=weight_artifact_to_proto(value.artifact),
        file_kind=_artifact_file_kind_to_proto(value.file_kind),
    )


def download_artifact_file_request_from_proto(
    value: pb.DownloadArtifactFileRequest,
) -> DownloadArtifactFileRequest:
    return DownloadArtifactFileRequest(
        context=context_from_proto(value.context),
        artifact=weight_artifact_from_proto(value.artifact),
        file_kind=_artifact_file_kind_from_proto(value.file_kind),
    )


def download_artifact_file_response_to_proto(
    value: DownloadArtifactFileResponse,
) -> pb.DownloadArtifactFileResponse:
    return pb.DownloadArtifactFileResponse(
        request_id=value.request_id,
        artifact=weight_artifact_to_proto(value.artifact),
        file_kind=_artifact_file_kind_to_proto(value.file_kind),
        path=value.path,
        contents=value.contents,
    )


def download_artifact_file_response_from_proto(
    value: pb.DownloadArtifactFileResponse,
) -> DownloadArtifactFileResponse:
    return DownloadArtifactFileResponse(
        request_id=value.request_id,
        artifact=weight_artifact_from_proto(value.artifact),
        file_kind=_artifact_file_kind_from_proto(value.file_kind),
        path=value.path,
        contents=value.contents,
    )
