from ganker.client import ServiceClient
from ganker.contracts import (
    AdamParams,
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
    WeightArtifact,
)


class FakeProxyTransport:
    def __init__(self):
        self.requests = []
        self.artifact = WeightArtifact(
            run_id="run-1",
            checkpoint_version=1,
            kind=ArtifactKind.DELTA,
            manifest_path="/tmp/manifest.json",
            payload_path="/tmp/payload.json",
        )

    def create_training_run(self, request: CreateTrainingRunRequest) -> CreateTrainingRunResponse:
        self.requests.append(request)
        return CreateTrainingRunResponse(
            request_id=request.context.request_id,
            run=TrainingRun(
                run_id="run-1",
                base_model=request.base_model,
                tuning_mode=request.tuning_mode,
                lora_rank=request.lora_rank,
                checkpoint_version=0,
            ),
        )

    def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        self.requests.append(request)
        return ForwardBackwardResponse(
            request_id=request.context.request_id,
            run_id=request.run_id,
            output=ForwardBackwardOutput(loss=0.1, metrics={"loss": 0.1}),
            gradient_version=1,
            usage=Usage(input_tokens=len(request.data[0].model_input.token_ids)),
        )

    def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        self.requests.append(request)
        return OptimStepResponse(
            request_id=request.context.request_id,
            run_id=request.run_id,
            optimizer_step=1,
            checkpoint_version=1,
            usage=Usage(training_steps=1),
        )

    def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        self.requests.append(request)
        return SaveWeightsResponse(request_id=request.context.request_id, artifact=self.artifact)

    def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        self.requests.append(request)
        return RefreshWeightsResponse(
            request_id=request.context.request_id,
            artifact=request.artifact or self.artifact,
        )

    def sample(self, request: SampleRequest) -> SampleResponse:
        self.requests.append(request)
        return SampleResponse(
            request_id=request.context.request_id,
            run_id=request.run_id,
            sequences=[SampledSequence(tokens=[1, 2], logprobs=[-0.1, -0.1])],
            artifact=self.artifact,
            usage=Usage(
                input_tokens=len(request.prompt.token_ids),
                output_tokens=request.sampling_params.max_tokens,
                samples=request.num_samples,
            ),
        )

    def get_telemetry_summary(
        self,
        request: GetTelemetrySummaryRequest,
    ) -> GetTelemetrySummaryResponse:
        self.requests.append(request)
        return GetTelemetrySummaryResponse(
            request_id=request.context.request_id,
            summary=TelemetrySummary(run_id=request.run_id),
        )

    def download_artifact_file(
        self,
        request: DownloadArtifactFileRequest,
    ) -> DownloadArtifactFileResponse:
        self.requests.append(request)
        return DownloadArtifactFileResponse(
            request_id=request.context.request_id,
            artifact=request.artifact,
            file_kind=request.file_kind,
            path=request.artifact.payload_path,
            contents=b"payload",
        )


def test_service_client_hides_proxy_transport_details():
    transport = FakeProxyTransport()
    service = ServiceClient(transport)

    training = service.create_lora_training_client(
        base_model="Qwen/Qwen3-8B",
        rank=16,
        request_id="req-create",
    )
    fb = training.forward_backward(
        Datum(
            model_input=ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={
                "target_tokens": TensorData.from_ints([2, 3, 4]),
                "weights": TensorData.from_floats([0.0, 1.0, 1.0]),
            },
        ),
        loss_fn="cross_entropy",
        request_id="req-fb",
    )
    step = training.optim_step(
        params=AdamParams(learning_rate=1e-4),
        request_id="req-step",
    )
    sampling = training.save_weights_and_get_sampling_client(request_id="req-sampler")
    sample = sampling.sample(
        ModelInput.from_ints([10]),
        SamplingParams(max_tokens=2),
        request_id="req-sample",
    )
    summary = sampling.get_telemetry_summary(request_id="req-summary")

    assert training.run_id == "run-1"
    assert training.run.tuning_mode == TuningMode.LORA
    assert fb.usage.input_tokens == 3
    assert step.optimizer_step == 1
    assert sampling.artifact == transport.artifact
    assert sample.sequences[0].tokens == [1, 2]
    assert summary.summary.run_id == "run-1"
    assert [type(request).__name__ for request in transport.requests] == [
        "CreateTrainingRunRequest",
        "ForwardBackwardRequest",
        "OptimStepRequest",
        "SaveWeightsRequest",
        "RefreshWeightsRequest",
        "SampleRequest",
        "GetTelemetrySummaryRequest",
    ]
