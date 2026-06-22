from ganker.contracts import (
    AdamParams,
    ArtifactFileKind,
    ArtifactKind,
    CreateTrainingRunRequest,
    Datum,
    DownloadArtifactFileRequest,
    ForwardBackwardRequest,
    ModelInput,
    OptimStepRequest,
    RefreshWeightsRequest,
    RequestContext,
    SampleRequest,
    SamplingParams,
    SaveWeightsRequest,
    TensorData,
    TuningMode,
    WeightArtifact,
)
from ganker.rpc import conversion as conv


def test_training_request_proto_round_trips():
    request = CreateTrainingRunRequest(
        context=RequestContext("req-create"),
        base_model="Qwen/Qwen3-0.6B",
        tuning_mode=TuningMode.LORA,
        lora_rank=8,
    )

    assert conv.create_training_run_request_from_proto(
        conv.create_training_run_request_to_proto(request)
    ) == request


def test_forward_backward_request_proto_round_trips():
    request = ForwardBackwardRequest(
        context=RequestContext("req-fb"),
        run_id="run-1",
        data=[
            Datum(
                model_input=ModelInput.from_ints([1, 2, 3]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_ints([2, 3, 4]),
                    "weights": TensorData.from_floats([1.0, 0.5, 1.0]),
                },
            )
        ],
        loss_fn="cross_entropy",
        loss_fn_config={"scale": 0.5},
    )

    assert conv.forward_backward_request_from_proto(
        conv.forward_backward_request_to_proto(request)
    ) == request


def test_optimizer_sampling_and_artifact_requests_proto_round_trip():
    artifact = WeightArtifact(
        run_id="run-1",
        checkpoint_version=2,
        kind=ArtifactKind.DELTA,
        manifest_path="/tmp/manifest.json",
        payload_path="/tmp/payload.json",
    )
    requests = [
        OptimStepRequest(
            context=RequestContext("req-step"),
            run_id="run-1",
            optimizer=AdamParams(learning_rate=1e-4),
        ),
        SaveWeightsRequest(
            context=RequestContext("req-save"),
            run_id="run-1",
            kind=ArtifactKind.DELTA,
        ),
        RefreshWeightsRequest(
            context=RequestContext("req-refresh"),
            run_id="run-1",
            artifact=artifact,
        ),
        SampleRequest(
            context=RequestContext("req-sample"),
            run_id="run-1",
            prompt=ModelInput.from_ints([10, 11]),
            sampling_params=SamplingParams(max_tokens=3, temperature=0.7, stop=["</s>"]),
            num_samples=2,
        ),
        DownloadArtifactFileRequest(
            context=RequestContext("req-download"),
            artifact=artifact,
            file_kind=ArtifactFileKind.PAYLOAD,
        ),
    ]

    assert conv.optim_step_request_from_proto(
        conv.optim_step_request_to_proto(requests[0])
    ) == requests[0]
    assert conv.save_weights_request_from_proto(
        conv.save_weights_request_to_proto(requests[1])
    ) == requests[1]
    assert conv.refresh_weights_request_from_proto(
        conv.refresh_weights_request_to_proto(requests[2])
    ) == requests[2]
    assert conv.sample_request_from_proto(conv.sample_request_to_proto(requests[3])) == requests[3]
    assert conv.download_artifact_file_request_from_proto(
        conv.download_artifact_file_request_to_proto(requests[4])
    ) == requests[4]


def test_refresh_weights_without_artifact_round_trips():
    request = RefreshWeightsRequest(
        context=RequestContext("req-refresh"),
        run_id="run-1",
        artifact=None,
    )

    assert conv.refresh_weights_request_from_proto(
        conv.refresh_weights_request_to_proto(request)
    ) == request
