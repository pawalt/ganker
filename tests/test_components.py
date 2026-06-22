from pathlib import Path

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.fake import FakeInferenceBackend, FakeTrainingBackend
from ganker.components import RolloutComponent, TelemetryComponent, TelemetryLedger, TrainingComponent
from ganker.contracts import (
    AdamParams,
    ArtifactKind,
    CreateTrainingRunRequest,
    Datum,
    ForwardBackwardRequest,
    GetTelemetrySummaryRequest,
    ModelInput,
    OptimStepRequest,
    RecordTelemetryRequest,
    RefreshWeightsRequest,
    RequestContext,
    SampleRequest,
    SamplingParams,
    SaveWeightsRequest,
    TensorData,
    TuningMode,
    Usage,
    UsageEvent,
)


def test_training_and_rollout_components_use_backends(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    training = TrainingComponent(FakeTrainingBackend(store))
    rollout = RolloutComponent(FakeInferenceBackend(store))

    created = training.create_training_run(
        CreateTrainingRunRequest(
            context=RequestContext("req-create"),
            base_model="Qwen/Qwen3-8B",
            tuning_mode=TuningMode.LORA,
            lora_rank=32,
        )
    )
    fb = training.forward_backward(
        ForwardBackwardRequest(
            context=RequestContext("req-fb"),
            run_id=created.run.run_id,
            data=[
                Datum(
                    model_input=ModelInput.from_ints([1, 2, 3]),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_ints([2, 3, 4]),
                        "weights": TensorData.from_floats([1.0, 1.0, 1.0]),
                    },
                )
            ],
            loss_fn="cross_entropy",
        )
    )
    step = training.optim_step(
        OptimStepRequest(
            context=RequestContext("req-step"),
            run_id=created.run.run_id,
            optimizer=AdamParams(learning_rate=1e-4),
        )
    )
    saved = training.save_weights(
        SaveWeightsRequest(
            context=RequestContext("req-save"),
            run_id=created.run.run_id,
            kind=ArtifactKind.DELTA,
        )
    )
    refreshed = rollout.refresh_weights(
        RefreshWeightsRequest(
            context=RequestContext("req-refresh"),
            run_id=created.run.run_id,
            artifact=saved.artifact,
        )
    )
    sample = rollout.sample(
        SampleRequest(
            context=RequestContext("req-sample"),
            run_id=created.run.run_id,
            prompt=ModelInput.from_ints([10]),
            sampling_params=SamplingParams(max_tokens=2),
        )
    )

    assert fb.usage.input_tokens == 3
    assert fb.output.metrics["loss"] > 0
    assert step.checkpoint_version == 1
    assert refreshed.artifact.manifest_path == saved.artifact.manifest_path
    assert sample.sequences[0].tokens == [12, 13]


def test_telemetry_component_records_and_summarizes():
    telemetry = TelemetryComponent(TelemetryLedger())

    telemetry.record(
        RecordTelemetryRequest(
            context=RequestContext("req-record"),
            event=UsageEvent(
                request_id="req-source",
                run_id="run-1",
                event_source="trainer",
                usage=Usage(input_tokens=5),
            ),
        )
    )
    summary = telemetry.get_summary(
        GetTelemetrySummaryRequest(
            context=RequestContext("req-summary"),
            run_id="run-1",
        )
    )

    assert summary.summary.event_count == 1
    assert summary.summary.total.input_tokens == 5
