from ganker.contracts import (
    ArtifactKind,
    Datum,
    ForwardBackwardOutput,
    ForwardBackwardResponse,
    ModelInput,
    OptimStepResponse,
    SaveWeightsResponse,
    TensorData,
    TrainingRun,
    TuningMode,
    Usage,
    WeightArtifact,
)

from examples.sft.loop import run_sft


def _datum(tokens: list[int]) -> Datum:
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_ints(tokens[1:] + [0]),
            "weights": TensorData.from_floats([1.0 for _ in tokens]),
        },
    )


class FakeTrainingClient:
    def __init__(self, run: TrainingRun):
        self.run = run
        self.forwarded = []
        self.saved_kinds = []
        self.step_count = 0

    @property
    def run_id(self) -> str:
        return self.run.run_id

    def forward_backward(self, batch, *, loss_fn):
        self.forwarded.append((batch, loss_fn))
        loss = 1.0 / len(self.forwarded)
        return ForwardBackwardResponse(
            request_id="",
            run_id=self.run_id,
            output=ForwardBackwardOutput(loss=loss, metrics={"loss": loss}),
            gradient_version=len(self.forwarded),
            usage=Usage(input_tokens=sum(len(datum.model_input.token_ids) for datum in batch)),
        )

    def optim_step(self, *, learning_rate):
        assert learning_rate == 1e-4
        self.step_count += 1
        return OptimStepResponse(
            request_id="",
            run_id=self.run_id,
            optimizer_step=self.step_count,
            checkpoint_version=self.step_count,
            usage=Usage(training_steps=1),
        )

    def save_weights(self, *, kind):
        self.saved_kinds.append(kind)
        artifact = WeightArtifact(
            run_id=self.run_id,
            checkpoint_version=self.step_count,
            kind=kind,
            manifest_path="/tmp/manifest.json",
            payload_path="/tmp/payload.json",
        )
        return SaveWeightsResponse(request_id="", artifact=artifact)


class FakeServiceClient:
    def __init__(self):
        self.created = []
        self.training = None

    def create_training_client(self, *, base_model, tuning, rank):
        self.created.append((base_model, tuning, rank))
        self.training = FakeTrainingClient(
            TrainingRun(
                run_id="run-1",
                base_model=base_model,
                tuning_mode=TuningMode.FULL,
                lora_rank=rank,
                checkpoint_version=0,
            )
        )
        return self.training


def test_run_sft_drives_public_training_client_and_saves_final_artifact():
    client = FakeServiceClient()
    dataset = [[_datum([1, 2, 3])], [_datum([4, 5, 6])]]

    summary = run_sft(
        client,
        base_model="local/tiny-config",
        dataset=dataset,
        tuning="full",
        lora_rank=0,
        learning_rate=1e-4,
        max_steps=2,
        save_every=0,
    )

    assert client.created == [("local/tiny-config", "full", 0)]
    assert summary.ok is True
    assert summary.steps == 2
    assert summary.losses == [1.0, 0.5]
    assert summary.final_loss == 0.5
    assert summary.optimizer_step == 2
    assert summary.checkpoint_version == 2
    assert summary.input_tokens == 6
    assert summary.training_steps == 2
    assert summary.artifact_path == "/tmp/payload.json"
    assert client.training is not None
    assert client.training.saved_kinds == [ArtifactKind.FULL]
    assert summary.to_dict()["ok"] is True


def test_run_sft_honors_max_steps_and_save_every():
    client = FakeServiceClient()
    dataset = [[_datum([1])], [_datum([2])], [_datum([3])]]

    summary = run_sft(
        client,
        base_model="local/tiny-config",
        dataset=dataset,
        max_steps=2,
        save_every=1,
    )

    assert summary.steps == 2
    assert client.training is not None
    assert len(client.training.forwarded) == 2
    assert client.training.saved_kinds == [ArtifactKind.FULL, ArtifactKind.FULL]
