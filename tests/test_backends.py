from pathlib import Path

import pytest

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.fake import FakeInferenceBackend, FakeTrainingBackend
from ganker.contracts import (
    AdamParams,
    ArtifactKind,
    Datum,
    ModelInput,
    SamplingParams,
    TensorData,
    TuningMode,
)
from ganker.errors import InvalidRequestError, NotFoundError


def _datum(tokens: list[int]) -> Datum:
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_ints(tokens[1:] + [0]),
            "weights": TensorData.from_floats([1.0 for _ in tokens]),
        },
    )


def test_fake_training_backend_runs_singleton_flow(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    backend = FakeTrainingBackend(store)

    run = backend.create_training_run(
        base_model="Qwen/Qwen3-8B",
        tuning_mode=TuningMode.LORA,
        lora_rank=32,
    )
    fb = backend.forward_backward(
        run_id=run.run_id,
        data=[_datum([10, 11, 12])],
        loss_fn="cross_entropy",
        loss_fn_config={},
    )
    step = backend.optim_step(run_id=run.run_id, params=AdamParams(learning_rate=1e-4))
    artifact = backend.save_weights(run_id=run.run_id, kind=ArtifactKind.DELTA)

    assert run.run_id == "run-000001"
    assert fb.gradient_version == 1
    assert fb.output.loss > 0
    assert fb.usage.input_tokens == 3
    assert step.optimizer_step == 1
    assert step.usage.training_steps == 1
    assert artifact.checkpoint_version == 1


def test_fake_training_backend_validates_run_and_loss(tmp_path: Path):
    backend = FakeTrainingBackend(FilesystemArtifactStore(tmp_path))

    with pytest.raises(InvalidRequestError):
        backend.create_training_run(base_model="", tuning_mode=TuningMode.LORA, lora_rank=32)

    with pytest.raises(NotFoundError):
        backend.forward_backward(
            run_id="missing",
            data=[_datum([1])],
            loss_fn="cross_entropy",
            loss_fn_config={},
        )


def test_fake_inference_backend_refreshes_and_samples(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    artifact = store.write(
        run_id="run-1",
        checkpoint_version=2,
        kind=ArtifactKind.FULL,
        payload={"checkpoint_version": 2},
    )
    backend = FakeInferenceBackend(store)

    loaded = backend.refresh_weights(run_id="run-1", artifact=artifact)
    sample = backend.sample(
        run_id="run-1",
        prompt=ModelInput.from_ints([5, 6]),
        sampling_params=SamplingParams(max_tokens=3),
        num_samples=1,
    )

    assert loaded.checkpoint_version == 2
    assert sample.sequences[0].tokens == [9, 10, 11]
    assert sample.usage.input_tokens == 2
    assert sample.usage.output_tokens == 3
    assert sample.usage.samples == 1


def test_fake_inference_backend_can_pull_latest_without_explicit_refresh(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    store.write(
        run_id="run-1",
        checkpoint_version=1,
        kind=ArtifactKind.DELTA,
        payload={},
    )
    backend = FakeInferenceBackend(store)

    sample = backend.sample(
        run_id="run-1",
        prompt=ModelInput.from_ints([]),
        sampling_params=SamplingParams(max_tokens=2),
        num_samples=1,
    )

    assert sample.artifact.checkpoint_version == 1
    assert sample.sequences[0].tokens == [2, 3]
