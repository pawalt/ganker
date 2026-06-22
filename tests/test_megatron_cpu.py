import json
from pathlib import Path

import pytest

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.factory import build_training_backend
from ganker.backends.megatron import (
    InstalledMegatronBridgeRuntime,
    MegatronRunHandle,
    MegatronTrainingBackend,
    datums_to_tensor_batch,
)
from ganker.config import MegatronBackendConfig
from ganker.contracts import (
    AdamParams,
    ArtifactKind,
    Datum,
    ForwardBackwardOutput,
    ModelInput,
    TensorData,
    TuningMode,
)
from ganker.errors import BackendUnavailableError, InvalidRequestError


pytestmark = pytest.mark.megatron_cpu


def _datum(tokens: list[int]) -> Datum:
    return Datum(
        model_input=ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_ints(tokens[1:] + [0]),
            "weights": TensorData.from_floats([1.0 for _ in tokens]),
        },
    )


def test_megatron_config_rejects_unknown_fields():
    with pytest.raises(ValueError, match="unknown Megatron backend config"):
        MegatronBackendConfig.from_mapping({"bogus": 1})


def test_factory_reports_missing_megatron_bridge(monkeypatch, tmp_path: Path):
    def unavailable():
        raise BackendUnavailableError("missing bridge")

    monkeypatch.setattr(
        "ganker.backends.megatron.InstalledMegatronBridgeRuntime.from_installed",
        unavailable,
    )

    with pytest.raises(BackendUnavailableError, match="missing bridge"):
        build_training_backend("megatron", tmp_path)


def test_installed_runtime_maps_config_to_bridge_provider():
    class FakeProvider:
        def __init__(self):
            self.tensor_model_parallel_size = 0
            self.pipeline_model_parallel_size = 0
            self.finalized = False

        def finalize(self):
            self.finalized = True

    class FakeAutoBridge:
        calls = []

        def __init__(self):
            self.provider = FakeProvider()

        @classmethod
        def from_hf_pretrained(cls, base_model, trust_remote_code):
            cls.calls.append((base_model, trust_remote_code))
            return cls()

        def to_megatron_provider(self, load_weights):
            self.load_weights = load_weights
            return self.provider

    class FakeBridgeModule:
        AutoBridge = FakeAutoBridge

    runtime = InstalledMegatronBridgeRuntime(FakeBridgeModule)
    handle = runtime.create_run(
        base_model="local/tiny-config",
        tuning_mode=TuningMode.LORA,
        lora_rank=8,
        config=MegatronBackendConfig(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            trust_remote_code=False,
            load_weights=False,
        ),
    )

    assert FakeAutoBridge.calls == [("local/tiny-config", False)]
    assert handle.provider.tensor_model_parallel_size == 2
    assert handle.provider.pipeline_model_parallel_size == 1
    assert handle.provider.finalized is True


def test_datums_to_tensor_batch_validates_required_inputs_without_torch():
    datum = Datum(
        model_input=ModelInput.from_ints([1, 2]),
        loss_fn_inputs={"weights": TensorData.from_floats([1.0, 1.0])},
    )

    with pytest.raises(InvalidRequestError, match="target_tokens"):
        datums_to_tensor_batch([datum], loss_fn="cross_entropy")


def test_datums_to_tensor_batch_uses_cpu_tensors_when_torch_is_available():
    torch = pytest.importorskip("torch")

    batch = datums_to_tensor_batch(
        [_datum([1, 2, 3]), _datum([4, 5, 6])],
        loss_fn="cross_entropy",
        device="cpu",
    )

    assert batch.input_ids.device.type == "cpu"
    assert batch.input_ids.dtype == torch.long
    assert batch.input_ids.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert batch.target_tokens.tolist() == [[2, 3, 0], [5, 6, 0]]
    assert batch.weights.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


class FakeMegatronRuntime:
    def __init__(self):
        self.created = []
        self.forwarded = []
        self.optimized = []
        self.saved = []

    def create_run(self, *, base_model, tuning_mode, lora_rank, config):
        self.created.append((base_model, tuning_mode, lora_rank, config))
        return MegatronRunHandle(bridge="fake-bridge", provider={"model": base_model})

    def forward_backward(self, *, handle, batch, loss_fn, loss_fn_config):
        self.forwarded.append((handle, batch.token_count, loss_fn, loss_fn_config))
        return ForwardBackwardOutput(loss=0.25, metrics={"loss": 0.25})

    def optim_step(self, *, handle, params):
        self.optimized.append((handle, params.learning_rate))

    def save_weights(self, *, handle, run_id, checkpoint_version, kind):
        self.saved.append((handle, run_id, checkpoint_version, kind))
        return {"runtime": "fake-megatron", "checkpoint_path": f"/tmp/{run_id}"}


def test_megatron_backend_lifecycle_with_mocked_runtime(tmp_path: Path):
    pytest.importorskip("torch")
    runtime = FakeMegatronRuntime()
    backend = MegatronTrainingBackend(
        FilesystemArtifactStore(tmp_path),
        config=MegatronBackendConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            tensor_device="cpu",
        ),
        runtime=runtime,
    )

    run = backend.create_training_run(
        base_model="local/tiny-config",
        tuning_mode=TuningMode.LORA,
        lora_rank=4,
    )
    fb = backend.forward_backward(
        run_id=run.run_id,
        data=[_datum([10, 11, 12])],
        loss_fn="cross_entropy",
        loss_fn_config={"label_smoothing": 0.0},
    )
    step = backend.optim_step(
        run_id=run.run_id,
        params=AdamParams(learning_rate=1e-4),
    )
    artifact = backend.save_weights(run_id=run.run_id, kind=ArtifactKind.DELTA)
    payload = json.loads(Path(artifact.payload_path).read_text())

    assert run.run_id == "meg-run-000001"
    assert fb.output.loss == 0.25
    assert fb.usage.input_tokens == 3
    assert step.optimizer_step == 1
    assert payload["artifact_format"] == "megatron"
    assert payload["backend"] == "megatron-bridge"
    assert payload["runtime"] == "fake-megatron"
    assert runtime.forwarded[0][1] == 3
    assert runtime.optimized[0][1] == 1e-4
    assert runtime.saved[0][2] == 1

