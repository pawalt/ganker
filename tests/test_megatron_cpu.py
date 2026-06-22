import json
from pathlib import Path

import pytest

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.factory import build_training_backend
from ganker.backends.megatron import (
    InstalledMegatronBridgeRuntime,
    InProcessMegatronCoreRuntime,
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


def test_factory_core_runtime_does_not_require_megatron_bridge(tmp_path: Path):
    backend = build_training_backend(
        "megatron",
        tmp_path,
        config={"runtime_kind": "core", "tensor_device": "cpu"},
    )

    assert backend.__class__.__name__ == "MegatronTrainingBackend"


def test_megatron_backend_rejects_unknown_runtime_kind(tmp_path: Path):
    with pytest.raises(InvalidRequestError, match="unknown Megatron runtime_kind"):
        MegatronTrainingBackend(
            FilesystemArtifactStore(tmp_path),
            config=MegatronBackendConfig(runtime_kind="bogus"),
        )


def test_core_runtime_validates_tiny_model_config_without_importing_megatron():
    runtime = InProcessMegatronCoreRuntime()

    with pytest.raises(InvalidRequestError, match="hidden_size must be divisible"):
        runtime._validate_config(
            MegatronBackendConfig(
                runtime_kind="core",
                hidden_size=30,
                num_attention_heads=8,
            )
        )


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
        self.shutdowns = []

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

    def shutdown(self, *, handle):
        self.shutdowns.append(handle)


def _backend(tmp_path: Path, runtime: FakeMegatronRuntime) -> MegatronTrainingBackend:
    return MegatronTrainingBackend(
        FilesystemArtifactStore(tmp_path),
        config=MegatronBackendConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            tensor_device="cpu",
        ),
        runtime=runtime,
    )


def _create_run(backend: MegatronTrainingBackend):
    return backend.create_training_run(
        base_model="local/tiny-config",
        tuning_mode=TuningMode.LORA,
        lora_rank=4,
    )


def _patch_tensor_batch(monkeypatch, token_count: int = 3):
    class FakeTensorBatch:
        def __init__(self, token_count: int):
            self.token_count = token_count
            self.input_ids = None
            self.target_tokens = None
            self.weights = None

    def fake_datums_to_tensor_batch(data, *, loss_fn, device):
        _ = loss_fn, device
        return FakeTensorBatch(token_count)

    monkeypatch.setitem(
        MegatronTrainingBackend.forward_backward.__globals__,
        "datums_to_tensor_batch",
        fake_datums_to_tensor_batch,
    )


def test_megatron_backend_lifecycle_with_mocked_runtime(tmp_path: Path, monkeypatch):
    _patch_tensor_batch(monkeypatch)
    runtime = FakeMegatronRuntime()
    backend = _backend(tmp_path, runtime)

    run = _create_run(backend)
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
    assert payload["run_status"] == "ready"
    assert payload["runtime"] == "fake-megatron"
    assert runtime.forwarded[0][1] == 3
    assert runtime.optimized[0][1] == 1e-4
    assert runtime.saved[0][2] == 1


def test_megatron_backend_requires_forward_backward_before_optim_step(tmp_path: Path):
    runtime = FakeMegatronRuntime()
    backend = _backend(tmp_path, runtime)
    run = _create_run(backend)

    with pytest.raises(InvalidRequestError, match="gradients_pending"):
        backend.optim_step(run_id=run.run_id, params=AdamParams(learning_rate=1e-4))

    assert runtime.optimized == []


def test_megatron_backend_rejects_second_forward_backward_before_step(
    tmp_path: Path,
    monkeypatch,
):
    _patch_tensor_batch(monkeypatch)
    runtime = FakeMegatronRuntime()
    backend = _backend(tmp_path, runtime)
    run = _create_run(backend)

    backend.forward_backward(
        run_id=run.run_id,
        data=[_datum([10, 11, 12])],
        loss_fn="cross_entropy",
        loss_fn_config={},
    )

    with pytest.raises(InvalidRequestError, match="gradients_pending"):
        backend.forward_backward(
            run_id=run.run_id,
            data=[_datum([10, 11, 12])],
            loss_fn="cross_entropy",
            loss_fn_config={},
        )

    assert len(runtime.forwarded) == 1


def test_megatron_backend_rejects_save_while_gradients_are_pending(
    tmp_path: Path,
    monkeypatch,
):
    _patch_tensor_batch(monkeypatch)
    runtime = FakeMegatronRuntime()
    backend = _backend(tmp_path, runtime)
    run = _create_run(backend)

    backend.forward_backward(
        run_id=run.run_id,
        data=[_datum([10, 11, 12])],
        loss_fn="cross_entropy",
        loss_fn_config={},
    )

    with pytest.raises(InvalidRequestError, match="gradients_pending"):
        backend.save_weights(run_id=run.run_id, kind=ArtifactKind.DELTA)

    assert runtime.saved == []


def test_megatron_backend_marks_runtime_forward_failure_failed(tmp_path: Path, monkeypatch):
    _patch_tensor_batch(monkeypatch)
    class FailingForwardRuntime(FakeMegatronRuntime):
        def forward_backward(self, *, handle, batch, loss_fn, loss_fn_config):
            super().forward_backward(
                handle=handle,
                batch=batch,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )
            raise RuntimeError("forward exploded")

    runtime = FailingForwardRuntime()
    backend = _backend(tmp_path, runtime)
    run = _create_run(backend)

    with pytest.raises(RuntimeError, match="forward exploded"):
        backend.forward_backward(
            run_id=run.run_id,
            data=[_datum([10, 11, 12])],
            loss_fn="cross_entropy",
            loss_fn_config={},
        )

    with pytest.raises(InvalidRequestError, match="failed"):
        backend.save_weights(run_id=run.run_id, kind=ArtifactKind.DELTA)


def test_megatron_backend_save_failure_leaves_run_ready_for_retry(tmp_path: Path):
    class FlakySaveRuntime(FakeMegatronRuntime):
        def __init__(self):
            super().__init__()
            self.fail_next_save = True

        def save_weights(self, *, handle, run_id, checkpoint_version, kind):
            if self.fail_next_save:
                self.fail_next_save = False
                raise RuntimeError("checkpoint exploded")
            return super().save_weights(
                handle=handle,
                run_id=run_id,
                checkpoint_version=checkpoint_version,
                kind=kind,
            )

    runtime = FlakySaveRuntime()
    backend = _backend(tmp_path, runtime)
    run = _create_run(backend)

    with pytest.raises(RuntimeError, match="checkpoint exploded"):
        backend.save_weights(run_id=run.run_id, kind=ArtifactKind.DELTA)

    artifact = backend.save_weights(run_id=run.run_id, kind=ArtifactKind.DELTA)

    assert artifact.checkpoint_version == 0
    assert len(runtime.saved) == 1


def test_megatron_backend_allows_only_one_active_run(tmp_path: Path):
    runtime = FakeMegatronRuntime()
    backend = _backend(tmp_path, runtime)

    run = _create_run(backend)
    with pytest.raises(InvalidRequestError, match="one active run"):
        _create_run(backend)

    backend.close()
    second_run = _create_run(backend)

    assert run.run_id == "meg-run-000001"
    assert second_run.run_id == "meg-run-000002"


def test_megatron_backend_close_shuts_down_runtime_and_closes_run(tmp_path: Path):
    runtime = FakeMegatronRuntime()
    backend = _backend(tmp_path, runtime)
    run = _create_run(backend)

    backend.close()

    assert len(runtime.shutdowns) == 1
    with pytest.raises(InvalidRequestError, match="closed"):
        backend.save_weights(run_id=run.run_id, kind=ArtifactKind.DELTA)
