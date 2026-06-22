"""Import-isolated Megatron Bridge training backend adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count
import importlib
from threading import RLock
from typing import Any, Protocol

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.base import ForwardBackwardResult, OptimStepResult
from ganker.config import MegatronBackendConfig
from ganker.contracts import (
    AdamParams,
    ArtifactKind,
    Datum,
    ForwardBackwardOutput,
    TensorData,
    TrainingRun,
    TuningMode,
    Usage,
    WeightArtifact,
)
from ganker.errors import BackendUnavailableError, InvalidRequestError, NotFoundError


@dataclass(frozen=True)
class MegatronTensorBatch:
    """Torch tensors produced from Ganker datums for Megatron preflight/runtime."""

    input_ids: Any
    target_tokens: Any
    weights: Any

    @property
    def token_count(self) -> int:
        shape = getattr(self.input_ids, "shape", ())
        if len(shape) == 2:
            return int(shape[0] * shape[1])
        return int(getattr(self.input_ids, "numel")())


@dataclass(frozen=True)
class MegatronRunHandle:
    bridge: Any
    provider: Any
    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    tokenizer: Any = None
    forward_backward_schedule: Any = None
    config: Any = None
    distributed_context: Any = None


class _RunStatus(Enum):
    READY = "ready"
    GRADIENTS_PENDING = "gradients_pending"
    CHECKPOINTING = "checkpointing"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class _RunState:
    run_id: str
    base_model: str
    tuning_mode: TuningMode
    lora_rank: int
    handle: MegatronRunHandle
    status: _RunStatus = _RunStatus.READY
    gradient_version: int = 0
    optimizer_step: int = 0
    checkpoint_version: int = 0
    lock: RLock = field(default_factory=RLock, repr=False)


class MegatronRuntime(Protocol):
    def create_run(
        self,
        *,
        base_model: str,
        tuning_mode: TuningMode,
        lora_rank: int,
        config: MegatronBackendConfig,
    ) -> MegatronRunHandle:
        ...

    def forward_backward(
        self,
        *,
        handle: MegatronRunHandle,
        batch: MegatronTensorBatch,
        loss_fn: str,
        loss_fn_config: dict[str, float],
    ) -> ForwardBackwardOutput:
        ...

    def optim_step(
        self,
        *,
        handle: MegatronRunHandle,
        params: AdamParams,
    ) -> None:
        ...

    def save_weights(
        self,
        *,
        handle: MegatronRunHandle,
        run_id: str,
        checkpoint_version: int,
        kind: ArtifactKind,
    ) -> dict[str, Any]:
        ...

    def shutdown(self, *, handle: MegatronRunHandle) -> None:
        ...


class InstalledMegatronBridgeRuntime:
    """Runtime that touches Megatron Bridge only when the backend is selected."""

    def __init__(self, bridge_module: Any):
        self._bridge = bridge_module

    @classmethod
    def from_installed(cls) -> "InstalledMegatronBridgeRuntime":
        try:
            bridge_module = importlib.import_module("megatron.bridge")
        except ImportError as exc:
            raise BackendUnavailableError(
                "Megatron backend requested, but megatron.bridge is not installed"
            ) from exc
        return cls(bridge_module)

    def create_run(
        self,
        *,
        base_model: str,
        tuning_mode: TuningMode,
        lora_rank: int,
        config: MegatronBackendConfig,
    ) -> MegatronRunHandle:
        _ = tuning_mode, lora_rank
        auto_bridge = self._bridge.AutoBridge.from_hf_pretrained(
            base_model,
            trust_remote_code=config.trust_remote_code,
        )
        provider = auto_bridge.to_megatron_provider(load_weights=config.load_weights)
        provider.tensor_model_parallel_size = config.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = config.pipeline_model_parallel_size
        if hasattr(provider, "finalize"):
            provider.finalize()
        return MegatronRunHandle(bridge=auto_bridge, provider=provider)

    def forward_backward(
        self,
        *,
        handle: MegatronRunHandle,
        batch: MegatronTensorBatch,
        loss_fn: str,
        loss_fn_config: dict[str, float],
    ) -> ForwardBackwardOutput:
        _ = handle, batch, loss_fn, loss_fn_config
        raise BackendUnavailableError(
            "real Megatron forward/backward requires a GPU worker runtime; "
            "use the CPU preflight tests locally or the Modal GPU test path"
        )

    def optim_step(
        self,
        *,
        handle: MegatronRunHandle,
        params: AdamParams,
    ) -> None:
        _ = handle, params
        raise BackendUnavailableError(
            "real Megatron optimizer steps require a GPU worker runtime"
        )

    def save_weights(
        self,
        *,
        handle: MegatronRunHandle,
        run_id: str,
        checkpoint_version: int,
        kind: ArtifactKind,
    ) -> dict[str, Any]:
        _ = handle, run_id, checkpoint_version, kind
        raise BackendUnavailableError(
            "real Megatron checkpoint writing requires a GPU worker runtime"
        )

    def shutdown(self, *, handle: MegatronRunHandle) -> None:
        _ = handle


def is_megatron_bridge_available() -> bool:
    try:
        importlib.import_module("megatron.bridge")
    except ImportError:
        return False
    return True


def datums_to_tensor_batch(
    data: list[Datum],
    *,
    loss_fn: str,
    device: str = "cpu",
) -> MegatronTensorBatch:
    """Convert basic Ganker datums into torch tensors for CPU preflight/runtime."""

    if loss_fn != "cross_entropy":
        raise InvalidRequestError(f"unsupported Megatron loss_fn: {loss_fn}")
    if not data:
        raise InvalidRequestError("data cannot be empty")

    token_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    weight_rows: list[list[float]] = []
    expected_length: int | None = None
    for index, datum in enumerate(data):
        tokens = [int(token) for token in datum.model_input.token_ids]
        if not tokens:
            raise InvalidRequestError(f"data[{index}].model_input.token_ids cannot be empty")
        targets = _required_tensor_values(datum, "target_tokens", index)
        weights = _required_tensor_values(datum, "weights", index)
        if len(targets) != len(tokens):
            raise InvalidRequestError(
                f"data[{index}].loss_fn_inputs['target_tokens'] length must match token_ids"
            )
        if len(weights) != len(tokens):
            raise InvalidRequestError(
                f"data[{index}].loss_fn_inputs['weights'] length must match token_ids"
            )
        if expected_length is None:
            expected_length = len(tokens)
        elif len(tokens) != expected_length:
            raise InvalidRequestError("all datums must have the same token length")

        token_rows.append(tokens)
        target_rows.append([int(value) for value in targets])
        weight_rows.append([float(value) for value in weights])

    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "torch is required for Megatron tensor conversion preflight"
        ) from exc

    return MegatronTensorBatch(
        input_ids=torch.tensor(token_rows, dtype=torch.long, device=device),
        target_tokens=torch.tensor(target_rows, dtype=torch.long, device=device),
        weights=torch.tensor(weight_rows, dtype=torch.float32, device=device),
    )


def _required_tensor_values(datum: Datum, name: str, datum_index: int) -> list[int | float]:
    try:
        tensor = datum.loss_fn_inputs[name]
    except KeyError as exc:
        raise InvalidRequestError(
            f"data[{datum_index}].loss_fn_inputs['{name}'] is required"
        ) from exc
    if not isinstance(tensor, TensorData):
        raise InvalidRequestError(
            f"data[{datum_index}].loss_fn_inputs['{name}'] must be TensorData"
        )
    return tensor.tolist()


class MegatronTrainingBackend:
    """Coordinator for Megatron Bridge training state.

    Local CPU tests inject a fake runtime. The default runtime only builds
    Megatron Bridge provider/config objects and fails clearly for real training
    until a GPU worker runtime is wired in.
    """

    def __init__(
        self,
        artifact_store: FilesystemArtifactStore,
        *,
        config: MegatronBackendConfig | None = None,
        runtime: MegatronRuntime | None = None,
    ):
        self._artifact_store = artifact_store
        self._config = config or MegatronBackendConfig()
        self._runtime = runtime or InstalledMegatronBridgeRuntime.from_installed()
        self._run_counter = count(1)
        self._runs: dict[str, _RunState] = {}
        self._lock = RLock()

    def create_training_run(
        self,
        *,
        base_model: str,
        tuning_mode: TuningMode,
        lora_rank: int,
    ) -> TrainingRun:
        if not base_model:
            raise InvalidRequestError("base_model is required")
        if tuning_mode not in (TuningMode.LORA, TuningMode.FULL):
            raise InvalidRequestError(f"unsupported tuning mode: {tuning_mode}")
        if tuning_mode == TuningMode.LORA and lora_rank <= 0:
            raise InvalidRequestError("lora_rank must be positive for LoRA runs")
        if tuning_mode == TuningMode.FULL and lora_rank < 0:
            raise InvalidRequestError("lora_rank cannot be negative")

        with self._lock:
            self._ensure_no_active_run()
            handle = self._runtime.create_run(
                base_model=base_model,
                tuning_mode=tuning_mode,
                lora_rank=lora_rank,
                config=self._config,
            )
            run_id = f"meg-run-{next(self._run_counter):06d}"
            state = _RunState(
                run_id=run_id,
                base_model=base_model,
                tuning_mode=tuning_mode,
                lora_rank=lora_rank,
                handle=handle,
            )
            self._runs[run_id] = state
            return self._run_message(state)

    def forward_backward(
        self,
        *,
        run_id: str,
        data: list[Datum],
        loss_fn: str,
        loss_fn_config: dict[str, float],
    ) -> ForwardBackwardResult:
        state = self._get_run(run_id)
        with state.lock:
            self._require_status(
                state,
                _RunStatus.READY,
                operation="forward_backward",
            )
            batch = datums_to_tensor_batch(
                data,
                loss_fn=loss_fn,
                device=self._config.tensor_device,
            )
            try:
                output = self._runtime.forward_backward(
                    handle=state.handle,
                    batch=batch,
                    loss_fn=loss_fn,
                    loss_fn_config=loss_fn_config,
                )
            except Exception:
                state.status = _RunStatus.FAILED
                raise

            state.gradient_version += 1
            state.status = _RunStatus.GRADIENTS_PENDING
            return ForwardBackwardResult(
                run_id=run_id,
                output=output,
                gradient_version=state.gradient_version,
                usage=Usage(input_tokens=batch.token_count),
            )

    def optim_step(
        self,
        *,
        run_id: str,
        params: AdamParams,
    ) -> OptimStepResult:
        state = self._get_run(run_id)
        if params.learning_rate <= 0:
            raise InvalidRequestError("learning_rate must be positive")

        with state.lock:
            self._require_status(
                state,
                _RunStatus.GRADIENTS_PENDING,
                operation="optim_step",
            )
            try:
                self._runtime.optim_step(handle=state.handle, params=params)
            except Exception:
                state.status = _RunStatus.FAILED
                raise

            state.optimizer_step += 1
            state.checkpoint_version += 1
            state.status = _RunStatus.READY
            return OptimStepResult(
                run_id=run_id,
                optimizer_step=state.optimizer_step,
                checkpoint_version=state.checkpoint_version,
                usage=Usage(training_steps=1),
            )

    def save_weights(self, *, run_id: str, kind: ArtifactKind) -> WeightArtifact:
        state = self._get_run(run_id)
        with state.lock:
            self._require_status(
                state,
                _RunStatus.READY,
                operation="save_weights",
            )
            state.status = _RunStatus.CHECKPOINTING
            try:
                runtime_payload = self._runtime.save_weights(
                    handle=state.handle,
                    run_id=run_id,
                    checkpoint_version=state.checkpoint_version,
                    kind=kind,
                ) or {}
                payload = {
                    "artifact_format": "megatron",
                    "backend": "megatron-bridge",
                    "base_model": state.base_model,
                    "checkpoint_version": state.checkpoint_version,
                    "gradient_version": state.gradient_version,
                    "global_batch_size": self._config.global_batch_size,
                    "lora_rank": state.lora_rank,
                    "micro_batch_size": self._config.micro_batch_size,
                    "optimizer_step": state.optimizer_step,
                    "pipeline_model_parallel_size": self._config.pipeline_model_parallel_size,
                    "run_id": state.run_id,
                    "run_status": _RunStatus.READY.value,
                    "sequence_length": self._config.sequence_length,
                    "tensor_model_parallel_size": self._config.tensor_model_parallel_size,
                    "tuning_mode": state.tuning_mode.name,
                }
                payload.update(runtime_payload)
                artifact = self._artifact_store.write(
                    run_id=state.run_id,
                    checkpoint_version=state.checkpoint_version,
                    kind=kind,
                    payload=payload,
                )
            except Exception:
                state.status = _RunStatus.READY
                raise

            state.status = _RunStatus.READY
            return artifact

    def close(self) -> None:
        with self._lock:
            states = list(self._runs.values())
        for state in states:
            with state.lock:
                if state.status is _RunStatus.CLOSED:
                    continue
                try:
                    self._runtime.shutdown(handle=state.handle)
                except Exception:
                    state.status = _RunStatus.FAILED
                else:
                    state.status = _RunStatus.CLOSED

    def _get_run(self, run_id: str) -> _RunState:
        if not run_id:
            raise InvalidRequestError("run_id is required")
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise NotFoundError(f"unknown run_id={run_id}") from exc

    def _ensure_no_active_run(self) -> None:
        active = [
            state.run_id
            for state in self._runs.values()
            if state.status not in (_RunStatus.FAILED, _RunStatus.CLOSED)
        ]
        if active:
            raise InvalidRequestError(
                "Megatron backend supports one active run per actor; "
                f"active run_id={active[0]}"
            )

    def _require_status(
        self,
        state: _RunState,
        expected: _RunStatus,
        *,
        operation: str,
    ) -> None:
        if state.status is expected:
            return
        raise InvalidRequestError(
            f"cannot {operation} run_id={state.run_id} while run status is "
            f"{state.status.value}; expected {expected.value}"
        )

    def _run_message(self, state: _RunState) -> TrainingRun:
        return TrainingRun(
            run_id=state.run_id,
            base_model=state.base_model,
            tuning_mode=state.tuning_mode,
            lora_rank=state.lora_rank,
            checkpoint_version=state.checkpoint_version,
        )
