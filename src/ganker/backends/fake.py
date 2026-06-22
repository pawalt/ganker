"""Local deterministic backends used by tests and development."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import Dict

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.base import ForwardBackwardResult, OptimStepResult, SampleResult
from ganker.contracts import (
    AdamParams,
    ArtifactKind,
    Datum,
    ForwardBackwardOutput,
    ModelInput,
    SampledSequence,
    SamplingParams,
    TensorData,
    TrainingRun,
    TuningMode,
    Usage,
    WeightArtifact,
)
from ganker.errors import InvalidRequestError, NotFoundError


@dataclass
class _RunState:
    run_id: str
    base_model: str
    tuning_mode: TuningMode
    lora_rank: int
    gradient_version: int = 0
    optimizer_step: int = 0
    checkpoint_version: int = 0


class FakeTrainingBackend:
    """Deterministic singleton training backend with no ML dependencies."""

    def __init__(self, artifact_store: FilesystemArtifactStore):
        self._artifact_store = artifact_store
        self._run_counter = count(1)
        self._runs: Dict[str, _RunState] = {}

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

        run_id = f"run-{next(self._run_counter):06d}"
        state = _RunState(
            run_id=run_id,
            base_model=base_model,
            tuning_mode=tuning_mode,
            lora_rank=lora_rank,
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
        if not data:
            raise InvalidRequestError("data cannot be empty")
        if not loss_fn:
            raise InvalidRequestError("loss_fn is required")

        token_count = 0
        loss_fn_outputs: list[dict[str, TensorData]] = []
        for datum in data:
            if not datum.model_input.token_ids:
                raise InvalidRequestError("datum.model_input.token_ids cannot be empty")
            token_count += len(datum.model_input.token_ids)
            loss_fn_outputs.append(
                {
                    "logprobs": TensorData.from_floats(
                        -0.01 * (index + 1)
                        for index, _ in enumerate(datum.model_input.token_ids)
                    )
                }
            )

        state.gradient_version += 1
        factor = {
            "cross_entropy": 1.0,
            "sft": 1.0,
            "dpo": 2.0,
            "importance_sampling": 2.0,
            "ppo": 3.0,
            "rl": 3.0,
        }.get(loss_fn, 1.0)
        param_bonus = sum(loss_fn_config.values()) / 1000.0
        fake_loss = (token_count * factor / 100.0) + param_bonus
        usage = Usage(input_tokens=token_count)
        return ForwardBackwardResult(
            run_id=run_id,
            output=ForwardBackwardOutput(
                loss=fake_loss,
                metrics={"loss": fake_loss},
                loss_fn_outputs=loss_fn_outputs,
            ),
            gradient_version=state.gradient_version,
            usage=usage,
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

        state.optimizer_step += 1
        state.checkpoint_version += 1
        return OptimStepResult(
            run_id=run_id,
            optimizer_step=state.optimizer_step,
            checkpoint_version=state.checkpoint_version,
            usage=Usage(training_steps=1),
        )

    def save_weights(self, *, run_id: str, kind: ArtifactKind) -> WeightArtifact:
        state = self._get_run(run_id)
        payload = {
            "backend": "fake",
            "base_model": state.base_model,
            "checkpoint_version": state.checkpoint_version,
            "gradient_version": state.gradient_version,
            "lora_rank": state.lora_rank,
            "optimizer_step": state.optimizer_step,
            "run_id": state.run_id,
            "tuning_mode": state.tuning_mode,
        }
        return self._artifact_store.write(
            run_id=state.run_id,
            checkpoint_version=state.checkpoint_version,
            kind=kind,
            payload=payload,
        )

    def _get_run(self, run_id: str) -> _RunState:
        if not run_id:
            raise InvalidRequestError("run_id is required")
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise NotFoundError(f"unknown run_id={run_id}") from exc

    def _run_message(self, state: _RunState) -> TrainingRun:
        return TrainingRun(
            run_id=state.run_id,
            base_model=state.base_model,
            tuning_mode=state.tuning_mode,
            lora_rank=state.lora_rank,
            checkpoint_version=state.checkpoint_version,
        )


class FakeInferenceBackend:
    """Deterministic rollout backend with filesystem weight refresh."""

    def __init__(self, artifact_store: FilesystemArtifactStore):
        self._artifact_store = artifact_store
        self._loaded_artifacts: Dict[str, WeightArtifact] = {}

    def refresh_weights(
        self,
        *,
        run_id: str,
        artifact: WeightArtifact | None,
    ) -> WeightArtifact:
        if not run_id:
            raise InvalidRequestError("run_id is required")
        if artifact is not None:
            if artifact.run_id and artifact.run_id != run_id:
                raise InvalidRequestError("artifact.run_id must match request.run_id")
            loaded = artifact
        else:
            loaded = self._artifact_store.latest(run_id)

        self._loaded_artifacts[run_id] = loaded
        return loaded

    def sample(
        self,
        *,
        run_id: str,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int,
    ) -> SampleResult:
        if not run_id:
            raise InvalidRequestError("run_id is required")
        if sampling_params.max_tokens <= 0:
            raise InvalidRequestError("sampling_params.max_tokens must be positive")
        if num_samples <= 0:
            raise InvalidRequestError("num_samples must be positive")

        artifact = self._loaded_artifacts.get(run_id)
        if artifact is None:
            artifact = self._artifact_store.latest(run_id)
            self._loaded_artifacts[run_id] = artifact

        base = prompt.token_ids[-1] if prompt.token_ids else 0
        sequences = []
        for sample_index in range(num_samples):
            tokens = [
                base
                + artifact.checkpoint_version
                + (sample_index * sampling_params.max_tokens)
                + offset
                + 1
                for offset in range(sampling_params.max_tokens)
            ]
            sequences.append(
                SampledSequence(
                    tokens=tokens,
                    logprobs=[-0.1 for _ in tokens],
                )
            )
        usage = Usage(
            input_tokens=len(prompt.token_ids),
            output_tokens=sampling_params.max_tokens * num_samples,
            samples=num_samples,
        )
        return SampleResult(
            run_id=run_id,
            sequences=sequences,
            artifact=artifact,
            usage=usage,
        )
