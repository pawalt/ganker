"""Config-driven backend construction."""

from pathlib import Path
from typing import Any

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.base import InferenceBackend, TrainingBackend
from ganker.backends.fake import FakeInferenceBackend, FakeTrainingBackend
from ganker.config import MegatronBackendConfig, SGLangBackendConfig
from ganker.errors import InvalidRequestError


def build_training_backend(
    kind: str,
    artifact_root: Path,
    *,
    config: dict[str, Any] | MegatronBackendConfig | None = None,
) -> TrainingBackend:
    if kind == "fake":
        return FakeTrainingBackend(FilesystemArtifactStore(artifact_root))
    if kind == "megatron":
        from ganker.backends.megatron import MegatronTrainingBackend

        if isinstance(config, MegatronBackendConfig):
            megatron_config = config
        else:
            try:
                megatron_config = MegatronBackendConfig.from_mapping(config)
            except ValueError as exc:
                raise InvalidRequestError(str(exc)) from exc
        return MegatronTrainingBackend(
            FilesystemArtifactStore(artifact_root),
            config=megatron_config,
        )
    raise InvalidRequestError(f"unknown training backend: {kind}")


def build_inference_backend(
    kind: str,
    artifact_root: Path,
    *,
    config: dict[str, Any] | SGLangBackendConfig | None = None,
) -> InferenceBackend:
    if kind == "fake":
        return FakeInferenceBackend(FilesystemArtifactStore(artifact_root))
    if kind == "sglang":
        from ganker.backends.sglang import SGLangInferenceBackend

        if isinstance(config, SGLangBackendConfig):
            sglang_config = config
        else:
            try:
                sglang_config = SGLangBackendConfig.from_mapping(config)
            except ValueError as exc:
                raise InvalidRequestError(str(exc)) from exc
        return SGLangInferenceBackend(
            FilesystemArtifactStore(artifact_root),
            config=sglang_config,
        )
    raise InvalidRequestError(f"unknown inference backend: {kind}")
