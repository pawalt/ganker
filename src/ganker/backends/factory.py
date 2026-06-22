"""Config-driven backend construction."""

from pathlib import Path

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.base import InferenceBackend, TrainingBackend
from ganker.backends.fake import FakeInferenceBackend, FakeTrainingBackend
from ganker.errors import InvalidRequestError


def build_training_backend(kind: str, artifact_root: Path) -> TrainingBackend:
    if kind == "fake":
        return FakeTrainingBackend(FilesystemArtifactStore(artifact_root))
    if kind == "megatron":
        from ganker.backends.megatron import MegatronTrainingBackend

        return MegatronTrainingBackend(FilesystemArtifactStore(artifact_root))
    raise InvalidRequestError(f"unknown training backend: {kind}")


def build_inference_backend(kind: str, artifact_root: Path) -> InferenceBackend:
    if kind == "fake":
        return FakeInferenceBackend(FilesystemArtifactStore(artifact_root))
    if kind == "sglang":
        from ganker.backends.sglang import SGLangInferenceBackend

        return SGLangInferenceBackend(FilesystemArtifactStore(artifact_root))
    raise InvalidRequestError(f"unknown inference backend: {kind}")
