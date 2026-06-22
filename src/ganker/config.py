"""Configuration for local Monarch orchestration."""

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class MeshSettings:
    """Local defaults for the singleton Monarch mesh."""

    artifact_root: Path = Path(".local_artifacts")
    monarch_transport: str = "tcp"
    training_backend: str = "fake"
    inference_backend: str = "fake"


def load_settings() -> MeshSettings:
    return MeshSettings(
        artifact_root=Path(os.getenv("GANKER_ARTIFACT_ROOT", ".local_artifacts")),
        monarch_transport=os.getenv("GANKER_MONARCH_TRANSPORT", "tcp"),
        training_backend=os.getenv("GANKER_TRAINING_BACKEND", "fake"),
        inference_backend=os.getenv("GANKER_INFERENCE_BACKEND", "fake"),
    )
