"""Import-isolated placeholder for the future Megatron training backend."""

from ganker.artifacts import FilesystemArtifactStore
from ganker.errors import BackendUnavailableError


class MegatronTrainingBackend:
    def __init__(self, artifact_store: FilesystemArtifactStore):
        self._artifact_store = artifact_store
        try:
            import megatron  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailableError(
                "Megatron backend requested, but megatron is not installed"
            ) from exc

        raise BackendUnavailableError("Megatron backend is not implemented yet")
