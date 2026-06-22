"""Import-isolated placeholder for the future sglang rollout backend."""

from ganker.artifacts import FilesystemArtifactStore
from ganker.errors import BackendUnavailableError


class SGLangInferenceBackend:
    def __init__(self, artifact_store: FilesystemArtifactStore):
        self._artifact_store = artifact_store
        try:
            import sglang  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailableError(
                "sglang backend requested, but sglang is not installed"
            ) from exc

        raise BackendUnavailableError("sglang backend is not implemented yet")
