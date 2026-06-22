import json
from pathlib import Path

import pytest

from ganker.artifacts import FilesystemArtifactStore
from ganker.contracts import ArtifactKind
from ganker.errors import NotFoundError


def test_artifact_store_writes_manifest_payload_and_latest_pointer(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)

    artifact = store.write(
        run_id="run-1",
        checkpoint_version=3,
        kind=ArtifactKind.DELTA,
        payload={"optimizer_step": 3},
    )

    assert Path(artifact.manifest_path).exists()
    assert Path(artifact.payload_path).exists()
    assert json.loads(Path(artifact.payload_path).read_text())["optimizer_step"] == 3

    latest = store.latest("run-1")
    assert latest.run_id == "run-1"
    assert latest.checkpoint_version == 3
    assert latest.kind == ArtifactKind.DELTA
    assert latest.manifest_path == artifact.manifest_path


def test_artifact_store_reports_missing_latest(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)

    with pytest.raises(NotFoundError):
        store.latest("missing-run")
