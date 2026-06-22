"""Filesystem-backed weight artifact store.

This emulates the future Modal Volume path while staying local and tiny.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ganker.contracts import ArtifactKind, WeightArtifact
from ganker.errors import InvalidRequestError, NotFoundError


class FilesystemArtifactStore:
    """Stores versioned checkpoint manifests under a local directory."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def write(
        self,
        *,
        run_id: str,
        checkpoint_version: int,
        kind: ArtifactKind,
        payload: Dict[str, Any],
    ) -> WeightArtifact:
        if not run_id:
            raise InvalidRequestError("run_id is required")
        if checkpoint_version < 0:
            raise InvalidRequestError("checkpoint_version must be non-negative")
        if kind not in (ArtifactKind.FULL, ArtifactKind.DELTA):
            raise InvalidRequestError(f"unsupported artifact kind: {kind}")

        run_dir = self.root / "weights" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        payload_path = run_dir / f"checkpoint-{checkpoint_version}.payload.json"
        manifest_path = run_dir / f"checkpoint-{checkpoint_version}.manifest.json"
        payload_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

        artifact = WeightArtifact(
            run_id=run_id,
            checkpoint_version=checkpoint_version,
            kind=kind,
            manifest_path=str(manifest_path),
            payload_path=str(payload_path),
        )
        manifest_path.write_text(
            json.dumps(self._artifact_to_dict(artifact), sort_keys=True, indent=2),
            encoding="utf-8",
        )
        (run_dir / "latest.json").write_text(
            json.dumps({"manifest_path": str(manifest_path)}, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        return artifact

    def latest(self, run_id: str) -> WeightArtifact:
        if not run_id:
            raise InvalidRequestError("run_id is required")

        latest_path = self.root / "weights" / run_id / "latest.json"
        if not latest_path.exists():
            raise NotFoundError(f"no artifact exists for run_id={run_id}")

        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        manifest_path = Path(latest["manifest_path"])
        if not manifest_path.exists():
            raise NotFoundError(f"artifact manifest is missing: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return self._artifact_from_dict(manifest)

    def _artifact_to_dict(self, artifact: WeightArtifact) -> Dict[str, Any]:
        return {
            "run_id": artifact.run_id,
            "checkpoint_version": artifact.checkpoint_version,
            "kind": artifact.kind.name,
            "manifest_path": artifact.manifest_path,
            "payload_path": artifact.payload_path,
        }

    def _artifact_from_dict(self, data: Dict[str, Any]) -> WeightArtifact:
        return WeightArtifact(
            run_id=str(data["run_id"]),
            checkpoint_version=int(data["checkpoint_version"]),
            kind=ArtifactKind[str(data["kind"])],
            manifest_path=str(data["manifest_path"]),
            payload_path=str(data["payload_path"]),
        )
