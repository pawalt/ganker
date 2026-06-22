"""Local Monarch orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monarch.actor import enable_transport, this_host

from ganker.actors import ProxyActor, RolloutActor, TelemetryActor, TrainingActor
from ganker.config import MegatronBackendConfig


@dataclass
class LocalMonarchMesh:
    """Handles for a locally spawned Monarch actor mesh."""

    procs: Any
    training: Any
    rollout: Any
    telemetry: Any
    proxy: Any

    def stop(self) -> None:
        for actor in (self.training, self.rollout, self.telemetry, self.proxy):
            actor.shutdown.choose().get(timeout=10)
        for actor in (self.proxy, self.training, self.rollout, self.telemetry):
            actor.stop().get(timeout=10)
        self.procs.stop().get(timeout=10)


def start_local_monarch_mesh(
    artifact_root: Path,
    *,
    transport: str = "tcp",
    training_backend: str = "fake",
    inference_backend: str = "fake",
    training_backend_config: dict[str, Any] | MegatronBackendConfig | None = None,
    inference_backend_config: dict[str, Any] | None = None,
) -> LocalMonarchMesh:
    """Spawn the singleton actors on a local Monarch process mesh.

    `transport="tcp"` keeps local behavior aligned with the intended IPv4 mesh.
    """

    enable_transport(transport)
    host = this_host()
    procs = host.spawn_procs(name="ganker-local")
    root = str(Path(artifact_root))
    if isinstance(training_backend_config, MegatronBackendConfig):
        training_config = training_backend_config.as_dict()
    else:
        training_config = training_backend_config

    training = procs.spawn("training", TrainingActor, root, training_backend, training_config)
    rollout = procs.spawn("rollout", RolloutActor, root, inference_backend, inference_backend_config)
    telemetry = procs.spawn("telemetry", TelemetryActor)
    proxy = procs.spawn("proxy", ProxyActor, training, rollout, telemetry)

    for actor in (training, rollout, telemetry, proxy):
        actor.initialized.get(timeout=10)

    return LocalMonarchMesh(
        procs=procs,
        training=training,
        rollout=rollout,
        telemetry=telemetry,
        proxy=proxy,
    )
