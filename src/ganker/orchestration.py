"""Local Monarch orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from monarch.actor import enable_transport, this_host

from ganker.actors import ProxyActor, RolloutActor, TelemetryActor, TrainingActor


@dataclass
class LocalMonarchMesh:
    """Handles for a locally spawned Monarch actor mesh."""

    procs: Any
    training: Any
    rollout: Any
    telemetry: Any
    proxy: Any

    def stop(self) -> None:
        for actor in (self.proxy, self.training, self.rollout, self.telemetry):
            actor.stop().get(timeout=10)
        self.procs.stop().get(timeout=10)


def start_local_monarch_mesh(
    artifact_root: Path,
    *,
    transport: str = "tcp",
) -> LocalMonarchMesh:
    """Spawn the singleton actors on a local Monarch process mesh.

    `transport="tcp"` keeps local behavior aligned with the intended IPv4 mesh.
    """

    enable_transport(transport)
    host = this_host()
    procs = host.spawn_procs(name="ganker-local")
    root = str(Path(artifact_root))

    training = procs.spawn("training", TrainingActor, root)
    rollout = procs.spawn("rollout", RolloutActor, root)
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
