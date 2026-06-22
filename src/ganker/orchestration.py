"""Local Monarch orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from typing import Any

from monarch.actor import enable_transport, this_host

from ganker.actors import (
    ControllerActor,
    ControllerProxyActor,
    ProxyActor,
    RolloutActor,
    TelemetryActor,
    TrainingActor,
)
from ganker.config import MegatronBackendConfig, SGLangBackendConfig
from ganker.distributed.monarch import attach_role_from_registry
from ganker.distributed.registry import EndpointAddress, InMemoryRunRegistry, RoleEndpoint


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


@dataclass
class LocalDistributedMonarchMesh:
    """Handles for a local attach-to-workers distributed topology."""

    registry: InMemoryRunRegistry
    worker_processes: list[subprocess.Popen]
    worker_tmpdir: str
    trainer_hosts: Any
    rollout_hosts: Any
    controller_procs: Any
    trainer_procs: Any
    rollout_procs: Any
    training: Any
    rollout: Any
    telemetry: Any
    controller: Any
    proxy: Any

    def stop(self) -> None:
        actors = (self.proxy, self.controller, self.training, self.rollout, self.telemetry)
        for actor in actors:
            try:
                actor.shutdown.choose().get(timeout=10)
            except Exception:
                pass
        for actor in actors:
            try:
                actor.stop().get(timeout=10)
            except Exception:
                pass
        for procs in (self.trainer_procs, self.rollout_procs, self.controller_procs):
            try:
                procs.stop().get(timeout=10)
            except Exception:
                pass
        for hosts in (self.trainer_hosts, self.rollout_hosts):
            try:
                hosts.stop().get(timeout=10)
            except Exception:
                pass
        for proc in self.worker_processes:
            if proc.poll() is not None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
        for proc in self.worker_processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                proc.wait(timeout=10)
        shutil.rmtree(self.worker_tmpdir, ignore_errors=True)


def start_local_monarch_mesh(
    artifact_root: Path,
    *,
    transport: str = "tcp",
    training_backend: str = "fake",
    inference_backend: str = "fake",
    training_backend_config: dict[str, Any] | MegatronBackendConfig | None = None,
    inference_backend_config: dict[str, Any] | SGLangBackendConfig | None = None,
) -> LocalMonarchMesh:
    """Spawn the singleton actors on a local Monarch process mesh.

    `transport="tcp"` keeps local behavior aligned with the intended IPv4 mesh.
    """

    enable_transport(transport)
    host = this_host()
    procs = host.spawn_procs(name="ganker_local")
    root = str(Path(artifact_root))
    if isinstance(training_backend_config, MegatronBackendConfig):
        training_config = training_backend_config.as_dict()
    else:
        training_config = training_backend_config
    if isinstance(inference_backend_config, SGLangBackendConfig):
        inference_config = inference_backend_config.as_dict()
    else:
        inference_config = inference_backend_config

    training = procs.spawn("training", TrainingActor, root, training_backend, training_config)
    rollout = procs.spawn("rollout", RolloutActor, root, inference_backend, inference_config)
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


def start_local_distributed_monarch_mesh(
    artifact_root: Path,
    *,
    deployment_id: str = "local",
    run_id: str = "run-000001",
    training_backend: str = "fake",
    inference_backend: str = "fake",
    training_backend_config: dict[str, Any] | MegatronBackendConfig | None = None,
    inference_backend_config: dict[str, Any] | SGLangBackendConfig | None = None,
) -> LocalDistributedMonarchMesh:
    """Spawn a CPU-only distributed topology through `attach_to_workers`.

    This mirrors the Modal design without requiring Modal: trainer and rollout
    worker listeners live in separate local subprocesses, publish endpoint
    metadata to a registry, and the controller attaches to those workers before
    spawning actors.
    """

    enable_transport("tcp")
    worker_tmpdir = tempfile.mkdtemp(prefix="ganker_workers_")
    registry = InMemoryRunRegistry()
    worker_processes: list[subprocess.Popen] = []

    try:
        for role in ("trainer", "rollout"):
            port = _reserve_tcp_port()
            address = f"tcp://127.0.0.1:{port}"
            worker_processes.append(_start_local_worker(address, f"ganker_{role}_0"))
            registry.put(
                RoleEndpoint(
                    deployment_id=deployment_id,
                    run_id=run_id,
                    role=role,
                    rank=0,
                    protocol="tcp",
                    addresses=(EndpointAddress(family="ipv4", host="127.0.0.1", port=port),),
                    status="ready",
                    region="local",
                )
            )

        trainer_hosts = attach_role_from_registry(
            registry,
            deployment_id=deployment_id,
            run_id=run_id,
            role="trainer",
            name="ganker_trainer_hosts",
            family="ipv4",
            transport="tcp",
        )
        rollout_hosts = attach_role_from_registry(
            registry,
            deployment_id=deployment_id,
            run_id=run_id,
            role="rollout",
            name="ganker_rollout_hosts",
            family="ipv4",
            transport="tcp",
        )
        controller_hosts = this_host()
        trainer_procs = trainer_hosts.spawn_procs(name="ganker_trainer")
        rollout_procs = rollout_hosts.spawn_procs(name="ganker_rollout")
        controller_procs = controller_hosts.spawn_procs(name="ganker_controller")

        root = str(Path(artifact_root))
        if isinstance(training_backend_config, MegatronBackendConfig):
            training_config = training_backend_config.as_dict()
        else:
            training_config = training_backend_config
        if isinstance(inference_backend_config, SGLangBackendConfig):
            inference_config = inference_backend_config.as_dict()
        else:
            inference_config = inference_backend_config

        training = trainer_procs.spawn(
            "training",
            TrainingActor,
            root,
            training_backend,
            training_config,
        )
        rollout = rollout_procs.spawn(
            "rollout",
            RolloutActor,
            root,
            inference_backend,
            inference_config,
        )
        telemetry = controller_procs.spawn("telemetry", TelemetryActor)
        controller = controller_procs.spawn("controller", ControllerActor, training, rollout, telemetry)
        proxy = controller_procs.spawn("proxy", ControllerProxyActor, controller)

        for actor in (training, rollout, telemetry, controller, proxy):
            actor.initialized.get(timeout=10)

        return LocalDistributedMonarchMesh(
            registry=registry,
            worker_processes=worker_processes,
            worker_tmpdir=worker_tmpdir,
            trainer_hosts=trainer_hosts,
            rollout_hosts=rollout_hosts,
            controller_procs=controller_procs,
            trainer_procs=trainer_procs,
            rollout_procs=rollout_procs,
            training=training,
            rollout=rollout,
            telemetry=telemetry,
            controller=controller,
            proxy=proxy,
        )
    except Exception:
        for proc in worker_processes:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    pass
        shutil.rmtree(worker_tmpdir, ignore_errors=True)
        raise


def _start_local_worker(address: str, process_name: str) -> subprocess.Popen:
    env = {**os.environ, "HYPERACTOR_PROCESS_NAME": process_name}
    cmd = [
        sys.executable,
        "-c",
        "from monarch.actor import run_worker_loop_forever; "
        f'run_worker_loop_forever(address="{address}", ca="trust_all_connections")',
    ]
    return subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _reserve_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
