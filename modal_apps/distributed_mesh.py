"""Modal smoke for Monarch attach-to-workers over i6pn.

Usage:

    source ~/.codex/modal.env
    modal run modal_apps/distributed_mesh.py --mode fake-distributed
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any
import uuid

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/ganker")
PYTHON_VERSION = os.getenv("GANKER_MODAL_PYTHON", "3.12")
REGION = os.getenv("GANKER_MODAL_REGION", "us-east-1")
REGISTRY_NAME = os.getenv("GANKER_DISTRIBUTED_REGISTRY", "ganker-distributed-registry")
MONARCH_PORT = int(os.getenv("GANKER_DISTRIBUTED_MONARCH_PORT", "26600"))
CONTROLLER_PORT = int(os.getenv("GANKER_DISTRIBUTED_CONTROLLER_PORT", "26610"))
TCP_SMOKE_PORT = int(os.getenv("GANKER_DISTRIBUTED_TCP_SMOKE_PORT", "26620"))


def _base_image():
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .apt_install("git", "curl")
        .uv_pip_install(
            "torch<3",
            "torchmonarch>=0.5.0",
        )
        .env(
            {
                "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}",
                "GANKER_ARTIFACT_ROOT": "/tmp/ganker-artifacts",
            }
        )
        .add_local_dir(
            PROJECT_ROOT,
            remote_path=str(REMOTE_ROOT),
            ignore=[
                ".git",
                ".jj",
                ".venv",
                ".pytest_cache",
                ".ruff_cache",
                "__pycache__",
                ".local_artifacts",
            ],
        )
    )


image = _base_image()
app = modal.App("ganker-distributed-mesh")
registry = modal.Dict.from_name(REGISTRY_NAME, create_if_missing=True)


def _add_remote_import_paths() -> None:
    for path in (REMOTE_ROOT, REMOTE_ROOT / "tests", REMOTE_ROOT / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True))


def _i6pn_address() -> str:
    return str(socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0])


def _endpoint_key(deployment_id: str, run_id: str, role: str, rank: int) -> str:
    _add_remote_import_paths()
    from ganker.distributed.registry import RoleKey

    return RoleKey(deployment_id, run_id, role, rank).storage_key


def _publish_worker_endpoint(
    *,
    deployment_id: str,
    run_id: str,
    role: str,
    rank: int,
    host: str,
    port: int,
    status: str,
) -> dict[str, Any]:
    _add_remote_import_paths()
    from ganker.distributed.registry import EndpointAddress, RoleEndpoint

    endpoint = RoleEndpoint(
        deployment_id=deployment_id,
        run_id=run_id,
        role=role,
        rank=rank,
        protocol="tcp",
        addresses=(EndpointAddress(family="ipv6", host=host, port=port),),
        status=status,
        region=REGION,
        metadata={
            "modal_app": "ganker-distributed-mesh",
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
        },
    )
    registry.put(endpoint.storage_key, endpoint.as_dict())
    return endpoint.as_dict()


def _wait_for_endpoint(
    *,
    deployment_id: str,
    run_id: str,
    role: str,
    rank: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    key = _endpoint_key(deployment_id, run_id, role, rank)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = registry.get(key)
        if payload is not None and payload.get("status") == "ready":
            return payload
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {key}")


@app.function(image=image, timeout=60 * 60, i6pn=True, region=REGION)
def monarch_worker_role(
    deployment_id: str,
    run_id: str,
    role: str,
    rank: int,
    port: int,
) -> dict[str, Any]:
    _add_remote_import_paths()
    host = _i6pn_address()
    address = f"tcp://[{host}]:{port}"
    env = {
        **os.environ,
        "HYPERACTOR_PROCESS_NAME": f"ganker_{deployment_id}_{run_id}_{role}_{rank}",
    }
    cmd = [
        sys.executable,
        "-c",
        "from monarch.actor import enable_transport, run_worker_loop_forever; "
        "enable_transport('tcp'); "
        f"run_worker_loop_forever(address={address!r}, ca='trust_all_connections')",
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        start_new_session=True,
    )
    try:
        time.sleep(2)
        if proc.poll() is not None:
            raise RuntimeError(
                f"Monarch worker exited before ready: role={role} code={proc.returncode}"
            )
        endpoint = _publish_worker_endpoint(
            deployment_id=deployment_id,
            run_id=run_id,
            role=role,
            rank=rank,
            host=host,
            port=port,
            status="ready",
        )
        print(json.dumps({"worker_ready": endpoint}, sort_keys=True), flush=True)
        proc.wait()
        return {"ok": True, "role": role, "rank": rank, "exit_code": proc.returncode}
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                proc.wait(timeout=10)


@app.function(image=image, timeout=60 * 60, i6pn=True, region=REGION)
def run_fake_distributed(
    deployment_id: str,
    run_id: str,
    artifact_root: str,
    port: int,
    controller_port: int,
    startup_timeout: int,
) -> dict[str, Any]:
    _add_remote_import_paths()

    from ganker.actors import ControllerActor, ControllerProxyActor, RolloutActor, TelemetryActor, TrainingActor
    from ganker.contracts import (
        AdamParams,
        CreateTrainingRunRequest,
        Datum,
        ForwardBackwardRequest,
        ModelInput,
        OptimStepRequest,
        RequestContext,
        TensorData,
        TuningMode,
    )
    from ganker.distributed.monarch import attach_role_endpoints
    from ganker.distributed.registry import RoleEndpoint
    from monarch.actor import enable_transport, this_host

    trainer_call = None
    rollout_call = None
    trainer_hosts = None
    rollout_hosts = None
    controller_procs = None
    trainer_procs = None
    rollout_procs = None
    try:
        controller_host = _i6pn_address()
        controller_transport = f"tcp://[{controller_host}]:{controller_port}"
        enable_transport(controller_transport)

        trainer_call = monarch_worker_role.spawn(deployment_id, run_id, "trainer", 0, port)
        rollout_call = monarch_worker_role.spawn(deployment_id, run_id, "rollout", 0, port)

        trainer_endpoint = RoleEndpoint.from_dict(
            _wait_for_endpoint(
                deployment_id=deployment_id,
                run_id=run_id,
                role="trainer",
                rank=0,
                timeout_seconds=startup_timeout,
            )
        )
        rollout_endpoint = RoleEndpoint.from_dict(
            _wait_for_endpoint(
                deployment_id=deployment_id,
                run_id=run_id,
                role="rollout",
                rank=0,
                timeout_seconds=startup_timeout,
            )
        )

        trainer_hosts = attach_role_endpoints(
            [trainer_endpoint],
            name=f"{deployment_id}_trainer",
            family="ipv6",
            transport=None,
        )
        rollout_hosts = attach_role_endpoints(
            [rollout_endpoint],
            name=f"{deployment_id}_rollout",
            family="ipv6",
            transport=None,
        )
        controller_procs = this_host().spawn_procs(name=f"{deployment_id}_controller")
        trainer_procs = trainer_hosts.spawn_procs(name=f"{deployment_id}_trainer")
        rollout_procs = rollout_hosts.spawn_procs(name=f"{deployment_id}_rollout")

        training = trainer_procs.spawn("training", TrainingActor, artifact_root, "fake", None)
        rollout = rollout_procs.spawn("rollout", RolloutActor, artifact_root, "fake", None)
        telemetry = controller_procs.spawn("telemetry", TelemetryActor)
        controller = controller_procs.spawn("controller", ControllerActor, training, rollout, telemetry)
        proxy = controller_procs.spawn("proxy", ControllerProxyActor, controller)
        for actor in (training, rollout, telemetry, controller, proxy):
            actor.initialized.get(timeout=30)

        created = proxy.create_training_run.choose(
            CreateTrainingRunRequest(
                context=RequestContext("modal-distributed-create"),
                base_model="Qwen/Qwen3-0.6B",
                tuning_mode=TuningMode.LORA,
                lora_rank=8,
            )
        ).get(timeout=30)
        fb = proxy.forward_backward.choose(
            ForwardBackwardRequest(
                context=RequestContext("modal-distributed-fb"),
                run_id=created.run.run_id,
                data=[
                    Datum(
                        model_input=ModelInput.from_ints([1, 2, 3, 4]),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_ints([2, 3, 4, 0]),
                            "weights": TensorData.from_floats([1.0, 1.0, 1.0, 1.0]),
                        },
                    )
                ],
            )
        ).get(timeout=30)
        step = proxy.optim_step.choose(
            OptimStepRequest(
                context=RequestContext("modal-distributed-step"),
                run_id=created.run.run_id,
                optimizer=AdamParams(learning_rate=1e-4),
            )
        ).get(timeout=30)

        return _json_safe(
            {
                "ok": True,
                "mode": "fake-distributed",
                "deployment_id": deployment_id,
                "run_id": created.run.run_id,
                "region": REGION,
                "controller_transport": controller_transport,
                "trainer_target": trainer_endpoint.target(family="ipv6"),
                "rollout_target": rollout_endpoint.target(family="ipv6"),
                "loss": fb.loss,
                "input_tokens": fb.usage.input_tokens,
                "optimizer_step": step.optimizer_step,
            }
        )
    finally:
        for mesh in (trainer_procs, rollout_procs, controller_procs):
            if mesh is not None:
                try:
                    mesh.stop().get(timeout=10)
                except Exception as exc:
                    print(f"[cleanup] proc mesh stop failed: {exc}", flush=True)
        for mesh in (trainer_hosts, rollout_hosts):
            if mesh is not None:
                try:
                    mesh.stop().get(timeout=10)
                except Exception as exc:
                    print(f"[cleanup] host mesh stop failed: {exc}", flush=True)
        if trainer_call is not None:
            trainer_call.cancel(terminate_containers=True)
        if rollout_call is not None:
            rollout_call.cancel(terminate_containers=True)


@app.function(image=image, timeout=10 * 60, i6pn=True, region=REGION)
def tcp_echo_role(
    deployment_id: str,
    run_id: str,
    port: int,
) -> dict[str, Any]:
    host = _i6pn_address()
    endpoint = _publish_worker_endpoint(
        deployment_id=deployment_id,
        run_id=run_id,
        role="tcp-echo",
        rank=0,
        host=host,
        port=port,
        status="ready",
    )
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        print(json.dumps({"tcp_echo_ready": endpoint}, sort_keys=True), flush=True)
        conn, peer = server.accept()
        with conn:
            payload = conn.recv(1024)
            conn.sendall(b"pong:" + payload)
        return {"ok": True, "endpoint": endpoint, "peer": repr(peer)}


@app.function(image=image, timeout=10 * 60, i6pn=True, region=REGION)
def run_tcp_smoke(
    deployment_id: str,
    run_id: str,
    port: int,
    startup_timeout: int,
) -> dict[str, Any]:
    call = None
    try:
        call = tcp_echo_role.spawn(deployment_id, run_id, port)
        endpoint = _wait_for_endpoint(
            deployment_id=deployment_id,
            run_id=run_id,
            role="tcp-echo",
            rank=0,
            timeout_seconds=startup_timeout,
        )
        host = endpoint["addresses"][0]["host"]
        with socket.create_connection((host, port), timeout=15) as client:
            client.sendall(b"ping")
            response = client.recv(1024)
        role_result = call.get(timeout=30)
        return _json_safe(
            {
                "ok": response == b"pong:ping",
                "mode": "tcp-smoke",
                "region": REGION,
                "target": f"tcp://[{host}]:{port}",
                "response": response.decode("utf-8"),
                "role_result": role_result,
            }
        )
    finally:
        if call is not None:
            call.cancel(terminate_containers=True)


@app.local_entrypoint()
def main(
    mode: str = "fake-distributed",
    artifact_root: str = "/tmp/ganker-distributed-artifacts",
    port: int = MONARCH_PORT,
    controller_port: int = CONTROLLER_PORT,
    startup_timeout: int = 120,
    deployment_id: str = "",
    run_id: str = "run-000001",
):
    if mode not in {"fake-distributed", "tcp-smoke"}:
        raise ValueError(f"unknown mode: {mode}")
    deployment = deployment_id or f"dev-{uuid.uuid4().hex[:8]}"
    if mode == "tcp-smoke":
        result = run_tcp_smoke.remote(deployment, run_id, port or TCP_SMOKE_PORT, startup_timeout)
    else:
        result = run_fake_distributed.remote(
            deployment,
            run_id,
            artifact_root,
            port,
            controller_port,
            startup_timeout,
        )
    print(result)
