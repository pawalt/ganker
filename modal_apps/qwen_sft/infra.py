"""Modal infra for one Qwen SFT deployment.

This file owns only infrastructure:

- a Megatron Bridge image for the controller and trainer worker,
- an SGLang image for rollout inference,
- Modal Volumes for artifacts and Hugging Face cache,
- i6pn worker rendezvous,
- Monarch `attach_to_workers`,
- and a single remote function that runs an importable SFT job against the mesh.

Deploy:

    source ~/.codex/modal.env
    GANKER_MODAL_GPU=A100 uv run modal deploy modal_apps/qwen_sft/infra.py
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Callable, cast

import modal
from monarch.actor import endpoint

from ganker.actors import RolloutActor, TrainingActor
from ganker.contracts import (
    RefreshWeightsRequest,
    RefreshWeightsResponse,
    SampleRequest,
    SampleResponse,
    SaveWeightsRequest,
    SaveWeightsResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REMOTE_ROOT = Path("/workspace/ganker")
MODEL = os.getenv("GANKER_QWEN_SFT_MODEL", "Qwen/Qwen3-0.6B")
APP_NAME = os.getenv("GANKER_QWEN_SFT_APP", "ganker-qwen-sft")
REGION = os.getenv("GANKER_MODAL_REGION", "us-east-1")
GPU = os.getenv("GANKER_MODAL_GPU", "L40S")
BRIDGE_BASE_IMAGE = os.getenv("GANKER_MODAL_BRIDGE_BASE_IMAGE", "nvcr.io/nvidia/pytorch:26.02-py3")
BRIDGE_REPO = os.getenv(
    "GANKER_MEGATRON_BRIDGE_REPO",
    "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
)
BRIDGE_REF = os.getenv("GANKER_MEGATRON_BRIDGE_REF", "v0.4.2")
BRIDGE_UV_VERSION = os.getenv("GANKER_MEGATRON_BRIDGE_UV_VERSION", "0.7.2")
TORCHMONARCH_VERSION = os.getenv("GANKER_MODAL_TORCHMONARCH_VERSION", "0.5.0")
SGLANG_IMAGE = os.getenv("GANKER_MODAL_SGLANG_IMAGE", "lmsysorg/sglang:v0.5.12")

REGISTRY_NAME = os.getenv("GANKER_QWEN_SFT_REGISTRY", "ganker-qwen-sft-registry")
ARTIFACT_VOLUME_NAME = os.getenv("GANKER_QWEN_SFT_ARTIFACT_VOLUME", "ganker-qwen-sft-artifacts")
HF_CACHE_VOLUME_NAME = os.getenv("GANKER_HF_CACHE_VOLUME", "huggingface-cache")
ARTIFACT_ROOT = Path(os.getenv("GANKER_QWEN_SFT_ARTIFACT_ROOT", "/vol/ganker-artifacts"))
ARTIFACT_MOUNT = str(ARTIFACT_ROOT)
MONARCH_PORT = int(os.getenv("GANKER_QWEN_SFT_MONARCH_PORT", "26600"))
CONTROLLER_PORT = int(os.getenv("GANKER_QWEN_SFT_CONTROLLER_PORT", "26610"))
SGLANG_PORT = int(os.getenv("GANKER_QWEN_SFT_SGLANG_PORT", "30000"))
SGLANG_CONTEXT_LENGTH = int(os.getenv("GANKER_QWEN_SFT_SGLANG_CONTEXT_LENGTH", "2048"))
SGLANG_MEM_FRACTION_STATIC = float(os.getenv("GANKER_QWEN_SFT_SGLANG_MEM_FRACTION_STATIC", "0.75"))


def _repo_ignore() -> list[str]:
    return [
        ".git",
        ".jj",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        ".local_artifacts",
    ]


def _hf_secrets() -> list[modal.Secret]:
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        return []
    return [
        modal.Secret.from_dict(
            {
                "HF_TOKEN": hf_token,
                "HUGGING_FACE_HUB_TOKEN": hf_token,
            }
        )
    ]


def _bridge_image() -> modal.Image:
    return (
        modal.Image.from_registry(BRIDGE_BASE_IMAGE)
        .apt_install("git", "curl")
        .run_commands(
            f"curl -LsSf https://astral.sh/uv/{BRIDGE_UV_VERSION}/install.sh | sh",
            "rm -rf /opt/Megatron-Bridge /opt/venv",
            (
                "git clone --depth 1 --branch "
                f"{BRIDGE_REF} --recurse-submodules --shallow-submodules "
                f"{BRIDGE_REPO} /opt/Megatron-Bridge"
            ),
            "/root/.local/bin/uv venv /opt/venv --system-site-packages",
            (
                "cd /opt/Megatron-Bridge && "
                "UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy "
                "/root/.local/bin/uv sync --frozen --only-group build"
            ),
            (
                "cd /opt/Megatron-Bridge && "
                "UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy "
                "MAX_JOBS=4 NVTE_BUILD_NUM_PHILOX_ROUNDS=3 "
                "/root/.local/bin/uv sync --link-mode copy --frozen --no-dev "
                "--no-install-package transformer-engine"
            ),
            (
                "UV_PROJECT_ENVIRONMENT=/opt/venv "
                f"/root/.local/bin/uv pip install --python /opt/venv/bin/python "
                f"torchmonarch=={TORCHMONARCH_VERSION}"
            ),
        )
        .env(
            {
                "PATH": "/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "VIRTUAL_ENV": "/opt/venv",
                "UV_PROJECT_ENVIRONMENT": "/opt/venv",
                "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}:/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM",
                "GANKER_ARTIFACT_ROOT": ARTIFACT_MOUNT,
            }
        )
        .add_local_dir(PROJECT_ROOT, remote_path=str(REMOTE_ROOT), ignore=_repo_ignore())
    )


def _sglang_image() -> modal.Image:
    return (
        modal.Image.from_registry(SGLANG_IMAGE)
        .entrypoint([])
        .apt_install("git", "curl")
        .uv_pip_install(
            "grpcio>=1.81.1",
            "protobuf>=6.33.6",
            f"torchmonarch=={TORCHMONARCH_VERSION}",
            "typing_extensions>=4.13",
        )
        .run_commands("rm -rf /root/.cache/huggingface")
        .env(
            {
                "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}",
                "GANKER_ARTIFACT_ROOT": ARTIFACT_MOUNT,
                "HF_HUB_CACHE": "/root/.cache/huggingface",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
            }
        )
        .add_local_dir(PROJECT_ROOT, remote_path=str(REMOTE_ROOT), ignore=_repo_ignore())
    )


bridge_image = _bridge_image()
sglang_image = _sglang_image()
app = modal.App(APP_NAME)
registry = modal.Dict.from_name(REGISTRY_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


@dataclass(frozen=True)
class JobContext:
    deployment_id: str
    region: str
    controller_transport: str
    trainer_target: str
    rollout_target: str

    def base_payload(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "region": self.region,
            "controller_transport": self.controller_transport,
            "trainer_target": self.trainer_target,
            "rollout_target": self.rollout_target,
        }

    def reload_artifacts(self) -> None:
        artifact_volume.reload()


class VolumeTrainingActor(TrainingActor):
    @endpoint
    def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        response = self._component.save_weights(request)
        artifact_volume.commit()
        return response


class VolumeRolloutActor(RolloutActor):
    @endpoint
    def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        artifact_volume.reload()
        return self._component.refresh_weights(request)

    @endpoint
    def sample(self, request: SampleRequest) -> SampleResponse:
        return self._component.sample(request)


def add_remote_import_paths() -> None:
    for path in (REMOTE_ROOT, REMOTE_ROOT / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True))


def bridge_training_config(
    *,
    micro_batch_size: int,
    sequence_length: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "runtime_kind": "bridge",
        "tensor_device": "cuda",
        "micro_batch_size": micro_batch_size,
        "global_batch_size": micro_batch_size,
        "sequence_length": sequence_length,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "seed": seed,
        "trust_remote_code": True,
        "load_weights": True,
    }


def sglang_inference_config(
    *,
    lora_rank: int,
    startup_timeout: int,
    port: int = SGLANG_PORT,
    context_length: int = SGLANG_CONTEXT_LENGTH,
    mem_fraction_static: float = SGLANG_MEM_FRACTION_STATIC,
) -> dict[str, Any]:
    from ganker.config import SGLangBackendConfig

    return SGLangBackendConfig(
        model_path=MODEL,
        launch_server=True,
        host="127.0.0.1",
        port=port,
        request_timeout=180,
        startup_timeout=float(startup_timeout),
        return_logprobs=True,
        enable_lora=True,
        max_lora_rank=max(lora_rank, 1),
        extra_server_args={
            "trust-remote-code": True,
            "context-length": context_length,
            "mem-fraction-static": mem_fraction_static,
            "chunked-prefill-size": min(1024, context_length),
        },
    ).as_dict()


def _i6pn_address() -> str:
    return str(socket.getaddrinfo("i6pn.modal.local", None, socket.AF_INET6)[0][4][0])


def _endpoint_key(deployment_id: str, run_id: str, role: str, rank: int) -> str:
    add_remote_import_paths()
    from ganker.distributed.registry import RoleKey

    return RoleKey(deployment_id, run_id, role, rank).storage_key


def _publish_endpoint(
    *,
    deployment_id: str,
    run_id: str,
    role: str,
    rank: int,
    host: str,
    port: int,
) -> dict[str, Any]:
    add_remote_import_paths()
    from ganker.distributed.registry import EndpointAddress, RoleEndpoint

    endpoint = RoleEndpoint(
        deployment_id=deployment_id,
        run_id=run_id,
        role=role,
        rank=rank,
        protocol="tcp",
        addresses=(EndpointAddress(family="ipv6", host=host, port=port),),
        status="ready",
        region=REGION,
        metadata={
            "modal_app": APP_NAME,
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
    timeout_seconds: int,
) -> Any:
    add_remote_import_paths()
    from ganker.distributed.registry import RoleEndpoint

    key = _endpoint_key(deployment_id, run_id, role, 0)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = registry.get(key)
        if payload is not None and payload.get("status") == "ready":
            return RoleEndpoint.from_dict(payload)
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for {key}")


@app.function(
    image=bridge_image,
    gpu=GPU,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def trainer_worker(deployment_id: str, run_id: str, role: str, rank: int, port: int) -> dict[str, Any]:
    return _run_worker_loop(deployment_id=deployment_id, run_id=run_id, role=role, rank=rank, port=port)


@app.function(
    image=sglang_image,
    gpu=GPU,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={
        ARTIFACT_MOUNT: artifact_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=_hf_secrets(),
)
def rollout_worker(deployment_id: str, run_id: str, role: str, rank: int, port: int) -> dict[str, Any]:
    return _run_worker_loop(deployment_id=deployment_id, run_id=run_id, role=role, rank=rank, port=port)


def _run_worker_loop(
    *,
    deployment_id: str,
    run_id: str,
    role: str,
    rank: int,
    port: int,
) -> dict[str, Any]:
    add_remote_import_paths()
    host = _i6pn_address()
    address = f"tcp://[{host}]:{port}"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from monarch.actor import enable_transport, run_worker_loop_forever; "
            "enable_transport('tcp'); "
            f"run_worker_loop_forever(address={address!r}, ca='trust_all_connections')",
        ],
        env={
            **os.environ,
            "HYPERACTOR_PROCESS_NAME": f"ganker_{deployment_id}_{run_id}_{role}_{rank}",
        },
        start_new_session=True,
    )
    try:
        time.sleep(2)
        if proc.poll() is not None:
            raise RuntimeError(f"Monarch worker exited before ready: role={role} code={proc.returncode}")
        endpoint = _publish_endpoint(
            deployment_id=deployment_id,
            run_id=run_id,
            role=role,
            rank=rank,
            host=host,
            port=port,
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


def _start_runtime(
    *,
    deployment_id: str,
    run_id: str,
    artifact_root: str,
    port: int,
    controller_port: int,
    startup_timeout: int,
    training_config: dict[str, Any],
    inference_config: dict[str, Any],
) -> dict[str, Any]:
    add_remote_import_paths()
    from ganker.actors import ControllerActor, ControllerProxyActor, TelemetryActor
    from ganker.distributed.monarch import attach_role_endpoints
    from monarch.actor import enable_transport, this_host

    runtime: dict[str, Any] = {"trainer_call": None, "rollout_call": None}
    try:
        controller_transport = f"tcp://[{_i6pn_address()}]:{controller_port}"
        enable_transport(controller_transport)
        runtime["controller_transport"] = controller_transport

        runtime["trainer_call"] = trainer_worker.spawn(deployment_id, run_id, "trainer", 0, port)
        runtime["rollout_call"] = rollout_worker.spawn(deployment_id, run_id, "rollout", 0, port)

        trainer_endpoint = _wait_for_endpoint(
            deployment_id=deployment_id,
            run_id=run_id,
            role="trainer",
            timeout_seconds=startup_timeout,
        )
        rollout_endpoint = _wait_for_endpoint(
            deployment_id=deployment_id,
            run_id=run_id,
            role="rollout",
            timeout_seconds=startup_timeout,
        )
        runtime["trainer_endpoint"] = trainer_endpoint
        runtime["rollout_endpoint"] = rollout_endpoint
        runtime["trainer_hosts"] = attach_role_endpoints(
            [trainer_endpoint],
            name=f"{deployment_id}_trainer",
            family="ipv6",
            transport=None,
        )
        runtime["rollout_hosts"] = attach_role_endpoints(
            [rollout_endpoint],
            name=f"{deployment_id}_rollout",
            family="ipv6",
            transport=None,
        )
        runtime["controller_procs"] = this_host().spawn_procs(name=f"{deployment_id}_controller")
        runtime["trainer_procs"] = runtime["trainer_hosts"].spawn_procs(name=f"{deployment_id}_trainer")
        runtime["rollout_procs"] = runtime["rollout_hosts"].spawn_procs(name=f"{deployment_id}_rollout")

        training = runtime["trainer_procs"].spawn(
            "training",
            VolumeTrainingActor,
            artifact_root,
            "megatron",
            training_config,
        )
        rollout = runtime["rollout_procs"].spawn(
            "rollout",
            VolumeRolloutActor,
            artifact_root,
            "sglang",
            inference_config,
        )
        telemetry = runtime["controller_procs"].spawn("telemetry", TelemetryActor)
        controller = runtime["controller_procs"].spawn("controller", ControllerActor, training, rollout, telemetry)
        proxy = runtime["controller_procs"].spawn("proxy", ControllerProxyActor, controller)
        for actor in (training, rollout, telemetry, controller, proxy):
            actor.initialized.get(timeout=30)

        runtime.update(
            {
                "training": training,
                "rollout": rollout,
                "telemetry": telemetry,
                "controller": controller,
                "proxy": proxy,
            }
        )
        return runtime
    except Exception:
        _stop_runtime(runtime)
        raise


def _start_training_runtime(
    *,
    deployment_id: str,
    run_id: str,
    artifact_root: str,
    port: int,
    controller_port: int,
    startup_timeout: int,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    add_remote_import_paths()
    from ganker.actors import ControllerActor, ControllerProxyActor, RolloutActor, TelemetryActor
    from ganker.distributed.monarch import attach_role_endpoints
    from monarch.actor import enable_transport, this_host

    runtime: dict[str, Any] = {"trainer_call": None, "rollout_call": None}
    try:
        controller_transport = f"tcp://[{_i6pn_address()}]:{controller_port}"
        enable_transport(controller_transport)
        runtime["controller_transport"] = controller_transport

        runtime["trainer_call"] = trainer_worker.spawn(deployment_id, run_id, "trainer", 0, port)

        trainer_endpoint = _wait_for_endpoint(
            deployment_id=deployment_id,
            run_id=run_id,
            role="trainer",
            timeout_seconds=startup_timeout,
        )
        runtime["trainer_endpoint"] = trainer_endpoint
        runtime["rollout_endpoint"] = None
        runtime["trainer_hosts"] = attach_role_endpoints(
            [trainer_endpoint],
            name=f"{deployment_id}_trainer",
            family="ipv6",
            transport=None,
        )
        runtime["controller_procs"] = this_host().spawn_procs(name=f"{deployment_id}_controller")
        runtime["trainer_procs"] = runtime["trainer_hosts"].spawn_procs(name=f"{deployment_id}_trainer")
        runtime["rollout_procs"] = None
        runtime["rollout_hosts"] = None

        training = runtime["trainer_procs"].spawn(
            "training",
            VolumeTrainingActor,
            artifact_root,
            "megatron",
            training_config,
        )
        rollout = runtime["controller_procs"].spawn("rollout", RolloutActor, artifact_root, "fake", {})
        telemetry = runtime["controller_procs"].spawn("telemetry", TelemetryActor)
        controller = runtime["controller_procs"].spawn("controller", ControllerActor, training, rollout, telemetry)
        proxy = runtime["controller_procs"].spawn("proxy", ControllerProxyActor, controller)
        for actor in (training, rollout, telemetry, controller, proxy):
            actor.initialized.get(timeout=30)

        runtime.update(
            {
                "training": training,
                "rollout": rollout,
                "telemetry": telemetry,
                "controller": controller,
                "proxy": proxy,
            }
        )
        return runtime
    except Exception:
        _stop_runtime(runtime)
        raise


def _stop_runtime(runtime: dict[str, Any] | None) -> None:
    if runtime is None:
        return
    for mesh in (runtime.get("trainer_procs"), runtime.get("rollout_procs"), runtime.get("controller_procs")):
        if mesh is not None:
            try:
                mesh.stop().get(timeout=10)
            except Exception as exc:
                print(f"[cleanup] proc mesh stop failed: {exc}", flush=True)
    for mesh in (runtime.get("trainer_hosts"), runtime.get("rollout_hosts")):
        if mesh is not None:
            try:
                mesh.stop().get(timeout=10)
            except Exception as exc:
                print(f"[cleanup] host mesh stop failed: {exc}", flush=True)
    for call in (runtime.get("trainer_call"), runtime.get("rollout_call")):
        if call is not None:
            call.cancel(terminate_containers=True)


def _context(deployment_id: str, runtime: dict[str, Any]) -> JobContext:
    rollout_endpoint = runtime.get("rollout_endpoint")
    return JobContext(
        deployment_id=deployment_id,
        region=REGION,
        controller_transport=runtime["controller_transport"],
        trainer_target=runtime["trainer_endpoint"].target(family="ipv6"),
        rollout_target=(
            rollout_endpoint.target(family="ipv6")
            if rollout_endpoint is not None
            else "controller-local-fake-rollout"
        ),
    )


def _load_job(job_module: str, job_function: str) -> Callable[..., dict[str, Any]]:
    add_remote_import_paths()
    module = importlib.import_module(job_module)
    job = getattr(module, job_function)
    if not callable(job):
        raise TypeError(f"{job_module}.{job_function} is not callable")
    return cast(Callable[..., dict[str, Any]], job)


@app.function(
    image=bridge_image,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_sft_job(
    deployment_id: str,
    run_id: str,
    artifact_root: str,
    port: int,
    controller_port: int,
    startup_timeout: int,
    sglang_startup_timeout: int,
    job_module: str,
    job_function: str,
    job_config: dict[str, Any],
) -> dict[str, Any]:
    add_remote_import_paths()
    from ganker.client import ServiceClient
    from ganker.transport import MonarchProxyTransport

    runtime = None
    client = None
    try:
        runtime = _start_runtime(
            deployment_id=deployment_id,
            run_id=run_id,
            artifact_root=artifact_root,
            port=port,
            controller_port=controller_port,
            startup_timeout=startup_timeout,
            training_config=bridge_training_config(
                micro_batch_size=int(job_config["micro_batch_size"]),
                sequence_length=int(job_config["sequence_length"]),
                seed=int(job_config["seed"]),
            ),
            inference_config=sglang_inference_config(
                lora_rank=int(job_config["lora_rank"]),
                startup_timeout=sglang_startup_timeout,
                port=int(job_config["sglang_port"]),
                context_length=int(job_config["sglang_context_length"]),
                mem_fraction_static=float(job_config["sglang_mem_fraction_static"]),
            ),
        )
        timeout = max(float(startup_timeout), float(sglang_startup_timeout)) + 300
        client = ServiceClient(_transport=MonarchProxyTransport(runtime["proxy"], timeout=timeout))
        job = _load_job(job_module, job_function)
        return json_safe(job(client, _context(deployment_id, runtime), dict(job_config)))
    finally:
        if client is not None:
            client.close()
        _stop_runtime(runtime)


@app.function(
    image=bridge_image,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={ARTIFACT_MOUNT: artifact_volume},
)
def run_training_job(
    deployment_id: str,
    run_id: str,
    artifact_root: str,
    port: int,
    controller_port: int,
    startup_timeout: int,
    job_module: str,
    job_function: str,
    job_config: dict[str, Any],
) -> dict[str, Any]:
    """Run a training-only job against the Bridge trainer mesh.

    The controller still gets a fake rollout actor so the public proxy contract
    remains complete, but no SGLang worker or rollout GPU is started.
    """

    add_remote_import_paths()
    from ganker.client import ServiceClient
    from ganker.transport import MonarchProxyTransport

    runtime = None
    client = None
    try:
        runtime = _start_training_runtime(
            deployment_id=deployment_id,
            run_id=run_id,
            artifact_root=artifact_root,
            port=port,
            controller_port=controller_port,
            startup_timeout=startup_timeout,
            training_config=bridge_training_config(
                micro_batch_size=int(job_config["micro_batch_size"]),
                sequence_length=int(job_config["sequence_length"]),
                seed=int(job_config["seed"]),
            ),
        )
        client = ServiceClient(
            _transport=MonarchProxyTransport(runtime["proxy"], timeout=float(startup_timeout) + 300)
        )
        job = _load_job(job_module, job_function)
        return json_safe(job(client, _context(deployment_id, runtime), dict(job_config)))
    finally:
        if client is not None:
            client.close()
        _stop_runtime(runtime)
