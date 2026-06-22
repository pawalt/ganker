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
from typing import Any, Literal, cast
import uuid

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/ganker")
PYTHON_VERSION = os.getenv("GANKER_MODAL_PYTHON", "3.12")
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
REGION = os.getenv("GANKER_MODAL_REGION", "us-east-1")
REGISTRY_NAME = os.getenv("GANKER_DISTRIBUTED_REGISTRY", "ganker-distributed-registry")
ARTIFACT_VOLUME_NAME = os.getenv("GANKER_DISTRIBUTED_ARTIFACT_VOLUME", "ganker-distributed-artifacts")
ARTIFACT_VOLUME_ROOT = Path(os.getenv("GANKER_DISTRIBUTED_ARTIFACT_ROOT", "/vol/ganker-artifacts"))
ARTIFACT_VOLUME_MOUNT = str(ARTIFACT_VOLUME_ROOT)
MONARCH_PORT = int(os.getenv("GANKER_DISTRIBUTED_MONARCH_PORT", "26600"))
CONTROLLER_PORT = int(os.getenv("GANKER_DISTRIBUTED_CONTROLLER_PORT", "26610"))
TCP_SMOKE_PORT = int(os.getenv("GANKER_DISTRIBUTED_TCP_SMOKE_PORT", "26620"))
SGLANG_PORT = int(os.getenv("GANKER_DISTRIBUTED_SGLANG_PORT", "30000"))
SGLANG_CONTEXT_LENGTH = int(os.getenv("GANKER_DISTRIBUTED_SGLANG_CONTEXT_LENGTH", "2048"))
SGLANG_MEM_FRACTION_STATIC = float(
    os.getenv("GANKER_DISTRIBUTED_SGLANG_MEM_FRACTION_STATIC", "0.75")
)


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
                "GANKER_ARTIFACT_ROOT": ARTIFACT_VOLUME_MOUNT,
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


def _bridge_image():
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
                "GANKER_ARTIFACT_ROOT": ARTIFACT_VOLUME_MOUNT,
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
                "GANKER_ARTIFACT_ROOT": ARTIFACT_VOLUME_MOUNT,
                "HF_HUB_CACHE": "/root/.cache/huggingface",
                "HF_XET_HIGH_PERFORMANCE": "1",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
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
bridge_image = _bridge_image()
sglang_image = _sglang_image()
app = modal.App("ganker-distributed-mesh")
registry = modal.Dict.from_name(REGISTRY_NAME, create_if_missing=True)
artifact_volume = modal.Volume.from_name(ARTIFACT_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


class ModalVolumeTrainingActor(TrainingActor):
    """Training actor that commits Modal Volume writes after saving weights."""

    @endpoint
    def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        response = self._component.save_weights(request)
        artifact_volume.commit()
        return response


class ModalVolumeRolloutActor(RolloutActor):
    """Rollout actor that reloads Modal Volume state before loading artifacts."""

    @endpoint
    def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        artifact_volume.reload()
        return self._component.refresh_weights(request)

    @endpoint
    def sample(self, request: SampleRequest) -> SampleResponse:
        return self._component.sample(request)


def _add_remote_import_paths() -> None:
    for path in (REMOTE_ROOT, REMOTE_ROOT / "tests", REMOTE_ROOT / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True))


def _sglang_backend_config(
    *,
    model: str,
    port: int,
    startup_timeout: int,
    context_length: int,
    mem_fraction_static: float,
    max_lora_rank: int,
) -> dict[str, Any]:
    from ganker.config import SGLangBackendConfig

    return SGLangBackendConfig(
        model_path=model,
        launch_server=True,
        host="127.0.0.1",
        port=port,
        request_timeout=180,
        startup_timeout=float(startup_timeout),
        return_logprobs=True,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
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


@app.function(
    image=image,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def monarch_worker_role(
    deployment_id: str,
    run_id: str,
    role: str,
    rank: int,
    port: int,
) -> dict[str, Any]:
    return _run_monarch_worker_loop(
        deployment_id=deployment_id,
        run_id=run_id,
        role=role,
        rank=rank,
        port=port,
    )


@app.function(
    image=bridge_image,
    gpu=GPU,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def monarch_bridge_trainer_worker_role(
    deployment_id: str,
    run_id: str,
    role: str,
    rank: int,
    port: int,
) -> dict[str, Any]:
    return _run_monarch_worker_loop(
        deployment_id=deployment_id,
        run_id=run_id,
        role=role,
        rank=rank,
        port=port,
    )


@app.function(
    image=sglang_image,
    gpu=GPU,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={
        ARTIFACT_VOLUME_MOUNT: artifact_volume,
        "/root/.cache/huggingface": hf_cache_volume,
    },
    secrets=_hf_secrets(),
)
def monarch_sglang_rollout_worker_role(
    deployment_id: str,
    run_id: str,
    role: str,
    rank: int,
    port: int,
) -> dict[str, Any]:
    return _run_monarch_worker_loop(
        deployment_id=deployment_id,
        run_id=run_id,
        role=role,
        rank=rank,
        port=port,
    )


def _run_monarch_worker_loop(
    *,
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


def _start_distributed_runtime(
    *,
    deployment_id: str,
    run_id: str,
    artifact_root: str,
    port: int,
    controller_port: int,
    startup_timeout: int,
    training_backend: str = "fake",
    training_backend_config: dict[str, Any] | None = None,
    inference_backend: str = "fake",
    inference_backend_config: dict[str, Any] | None = None,
    trainer_worker: Any = monarch_worker_role,
    rollout_worker: Any = monarch_worker_role,
) -> dict[str, Any]:
    _add_remote_import_paths()

    from ganker.actors import ControllerActor, ControllerProxyActor, TelemetryActor
    from ganker.distributed.monarch import attach_role_endpoints
    from ganker.distributed.registry import RoleEndpoint
    from monarch.actor import enable_transport, this_host

    runtime: dict[str, Any] = {
        "trainer_call": None,
        "rollout_call": None,
        "trainer_hosts": None,
        "rollout_hosts": None,
        "controller_procs": None,
        "trainer_procs": None,
        "rollout_procs": None,
    }
    try:
        controller_host = _i6pn_address()
        controller_transport = f"tcp://[{controller_host}]:{controller_port}"
        enable_transport(controller_transport)
        runtime["controller_transport"] = controller_transport

        runtime["trainer_call"] = trainer_worker.spawn(deployment_id, run_id, "trainer", 0, port)
        runtime["rollout_call"] = rollout_worker.spawn(deployment_id, run_id, "rollout", 0, port)

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
            ModalVolumeTrainingActor,
            artifact_root,
            training_backend,
            training_backend_config,
        )
        rollout = runtime["rollout_procs"].spawn(
            "rollout",
            ModalVolumeRolloutActor,
            artifact_root,
            inference_backend,
            inference_backend_config,
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
        _stop_distributed_runtime(runtime)
        raise


def _stop_distributed_runtime(runtime: dict[str, Any] | None) -> None:
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


@app.function(
    image=image,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def run_fake_distributed(
    deployment_id: str,
    run_id: str,
    artifact_root: str,
    port: int,
    controller_port: int,
    startup_timeout: int,
) -> dict[str, Any]:
    _add_remote_import_paths()

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

    runtime = None
    try:
        runtime = _start_distributed_runtime(
            deployment_id=deployment_id,
            run_id=run_id,
            artifact_root=artifact_root,
            port=port,
            controller_port=controller_port,
            startup_timeout=startup_timeout,
        )
        proxy = runtime["proxy"]
        trainer_endpoint = runtime["trainer_endpoint"]
        rollout_endpoint = runtime["rollout_endpoint"]
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
                "controller_transport": runtime["controller_transport"],
                "trainer_target": trainer_endpoint.target(family="ipv6"),
                "rollout_target": rollout_endpoint.target(family="ipv6"),
                "loss": fb.loss,
                "input_tokens": fb.usage.input_tokens,
                "optimizer_step": step.optimizer_step,
            }
        )
    finally:
        _stop_distributed_runtime(runtime)


@app.function(
    image=image,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def run_distributed_sft(
    deployment_id: str,
    run_id: str,
    artifact_root: str,
    port: int,
    controller_port: int,
    startup_timeout: int,
    dataset_path: str,
    base_model: str,
    tuning: str,
    lora_rank: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    vocab_size: int,
    seed: int,
) -> dict[str, Any]:
    _add_remote_import_paths()
    if tuning not in {"full", "lora"}:
        raise ValueError("tuning must be 'full' or 'lora'")
    if tuning == "lora" and lora_rank <= 0:
        raise ValueError("lora_rank must be positive for LoRA")
    tuning_literal = cast(Literal["full", "lora"], tuning)

    from examples.sft import SFTDataConfig, ToyTokenizer, load_jsonl_sft_batches, run_sft
    from ganker.client import SamplingClient, ServiceClient, TrainingClient
    from ganker.contracts import (
        ArtifactKind,
        ModelInput,
        SamplingParams,
        TrainingRun,
        TuningMode,
        WeightArtifact,
    )
    from ganker.transport import MonarchProxyTransport

    tokenizer = ToyTokenizer(vocab_size=vocab_size)
    batches = load_jsonl_sft_batches(
        dataset_path,
        tokenizer=tokenizer,
        config=SFTDataConfig(
            sequence_length=sequence_length,
            batch_size=micro_batch_size,
            shuffle=True,
            seed=seed,
        ),
    )

    runtime = None
    client = None
    try:
        runtime = _start_distributed_runtime(
            deployment_id=deployment_id,
            run_id=run_id,
            artifact_root=artifact_root,
            port=port,
            controller_port=controller_port,
            startup_timeout=startup_timeout,
        )
        client = ServiceClient(
            _transport=MonarchProxyTransport(runtime["proxy"], timeout=60),
        )
        summary = run_sft(
            client,
            base_model=base_model,
            dataset=batches,
            tuning=tuning_literal,
            lora_rank=lora_rank if tuning == "lora" else 0,
            learning_rate=learning_rate,
            max_steps=max_steps,
            save_every=save_every,
        )

        artifact_kind = ArtifactKind.DELTA if tuning == "lora" else ArtifactKind.FULL
        tuning_mode = TuningMode.LORA if tuning == "lora" else TuningMode.FULL
        artifact = WeightArtifact(
            run_id=summary.run_id,
            checkpoint_version=summary.checkpoint_version,
            kind=artifact_kind,
            manifest_path=summary.manifest_path,
            payload_path=summary.artifact_path,
        )
        training_run = TrainingRun(
            run_id=summary.run_id,
            base_model=base_model,
            tuning_mode=tuning_mode,
            lora_rank=lora_rank if tuning == "lora" else 0,
            checkpoint_version=summary.checkpoint_version,
        )
        training = TrainingClient(service=client, run=training_run)
        refreshed = training.refresh_weights(
            artifact,
            request_id="modal-distributed-sft-refresh",
        )
        sampler = SamplingClient(service=client, run=training_run, artifact=refreshed.artifact)
        sample = sampler.sample(
            ModelInput.from_ints([7, 8]),
            SamplingParams(max_tokens=4, temperature=0.7, top_p=0.9),
            request_id="modal-distributed-sft-sample",
        )
        telemetry = sampler.get_telemetry_summary(
            request_id="modal-distributed-sft-telemetry",
        )
        artifact_volume.reload()

        payload = {
            "ok": True,
            "mode": "sft-distributed",
            "deployment_id": deployment_id,
            "region": REGION,
            "controller_transport": runtime["controller_transport"],
            "trainer_target": runtime["trainer_endpoint"].target(family="ipv6"),
            "rollout_target": runtime["rollout_endpoint"].target(family="ipv6"),
            "dataset_path": dataset_path,
            "batch_count": len(batches),
            "sample_tokens": sample.sequences[0].tokens,
            "sample_checkpoint_version": sample.artifact.checkpoint_version,
            "telemetry_events": telemetry.summary.event_count,
            "telemetry_input_tokens": telemetry.summary.total.input_tokens,
            "telemetry_output_tokens": telemetry.summary.total.output_tokens,
            "telemetry_training_steps": telemetry.summary.total.training_steps,
            "telemetry_samples": telemetry.summary.total.samples,
            **summary.to_dict(),
        }
        payload["artifact_exists"] = Path(summary.artifact_path).exists()
        payload["manifest_exists"] = Path(summary.manifest_path).exists()
        return _json_safe(payload)
    finally:
        if client is not None:
            client.close()
        _stop_distributed_runtime(runtime)


@app.function(
    image=bridge_image,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def run_qwen_bridge_distributed_sft(
    deployment_id: str,
    run_id: str,
    artifact_root: str,
    port: int,
    controller_port: int,
    startup_timeout: int,
    dataset_path: str,
    base_model: str,
    tuning: str,
    lora_rank: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    _add_remote_import_paths()
    if tuning not in {"full", "lora"}:
        raise ValueError("tuning must be 'full' or 'lora'")
    if tuning == "lora" and lora_rank <= 0:
        raise ValueError("lora_rank must be positive for LoRA")
    tuning_literal = cast(Literal["full", "lora"], tuning)

    from examples.sft import HFAutoTokenizerAdapter, SFTDataConfig, load_jsonl_sft_batches, run_sft
    from ganker.client import SamplingClient, ServiceClient, TrainingClient
    from ganker.contracts import (
        ArtifactKind,
        ModelInput,
        SamplingParams,
        TrainingRun,
        TuningMode,
        WeightArtifact,
    )
    from ganker.transport import MonarchProxyTransport

    tokenizer = HFAutoTokenizerAdapter.from_pretrained(base_model)
    batches = load_jsonl_sft_batches(
        dataset_path,
        tokenizer=tokenizer,
        config=SFTDataConfig(
            sequence_length=sequence_length,
            batch_size=micro_batch_size,
            shuffle=True,
            seed=seed,
        ),
    )

    runtime = None
    client = None
    try:
        runtime = _start_distributed_runtime(
            deployment_id=deployment_id,
            run_id=run_id,
            artifact_root=artifact_root,
            port=port,
            controller_port=controller_port,
            startup_timeout=startup_timeout,
            training_backend="megatron",
            training_backend_config={
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
            },
            inference_backend="fake",
            trainer_worker=monarch_bridge_trainer_worker_role,
            rollout_worker=monarch_worker_role,
        )
        client = ServiceClient(
            _transport=MonarchProxyTransport(runtime["proxy"], timeout=120),
        )
        summary = run_sft(
            client,
            base_model=base_model,
            dataset=batches,
            tuning=tuning_literal,
            lora_rank=lora_rank if tuning == "lora" else 0,
            learning_rate=learning_rate,
            max_steps=max_steps,
            save_every=save_every,
        )

        artifact_kind = ArtifactKind.DELTA if tuning == "lora" else ArtifactKind.FULL
        tuning_mode = TuningMode.LORA if tuning == "lora" else TuningMode.FULL
        artifact = WeightArtifact(
            run_id=summary.run_id,
            checkpoint_version=summary.checkpoint_version,
            kind=artifact_kind,
            manifest_path=summary.manifest_path,
            payload_path=summary.artifact_path,
        )
        training_run = TrainingRun(
            run_id=summary.run_id,
            base_model=base_model,
            tuning_mode=tuning_mode,
            lora_rank=lora_rank if tuning == "lora" else 0,
            checkpoint_version=summary.checkpoint_version,
        )
        training = TrainingClient(service=client, run=training_run)
        refreshed = training.refresh_weights(
            artifact,
            request_id="modal-distributed-qwen-refresh",
        )
        sampler = SamplingClient(service=client, run=training_run, artifact=refreshed.artifact)
        sample = sampler.sample(
            ModelInput.from_ints([7, 8]),
            SamplingParams(max_tokens=4, temperature=0.7, top_p=0.9),
            request_id="modal-distributed-qwen-sample",
        )
        telemetry = sampler.get_telemetry_summary(
            request_id="modal-distributed-qwen-telemetry",
        )
        artifact_volume.reload()

        payload = {
            "ok": True,
            "mode": "qwen-bridge-sft-distributed",
            "runtime_kind": "bridge",
            "bridge_base_image": BRIDGE_BASE_IMAGE,
            "bridge_ref": BRIDGE_REF,
            "deployment_id": deployment_id,
            "region": REGION,
            "controller_transport": runtime["controller_transport"],
            "trainer_target": runtime["trainer_endpoint"].target(family="ipv6"),
            "rollout_target": runtime["rollout_endpoint"].target(family="ipv6"),
            "dataset_path": dataset_path,
            "batch_count": len(batches),
            "tuning": tuning,
            "lora_rank": lora_rank if tuning == "lora" else 0,
            "sample_tokens": sample.sequences[0].tokens,
            "sample_checkpoint_version": sample.artifact.checkpoint_version,
            "telemetry_events": telemetry.summary.event_count,
            "telemetry_input_tokens": telemetry.summary.total.input_tokens,
            "telemetry_output_tokens": telemetry.summary.total.output_tokens,
            "telemetry_training_steps": telemetry.summary.total.training_steps,
            "telemetry_samples": telemetry.summary.total.samples,
            **summary.to_dict(),
        }
        payload["artifact_exists"] = Path(summary.artifact_path).exists()
        payload["manifest_exists"] = Path(summary.manifest_path).exists()
        if payload["artifact_exists"]:
            artifact_payload = json.loads(Path(summary.artifact_path).read_text())
            payload["artifact_format"] = artifact_payload.get("artifact_format")
            for key in (
                "hf_checkpoint_path",
                "hf_adapter_path",
                "hf_weights_path",
                "hf_weights_index_path",
                "hf_adapter_config_path",
                "hf_adapter_weights_path",
                "hf_checkpoint_bytes",
                "hf_weight_count",
                "hf_weight_format",
            ):
                if key in artifact_payload:
                    payload[key] = artifact_payload[key]
            for key in (
                "hf_checkpoint_path",
                "hf_adapter_path",
                "hf_weights_path",
                "hf_weights_index_path",
                "hf_adapter_config_path",
                "hf_adapter_weights_path",
            ):
                if key in payload:
                    payload[f"{key}_exists"] = Path(payload[key]).exists()
        return _json_safe(payload)
    finally:
        if client is not None:
            client.close()
        _stop_distributed_runtime(runtime)


@app.function(
    image=bridge_image,
    timeout=60 * 60,
    i6pn=True,
    region=REGION,
    volumes={ARTIFACT_VOLUME_MOUNT: artifact_volume},
)
def run_qwen_bridge_sglang_distributed_sft(
    deployment_id: str,
    run_id: str,
    artifact_root: str,
    port: int,
    controller_port: int,
    startup_timeout: int,
    dataset_path: str,
    base_model: str,
    tuning: str,
    lora_rank: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    seed: int,
    prompt: str,
    sample_max_tokens: int,
    sglang_port: int,
    sglang_startup_timeout: int,
    sglang_context_length: int,
    sglang_mem_fraction_static: float,
) -> dict[str, Any]:
    _add_remote_import_paths()
    if tuning not in {"full", "lora"}:
        raise ValueError("tuning must be 'full' or 'lora'")
    if tuning == "lora" and lora_rank <= 0:
        raise ValueError("lora_rank must be positive for LoRA")
    if sample_max_tokens <= 0:
        raise ValueError("sample_max_tokens must be positive")
    tuning_literal = cast(Literal["full", "lora"], tuning)

    from examples.sft import HFAutoTokenizerAdapter, SFTDataConfig, load_jsonl_sft_batches, run_sft
    from ganker.client import SamplingClient, ServiceClient, TrainingClient
    from ganker.contracts import (
        ArtifactKind,
        SamplingParams,
        TrainingRun,
        TuningMode,
        WeightArtifact,
    )
    from ganker.transport import MonarchProxyTransport

    tokenizer = HFAutoTokenizerAdapter.from_pretrained(base_model)
    batches = load_jsonl_sft_batches(
        dataset_path,
        tokenizer=tokenizer,
        config=SFTDataConfig(
            sequence_length=sequence_length,
            batch_size=micro_batch_size,
            shuffle=True,
            seed=seed,
        ),
    )

    runtime = None
    client = None
    try:
        runtime = _start_distributed_runtime(
            deployment_id=deployment_id,
            run_id=run_id,
            artifact_root=artifact_root,
            port=port,
            controller_port=controller_port,
            startup_timeout=startup_timeout,
            training_backend="megatron",
            training_backend_config={
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
            },
            inference_backend="sglang",
            inference_backend_config=_sglang_backend_config(
                model=base_model,
                port=sglang_port,
                startup_timeout=sglang_startup_timeout,
                context_length=sglang_context_length,
                mem_fraction_static=sglang_mem_fraction_static,
                max_lora_rank=max(lora_rank, 1),
            ),
            trainer_worker=monarch_bridge_trainer_worker_role,
            rollout_worker=monarch_sglang_rollout_worker_role,
        )
        client_timeout = max(float(startup_timeout), float(sglang_startup_timeout)) + 300
        client = ServiceClient(
            _transport=MonarchProxyTransport(runtime["proxy"], timeout=client_timeout),
        )
        summary = run_sft(
            client,
            base_model=base_model,
            dataset=batches,
            tuning=tuning_literal,
            lora_rank=lora_rank if tuning == "lora" else 0,
            learning_rate=learning_rate,
            max_steps=max_steps,
            save_every=save_every,
        )

        artifact_kind = ArtifactKind.DELTA if tuning == "lora" else ArtifactKind.FULL
        tuning_mode = TuningMode.LORA if tuning == "lora" else TuningMode.FULL
        artifact = WeightArtifact(
            run_id=summary.run_id,
            checkpoint_version=summary.checkpoint_version,
            kind=artifact_kind,
            manifest_path=summary.manifest_path,
            payload_path=summary.artifact_path,
        )
        training_run = TrainingRun(
            run_id=summary.run_id,
            base_model=base_model,
            tuning_mode=tuning_mode,
            lora_rank=lora_rank if tuning == "lora" else 0,
            checkpoint_version=summary.checkpoint_version,
        )
        training = TrainingClient(service=client, run=training_run)
        refreshed = training.refresh_weights(
            artifact,
            request_id="modal-distributed-qwen-sglang-refresh",
        )
        sampler = SamplingClient(service=client, run=training_run, artifact=refreshed.artifact)
        sample = sampler.sample_text(
            prompt,
            SamplingParams(max_tokens=sample_max_tokens, temperature=0.0, top_p=1.0),
            request_id="modal-distributed-qwen-sglang-sample",
        )
        telemetry = sampler.get_telemetry_summary(
            request_id="modal-distributed-qwen-sglang-telemetry",
        )
        artifact_volume.reload()

        payload = {
            "ok": True,
            "mode": "qwen-bridge-sglang-distributed",
            "runtime_kind": "bridge",
            "inference_backend": "sglang",
            "bridge_base_image": BRIDGE_BASE_IMAGE,
            "bridge_ref": BRIDGE_REF,
            "sglang_image": SGLANG_IMAGE,
            "sglang_port": sglang_port,
            "sglang_context_length": sglang_context_length,
            "sglang_mem_fraction_static": sglang_mem_fraction_static,
            "deployment_id": deployment_id,
            "region": REGION,
            "controller_transport": runtime["controller_transport"],
            "trainer_target": runtime["trainer_endpoint"].target(family="ipv6"),
            "rollout_target": runtime["rollout_endpoint"].target(family="ipv6"),
            "dataset_path": dataset_path,
            "batch_count": len(batches),
            "tuning": tuning,
            "lora_rank": lora_rank if tuning == "lora" else 0,
            "prompt": prompt,
            "sample_text": sample.sequences[0].text,
            "sample_tokens": sample.sequences[0].tokens,
            "sample_stop_reason": sample.sequences[0].stop_reason,
            "sample_checkpoint_version": sample.artifact.checkpoint_version,
            "telemetry_events": telemetry.summary.event_count,
            "telemetry_input_tokens": telemetry.summary.total.input_tokens,
            "telemetry_output_tokens": telemetry.summary.total.output_tokens,
            "telemetry_training_steps": telemetry.summary.total.training_steps,
            "telemetry_samples": telemetry.summary.total.samples,
            **summary.to_dict(),
        }
        payload["artifact_exists"] = Path(summary.artifact_path).exists()
        payload["manifest_exists"] = Path(summary.manifest_path).exists()
        if payload["artifact_exists"]:
            artifact_payload = json.loads(Path(summary.artifact_path).read_text())
            payload["artifact_format"] = artifact_payload.get("artifact_format")
            for key in (
                "hf_checkpoint_path",
                "hf_adapter_path",
                "hf_weights_path",
                "hf_weights_index_path",
                "hf_adapter_config_path",
                "hf_adapter_weights_path",
                "hf_checkpoint_bytes",
                "hf_weight_count",
                "hf_weight_format",
            ):
                if key in artifact_payload:
                    payload[key] = artifact_payload[key]
            for key in (
                "hf_checkpoint_path",
                "hf_adapter_path",
                "hf_weights_path",
                "hf_weights_index_path",
                "hf_adapter_config_path",
                "hf_adapter_weights_path",
            ):
                if key in payload:
                    payload[f"{key}_exists"] = Path(payload[key]).exists()
        return _json_safe(payload)
    finally:
        if client is not None:
            client.close()
        _stop_distributed_runtime(runtime)


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
    dataset_path: str = str(REMOTE_ROOT / "examples" / "tiny_sft.jsonl"),
    artifact_root: str = str(ARTIFACT_VOLUME_ROOT),
    base_model: str = "Qwen/Qwen3-0.6B",
    tuning: str = "lora",
    lora_rank: int = 8,
    max_steps: int = 1,
    save_every: int = 0,
    learning_rate: float = 1e-4,
    sequence_length: int = 32,
    micro_batch_size: int = 1,
    vocab_size: int = 128,
    seed: int = 1234,
    prompt: str = "Answer in one short sentence: what is 2+2?",
    sample_max_tokens: int = 12,
    sglang_port: int = SGLANG_PORT,
    sglang_startup_timeout: int = 900,
    sglang_context_length: int = SGLANG_CONTEXT_LENGTH,
    sglang_mem_fraction_static: float = SGLANG_MEM_FRACTION_STATIC,
    port: int = MONARCH_PORT,
    controller_port: int = CONTROLLER_PORT,
    startup_timeout: int = 120,
    deployment_id: str = "",
    run_id: str = "run-000001",
):
    if mode not in {
        "fake-distributed",
        "qwen-bridge-sft-distributed",
        "qwen-bridge-sglang-distributed",
        "sft-distributed",
        "tcp-smoke",
    }:
        raise ValueError(f"unknown mode: {mode}")
    deployment = deployment_id or f"dev-{uuid.uuid4().hex[:8]}"
    if mode == "tcp-smoke":
        result = run_tcp_smoke.remote(deployment, run_id, port or TCP_SMOKE_PORT, startup_timeout)
    elif mode == "sft-distributed":
        result = run_distributed_sft.remote(
            deployment,
            run_id,
            artifact_root,
            port,
            controller_port,
            startup_timeout,
            dataset_path,
            base_model,
            tuning,
            lora_rank,
            max_steps,
            save_every,
            learning_rate,
            sequence_length,
            micro_batch_size,
            vocab_size,
            seed,
        )
    elif mode == "qwen-bridge-sft-distributed":
        result = run_qwen_bridge_distributed_sft.remote(
            deployment,
            run_id,
            artifact_root,
            port,
            controller_port,
            startup_timeout,
            dataset_path,
            base_model,
            tuning,
            lora_rank,
            max_steps,
            save_every,
            learning_rate,
            sequence_length,
            micro_batch_size,
            seed,
        )
    elif mode == "qwen-bridge-sglang-distributed":
        result = run_qwen_bridge_sglang_distributed_sft.remote(
            deployment,
            run_id,
            artifact_root,
            port,
            controller_port,
            startup_timeout,
            dataset_path,
            base_model,
            tuning,
            lora_rank,
            max_steps,
            save_every,
            learning_rate,
            sequence_length,
            micro_batch_size,
            seed,
            prompt,
            sample_max_tokens,
            sglang_port,
            sglang_startup_timeout,
            sglang_context_length,
            sglang_mem_fraction_static,
        )
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
