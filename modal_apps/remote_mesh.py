"""Modal entrypoint for remote gRPC access to a singleton Ganker mesh.

Usage:

    source ~/.codex/modal.env
    modal run modal_apps/remote_mesh.py --mode grpc-smoke-fake
    modal run modal_apps/remote_mesh.py --mode grpc-smoke-qwen-lora
    modal run modal_apps/remote_mesh.py --mode serve --serve-seconds 600
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/ganker")
PYTHON_VERSION = os.getenv("GANKER_MODAL_PYTHON", "3.12")
GPU = os.getenv("GANKER_MODAL_GPU", "L40S")
BASE_IMAGE = os.getenv("GANKER_MODAL_BASE_IMAGE", "")
BRIDGE_BASE_IMAGE = os.getenv("GANKER_MODAL_BRIDGE_BASE_IMAGE", "nvcr.io/nvidia/pytorch:26.02-py3")
BRIDGE_REPO = os.getenv(
    "GANKER_MEGATRON_BRIDGE_REPO",
    "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git",
)
BRIDGE_REF = os.getenv("GANKER_MEGATRON_BRIDGE_REF", "v0.4.2")
BRIDGE_UV_VERSION = os.getenv("GANKER_MEGATRON_BRIDGE_UV_VERSION", "0.7.2")
TORCHMONARCH_VERSION = os.getenv("GANKER_MODAL_TORCHMONARCH_VERSION", "0.5.0")


def _base_image():
    if BASE_IMAGE:
        return modal.Image.from_registry(BASE_IMAGE, add_python=PYTHON_VERSION)
    return modal.Image.debian_slim(python_version=PYTHON_VERSION)


def _add_project(image, *, env: dict[str, str] | None = None):
    image_env = {
        "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}",
        "GANKER_ARTIFACT_ROOT": "/tmp/ganker-artifacts",
    }
    if env is not None:
        image_env.update(env)
    return (
        image.env(image_env)
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


def _common_image():
    return _add_project(
        _base_image()
        .apt_install("git", "curl")
        .uv_pip_install(
            "grpcio>=1.81.1",
            "protobuf>=6.33.6",
            "torch<3",
            "torchmonarch>=0.5.0",
            "pytest>=8.0",
        )
    )


def _bridge_image():
    return _add_project(
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
                "/root/.local/bin/uv pip install --python /opt/venv/bin/python "
                f"grpcio>=1.81.1 protobuf>=6.33.6 torchmonarch=={TORCHMONARCH_VERSION}"
            ),
        ),
        env={
            "PATH": "/opt/venv/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "VIRTUAL_ENV": "/opt/venv",
            "UV_PROJECT_ENVIRONMENT": "/opt/venv",
            "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}:/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM",
            "GANKER_ARTIFACT_ROOT": "/tmp/ganker-artifacts",
        },
    )


image = _common_image()
bridge_image = _bridge_image()
app = modal.App("ganker-remote-mesh")


def _add_remote_import_paths() -> None:
    for path in (REMOTE_ROOT, REMOTE_ROOT / "tests", REMOTE_ROOT / "src"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True))


def _run_grpc_fake_smoke(artifact_root: str) -> dict[str, Any]:
    _add_remote_import_paths()

    from ganker import ServiceClient
    from ganker.contracts import ArtifactFileKind, Datum, ModelInput, SamplingParams, TensorData
    from ganker.rpc.server import start_grpc_proxy_server

    server = None
    client = None
    try:
        server = start_grpc_proxy_server(
            bind="127.0.0.1:0",
            artifact_root=Path(artifact_root),
            training_backend="fake",
            inference_backend="fake",
            timeout=60,
        )
        client = ServiceClient.connect_grpc(server.bound_address, timeout=60)
        training = client.create_lora_training_client(
            base_model="Qwen/Qwen3-0.6B",
            rank=8,
            request_id="modal-grpc-create",
        )
        fb = training.forward_backward(
            Datum(
                model_input=ModelInput.from_ints([1, 2, 3, 4]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_ints([2, 3, 4, 0]),
                    "weights": TensorData.from_floats([1.0, 1.0, 1.0, 1.0]),
                },
            ),
            request_id="modal-grpc-fb",
        )
        step = training.optim_step(learning_rate=1e-4, request_id="modal-grpc-step")
        sampling = training.save_weights_and_get_sampling_client(request_id="modal-grpc-save")
        sample = sampling.sample(
            ModelInput.from_ints([100, 101]),
            SamplingParams(max_tokens=2),
            request_id="modal-grpc-sample",
        )
        downloaded = client.download_artifact_file(
            sampling.artifact,
            file_kind=ArtifactFileKind.PAYLOAD,
            request_id="modal-grpc-download",
        )
        return _json_safe(
            {
                "ok": True,
                "mode": "grpc-smoke-fake",
                "server": server.bound_address,
                "run_id": training.run_id,
                "loss": fb.loss,
                "optimizer_step": step.optimizer_step,
                "checkpoint_version": step.checkpoint_version,
                "sample_tokens": sample.sequences[0].tokens,
                "artifact_payload_bytes": len(downloaded.contents),
            }
        )
    finally:
        if client is not None:
            client.close()
        if server is not None:
            server.stop()


def _run_grpc_qwen_lora_smoke(
    *,
    artifact_root: str,
    dataset_path: str,
    base_model: str,
    lora_rank: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    _add_remote_import_paths()

    from examples.sft import HFAutoTokenizerAdapter, SFTDataConfig, load_jsonl_sft_batches, run_sft
    from ganker import ServiceClient
    from ganker.contracts import ArtifactFileKind, ArtifactKind, WeightArtifact
    from ganker.rpc.server import start_grpc_proxy_server

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

    server = None
    client = None
    try:
        server = start_grpc_proxy_server(
            bind="127.0.0.1:0",
            artifact_root=Path(artifact_root),
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
            timeout=60,
        )
        client = ServiceClient.connect_grpc(server.bound_address, timeout=60)
        summary = run_sft(
            client,
            base_model=base_model,
            dataset=batches,
            tuning="lora",
            lora_rank=lora_rank,
            learning_rate=learning_rate,
            max_steps=max_steps,
            save_every=save_every,
        )
        artifact = WeightArtifact(
            run_id=summary.run_id,
            checkpoint_version=summary.checkpoint_version,
            kind=ArtifactKind.DELTA,
            manifest_path=summary.manifest_path,
            payload_path=summary.artifact_path,
        )
        payload = client.download_artifact_file(
            artifact,
            file_kind=ArtifactFileKind.PAYLOAD,
            request_id="modal-grpc-qwen-download",
        )
        return _json_safe(
            {
                "ok": True,
                "mode": "grpc-smoke-qwen-lora",
                "server": server.bound_address,
                "base_model": base_model,
                "batch_count": len(batches),
                "artifact_payload_bytes": len(payload.contents),
                **summary.to_dict(),
            }
        )
    finally:
        if client is not None:
            client.close()
        if server is not None:
            server.stop()


def _run_serve(
    *,
    artifact_root: str,
    port: int,
    serve_seconds: int,
    bearer_token: str,
) -> dict[str, Any]:
    _add_remote_import_paths()

    from ganker.rpc.server import start_grpc_proxy_server

    server = start_grpc_proxy_server(
        bind=f"0.0.0.0:{port}",
        artifact_root=Path(artifact_root),
        training_backend="fake",
        inference_backend="fake",
        bearer_token=bearer_token or None,
        timeout=60,
    )
    try:
        with modal.forward(port, unencrypted=True) as tunnel:
            payload = {
                "ok": True,
                "mode": "serve",
                "server": server.bound_address,
                "tcp_socket": tunnel.tcp_socket,
                "serve_seconds": serve_seconds,
                "bearer_token_required": bool(bearer_token),
            }
            print(_json_safe(payload), flush=True)
            time.sleep(serve_seconds)
            return _json_safe(payload)
    finally:
        server.stop()


@app.function(image=image, timeout=60 * 60)
def run_remote(
    mode: str,
    artifact_root: str,
    port: int,
    serve_seconds: int,
    bearer_token: str,
) -> dict[str, Any]:
    if mode == "grpc-smoke-fake":
        return _run_grpc_fake_smoke(artifact_root)
    if mode == "serve":
        return _run_serve(
            artifact_root=artifact_root,
            port=port,
            serve_seconds=serve_seconds,
            bearer_token=bearer_token,
        )
    raise ValueError(f"unknown mode for CPU image: {mode}")


@app.function(gpu=GPU, image=bridge_image, timeout=60 * 60)
def run_bridge_remote(
    mode: str,
    artifact_root: str,
    dataset_path: str,
    base_model: str,
    lora_rank: int,
    max_steps: int,
    save_every: int,
    learning_rate: float,
    sequence_length: int,
    micro_batch_size: int,
    seed: int,
) -> dict[str, Any]:
    if mode == "grpc-smoke-qwen-lora":
        return _run_grpc_qwen_lora_smoke(
            artifact_root=artifact_root,
            dataset_path=dataset_path,
            base_model=base_model,
            lora_rank=lora_rank,
            max_steps=max_steps,
            save_every=save_every,
            learning_rate=learning_rate,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            seed=seed,
        )
    raise ValueError(f"unknown bridge mode: {mode}")


@app.local_entrypoint()
def main(
    mode: str = "grpc-smoke-fake",
    artifact_root: str = "/tmp/ganker-remote-mesh",
    dataset_path: str = str(REMOTE_ROOT / "examples" / "tiny_sft.jsonl"),
    base_model: str = "Qwen/Qwen3-0.6B",
    lora_rank: int = 8,
    max_steps: int = 1,
    save_every: int = 0,
    learning_rate: float = 1e-4,
    sequence_length: int = 64,
    micro_batch_size: int = 1,
    seed: int = 1234,
    port: int = 50051,
    serve_seconds: int = 600,
    bearer_token: str = "",
):
    if mode == "grpc-smoke-qwen-lora":
        result = run_bridge_remote.remote(
            mode,
            artifact_root,
            dataset_path,
            base_model,
            lora_rank,
            max_steps,
            save_every,
            learning_rate,
            sequence_length,
            micro_batch_size,
            seed,
        )
    else:
        result = run_remote.remote(
            mode,
            artifact_root,
            port,
            serve_seconds,
            bearer_token,
        )
    print(result)
