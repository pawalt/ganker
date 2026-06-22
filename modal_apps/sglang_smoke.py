"""Modal entrypoint for real SGLang GPU smoke tests.

Usage:

    source ~/.codex/modal.env
    modal run modal_apps/sglang_smoke.py --mode client
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/ganker")
GPU = os.getenv("GANKER_MODAL_GPU", "L40S")
SGLANG_IMAGE = os.getenv("GANKER_MODAL_SGLANG_IMAGE", "lmsysorg/sglang:v0.5.12")
DEFAULT_MODEL = os.getenv("GANKER_SGLANG_MODEL", "Qwen/Qwen3-0.6B")


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


def _sglang_image() -> modal.Image:
    return (
        modal.Image.from_registry(SGLANG_IMAGE)
        .entrypoint([])
        .apt_install("git", "curl")
        .uv_pip_install(
            "grpcio>=1.81.1",
            "protobuf>=6.33.6",
            "torchmonarch>=0.5.0",
            "typing_extensions>=4.13",
        )
        .run_commands("rm -rf /root/.cache/huggingface")
        .env(
            {
                "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT / 'src'}",
                "GANKER_ARTIFACT_ROOT": "/tmp/ganker-sglang-smoke",
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


hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
image = _sglang_image()
app = modal.App("ganker-sglang-smoke")


def _add_remote_import_paths() -> None:
    for path in (REMOTE_ROOT, REMOTE_ROOT / "src", REMOTE_ROOT / "tests"):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def _json_safe(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True))


def _runtime_config(
    *,
    model: str,
    port: int,
    startup_timeout: int,
    context_length: int,
    mem_fraction_static: float,
) -> Any:
    from ganker.config import SGLangBackendConfig

    return SGLangBackendConfig(
        model_path=model,
        launch_server=True,
        host="127.0.0.1",
        port=port,
        request_timeout=120,
        startup_timeout=float(startup_timeout),
        return_logprobs=True,
        enable_lora=False,
        extra_server_args={
            "trust-remote-code": True,
            "context-length": context_length,
            "mem-fraction-static": mem_fraction_static,
            "chunked-prefill-size": min(1024, context_length),
        },
    )


def _sample_response_payload(
    *,
    mode: str,
    model: str,
    sample: Any,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sequence = sample.sequences[0]
    payload: dict[str, Any] = {
        "ok": True,
        "mode": mode,
        "model": model,
        "text": sequence.text,
        "tokens": sequence.tokens,
        "logprobs": sequence.logprobs,
        "stop_reason": sequence.stop_reason,
        "usage": {
            "input_tokens": sample.usage.input_tokens,
            "output_tokens": sample.usage.output_tokens,
            "samples": sample.usage.samples,
        },
        "artifact": {
            "run_id": sample.artifact.run_id,
            "checkpoint_version": sample.artifact.checkpoint_version,
            "payload_path": sample.artifact.payload_path,
        },
    }
    if extra:
        payload.update(extra)
    return _json_safe(payload)


def _run_backend_smoke(
    *,
    artifact_root: str,
    model: str,
    prompt: str,
    max_tokens: int,
    port: int,
    startup_timeout: int,
    context_length: int,
    mem_fraction_static: float,
) -> dict[str, Any]:
    _add_remote_import_paths()

    from ganker.artifacts import FilesystemArtifactStore
    from ganker.backends.sglang import SGLangInferenceBackend
    from ganker.contracts import ArtifactKind, ModelInput, SamplingParams

    store = FilesystemArtifactStore(Path(artifact_root))
    artifact = store.write(
        run_id="sglang-backend-smoke",
        checkpoint_version=0,
        kind=ArtifactKind.FULL,
        payload={"base_model": model},
    )
    backend = SGLangInferenceBackend(
        store,
        config=_runtime_config(
            model=model,
            port=port,
            startup_timeout=startup_timeout,
            context_length=context_length,
            mem_fraction_static=mem_fraction_static,
        ),
    )
    try:
        backend.refresh_weights(run_id=artifact.run_id, artifact=artifact)
        sample = backend.sample(
            run_id=artifact.run_id,
            prompt=ModelInput.from_text(prompt),
            sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0),
            num_samples=1,
        )
        return _sample_response_payload(mode="backend", model=model, sample=sample)
    finally:
        backend.close()


def _run_client_smoke(
    *,
    artifact_root: str,
    model: str,
    prompt: str,
    max_tokens: int,
    port: int,
    startup_timeout: int,
    context_length: int,
    mem_fraction_static: float,
) -> dict[str, Any]:
    _add_remote_import_paths()

    from ganker import ServiceClient
    from ganker.contracts import SamplingParams

    client = ServiceClient.local(
        Path(artifact_root),
        training_backend="fake",
        inference_backend="sglang",
        inference_backend_config=_runtime_config(
            model=model,
            port=port,
            startup_timeout=startup_timeout,
            context_length=context_length,
            mem_fraction_static=mem_fraction_static,
        ),
        timeout=startup_timeout + 180,
    )
    try:
        sampler = client.create_sampling_client(
            base_model=model,
            request_id="modal-sglang-create",
        )
        sample = sampler.sample_text(
            prompt,
            SamplingParams(max_tokens=max_tokens, temperature=0.0, top_p=1.0),
            request_id="modal-sglang-sample",
        )
        telemetry = sampler.get_telemetry_summary(request_id="modal-sglang-telemetry")
        return _sample_response_payload(
            mode="client",
            model=model,
            sample=sample,
            extra={
                "telemetry": {
                    "event_count": telemetry.summary.event_count,
                    "output_tokens": telemetry.summary.total.output_tokens,
                    "samples": telemetry.summary.total.samples,
                }
            },
        )
    finally:
        client.close()


@app.function(
    gpu=GPU,
    image=image,
    timeout=60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache_volume},
    secrets=_hf_secrets(),
)
def run_remote(
    mode: str,
    artifact_root: str,
    model: str,
    prompt: str,
    max_tokens: int,
    port: int,
    startup_timeout: int,
    context_length: int,
    mem_fraction_static: float,
) -> dict[str, Any]:
    if mode == "backend":
        return _run_backend_smoke(
            artifact_root=artifact_root,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            port=port,
            startup_timeout=startup_timeout,
            context_length=context_length,
            mem_fraction_static=mem_fraction_static,
        )
    if mode == "client":
        return _run_client_smoke(
            artifact_root=artifact_root,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            port=port,
            startup_timeout=startup_timeout,
            context_length=context_length,
            mem_fraction_static=mem_fraction_static,
        )
    raise ValueError(f"unknown mode: {mode}")


@app.local_entrypoint()
def main(
    mode: str = "client",
    artifact_root: str = "/tmp/ganker-sglang-smoke",
    model: str = DEFAULT_MODEL,
    prompt: str = "Say hello from SGLang in one short sentence.",
    max_tokens: int = 12,
    port: int = 30000,
    startup_timeout: int = 900,
    context_length: int = 2048,
    mem_fraction_static: float = 0.75,
):
    result = run_remote.remote(
        mode,
        artifact_root,
        model,
        prompt,
        max_tokens,
        port,
        startup_timeout,
        context_length,
        mem_fraction_static,
    )
    print(result)
