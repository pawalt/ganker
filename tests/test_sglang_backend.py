from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.sglang import (
    SGLangGeneration,
    SGLangHTTPRuntime,
    SGLangInferenceBackend,
    SGLangLoadedArtifact,
)
from ganker.config import SGLangBackendConfig
from ganker.contracts import ArtifactKind, ModelInput, SamplingParams, WeightArtifact
from ganker.errors import InvalidRequestError


class FakeRuntime:
    def __init__(self):
        self.refreshes = []
        self.samples = []

    def refresh_weights(
        self,
        *,
        run_id: str,
        artifact: WeightArtifact,
        payload: dict[str, Any],
    ) -> SGLangLoadedArtifact:
        self.refreshes.append((run_id, artifact, payload))
        return SGLangLoadedArtifact(
            artifact=artifact,
            artifact_format=str(payload["artifact_format"]),
            model_path=str(payload.get("base_model", "base")),
            lora_name="adapter-1",
            lora_path=str(payload.get("hf_adapter_path", "")),
            payload=payload,
        )

    def sample(
        self,
        *,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int,
        artifact: SGLangLoadedArtifact,
    ) -> list[SGLangGeneration]:
        self.samples.append((prompt, sampling_params, num_samples, artifact))
        return [
            SGLangGeneration(
                text=f"{prompt.text} response {index}",
                token_ids=[10 + index],
                logprobs=[-0.1],
                prompt_tokens=2,
                output_tokens=1,
                stop_reason="stop",
            )
            for index in range(num_samples)
        ]

    def close(self) -> None:
        pass


class FakeHTTPClient:
    def __init__(self):
        self.calls = []

    def get(self, url: str, *, timeout: float) -> bytes:
        self.calls.append(("GET", url, timeout))
        return b"ok"

    def post_json(self, url: str, payload: dict[str, Any], *, timeout: float) -> Any:
        self.calls.append(("POST", url, payload, timeout))
        if url.endswith("/load_lora_adapter"):
            return {"success": True}
        if url.endswith("/generate"):
            return {
                "text": "hello response",
                "output_ids": [7, 8],
                "meta_info": {
                    "prompt_tokens": 2,
                    "output_tokens": 2,
                    "output_token_logprobs": [
                        [-0.1, 7, " hello"],
                        [-0.2, 8, " response"],
                    ],
                    "finish_reason": {"type": "stop"},
                },
            }
        raise AssertionError(f"unexpected URL: {url}")


def test_sglang_backend_uses_runtime_for_refresh_and_sampling(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    artifact = store.write(
        run_id="run-1",
        checkpoint_version=5,
        kind=ArtifactKind.DELTA,
        payload={
            "artifact_format": "hf-lora-adapter",
            "base_model": "Qwen/Qwen3-0.6B",
            "hf_adapter_path": "/checkpoints/adapter",
        },
    )
    runtime = FakeRuntime()
    backend = SGLangInferenceBackend(store, runtime=runtime)

    loaded = backend.refresh_weights(run_id="run-1", artifact=artifact)
    sample = backend.sample(
        run_id="run-1",
        prompt=ModelInput.from_text("hello"),
        sampling_params=SamplingParams(max_tokens=4, top_p=0.9),
        num_samples=2,
    )

    assert loaded == artifact
    assert runtime.refreshes[0][2]["hf_adapter_path"] == "/checkpoints/adapter"
    assert sample.sequences[0].text == "hello response 0"
    assert sample.sequences[1].tokens == [11]
    assert sample.usage.input_tokens == 4
    assert sample.usage.output_tokens == 2
    assert sample.usage.samples == 2


def test_sglang_backend_rejects_raw_megatron_artifacts(tmp_path: Path):
    store = FilesystemArtifactStore(tmp_path)
    artifact = store.write(
        run_id="run-1",
        checkpoint_version=1,
        kind=ArtifactKind.DELTA,
        payload={
            "artifact_format": "megatron-core-torch-state-dict",
            "checkpoint_path": "/checkpoints/raw",
        },
    )
    backend = SGLangInferenceBackend(
        store,
        config=SGLangBackendConfig(base_url="http://sglang"),
    )

    with pytest.raises(InvalidRequestError, match="SGLang requires artifact_format"):
        backend.refresh_weights(run_id="run-1", artifact=artifact)


def test_sglang_http_runtime_loads_lora_and_generates_text():
    http = FakeHTTPClient()
    runtime = SGLangHTTPRuntime(
        SGLangBackendConfig(
            base_url="http://sglang:30000",
            request_timeout=12,
            return_logprobs=True,
        ),
        http_client=http,
    )
    artifact = WeightArtifact(
        run_id="run-1",
        checkpoint_version=5,
        kind=ArtifactKind.DELTA,
        manifest_path="/tmp/manifest.json",
        payload_path="/tmp/payload.json",
    )

    loaded = runtime.refresh_weights(
        run_id="run-1",
        artifact=artifact,
        payload={
            "artifact_format": "hf-lora-adapter",
            "base_model": "Qwen/Qwen3-0.6B",
            "hf_adapter_path": "/checkpoints/adapter",
        },
    )
    generations = runtime.sample(
        prompt=ModelInput.from_text("hello"),
        sampling_params=SamplingParams(
            max_tokens=2,
            temperature=0.3,
            top_p=0.8,
            stop=["</s>"],
        ),
        num_samples=1,
        artifact=loaded,
    )

    assert loaded.lora_name == "run-1-ckpt-5"
    load_call = http.calls[0]
    assert load_call == (
        "POST",
        "http://sglang:30000/load_lora_adapter",
        {"lora_name": "run-1-ckpt-5", "lora_path": "/checkpoints/adapter"},
        12,
    )
    generate_call = http.calls[1]
    assert generate_call[0] == "POST"
    assert generate_call[1] == "http://sglang:30000/generate"
    payload = generate_call[2]
    assert payload["text"] == "hello"
    assert payload["lora_path"] == "run-1-ckpt-5"
    assert payload["return_logprob"] is True
    assert payload["logprob_start_len"] == -1
    assert payload["sampling_params"] == {
        "max_new_tokens": 2,
        "temperature": 0.3,
        "top_p": 0.8,
        "stop": ["</s>"],
    }
    assert generations == [
        SGLangGeneration(
            text="hello response",
            token_ids=[7, 8],
            logprobs=[-0.1, -0.2],
            prompt_tokens=2,
            output_tokens=2,
            stop_reason="stop",
        )
    ]


def test_sglang_http_runtime_supports_full_checkpoint_token_prompts():
    http = FakeHTTPClient()
    runtime = SGLangHTTPRuntime(
        SGLangBackendConfig(
            base_url="http://sglang:30000",
            model_path="Qwen/Qwen3-0.6B",
            return_logprobs=False,
        ),
        http_client=http,
    )
    artifact = WeightArtifact(
        run_id="run-1",
        checkpoint_version=2,
        kind=ArtifactKind.FULL,
        manifest_path="/tmp/manifest.json",
        payload_path="/tmp/payload.json",
    )

    loaded = runtime.refresh_weights(
        run_id="run-1",
        artifact=artifact,
        payload={
            "artifact_format": "hf-full-safetensors",
            "hf_checkpoint_path": "/checkpoints/full",
        },
    )
    runtime.sample(
        prompt=ModelInput.from_ints([1, 2, 3]),
        sampling_params=SamplingParams(max_tokens=1),
        num_samples=1,
        artifact=loaded,
    )

    assert loaded.model_path == "/checkpoints/full"
    assert len(http.calls) == 1
    payload = http.calls[0][2]
    assert payload["input_ids"] == [1, 2, 3]
    assert "lora_path" not in payload
    assert "return_logprob" not in payload


def test_sglang_http_runtime_builds_launch_command():
    runtime = SGLangHTTPRuntime(
        SGLangBackendConfig(
            host="0.0.0.0",
            port=9000,
            tensor_parallel_size=2,
            data_parallel_size=2,
            max_lora_rank=64,
            served_model_name="qwen",
            extra_server_args={"log-level": "warning", "--disable-radix-cache": True},
        )
    )

    cmd = runtime.server_cmd("Qwen/Qwen3-0.6B")

    assert cmd[:3] == [cmd[0], "-m", "sglang.launch_server"]
    assert "--model-path" in cmd
    assert "Qwen/Qwen3-0.6B" in cmd
    assert ["--tp", "2"] == cmd[cmd.index("--tp") : cmd.index("--tp") + 2]
    assert "--enable-dp-attention" in cmd
    assert "--enable-lora" in cmd
    assert ["--max-lora-rank", "64"] == cmd[
        cmd.index("--max-lora-rank") : cmd.index("--max-lora-rank") + 2
    ]
    assert "--disable-radix-cache" in cmd
    assert ["--log-level", "warning"] == cmd[
        cmd.index("--log-level") : cmd.index("--log-level") + 2
    ]
