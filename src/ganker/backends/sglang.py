"""SGLang-backed rollout backend.

This adapter deliberately talks to SGLang over its HTTP API instead of
importing `sglang`, so local tests can exercise request construction and
artifact handling without CUDA, model weights, or the SGLang package.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.base import SampleResult
from ganker.config import SGLangBackendConfig
from ganker.contracts import (
    ModelInput,
    SampledSequence,
    SamplingParams,
    Usage,
    WeightArtifact,
)
from ganker.errors import BackendUnavailableError, InvalidRequestError


_BASE_MODEL_FORMAT = "base-model"
_HF_FULL_FORMAT = "hf-full-safetensors"
_HF_LORA_FORMAT = "hf-lora-adapter"


@dataclass(frozen=True)
class SGLangLoadedArtifact:
    artifact: WeightArtifact
    artifact_format: str
    model_path: str
    lora_name: str = ""
    lora_path: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SGLangGeneration:
    text: str
    token_ids: list[int]
    logprobs: list[float]
    prompt_tokens: int
    output_tokens: int
    stop_reason: str = "length"


class SGLangRuntime(Protocol):
    def refresh_weights(
        self,
        *,
        run_id: str,
        artifact: WeightArtifact,
        payload: dict[str, Any],
    ) -> SGLangLoadedArtifact:
        ...

    def sample(
        self,
        *,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int,
        artifact: SGLangLoadedArtifact,
    ) -> list[SGLangGeneration]:
        ...

    def close(self) -> None:
        ...


class SGLangHTTPClient(Protocol):
    def get(self, url: str, *, timeout: float) -> bytes:
        ...

    def post_json(self, url: str, payload: dict[str, Any], *, timeout: float) -> Any:
        ...


class UrllibSGLangHTTPClient:
    """Tiny blocking JSON client for SGLang's native HTTP API."""

    def get(self, url: str, *, timeout: float) -> bytes:
        request = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            raise BackendUnavailableError(_http_error_message(exc, url)) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendUnavailableError(f"SGLang request failed for {url}: {exc}") from exc

    def post_json(self, url: str, payload: dict[str, Any], *, timeout: float) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise BackendUnavailableError(_http_error_message(exc, url)) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BackendUnavailableError(f"SGLang request failed for {url}: {exc}") from exc

        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {"text": raw.decode("utf-8", errors="replace")}


class SGLangInferenceBackend:
    """Inference backend that samples via an SGLang HTTP server."""

    def __init__(
        self,
        artifact_store: FilesystemArtifactStore,
        *,
        config: SGLangBackendConfig | None = None,
        runtime: SGLangRuntime | None = None,
    ):
        self._artifact_store = artifact_store
        self._runtime = runtime or SGLangHTTPRuntime(config or SGLangBackendConfig())
        self._loaded_artifacts: dict[str, SGLangLoadedArtifact] = {}

    def refresh_weights(
        self,
        *,
        run_id: str,
        artifact: WeightArtifact | None,
    ) -> WeightArtifact:
        if not run_id:
            raise InvalidRequestError("run_id is required")
        if artifact is not None and artifact.run_id and artifact.run_id != run_id:
            raise InvalidRequestError("artifact.run_id must match request.run_id")

        selected = artifact or self._artifact_store.latest(run_id)
        payload = _read_artifact_payload(selected)
        loaded = self._runtime.refresh_weights(
            run_id=run_id,
            artifact=selected,
            payload=payload,
        )
        self._loaded_artifacts[run_id] = loaded
        return loaded.artifact

    def sample(
        self,
        *,
        run_id: str,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int,
    ) -> SampleResult:
        if not run_id:
            raise InvalidRequestError("run_id is required")
        _validate_sampling_request(prompt, sampling_params, num_samples)

        loaded = self._loaded_artifacts.get(run_id)
        if loaded is None:
            artifact = self._artifact_store.latest(run_id)
            payload = _read_artifact_payload(artifact)
            loaded = self._runtime.refresh_weights(
                run_id=run_id,
                artifact=artifact,
                payload=payload,
            )
            self._loaded_artifacts[run_id] = loaded

        generations = self._runtime.sample(
            prompt=prompt,
            sampling_params=sampling_params,
            num_samples=num_samples,
            artifact=loaded,
        )
        return SampleResult(
            run_id=run_id,
            sequences=[
                SampledSequence(
                    text=item.text,
                    tokens=item.token_ids,
                    logprobs=item.logprobs,
                    stop_reason=item.stop_reason,
                )
                for item in generations
            ],
            artifact=loaded.artifact,
            usage=Usage(
                input_tokens=sum(item.prompt_tokens for item in generations),
                output_tokens=sum(item.output_tokens for item in generations),
                samples=len(generations),
            ),
        )

    def close(self) -> None:
        self._runtime.close()


class SGLangHTTPRuntime:
    """HTTP implementation of the SGLang runtime contract."""

    def __init__(
        self,
        config: SGLangBackendConfig | None = None,
        *,
        http_client: SGLangHTTPClient | None = None,
    ):
        self._config = config or SGLangBackendConfig()
        self._http = http_client or UrllibSGLangHTTPClient()
        self._process: subprocess.Popen | None = None
        self._active_model_path = ""
        self._loaded_loras: dict[str, str] = {}

    def refresh_weights(
        self,
        *,
        run_id: str,
        artifact: WeightArtifact,
        payload: dict[str, Any],
    ) -> SGLangLoadedArtifact:
        artifact_format = str(payload.get("artifact_format") or "")
        if not artifact_format and payload.get("base_model"):
            artifact_format = _BASE_MODEL_FORMAT

        if artifact_format in (_BASE_MODEL_FORMAT, _HF_FULL_FORMAT):
            model_path = self._full_model_path(payload)
            self._ensure_server(model_path)
            return SGLangLoadedArtifact(
                artifact=artifact,
                artifact_format=artifact_format,
                model_path=model_path,
                payload=payload,
            )

        if artifact_format == _HF_LORA_FORMAT:
            model_path = self._lora_base_model_path(payload)
            lora_path = _required_str(payload, "hf_adapter_path")
            lora_name = _adapter_name(run_id, artifact.checkpoint_version)
            self._ensure_server(model_path)
            self._load_lora_adapter(lora_name, lora_path)
            return SGLangLoadedArtifact(
                artifact=artifact,
                artifact_format=artifact_format,
                model_path=model_path,
                lora_name=lora_name,
                lora_path=lora_path,
                payload=payload,
            )

        raise InvalidRequestError(
            "SGLang requires artifact_format "
            f"{_HF_FULL_FORMAT!r}, {_HF_LORA_FORMAT!r}, or a base_model payload; "
            f"got {artifact_format!r}"
        )

    def sample(
        self,
        *,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        num_samples: int,
        artifact: SGLangLoadedArtifact,
    ) -> list[SGLangGeneration]:
        return [
            self._generate_once(
                prompt=prompt,
                sampling_params=sampling_params,
                artifact=artifact,
            )
            for _ in range(num_samples)
        ]

    def close(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    @property
    def base_url(self) -> str:
        configured = self._config.base_url.rstrip("/")
        if configured:
            return configured
        return f"http://{self._config.host}:{self._config.port}"

    def server_cmd(self, model_path: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--host",
            self._config.host,
            "--port",
            str(self._config.port),
            "--model-path",
            model_path,
        ]
        if self._config.served_model_name:
            cmd.extend(["--served-model-name", self._config.served_model_name])
        if self._config.tensor_parallel_size is not None:
            cmd.extend(["--tp", str(self._config.tensor_parallel_size)])
        if self._config.data_parallel_size is not None:
            cmd.extend(
                [
                    "--dp",
                    str(self._config.data_parallel_size),
                    "--enable-dp-attention",
                ]
            )
        if self._config.enable_lora:
            cmd.append("--enable-lora")
            cmd.extend(["--max-lora-rank", str(self._config.max_lora_rank)])
            cmd.extend(["--lora-target-modules", "all"])

        for key, value in self._config.extra_server_args.items():
            if value is False or value is None:
                continue
            arg = key if str(key).startswith("--") else f"--{key}"
            cmd.append(arg)
            if value is not True and value != "":
                cmd.append(str(value))
        return cmd

    def _full_model_path(self, payload: dict[str, Any]) -> str:
        artifact_path = _optional_str(payload, "hf_checkpoint_path") or _optional_str(
            payload,
            "checkpoint_path",
        )
        if artifact_path:
            return artifact_path
        return self._config.model_path or _required_str(payload, "base_model")

    def _lora_base_model_path(self, payload: dict[str, Any]) -> str:
        return self._config.model_path or _required_str(payload, "base_model")

    def _ensure_server(self, model_path: str) -> None:
        if not self._config.launch_server:
            return
        if self._process is not None and self._process.poll() is not None:
            self._process = None
            self._active_model_path = ""
            self._loaded_loras.clear()
        if self._process is not None and self._active_model_path == model_path:
            return
        self.close()
        self._loaded_loras.clear()
        self._process = subprocess.Popen(self.server_cmd(model_path))
        self._active_model_path = model_path
        self._wait_for_health()

    def _wait_for_health(self) -> None:
        if self._process is None:
            return
        deadline = time.monotonic() + self._config.startup_timeout
        url = self._url("/health")
        while time.monotonic() < deadline:
            if (returncode := self._process.poll()) is not None:
                raise BackendUnavailableError(
                    "SGLang server exited during startup "
                    f"with return code {returncode}"
                )
            try:
                self._http.get(url, timeout=min(5.0, self._config.request_timeout))
                return
            except BackendUnavailableError:
                time.sleep(1.0)
        raise BackendUnavailableError(
            "SGLang health check timed out after "
            f"{self._config.startup_timeout}s"
        )

    def _load_lora_adapter(self, lora_name: str, lora_path: str) -> None:
        if not self._config.enable_lora:
            raise InvalidRequestError("SGLang LoRA artifact requires enable_lora=True")
        if self._loaded_loras.get(lora_name) == lora_path:
            return
        self._http.post_json(
            self._url("/load_lora_adapter"),
            {"lora_name": lora_name, "lora_path": lora_path},
            timeout=self._config.request_timeout,
        )
        self._loaded_loras[lora_name] = lora_path

    def _generate_once(
        self,
        *,
        prompt: ModelInput,
        sampling_params: SamplingParams,
        artifact: SGLangLoadedArtifact,
    ) -> SGLangGeneration:
        payload = _generate_payload(
            prompt=prompt,
            sampling_params=sampling_params,
            return_logprobs=self._config.return_logprobs,
        )
        if artifact.lora_name:
            payload["lora_path"] = artifact.lora_name
        response = self._http.post_json(
            self._url("/generate"),
            payload,
            timeout=self._config.request_timeout,
        )
        item = response[0] if isinstance(response, list) else response
        if not isinstance(item, dict):
            raise BackendUnavailableError(
                "SGLang /generate returned an unexpected response type"
            )
        return _generation_from_response(
            item,
            prompt=prompt,
            sampling_params=sampling_params,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"


def _validate_sampling_request(
    prompt: ModelInput,
    sampling_params: SamplingParams,
    num_samples: int,
) -> None:
    if not prompt.token_ids and not prompt.text:
        raise InvalidRequestError("prompt must contain token_ids or text")
    if prompt.token_ids and prompt.text:
        raise InvalidRequestError("prompt cannot contain both token_ids and text")
    if sampling_params.max_tokens <= 0:
        raise InvalidRequestError("sampling_params.max_tokens must be positive")
    if sampling_params.temperature < 0:
        raise InvalidRequestError("sampling_params.temperature cannot be negative")
    if sampling_params.top_p <= 0 or sampling_params.top_p > 1:
        raise InvalidRequestError("sampling_params.top_p must be in (0, 1]")
    if num_samples <= 0:
        raise InvalidRequestError("num_samples must be positive")


def _read_artifact_payload(artifact: WeightArtifact) -> dict[str, Any]:
    path = Path(artifact.payload_path)
    if not path.exists():
        raise InvalidRequestError(f"artifact payload is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidRequestError(f"artifact payload is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise InvalidRequestError("artifact payload must be a JSON object")
    return payload


def _generate_payload(
    *,
    prompt: ModelInput,
    sampling_params: SamplingParams,
    return_logprobs: bool,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "sampling_params": {
            "max_new_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
        }
    }
    if sampling_params.stop:
        request["sampling_params"]["stop"] = list(sampling_params.stop)
    if prompt.text:
        request["text"] = prompt.text
    else:
        request["input_ids"] = list(prompt.token_ids)
    if return_logprobs:
        request["return_logprob"] = True
        request["logprob_start_len"] = -1
    return request


def _generation_from_response(
    item: dict[str, Any],
    *,
    prompt: ModelInput,
    sampling_params: SamplingParams,
) -> SGLangGeneration:
    meta = item.get("meta_info")
    if not isinstance(meta, dict):
        meta = {}

    token_ids = _token_ids_from_response(item, meta)
    logprobs = _logprobs_from_meta(meta)
    text = str(item.get("text") or item.get("output_text") or "")

    prompt_tokens = _int_from_any(meta.get("prompt_tokens"))
    if prompt_tokens is None:
        prompt_tokens = len(prompt.token_ids) or len(prompt.text.split()) or 1
    output_tokens = (
        _int_from_any(meta.get("completion_tokens"))
        or _int_from_any(meta.get("output_tokens"))
        or len(token_ids)
        or len(logprobs)
        or len(text.split())
        or sampling_params.max_tokens
    )

    return SGLangGeneration(
        text=text,
        token_ids=token_ids,
        logprobs=logprobs,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        stop_reason=_stop_reason(item, meta),
    )


def _token_ids_from_response(item: dict[str, Any], meta: dict[str, Any]) -> list[int]:
    for key in ("output_ids", "token_ids", "tokens"):
        value = item.get(key)
        if isinstance(value, list):
            return [int(token) for token in value if _is_int_like(token)]

    logprob_entries = meta.get("output_token_logprobs")
    if isinstance(logprob_entries, list):
        tokens = []
        for entry in logprob_entries:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                token_id = entry[1]
            elif isinstance(entry, dict):
                token_id = entry.get("token_id") or entry.get("token")
            else:
                continue
            parsed = _int_from_any(token_id)
            if parsed is not None:
                tokens.append(parsed)
        return tokens
    return []


def _logprobs_from_meta(meta: dict[str, Any]) -> list[float]:
    entries = meta.get("output_token_logprobs")
    if not isinstance(entries, list):
        return []
    logprobs = []
    for entry in entries:
        if isinstance(entry, (list, tuple)) and entry:
            value = entry[0]
        elif isinstance(entry, dict):
            value = entry.get("logprob") or entry.get("logprob_value")
        else:
            continue
        parsed = _float_from_any(value)
        if parsed is not None:
            logprobs.append(parsed)
    return logprobs


def _stop_reason(item: dict[str, Any], meta: dict[str, Any]) -> str:
    value = (
        item.get("stop_reason")
        or item.get("finish_reason")
        or meta.get("finish_reason")
        or meta.get("finish_reasons")
        or "length"
    )
    if isinstance(value, dict):
        return str(value.get("type") or value.get("reason") or "length")
    if isinstance(value, list) and value:
        return _stop_reason({"finish_reason": value[0]}, {})
    return str(value)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = _optional_str(payload, key)
    if value:
        return value
    raise InvalidRequestError(f"SGLang artifact payload is missing {key!r}")


def _optional_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        return ""
    text = str(value)
    return text if text else ""


def _adapter_name(run_id: str, checkpoint_version: int) -> str:
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip("-")
    return f"{safe_run_id or 'run'}-ckpt-{checkpoint_version}"


def _is_int_like(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _int_from_any(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_from_any(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _http_error_message(exc: urllib.error.HTTPError, url: str) -> str:
    body = exc.read().decode("utf-8", errors="replace")
    suffix = f": {body}" if body else ""
    return f"SGLang HTTP {exc.code} for {url}{suffix}"
