from __future__ import annotations

from dataclasses import dataclass, fields
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Literal


SmokeMode = Literal["env", "pytest-cpu", "megatron", "ganker"]
DEFAULT_ARTIFACT_ROOT = "/tmp/ganker-megatron-smoke"


@dataclass(frozen=True)
class SmokeConfig:
    mode: SmokeMode = "env"
    artifact_root: str = DEFAULT_ARTIFACT_ROOT
    device: Literal["auto", "cuda", "cpu"] = "auto"
    allow_cpu: bool = False
    base_model: str = "local/tiny-config"
    lora_rank: int = 4
    num_steps: int = 1
    micro_batch_size: int = 1
    sequence_length: int = 16
    vocab_size: int = 128
    hidden_size: int = 32
    num_layers: int = 2
    num_attention_heads: int = 4
    tensor_parallel: int = 1
    pipeline_parallel: int = 1
    learning_rate: float = 1e-4
    seed: int = 1234
    output_json: str = ""

    @classmethod
    def from_namespace(cls, namespace: Any) -> "SmokeConfig":
        values = {
            field.name: getattr(namespace, field.name)
            for field in fields(cls)
            if hasattr(namespace, field.name)
        }
        return cls(**values)

    @property
    def artifact_path(self) -> Path:
        return Path(self.artifact_root)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_command(command: list[str], timeout: int = 20) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        return {"available": False}
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def result_to_json(result: dict[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True)


def write_result_json(path: str, result: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result_to_json(result) + "\n")

