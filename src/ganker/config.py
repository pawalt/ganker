"""Configuration for local Monarch orchestration."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MegatronBackendConfig:
    """Small subset of Megatron Bridge config needed by the adapter."""

    runtime_kind: str = "bridge"
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    micro_batch_size: int = 1
    global_batch_size: int = 1
    sequence_length: int = 16
    trust_remote_code: bool = True
    load_weights: bool = False
    tensor_device: str = "cpu"
    vocab_size: int = 128
    hidden_size: int = 32
    num_layers: int = 2
    num_attention_heads: int = 4
    seed: int = 1234

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "MegatronBackendConfig":
        if values is None:
            return cls()
        allowed = set(cls.__dataclass_fields__)
        unexpected = sorted(set(values) - allowed)
        if unexpected:
            joined = ", ".join(unexpected)
            raise ValueError(f"unknown Megatron backend config field(s): {joined}")
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime_kind": self.runtime_kind,
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.pipeline_model_parallel_size,
            "micro_batch_size": self.micro_batch_size,
            "global_batch_size": self.global_batch_size,
            "sequence_length": self.sequence_length,
            "trust_remote_code": self.trust_remote_code,
            "load_weights": self.load_weights,
            "tensor_device": self.tensor_device,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class MeshSettings:
    """Local defaults for the singleton Monarch mesh."""

    artifact_root: Path = Path(".local_artifacts")
    monarch_transport: str = "tcp"
    training_backend: str = "fake"
    inference_backend: str = "fake"
    megatron: MegatronBackendConfig = field(default_factory=MegatronBackendConfig)


def load_settings() -> MeshSettings:
    return MeshSettings(
        artifact_root=Path(os.getenv("GANKER_ARTIFACT_ROOT", ".local_artifacts")),
        monarch_transport=os.getenv("GANKER_MONARCH_TRANSPORT", "tcp"),
        training_backend=os.getenv("GANKER_TRAINING_BACKEND", "fake"),
        inference_backend=os.getenv("GANKER_INFERENCE_BACKEND", "fake"),
        megatron=MegatronBackendConfig(
            runtime_kind=os.getenv("GANKER_MEGATRON_RUNTIME", "bridge"),
            tensor_model_parallel_size=int(os.getenv("GANKER_MEGATRON_TP", "1")),
            pipeline_model_parallel_size=int(os.getenv("GANKER_MEGATRON_PP", "1")),
            micro_batch_size=int(os.getenv("GANKER_MEGATRON_MICRO_BATCH_SIZE", "1")),
            global_batch_size=int(os.getenv("GANKER_MEGATRON_GLOBAL_BATCH_SIZE", "1")),
            sequence_length=int(os.getenv("GANKER_MEGATRON_SEQUENCE_LENGTH", "16")),
            trust_remote_code=os.getenv("GANKER_MEGATRON_TRUST_REMOTE_CODE", "1") != "0",
            load_weights=os.getenv("GANKER_MEGATRON_LOAD_WEIGHTS", "0") == "1",
            tensor_device=os.getenv("GANKER_MEGATRON_TENSOR_DEVICE", "cpu"),
            vocab_size=int(os.getenv("GANKER_MEGATRON_VOCAB_SIZE", "128")),
            hidden_size=int(os.getenv("GANKER_MEGATRON_HIDDEN_SIZE", "32")),
            num_layers=int(os.getenv("GANKER_MEGATRON_NUM_LAYERS", "2")),
            num_attention_heads=int(os.getenv("GANKER_MEGATRON_NUM_ATTENTION_HEADS", "4")),
            seed=int(os.getenv("GANKER_MEGATRON_SEED", "1234")),
        ),
    )
