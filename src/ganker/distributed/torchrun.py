"""Small contracts for launching distributed PyTorch jobs.

This module intentionally has no Modal, torch, or Megatron imports. Modal apps
and local tests can share the same validation and argv construction without
pulling in GPU-only dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class MegatronRankInfo:
    """Rank metadata needed by torchrun entrypoints.

    This module stays import-isolated from Megatron. Real Megatron jobs should
    prefer `parallel_state` after model-parallel initialization; the pure helper
    below exists for validation, local tests, and fallback metadata.
    """

    global_rank: int
    world_size: int
    data_parallel_rank: int
    data_parallel_size: int
    tensor_model_parallel_rank: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_rank: int
    pipeline_model_parallel_size: int

    @property
    def is_artifact_writer(self) -> bool:
        return (
            self.data_parallel_rank == 0
            and self.tensor_model_parallel_rank == 0
            and self.pipeline_model_parallel_rank == 0
        )

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "global_rank": self.global_rank,
            "world_size": self.world_size,
            "data_parallel_rank": self.data_parallel_rank,
            "data_parallel_size": self.data_parallel_size,
            "tensor_model_parallel_rank": self.tensor_model_parallel_rank,
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "pipeline_model_parallel_rank": self.pipeline_model_parallel_rank,
            "pipeline_model_parallel_size": self.pipeline_model_parallel_size,
            "is_artifact_writer": self.is_artifact_writer,
        }


@dataclass(frozen=True)
class DistributedTrainingConfig:
    n_nodes: int
    gpus_per_node: int
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    micro_batch_size: int = 1
    global_batch_size: int | None = None

    def __post_init__(self) -> None:
        if self.n_nodes <= 0:
            raise ValueError("n_nodes must be positive")
        if self.gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive")
        if self.tensor_model_parallel_size <= 0:
            raise ValueError("tensor_model_parallel_size must be positive")
        if self.pipeline_model_parallel_size <= 0:
            raise ValueError("pipeline_model_parallel_size must be positive")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.global_batch_size is not None and self.global_batch_size <= 0:
            raise ValueError("global_batch_size must be positive")
        if self.world_size % self.model_parallel_size:
            raise ValueError(
                "world_size must be divisible by tensor_model_parallel_size * "
                "pipeline_model_parallel_size"
            )
        if self.effective_global_batch_size % (self.micro_batch_size * self.data_parallel_size):
            raise ValueError(
                "global_batch_size must be divisible by micro_batch_size * data_parallel_size"
            )

    @property
    def world_size(self) -> int:
        return self.n_nodes * self.gpus_per_node

    @property
    def model_parallel_size(self) -> int:
        return self.tensor_model_parallel_size * self.pipeline_model_parallel_size

    @property
    def data_parallel_size(self) -> int:
        return self.world_size // self.model_parallel_size

    @property
    def effective_global_batch_size(self) -> int:
        if self.global_batch_size is not None:
            return self.global_batch_size
        return self.micro_batch_size * self.data_parallel_size

    @property
    def grad_accum_steps(self) -> int:
        return self.effective_global_batch_size // (
            self.micro_batch_size * self.data_parallel_size
        )

    def require_dp_only(self) -> None:
        if self.tensor_model_parallel_size != 1 or self.pipeline_model_parallel_size != 1:
            raise ValueError("the first multinode SFT implementation supports DP-only: tp=1, pp=1")
        if self.grad_accum_steps != 1:
            raise ValueError("the first multinode SFT implementation requires grad_accum_steps=1")

    def require_supported_model_parallel(self, *, allow_pipeline_parallel: bool = False) -> None:
        """Validate the supported model-parallel shape for the current SFT path."""

        if self.pipeline_model_parallel_size != 1 and not allow_pipeline_parallel:
            raise ValueError(
                "pipeline_model_parallel_size > 1 is not supported by the current "
                "Qwen SFT entrypoint; TP-only model parallelism is supported first"
            )

    def as_dict(self) -> dict[str, int]:
        payload = asdict(self)
        payload["global_batch_size"] = self.effective_global_batch_size
        payload["world_size"] = self.world_size
        payload["data_parallel_size"] = self.data_parallel_size
        payload["grad_accum_steps"] = self.grad_accum_steps
        return payload


@dataclass(frozen=True)
class TorchrunLaunchConfig:
    nnodes: int
    nproc_per_node: int
    node_rank: int
    master_addr: str
    master_port: int = 29500

    def __post_init__(self) -> None:
        if self.nnodes <= 0:
            raise ValueError("nnodes must be positive")
        if self.nproc_per_node <= 0:
            raise ValueError("nproc_per_node must be positive")
        if self.node_rank < 0:
            raise ValueError("node_rank cannot be negative")
        if self.node_rank >= self.nnodes:
            raise ValueError("node_rank must be less than nnodes")
        if not self.master_addr:
            raise ValueError("master_addr is required")
        if not (0 < self.master_port < 65536):
            raise ValueError("master_port must be a TCP port")

    def distributed_run_args(self, entrypoint: str, entrypoint_args: Sequence[str] = ()) -> list[str]:
        if not entrypoint:
            raise ValueError("entrypoint is required")
        return [
            f"--nnodes={self.nnodes}",
            f"--nproc-per-node={self.nproc_per_node}",
            f"--node-rank={self.node_rank}",
            f"--master-addr={self.master_addr}",
            f"--master-port={self.master_port}",
            entrypoint,
            *entrypoint_args,
        ]

    def torchrun_argv(self, entrypoint: str, entrypoint_args: Sequence[str] = ()) -> list[str]:
        return ["torchrun", *self.distributed_run_args(entrypoint, entrypoint_args)]

    def env(self) -> dict[str, str]:
        return {
            "NNODES": str(self.nnodes),
            "NPROC_PER_NODE": str(self.nproc_per_node),
            "NODE_RANK": str(self.node_rank),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_gpu_count(gpu_spec: str, *, default: int = 1) -> int:
    """Return the per-node GPU count from Modal specs such as ``H100:8``."""

    if not gpu_spec:
        return default
    if ":" not in gpu_spec:
        return default
    suffix = gpu_spec.rsplit(":", 1)[1]
    try:
        count = int(suffix)
    except ValueError as exc:
        raise ValueError(f"invalid Modal GPU spec: {gpu_spec!r}") from exc
    if count <= 0:
        raise ValueError(f"invalid Modal GPU spec: {gpu_spec!r}")
    return count


def training_config_from_mapping(values: Mapping[str, Any]) -> DistributedTrainingConfig:
    return DistributedTrainingConfig(
        n_nodes=int(values["n_nodes"]),
        gpus_per_node=int(values["gpus_per_node"]),
        tensor_model_parallel_size=int(values.get("tensor_model_parallel_size", 1)),
        pipeline_model_parallel_size=int(values.get("pipeline_model_parallel_size", 1)),
        micro_batch_size=int(values.get("micro_batch_size", 1)),
        global_batch_size=(
            int(values["global_batch_size"]) if values.get("global_batch_size") is not None else None
        ),
    )


def entrypoint_args_from_mapping(values: Mapping[str, Any], *, result_path: str) -> list[str]:
    args = [
        "--mode",
        str(values["mode"]),
        "--result-path",
        result_path,
        "--dataset-path",
        str(values["dataset_path"]),
        "--artifact-root",
        str(values["artifact_root"]),
        "--base-model",
        str(values["base_model"]),
        "--lora-rank",
        str(int(values["lora_rank"])),
        "--learning-rate",
        str(float(values["learning_rate"])),
        "--max-steps",
        str(int(values["max_steps"])),
        "--sequence-length",
        str(int(values["sequence_length"])),
        "--micro-batch-size",
        str(int(values["micro_batch_size"])),
        "--global-batch-size",
        str(int(values["global_batch_size"])),
        "--tensor-model-parallel-size",
        str(int(values.get("tensor_model_parallel_size", 1))),
        "--pipeline-model-parallel-size",
        str(int(values.get("pipeline_model_parallel_size", 1))),
        "--seed",
        str(int(values["seed"])),
        "--comparison-id",
        str(values["comparison_id"]),
    ]
    save_every = int(values.get("save_every", 0))
    if save_every:
        args.extend(["--save-every", str(save_every)])
    return args


def select_data_parallel_item(
    items: Sequence[T],
    *,
    step: int,
    data_parallel_rank: int,
    data_parallel_size: int,
) -> T:
    if not items:
        raise ValueError("items cannot be empty")
    if step < 0:
        raise ValueError("step cannot be negative")
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive")
    if data_parallel_rank < 0 or data_parallel_rank >= data_parallel_size:
        raise ValueError("data_parallel_rank must be in [0, data_parallel_size)")
    index = (step * data_parallel_size + data_parallel_rank) % len(items)
    return items[index]


def select_data_parallel_items(
    items: Sequence[T],
    *,
    step: int,
    data_parallel_rank: int,
    data_parallel_size: int,
    grad_accum_steps: int,
) -> list[T]:
    """Select all local microbatches for one logical optimizer step."""

    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    return [
        select_data_parallel_item(
            items,
            step=step * grad_accum_steps + microstep,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
        )
        for microstep in range(grad_accum_steps)
    ]


def rank_info_from_global_rank(
    config: DistributedTrainingConfig,
    *,
    global_rank: int,
) -> MegatronRankInfo:
    """Return deterministic rank info for contiguous TP/PP replica groups."""

    if global_rank < 0 or global_rank >= config.world_size:
        raise ValueError("global_rank must be in [0, world_size)")
    model_parallel_rank = global_rank % config.model_parallel_size
    return MegatronRankInfo(
        global_rank=global_rank,
        world_size=config.world_size,
        data_parallel_rank=global_rank // config.model_parallel_size,
        data_parallel_size=config.data_parallel_size,
        tensor_model_parallel_rank=model_parallel_rank % config.tensor_model_parallel_size,
        tensor_model_parallel_size=config.tensor_model_parallel_size,
        pipeline_model_parallel_rank=model_parallel_rank // config.tensor_model_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
    )


def rank_result_path(result_path: str | Path, rank: int) -> Path:
    path = Path(result_path)
    return path.with_name(f"{path.stem}.rank-{rank:05d}{path.suffix}")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rank_results(result_path: str | Path, ranks: Iterable[int]) -> list[dict[str, Any]]:
    return [read_json(rank_result_path(result_path, rank)) for rank in ranks]
