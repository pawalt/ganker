"""Import-isolated Megatron training backend adapters."""

# pyright: reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from itertools import count
import importlib
import json
import os
from pathlib import Path
import socket
from threading import RLock
from typing import Any, Protocol

from ganker.artifacts import FilesystemArtifactStore
from ganker.backends.base import ForwardBackwardResult, OptimStepResult
from ganker.config import MegatronBackendConfig
from ganker.contracts import (
    AdamParams,
    ArtifactKind,
    Datum,
    ForwardBackwardOutput,
    TensorData,
    TrainingRun,
    TuningMode,
    Usage,
    WeightArtifact,
)
from ganker.errors import BackendUnavailableError, InvalidRequestError, NotFoundError


@dataclass(frozen=True)
class MegatronTensorBatch:
    """Torch tensors produced from Ganker datums for Megatron preflight/runtime."""

    input_ids: Any
    target_tokens: Any
    weights: Any

    @property
    def token_count(self) -> int:
        shape = getattr(self.input_ids, "shape", ())
        if len(shape) == 2:
            return int(shape[0] * shape[1])
        return int(getattr(self.input_ids, "numel")())


@dataclass(frozen=True)
class MegatronRunHandle:
    bridge: Any
    provider: Any
    model: Any = None
    optimizer: Any = None
    scheduler: Any = None
    tokenizer: Any = None
    forward_backward_schedule: Any = None
    config: Any = None
    distributed_context: Any = None
    base_model: str = ""
    tuning_mode: TuningMode = TuningMode.FULL
    lora_rank: int = 0
    peft_config: Any = None


class _RunStatus(Enum):
    READY = "ready"
    GRADIENTS_PENDING = "gradients_pending"
    CHECKPOINTING = "checkpointing"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass
class _RunState:
    run_id: str
    base_model: str
    tuning_mode: TuningMode
    lora_rank: int
    handle: MegatronRunHandle
    status: _RunStatus = _RunStatus.READY
    gradient_version: int = 0
    optimizer_step: int = 0
    checkpoint_version: int = 0
    lock: RLock = field(default_factory=RLock, repr=False)


class MegatronRuntime(Protocol):
    def create_run(
        self,
        *,
        base_model: str,
        tuning_mode: TuningMode,
        lora_rank: int,
        config: MegatronBackendConfig,
    ) -> MegatronRunHandle:
        ...

    def forward_backward(
        self,
        *,
        handle: MegatronRunHandle,
        batch: MegatronTensorBatch,
        loss_fn: str,
        loss_fn_config: dict[str, float],
    ) -> ForwardBackwardOutput:
        ...

    def optim_step(
        self,
        *,
        handle: MegatronRunHandle,
        params: AdamParams,
    ) -> None:
        ...

    def save_weights(
        self,
        *,
        handle: MegatronRunHandle,
        run_id: str,
        checkpoint_version: int,
        kind: ArtifactKind,
    ) -> dict[str, Any]:
        ...

    def shutdown(self, *, handle: MegatronRunHandle) -> None:
        ...


class InstalledMegatronBridgeRuntime:
    """Runtime that touches Megatron Bridge only when the backend is selected."""

    def __init__(self, bridge_module: Any):
        self._bridge = bridge_module

    @classmethod
    def from_installed(cls) -> "InstalledMegatronBridgeRuntime":
        try:
            bridge_module = importlib.import_module("megatron.bridge")
        except ImportError as exc:
            raise BackendUnavailableError(
                "Megatron backend requested, but megatron.bridge is not installed"
            ) from exc
        return cls(bridge_module)

    def create_run(
        self,
        *,
        base_model: str,
        tuning_mode: TuningMode,
        lora_rank: int,
        config: MegatronBackendConfig,
    ) -> MegatronRunHandle:
        auto_bridge = self._bridge.AutoBridge.from_hf_pretrained(
            base_model,
            trust_remote_code=config.trust_remote_code,
        )
        provider = _to_megatron_provider(
            auto_bridge,
            load_weights=config.load_weights,
            hf_path=base_model if config.load_weights else None,
        )
        provider.tensor_model_parallel_size = config.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = config.pipeline_model_parallel_size
        if hasattr(provider, "seq_length"):
            provider.seq_length = config.sequence_length
        if hasattr(provider, "micro_batch_size"):
            provider.micro_batch_size = config.micro_batch_size
        if hasattr(provider, "global_batch_size"):
            provider.global_batch_size = config.global_batch_size
        peft_config = None

        # Unit tests use a fake provider to validate config mapping without
        # requiring torch, Megatron, CUDA, or model downloads.
        if not hasattr(provider, "provide_distributed_model"):
            if hasattr(provider, "finalize"):
                provider.finalize()
            return MegatronRunHandle(
                bridge=auto_bridge,
                provider=provider,
                base_model=base_model,
                tuning_mode=tuning_mode,
                lora_rank=lora_rank,
            )

        torch = _import_torch()
        device = _resolve_device(config.tensor_device, torch)
        _initialize_distributed(
            device=device,
            tensor_parallel=config.tensor_model_parallel_size,
            pipeline_parallel=config.pipeline_model_parallel_size,
        )
        torch.manual_seed(config.seed)
        if device == "cuda":
            from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

            model_parallel_cuda_manual_seed(config.seed)

        if device == "cuda":
            if hasattr(provider, "bf16"):
                provider.bf16 = True
            if hasattr(provider, "fp16"):
                provider.fp16 = False
            if hasattr(provider, "params_dtype"):
                provider.params_dtype = torch.bfloat16
            if hasattr(provider, "pipeline_dtype"):
                provider.pipeline_dtype = torch.bfloat16
        if hasattr(provider, "finalize"):
            provider.finalize()

        if tuning_mode is TuningMode.LORA:
            peft_config = _build_lora_config(lora_rank)
            provider.register_pre_wrap_hook(
                lambda chunks: _apply_peft_config(peft_config, chunks, training=True)
            )

        from megatron.core.distributed import DistributedDataParallelConfig
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        model = provider.provide_distributed_model(
            ddp_config=DistributedDataParallelConfig(),
            wrap_with_ddp=True,
        )

        optimizer_params = list(_trainable_model_parameters(model))
        if not optimizer_params:
            raise BackendUnavailableError("Megatron Bridge model has no trainable parameters")
        optimizer = torch.optim.Adam(optimizer_params, lr=1e-4)
        return MegatronRunHandle(
            bridge=auto_bridge,
            provider=provider,
            model=model,
            optimizer=optimizer,
            forward_backward_schedule=get_forward_backward_func(),
            config={
                "device": device,
                "global_batch_size": config.global_batch_size,
                "micro_batch_size": config.micro_batch_size,
                "runtime": "megatron-bridge",
            },
            base_model=base_model,
            tuning_mode=tuning_mode,
            lora_rank=lora_rank,
            peft_config=peft_config,
        )

    def forward_backward(
        self,
        *,
        handle: MegatronRunHandle,
        batch: MegatronTensorBatch,
        loss_fn: str,
        loss_fn_config: dict[str, float],
    ) -> ForwardBackwardOutput:
        _ = loss_fn_config
        if loss_fn != "cross_entropy":
            raise InvalidRequestError(f"unsupported Megatron Bridge loss_fn: {loss_fn}")
        if handle.model is None or handle.optimizer is None or handle.forward_backward_schedule is None:
            raise BackendUnavailableError("Megatron Bridge runtime handle is not initialized")

        sequence_length = int(batch.input_ids.shape[1])
        micro_batch_size = int(handle.config.get("micro_batch_size", batch.input_ids.shape[0]))
        microbatches = _split_tensor_batch(batch, micro_batch_size=micro_batch_size)
        handle.optimizer.zero_grad(set_to_none=True)
        losses_reduced = handle.forward_backward_schedule(
            forward_step_func=_core_forward_step_func,
            data_iterator=iter([_core_runtime_batch(microbatch) for microbatch in microbatches]),
            model=handle.model,
            num_microbatches=len(microbatches),
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=sequence_length,
            forward_only=False,
        )
        loss = _extract_loss(losses_reduced)
        return ForwardBackwardOutput(loss=loss, metrics={"loss": loss})

    def optim_step(
        self,
        *,
        handle: MegatronRunHandle,
        params: AdamParams,
    ) -> None:
        if handle.optimizer is None:
            raise BackendUnavailableError("Megatron Bridge optimizer is not initialized")
        for group in handle.optimizer.param_groups:
            group["lr"] = params.learning_rate
        handle.optimizer.step()
        handle.optimizer.zero_grad(set_to_none=True)

    def save_weights(
        self,
        *,
        handle: MegatronRunHandle,
        run_id: str,
        checkpoint_version: int,
        kind: ArtifactKind,
    ) -> dict[str, Any]:
        _ = kind
        if handle.bridge is None or handle.model is None:
            raise BackendUnavailableError("Megatron Bridge model is not initialized")
        checkpoint_dir = Path(os.getenv("GANKER_ARTIFACT_ROOT", "/tmp/ganker-artifacts"))
        checkpoint_dir = Path(
            os.getenv(
                "GANKER_MEGATRON_CHECKPOINT_ROOT",
                str(checkpoint_dir),
            )
        )
        artifact_subdir = "hf-lora" if handle.tuning_mode is TuningMode.LORA else "hf-full"
        checkpoint_dir = checkpoint_dir / artifact_subdir / run_id / f"checkpoint-{checkpoint_version:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if handle.tuning_mode is TuningMode.LORA:
            hf_payload = _save_bridge_lora_adapter(
                bridge=handle.bridge,
                model=handle.model,
                base_model=handle.base_model,
                checkpoint_dir=checkpoint_dir,
                peft_config=handle.peft_config,
            )
            artifact_format = "hf-lora-adapter"
        else:
            hf_payload = _save_bridge_hf_checkpoint(
                bridge=handle.bridge,
                model=handle.model,
                base_model=handle.base_model,
                checkpoint_dir=checkpoint_dir,
            )
            artifact_format = "hf-full-safetensors"
        return {
            "backend": "megatron-bridge",
            "runtime": "megatron-bridge",
            "artifact_format": artifact_format,
            "checkpoint_path": str(checkpoint_dir),
            **hf_payload,
        }

    def shutdown(self, *, handle: MegatronRunHandle) -> None:
        if handle.model is not None:
            _destroy_distributed()


class InProcessMegatronCoreRuntime:
    """Tiny in-process Megatron-Core runtime for Modal smoke testing.

    This is not the production Bridge/HF conversion path. It exists to prove
    the Ganker lifecycle against Megatron-Core's real forward/backward schedule
    without installing Megatron Bridge.
    """

    def create_run(
        self,
        *,
        base_model: str,
        tuning_mode: TuningMode,
        lora_rank: int,
        config: MegatronBackendConfig,
    ) -> MegatronRunHandle:
        _ = base_model, tuning_mode, lora_rank
        torch = _import_torch()
        self._validate_config(config)
        device = _resolve_device(config.tensor_device, torch)
        _initialize_distributed(
            device=device,
            tensor_parallel=config.tensor_model_parallel_size,
            pipeline_parallel=config.pipeline_model_parallel_size,
        )
        torch.manual_seed(config.seed)
        if device == "cuda":
            from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

            model_parallel_cuda_manual_seed(config.seed)

        model = _build_tiny_gpt_model(config).to(torch.device(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        return MegatronRunHandle(
            bridge=None,
            provider={"runtime": "megatron-core"},
            model=model,
            optimizer=optimizer,
            forward_backward_schedule=get_forward_backward_func(),
            config={
                "device": device,
                "global_batch_size": config.global_batch_size,
                "micro_batch_size": config.micro_batch_size,
                "runtime": "megatron-core",
            },
        )

    def forward_backward(
        self,
        *,
        handle: MegatronRunHandle,
        batch: MegatronTensorBatch,
        loss_fn: str,
        loss_fn_config: dict[str, float],
    ) -> ForwardBackwardOutput:
        _ = loss_fn_config
        if loss_fn != "cross_entropy":
            raise InvalidRequestError(f"unsupported Megatron-Core loss_fn: {loss_fn}")
        if handle.model is None or handle.optimizer is None or handle.forward_backward_schedule is None:
            raise BackendUnavailableError("Megatron-Core runtime handle is not initialized")

        sequence_length = int(batch.input_ids.shape[1])
        micro_batch_size = int(handle.config.get("micro_batch_size", batch.input_ids.shape[0]))
        microbatches = _split_tensor_batch(batch, micro_batch_size=micro_batch_size)
        handle.optimizer.zero_grad(set_to_none=True)
        losses_reduced = handle.forward_backward_schedule(
            forward_step_func=_core_forward_step_func,
            data_iterator=iter([_core_runtime_batch(microbatch) for microbatch in microbatches]),
            model=handle.model,
            num_microbatches=len(microbatches),
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=sequence_length,
            forward_only=False,
        )
        loss = _extract_loss(losses_reduced)
        return ForwardBackwardOutput(loss=loss, metrics={"loss": loss})

    def optim_step(
        self,
        *,
        handle: MegatronRunHandle,
        params: AdamParams,
    ) -> None:
        if handle.optimizer is None:
            raise BackendUnavailableError("Megatron-Core optimizer is not initialized")
        for group in handle.optimizer.param_groups:
            group["lr"] = params.learning_rate
        handle.optimizer.step()
        handle.optimizer.zero_grad(set_to_none=True)

    def save_weights(
        self,
        *,
        handle: MegatronRunHandle,
        run_id: str,
        checkpoint_version: int,
        kind: ArtifactKind,
    ) -> dict[str, Any]:
        if handle.model is None:
            raise BackendUnavailableError("Megatron-Core model is not initialized")
        torch = _import_torch()
        checkpoint_dir = Path(os.getenv("GANKER_ARTIFACT_ROOT", "/tmp/ganker-artifacts"))
        checkpoint_dir = checkpoint_dir / "megatron-core" / run_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint-{checkpoint_version:06d}.pt"
        torch.save(handle.model.state_dict(), checkpoint_path)
        return {
            "backend": "megatron-core",
            "runtime": "megatron-core",
            "artifact_format": "megatron-core-torch-state-dict",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "kind": kind.name,
        }

    def shutdown(self, *, handle: MegatronRunHandle) -> None:
        _ = handle
        _destroy_distributed()

    def _validate_config(self, config: MegatronBackendConfig) -> None:
        if config.tensor_model_parallel_size != 1:
            raise InvalidRequestError("core runtime smoke supports tensor_model_parallel_size=1")
        if config.pipeline_model_parallel_size != 1:
            raise InvalidRequestError("core runtime smoke supports pipeline_model_parallel_size=1")
        if config.micro_batch_size <= 0:
            raise InvalidRequestError("micro_batch_size must be positive")
        if config.sequence_length <= 0:
            raise InvalidRequestError("sequence_length must be positive")
        if config.hidden_size <= 0:
            raise InvalidRequestError("hidden_size must be positive")
        if config.num_layers <= 0:
            raise InvalidRequestError("num_layers must be positive")
        if config.num_attention_heads <= 0:
            raise InvalidRequestError("num_attention_heads must be positive")
        if config.hidden_size % config.num_attention_heads:
            raise InvalidRequestError("hidden_size must be divisible by num_attention_heads")


def _build_runtime(config: MegatronBackendConfig) -> MegatronRuntime:
    if config.runtime_kind == "bridge":
        return InstalledMegatronBridgeRuntime.from_installed()
    if config.runtime_kind == "core":
        return InProcessMegatronCoreRuntime()
    raise InvalidRequestError(f"unknown Megatron runtime_kind: {config.runtime_kind}")


def is_megatron_bridge_available() -> bool:
    try:
        importlib.import_module("megatron.bridge")
    except ImportError:
        return False
    return True


def datums_to_tensor_batch(
    data: list[Datum],
    *,
    loss_fn: str,
    device: str = "cpu",
) -> MegatronTensorBatch:
    """Convert basic Ganker datums into torch tensors for CPU preflight/runtime."""

    if loss_fn != "cross_entropy":
        raise InvalidRequestError(f"unsupported Megatron loss_fn: {loss_fn}")
    if not data:
        raise InvalidRequestError("data cannot be empty")

    token_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    weight_rows: list[list[float]] = []
    expected_length: int | None = None
    for index, datum in enumerate(data):
        tokens = [int(token) for token in datum.model_input.token_ids]
        if not tokens:
            raise InvalidRequestError(f"data[{index}].model_input.token_ids cannot be empty")
        targets = _required_tensor_values(datum, "target_tokens", index)
        weights = _required_tensor_values(datum, "weights", index)
        if len(targets) != len(tokens):
            raise InvalidRequestError(
                f"data[{index}].loss_fn_inputs['target_tokens'] length must match token_ids"
            )
        if len(weights) != len(tokens):
            raise InvalidRequestError(
                f"data[{index}].loss_fn_inputs['weights'] length must match token_ids"
            )
        if expected_length is None:
            expected_length = len(tokens)
        elif len(tokens) != expected_length:
            raise InvalidRequestError("all datums must have the same token length")

        token_rows.append(tokens)
        target_rows.append([int(value) for value in targets])
        weight_rows.append([float(value) for value in weights])

    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError(
            "torch is required for Megatron tensor conversion preflight"
        ) from exc

    return MegatronTensorBatch(
        input_ids=torch.tensor(token_rows, dtype=torch.long, device=device),
        target_tokens=torch.tensor(target_rows, dtype=torch.long, device=device),
        weights=torch.tensor(weight_rows, dtype=torch.float32, device=device),
    )


def _required_tensor_values(datum: Datum, name: str, datum_index: int) -> list[int | float]:
    try:
        tensor = datum.loss_fn_inputs[name]
    except KeyError as exc:
        raise InvalidRequestError(
            f"data[{datum_index}].loss_fn_inputs['{name}'] is required"
        ) from exc
    if not isinstance(tensor, TensorData):
        raise InvalidRequestError(
            f"data[{datum_index}].loss_fn_inputs['{name}'] must be TensorData"
        )
    return tensor.tolist()


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailableError("torch is required for Megatron-Core runtime") from exc
    return torch


def _model_parameters(model: Any):
    chunks = model if isinstance(model, list) else [model]
    for chunk in chunks:
        yield from chunk.parameters()


def _trainable_model_parameters(model: Any):
    for param in _model_parameters(model):
        if getattr(param, "requires_grad", True):
            yield param


def _build_lora_config(lora_rank: int) -> Any:
    LoRA = importlib.import_module("megatron.bridge.peft.lora").LoRA
    return LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=lora_rank,
        alpha=2 * lora_rank,
        dropout=0.0,
    )


def _apply_peft_config(peft_config: Any, model: Any, *, training: bool) -> Any:
    transformed = peft_config(model, training=training)
    set_params_to_save = getattr(peft_config, "set_params_to_save", None)
    if callable(set_params_to_save):
        set_params_to_save(transformed)
    return transformed


def _to_megatron_provider(auto_bridge: Any, *, load_weights: bool, hf_path: str | None) -> Any:
    try:
        return auto_bridge.to_megatron_provider(load_weights=load_weights, hf_path=hf_path)
    except TypeError as exc:
        if "hf_path" not in str(exc):
            raise
        return auto_bridge.to_megatron_provider(load_weights=load_weights)


def _save_bridge_hf_checkpoint(
    *,
    bridge: Any,
    model: Any,
    base_model: str,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.save_pretrained(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(checkpoint_dir)

    try:
        from transformers import GenerationConfig

        generation_config = GenerationConfig.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
        generation_config.save_pretrained(checkpoint_dir)
    except Exception:
        pass

    state_dict: dict[str, Any] = {}
    for item in bridge.export_hf_weights(model, cpu=True, show_progress=False):
        name = getattr(item, "param_name", None)
        weight = getattr(item, "weight", None)
        if name is None or weight is None:
            name, weight = item
        state_dict[str(name)] = weight.detach().cpu().contiguous().clone()

    weights_payload = _write_safetensors_state_dict(
        state_dict,
        checkpoint_dir=checkpoint_dir,
    )
    return {
        "hf_checkpoint_path": str(checkpoint_dir),
        "hf_weight_format": "safetensors",
        "hf_checkpoint_bytes": sum(
            path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file()
        ),
        "hf_weight_count": len(state_dict),
        **weights_payload,
    }


def _save_bridge_lora_adapter(
    *,
    bridge: Any,
    model: Any,
    base_model: str,
    checkpoint_dir: Path,
    peft_config: Any,
) -> dict[str, Any]:
    if peft_config is None:
        raise BackendUnavailableError("LoRA run is missing PEFT config")
    bridge.save_hf_adapter(
        model,
        checkpoint_dir,
        peft_config=peft_config,
        base_model_name_or_path=base_model,
        show_progress=False,
    )
    weights_path = checkpoint_dir / "adapter_model.safetensors"
    config_path = checkpoint_dir / "adapter_config.json"
    if not weights_path.exists():
        return {
            "hf_adapter_path": str(checkpoint_dir),
            "hf_adapter_config_path": str(config_path),
            "hf_adapter_weights_path": str(weights_path),
            "hf_adapter_written": False,
            "hf_weight_format": "safetensors",
            "hf_checkpoint_bytes": sum(
                path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file()
            ),
            "hf_weight_count": 0,
        }
    return {
        "hf_adapter_path": str(checkpoint_dir),
        "hf_adapter_config_path": str(config_path),
        "hf_adapter_weights_path": str(weights_path),
        "hf_adapter_written": True,
        "hf_weight_format": "safetensors",
        "hf_checkpoint_bytes": sum(
            path.stat().st_size for path in checkpoint_dir.rglob("*") if path.is_file()
        ),
        "hf_weight_count": _safetensors_tensor_count(weights_path),
    }


def _write_safetensors_state_dict(
    state_dict: dict[str, Any],
    *,
    checkpoint_dir: Path,
    max_shard_size: str = "5GB",
) -> dict[str, Any]:
    from huggingface_hub import split_torch_state_dict_into_shards
    from safetensors.torch import save_file

    plan = split_torch_state_dict_into_shards(
        state_dict,
        max_shard_size=max_shard_size,
        filename_pattern="model{suffix}.safetensors",
    )
    weight_files: list[str] = []
    for filename, tensor_names in plan.filename_to_tensors.items():
        shard = {name: state_dict[name] for name in tensor_names}
        path = checkpoint_dir / filename
        save_file(shard, path, metadata={"format": "pt"})
        weight_files.append(str(path))
    payload: dict[str, Any] = {
        "hf_weight_files": weight_files,
    }
    if plan.is_sharded:
        index_path = checkpoint_dir / "model.safetensors.index.json"
        index = {"metadata": plan.metadata, "weight_map": plan.tensor_to_filename}
        index_path.write_text(json.dumps(index, sort_keys=True, indent=2), encoding="utf-8")
        payload["hf_weights_index_path"] = str(index_path)
    else:
        payload["hf_weights_path"] = weight_files[0]
    return payload


def _safetensors_tensor_count(path: Path) -> int:
    try:
        from safetensors import safe_open
    except ImportError:
        return 0

    with safe_open(path, framework="pt", device="cpu") as handle:
        return len(handle.keys())


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_device(requested: str, torch) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise BackendUnavailableError("CUDA was requested but torch.cuda.is_available() is false")
    if requested not in ("cpu", "cuda"):
        raise InvalidRequestError(f"unsupported tensor_device: {requested}")
    return requested


def _initialize_distributed(*, device: str, tensor_parallel: int, pipeline_parallel: int) -> None:
    torch = _import_torch()
    import torch.distributed as dist
    from megatron.core import parallel_state

    rank = int(os.environ.setdefault("RANK", "0"))
    world_size = int(os.environ.setdefault("WORLD_SIZE", "1"))
    local_rank = int(os.environ.setdefault("LOCAL_RANK", str(rank)))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))

    if device == "cuda":
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel,
            pipeline_model_parallel_size=pipeline_parallel,
        )


def _destroy_distributed() -> None:
    try:
        import torch.distributed as dist
        from megatron.core import parallel_state
    except Exception:
        return

    try:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _build_tiny_gpt_model(config: MegatronBackendConfig):
    torch = _import_torch()
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    transformer_config = TransformerConfig(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )
    return GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=config.vocab_size,
        max_sequence_length=config.sequence_length,
    )


def _core_runtime_batch(batch: MegatronTensorBatch) -> dict[str, Any]:
    torch = _import_torch()
    micro_batch_size, sequence_length = batch.input_ids.shape
    device = batch.input_ids.device
    position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
    position_ids = position_ids.expand(micro_batch_size, -1)
    attention_mask = torch.tril(
        torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device)
    )
    attention_mask = ~attention_mask.view(1, 1, sequence_length, sequence_length)
    return {
        "tokens": batch.input_ids,
        "labels": batch.target_tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "loss_mask": batch.weights,
    }


def _split_tensor_batch(batch: MegatronTensorBatch, *, micro_batch_size: int) -> list[MegatronTensorBatch]:
    if micro_batch_size <= 0:
        raise InvalidRequestError("micro_batch_size must be positive")
    total_batch_size = int(batch.input_ids.shape[0])
    if total_batch_size <= 0:
        raise InvalidRequestError("batch must contain at least one datum")
    if total_batch_size % micro_batch_size:
        raise InvalidRequestError(
            "logical batch size must be divisible by configured micro_batch_size: "
            f"batch_size={total_batch_size}, micro_batch_size={micro_batch_size}"
        )
    if total_batch_size == micro_batch_size:
        return [batch]
    microbatches: list[MegatronTensorBatch] = []
    for start in range(0, total_batch_size, micro_batch_size):
        end = start + micro_batch_size
        microbatches.append(
            MegatronTensorBatch(
                input_ids=batch.input_ids[start:end],
                target_tokens=batch.target_tokens[start:end],
                weights=batch.weights[start:end],
            )
        )
    return microbatches


def _core_forward_step_func(data_iterator, model):
    torch = _import_torch()

    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.reshape(-1).float()
        loss = torch.sum(losses.reshape(-1) * loss_mask) / loss_mask.sum()
        return loss, {"lm loss": loss.detach()}

    data = next(data_iterator)
    output_tensor = model(
        data["tokens"],
        data["position_ids"],
        data["attention_mask"],
        labels=data["labels"],
    )
    return output_tensor, partial(loss_func, data["loss_mask"])


def _extract_loss(losses_reduced: Any) -> float:
    if not losses_reduced:
        return 0.0
    values: list[float] = []
    for item in losses_reduced:
        if isinstance(item, dict):
            value = item.get("lm loss", item.get("loss", 0.0))
        else:
            value = item
        if hasattr(value, "item"):
            value = value.item()
        values.append(float(value))
    if not values:
        return 0.0
    return sum(values) / len(values)


class MegatronTrainingBackend:
    """Coordinator for Megatron Bridge training state.

    Local CPU tests inject a fake runtime. The default runtime only builds
    Megatron Bridge provider/config objects and fails clearly for real training
    until a GPU worker runtime is wired in.
    """

    def __init__(
        self,
        artifact_store: FilesystemArtifactStore,
        *,
        config: MegatronBackendConfig | None = None,
        runtime: MegatronRuntime | None = None,
    ):
        self._artifact_store = artifact_store
        self._config = config or MegatronBackendConfig()
        self._runtime = runtime or _build_runtime(self._config)
        self._run_counter = count(1)
        self._runs: dict[str, _RunState] = {}
        self._lock = RLock()

    def create_training_run(
        self,
        *,
        base_model: str,
        tuning_mode: TuningMode,
        lora_rank: int,
    ) -> TrainingRun:
        if not base_model:
            raise InvalidRequestError("base_model is required")
        if tuning_mode not in (TuningMode.LORA, TuningMode.FULL):
            raise InvalidRequestError(f"unsupported tuning mode: {tuning_mode}")
        if tuning_mode == TuningMode.LORA and lora_rank <= 0:
            raise InvalidRequestError("lora_rank must be positive for LoRA runs")
        if tuning_mode == TuningMode.FULL and lora_rank < 0:
            raise InvalidRequestError("lora_rank cannot be negative")

        with self._lock:
            self._ensure_no_active_run()
            handle = self._runtime.create_run(
                base_model=base_model,
                tuning_mode=tuning_mode,
                lora_rank=lora_rank,
                config=self._config,
            )
            run_id = f"meg-run-{next(self._run_counter):06d}"
            state = _RunState(
                run_id=run_id,
                base_model=base_model,
                tuning_mode=tuning_mode,
                lora_rank=lora_rank,
                handle=handle,
            )
            self._runs[run_id] = state
            return self._run_message(state)

    def forward_backward(
        self,
        *,
        run_id: str,
        data: list[Datum],
        loss_fn: str,
        loss_fn_config: dict[str, float],
    ) -> ForwardBackwardResult:
        state = self._get_run(run_id)
        with state.lock:
            self._require_status(
                state,
                _RunStatus.READY,
                operation="forward_backward",
            )
            batch = datums_to_tensor_batch(
                data,
                loss_fn=loss_fn,
                device=self._config.tensor_device,
            )
            try:
                output = self._runtime.forward_backward(
                    handle=state.handle,
                    batch=batch,
                    loss_fn=loss_fn,
                    loss_fn_config=loss_fn_config,
                )
            except Exception:
                state.status = _RunStatus.FAILED
                raise

            state.gradient_version += 1
            state.status = _RunStatus.GRADIENTS_PENDING
            return ForwardBackwardResult(
                run_id=run_id,
                output=output,
                gradient_version=state.gradient_version,
                usage=Usage(input_tokens=batch.token_count),
            )

    def optim_step(
        self,
        *,
        run_id: str,
        params: AdamParams,
    ) -> OptimStepResult:
        state = self._get_run(run_id)
        if params.learning_rate <= 0:
            raise InvalidRequestError("learning_rate must be positive")

        with state.lock:
            self._require_status(
                state,
                _RunStatus.GRADIENTS_PENDING,
                operation="optim_step",
            )
            try:
                self._runtime.optim_step(handle=state.handle, params=params)
            except Exception:
                state.status = _RunStatus.FAILED
                raise

            state.optimizer_step += 1
            state.checkpoint_version += 1
            state.status = _RunStatus.READY
            return OptimStepResult(
                run_id=run_id,
                optimizer_step=state.optimizer_step,
                checkpoint_version=state.checkpoint_version,
                usage=Usage(training_steps=1),
            )

    def save_weights(self, *, run_id: str, kind: ArtifactKind) -> WeightArtifact:
        state = self._get_run(run_id)
        with state.lock:
            self._require_status(
                state,
                _RunStatus.READY,
                operation="save_weights",
            )
            state.status = _RunStatus.CHECKPOINTING
            try:
                runtime_payload = self._runtime.save_weights(
                    handle=state.handle,
                    run_id=run_id,
                    checkpoint_version=state.checkpoint_version,
                    kind=kind,
                ) or {}
                payload = {
                    "artifact_format": "megatron",
                    "backend": "megatron-bridge",
                    "base_model": state.base_model,
                    "checkpoint_version": state.checkpoint_version,
                    "gradient_version": state.gradient_version,
                    "global_batch_size": self._config.global_batch_size,
                    "lora_rank": state.lora_rank,
                    "micro_batch_size": self._config.micro_batch_size,
                    "optimizer_step": state.optimizer_step,
                    "pipeline_model_parallel_size": self._config.pipeline_model_parallel_size,
                    "run_id": state.run_id,
                    "run_status": _RunStatus.READY.value,
                    "sequence_length": self._config.sequence_length,
                    "tensor_model_parallel_size": self._config.tensor_model_parallel_size,
                    "tuning_mode": state.tuning_mode.name,
                }
                payload.update(runtime_payload)
                artifact = self._artifact_store.write(
                    run_id=state.run_id,
                    checkpoint_version=state.checkpoint_version,
                    kind=kind,
                    payload=payload,
                )
            except Exception:
                state.status = _RunStatus.READY
                raise

            state.status = _RunStatus.READY
            return artifact

    def close(self) -> None:
        with self._lock:
            states = list(self._runs.values())
        for state in states:
            with state.lock:
                if state.status is _RunStatus.CLOSED:
                    continue
                try:
                    self._runtime.shutdown(handle=state.handle)
                except Exception:
                    state.status = _RunStatus.FAILED
                else:
                    state.status = _RunStatus.CLOSED

    def _get_run(self, run_id: str) -> _RunState:
        if not run_id:
            raise InvalidRequestError("run_id is required")
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise NotFoundError(f"unknown run_id={run_id}") from exc

    def _ensure_no_active_run(self) -> None:
        active = [
            state.run_id
            for state in self._runs.values()
            if state.status not in (_RunStatus.FAILED, _RunStatus.CLOSED)
        ]
        if active:
            raise InvalidRequestError(
                "Megatron backend supports one active run per actor; "
                f"active run_id={active[0]}"
            )

    def _require_status(
        self,
        state: _RunState,
        expected: _RunStatus,
        *,
        operation: str,
    ) -> None:
        if state.status is expected:
            return
        raise InvalidRequestError(
            f"cannot {operation} run_id={state.run_id} while run status is "
            f"{state.status.value}; expected {expected.value}"
        )

    def _run_message(self, state: _RunState) -> TrainingRun:
        return TrainingRun(
            run_id=state.run_id,
            base_model=state.base_model,
            tuning_mode=state.tuning_mode,
            lora_rank=state.lora_rank,
            checkpoint_version=state.checkpoint_version,
        )
