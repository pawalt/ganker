from __future__ import annotations

from functools import partial
import os
from pathlib import Path
import socket
from typing import Any

from .common import SmokeConfig, package_version


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def initialize_distributed(
    *,
    device: str,
    tensor_parallel: int,
    pipeline_parallel: int,
) -> None:
    import torch
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


def destroy_distributed() -> None:
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


def model_provider(
    *,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    vocab_size: int,
    sequence_length: int,
):
    import torch
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer.transformer_config import TransformerConfig

    transformer_config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
    )
    return GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab_size,
        max_sequence_length=sequence_length,
    )


def synthetic_batches(
    *,
    device: str,
    micro_batch_size: int,
    sequence_length: int,
    vocab_size: int,
    seed: int,
):
    import torch

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
    position_ids = position_ids.expand(micro_batch_size, -1)
    attention_mask = torch.tril(
        torch.ones((sequence_length, sequence_length), dtype=torch.bool, device=device)
    )
    attention_mask = ~attention_mask.view(1, 1, sequence_length, sequence_length)

    while True:
        tokens = torch.randint(
            low=0,
            high=vocab_size,
            size=(micro_batch_size, sequence_length),
            generator=generator,
            device=device,
        )
        labels = torch.roll(tokens, shifts=-1, dims=1)
        labels[:, -1] = 0
        loss_mask = torch.ones((micro_batch_size, sequence_length), device=device)
        yield {
            "tokens": tokens,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }


def forward_step_func(data_iterator, model):
    import torch

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
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


def run_megatron_core_smoke(config: SmokeConfig) -> dict[str, Any]:
    import torch
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    cuda_available = bool(torch.cuda.is_available())
    if config.device == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = config.device
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device == "cpu" and not config.allow_cpu:
        raise RuntimeError("CPU Megatron smoke requires --allow-cpu; use GPU for real smoke")

    initialize_distributed(
        device=device,
        tensor_parallel=config.tensor_parallel,
        pipeline_parallel=config.pipeline_parallel,
    )
    try:
        torch.manual_seed(config.seed)
        if device == "cuda":
            from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

            model_parallel_cuda_manual_seed(config.seed)

        model = model_provider(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            vocab_size=config.vocab_size,
            sequence_length=config.sequence_length,
        ).to(torch.device(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        data_iterator = synthetic_batches(
            device=device,
            micro_batch_size=config.micro_batch_size,
            sequence_length=config.sequence_length,
            vocab_size=config.vocab_size,
            seed=config.seed,
        )
        forward_backward_func = get_forward_backward_func()

        losses: list[float] = []
        for _ in range(config.num_steps):
            optimizer.zero_grad(set_to_none=True)
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=1,
                seq_length=config.sequence_length,
                micro_batch_size=config.micro_batch_size,
                decoder_seq_length=config.sequence_length,
                forward_only=False,
            )
            optimizer.step()
            if losses_reduced:
                first = losses_reduced[0]
                loss_value = first.get("lm loss") if isinstance(first, dict) else first
                if hasattr(loss_value, "item"):
                    loss_value = loss_value.item()
                losses.append(float(loss_value))

        checkpoint_dir = Path(config.artifact_root)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "megatron-core-smoke-model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        return {
            "ok": True,
            "mode": "megatron",
            "device": device,
            "cuda_available": cuda_available,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "torch_version": str(torch.__version__),
            "megatron_core_version": package_version("megatron-core"),
            "num_steps": config.num_steps,
            "losses": losses,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "model": {
                "num_layers": config.num_layers,
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "vocab_size": config.vocab_size,
                "sequence_length": config.sequence_length,
            },
        }
    finally:
        destroy_distributed()
