#!/usr/bin/env python
"""Small Modal-friendly smoke tests for real Megatron execution.

The real training mode uses Megatron-Core directly because
`get_forward_backward_func()` is the low-level primitive Ganker needs. The file
keeps the historical "bridge" name because the production backend will still
use Megatron Bridge for model conversion/runtime setup.
"""

from __future__ import annotations

import argparse
from functools import partial
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import traceback
from typing import Any


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


def collect_env() -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "mode": "env",
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "ganker": package_version("ganker"),
            "torch": package_version("torch"),
            "megatron-core": package_version("megatron-core"),
            "megatron-bridge": package_version("megatron-bridge"),
            "torchmonarch": package_version("torchmonarch"),
        },
        "nvidia_smi": run_command(["nvidia-smi"]),
    }

    try:
        import torch
    except Exception as exc:
        result["torch"] = {"imported": False, "error": repr(exc)}
        return result

    cuda_available = bool(torch.cuda.is_available())
    result["torch"] = {
        "imported": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
    }

    result["imports"] = {}
    for module in (
        "megatron.core",
        "megatron.core.pipeline_parallel.schedules",
        "megatron.bridge",
    ):
        try:
            __import__(module)
        except Exception as exc:
            result["imports"][module] = {"ok": False, "error": repr(exc)}
        else:
            result["imports"][module] = {"ok": True}
    return result


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def initialize_distributed(*, device: str, tensor_parallel: int, pipeline_parallel: int):
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


def run_megatron_core_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    cuda_available = bool(torch.cuda.is_available())
    if args.device == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = args.device
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError("CPU Megatron smoke requires --allow-cpu; use GPU for real smoke")

    initialize_distributed(
        device=device,
        tensor_parallel=args.tensor_parallel,
        pipeline_parallel=args.pipeline_parallel,
    )
    try:
        torch.manual_seed(args.seed)
        if device == "cuda":
            from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

            model_parallel_cuda_manual_seed(args.seed)

        model = model_provider(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_attention_heads=args.num_attention_heads,
            vocab_size=args.vocab_size,
            sequence_length=args.sequence_length,
        ).to(torch.device(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        data_iterator = synthetic_batches(
            device=device,
            micro_batch_size=args.micro_batch_size,
            sequence_length=args.sequence_length,
            vocab_size=args.vocab_size,
            seed=args.seed,
        )
        forward_backward_func = get_forward_backward_func()

        losses: list[float] = []
        for _ in range(args.num_steps):
            optimizer.zero_grad(set_to_none=True)
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=1,
                seq_length=args.sequence_length,
                micro_batch_size=args.micro_batch_size,
                decoder_seq_length=args.sequence_length,
                forward_only=False,
            )
            optimizer.step()
            if losses_reduced:
                first = losses_reduced[0]
                loss_value = first.get("lm loss") if isinstance(first, dict) else first
                if hasattr(loss_value, "item"):
                    loss_value = loss_value.item()
                losses.append(float(loss_value))

        checkpoint_dir = Path(args.artifact_root)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "megatron-core-smoke-model.pt"
        torch.save(model.state_dict(), checkpoint_path)

        return {
            "ok": True,
            "mode": "megatron",
            "device": device,
            "cuda_available": cuda_available,
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "torch_version": torch.__version__,
            "megatron_core_version": package_version("megatron-core"),
            "num_steps": args.num_steps,
            "losses": losses,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "model": {
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "num_attention_heads": args.num_attention_heads,
                "vocab_size": args.vocab_size,
                "sequence_length": args.sequence_length,
            },
        }
    finally:
        destroy_distributed()


def run_ganker_boundary_probe(args: argparse.Namespace) -> dict[str, Any]:
    from ganker import ServiceClient
    from ganker.contracts import Datum, ModelInput, TensorData
    from ganker.errors import BackendUnavailableError, InvalidRequestError

    datum = Datum(
        model_input=ModelInput.from_ints([1, 2, 3, 4]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_ints([2, 3, 4, 0]),
            "weights": TensorData.from_floats([1.0, 1.0, 1.0, 1.0]),
        },
    )

    try:
        with ServiceClient.local(
            Path(args.artifact_root),
            training_backend="megatron",
            training_backend_config={
                "tensor_device": args.device if args.device != "auto" else "cuda",
                "micro_batch_size": args.micro_batch_size,
                "global_batch_size": args.micro_batch_size,
                "sequence_length": args.sequence_length,
                "load_weights": False,
            },
            timeout=60,
        ) as client:
            training = client.create_lora_training_client(
                base_model=args.base_model,
                rank=args.lora_rank,
            )
            training.forward_backward(datum, loss_fn="cross_entropy")
    except (BackendUnavailableError, InvalidRequestError, RuntimeError) as exc:
        return {
            "ok": True,
            "mode": "ganker",
            "wired": False,
            "reason": str(exc),
            "note": "The direct Megatron-Core smoke is implemented; the Ganker Megatron runtime is still a stub.",
        }

    return {
        "ok": True,
        "mode": "ganker",
        "wired": True,
        "note": "Ganker Megatron path completed; this is expected only after InProcessMegatronRuntime is implemented.",
    }


def run_pytest_cpu() -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-m",
            "not megatron",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "ok": completed.returncode == 0,
        "mode": "pytest-cpu",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("env", "pytest-cpu", "megatron", "ganker"),
        default="env",
    )
    parser.add_argument("--artifact-root", default="/tmp/ganker-megatron-smoke")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--base-model", default="local/tiny-config")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-json", default="")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "env":
        return collect_env()
    if args.mode == "pytest-cpu":
        return run_pytest_cpu()
    if args.mode == "megatron":
        return run_megatron_core_smoke(args)
    if args.mode == "ganker":
        return run_ganker_boundary_probe(args)
    raise AssertionError(f"unhandled mode: {args.mode}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        result = {
            "ok": False,
            "mode": args.mode,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
