from __future__ import annotations

import argparse
import traceback
from typing import Any

from .common import SmokeConfig, result_to_json, write_result_json
from .env_smoke import collect_env
from .ganker_smoke import run_ganker_boundary_probe
from .megatron_core_smoke import run_megatron_core_smoke
from .pytest_cpu_smoke import run_pytest_cpu


DESCRIPTION = """Small Modal-friendly smoke tests for real Megatron execution.

The real training mode uses Megatron-Core directly because
`get_forward_backward_func()` is the low-level primitive Ganker needs. The
historical script keeps the "bridge" name because the production backend will
still use Megatron Bridge for model conversion/runtime setup.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "--mode",
        choices=("env", "pytest-cpu", "megatron", "ganker"),
        default="env",
    )
    parser.add_argument("--artifact-root", default=SmokeConfig.artifact_root)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--base-model", default=SmokeConfig.base_model)
    parser.add_argument("--lora-rank", type=int, default=SmokeConfig.lora_rank)
    parser.add_argument("--num-steps", type=int, default=SmokeConfig.num_steps)
    parser.add_argument("--micro-batch-size", type=int, default=SmokeConfig.micro_batch_size)
    parser.add_argument("--sequence-length", type=int, default=SmokeConfig.sequence_length)
    parser.add_argument("--vocab-size", type=int, default=SmokeConfig.vocab_size)
    parser.add_argument("--hidden-size", type=int, default=SmokeConfig.hidden_size)
    parser.add_argument("--num-layers", type=int, default=SmokeConfig.num_layers)
    parser.add_argument(
        "--num-attention-heads",
        type=int,
        default=SmokeConfig.num_attention_heads,
    )
    parser.add_argument("--tensor-parallel", type=int, default=SmokeConfig.tensor_parallel)
    parser.add_argument("--pipeline-parallel", type=int, default=SmokeConfig.pipeline_parallel)
    parser.add_argument("--learning-rate", type=float, default=SmokeConfig.learning_rate)
    parser.add_argument("--seed", type=int, default=SmokeConfig.seed)
    parser.add_argument("--output-json", default="")
    return parser


def run(config: SmokeConfig) -> dict[str, Any]:
    if config.mode == "env":
        return collect_env()
    if config.mode == "pytest-cpu":
        return run_pytest_cpu()
    if config.mode == "megatron":
        return run_megatron_core_smoke(config)
    if config.mode == "ganker":
        return run_ganker_boundary_probe(config)
    raise AssertionError(f"unhandled mode: {config.mode}")


def run_from_argv(argv: list[str] | None = None) -> dict[str, Any]:
    config = SmokeConfig.from_namespace(build_parser().parse_args(argv))
    try:
        result = run(config)
    except Exception as exc:
        result = {
            "ok": False,
            "mode": config.mode,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    if config.output_json:
        write_result_json(config.output_json, result)
    return result


def main(argv: list[str] | None = None) -> int:
    result = run_from_argv(argv)
    print(result_to_json(result))
    return 0 if result.get("ok") else 1

