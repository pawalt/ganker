from __future__ import annotations

from typing import Any

from .common import SmokeConfig


def run_ganker_boundary_probe(config: SmokeConfig) -> dict[str, Any]:
    from ganker import ServiceClient
    from ganker.contracts import Datum, ModelInput, TensorData

    datum = Datum(
        model_input=ModelInput.from_ints([1, 2, 3, 4]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_ints([2, 3, 4, 0]),
            "weights": TensorData.from_floats([1.0, 1.0, 1.0, 1.0]),
        },
    )

    client = None
    close_error = None
    try:
        client = ServiceClient.local(
            config.artifact_path,
            training_backend="megatron",
            training_backend_config={
                "runtime_kind": "core",
                "tensor_device": config.device if config.device != "auto" else "cuda",
                "micro_batch_size": config.micro_batch_size,
                "global_batch_size": config.micro_batch_size,
                "sequence_length": config.sequence_length,
                "vocab_size": config.vocab_size,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "num_attention_heads": config.num_attention_heads,
                "seed": config.seed,
                "load_weights": False,
            },
            timeout=60,
        )
        training = client.create_lora_training_client(
            base_model=config.base_model,
            rank=config.lora_rank,
        )
        fb = training.forward_backward(datum, loss_fn="cross_entropy")
        step = training.optim_step(learning_rate=config.learning_rate)
        saved = training.save_weights()
    except Exception as exc:
        if client is not None:
            try:
                client.close()
            except Exception as shutdown_exc:
                close_error = str(shutdown_exc)
        return {
            "ok": True,
            "mode": "ganker",
            "wired": False,
            "reason": str(exc),
            "close_error": close_error,
            "note": "Ganker attempted the in-process Megatron-Core runtime path.",
        }
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    return {
        "ok": True,
        "mode": "ganker",
        "wired": True,
        "loss": fb.loss,
        "optimizer_step": step.optimizer_step,
        "checkpoint_version": step.checkpoint_version,
        "artifact_path": saved.artifact.payload_path,
        "note": "Ganker Megatron-Core runtime path completed through the public client.",
    }

