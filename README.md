# Ganker

Local singleton prototype for a Tinker-style training system orchestrated with PyTorch Monarch.

The public API is a Python client:

```python
from ganker import ServiceClient
from ganker.contracts import AdamParams, Datum, ModelInput, SamplingParams, TensorData

with ServiceClient.local("/tmp/ganker-artifacts") as client:
    training_client = client.create_lora_training_client(
        base_model="Qwen/Qwen3-8B",
        rank=32,
    )

    datum = Datum(
        model_input=ModelInput.from_ints(tokens=[1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_ints([2, 3, 4]),
            "weights": TensorData.from_floats([0.0, 1.0, 1.0]),
        },
    )

    training_client.forward_backward([datum], loss_fn="cross_entropy")
    training_client.optim_step(params=AdamParams(learning_rate=1e-4))

    sampling_client = training_client.save_weights_and_get_sampling_client()
    sample = sampling_client.sample(
        ModelInput.from_ints(tokens=[1, 2, 3]),
        SamplingParams(max_tokens=8),
    )
    print(sample.sequences[0].tokens)
```

Internally, the project models the high-level flows between:

- a proxy actor behind the public client,
- a training server,
- a rollout server,
- and a general `TelemetryActor`.

Production backends are expected to be Megatron for training and `sglang` for
inference. Local development defaults to fake backends so the contracts and
component behavior can be tested without GPU dependencies.

Internal orchestration uses Monarch actor endpoints instead of gRPC. The client
speaks a clear proxy API through a transport adapter and does not call Monarch
actor handles directly.

## Development

```bash
uv run pytest
```

The default suite stays CPU-only and does not require Megatron Bridge, `sglang`,
CUDA, model weights, or checkpoints.

Megatron adapter preflight tests run locally without a GPU:

```bash
uv run pytest -m megatron_cpu
```

Those tests cover import isolation, backend config mapping, Datum-to-tensor
conversion when torch is installed, and mocked Megatron backend lifecycle. They
do not run real Megatron forward/backward.

Real Megatron execution should be tested through the future Modal GPU path:

```bash
GANKER_RUN_MEGATRON_TESTS=1 uv run pytest -m megatron
```

See `architecture/` for the local orchestration diagrams.
