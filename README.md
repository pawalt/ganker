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
conversion when torch is installed, and the mocked Megatron runtime lifecycle.
They do not run real Megatron forward/backward.

Real Megatron-Core execution is tested through the Modal GPU smoke path:

```bash
source ~/.codex/modal.env
modal run modal_apps/megatron_smoke.py --mode env
modal run modal_apps/megatron_smoke.py --mode pytest-cpu
modal run modal_apps/megatron_smoke.py --mode megatron
```

The `env` mode reports CUDA, torch, Megatron-Core, and Megatron Bridge
availability. The `pytest-cpu` mode runs the repo's CPU suite inside the Modal
image. The `megatron` mode runs a tiny synthetic GPT training step using
Megatron-Core's `get_forward_backward_func()`, takes one optimizer step, and
writes a checkpoint under `/tmp/ganker-megatron-smoke` in the Modal container.

Useful overrides:

```bash
GANKER_MODAL_GPU=A100 modal run modal_apps/megatron_smoke.py --mode megatron
GANKER_MODAL_BASE_IMAGE=nvcr.io/nvidia/pytorch:<tag> modal run modal_apps/megatron_smoke.py --mode megatron
modal run modal_apps/megatron_smoke.py --mode megatron --num-steps 2 --sequence-length 32
```

The current `--mode ganker` path is a boundary probe: it verifies that the
public client can reach the Megatron backend, but it will report `wired: false`
until `InProcessMegatronRuntime` is implemented behind `MegatronTrainingBackend`.

See `architecture/` for the local orchestration diagrams.
