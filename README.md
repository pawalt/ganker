# Ganker

Ganker is a local-first prototype of a Tinker-style training service. The
client talks to a small Python API, while the service side can orchestrate a
proxy, trainer, rollout server, and telemetry actor behind that API.

The current implementation is aimed at proving the contracts and end-to-end
plumbing for:

- granular `forward_backward` and `optim_step` training calls,
- LoRA checkpoint export through Megatron Bridge,
- SGLang-backed sampling through HTTP,
- local Monarch orchestration for development,
- Modal GPU jobs for real Megatron/SGLang execution,
- lightweight tests for each component without requiring local GPUs.

## Status

Working today:

- Public Python client API: `ServiceClient`, `TrainingClient`, and `SamplingClient`.
- Local fake training and inference backends for CPU-only development.
- Local Monarch mesh orchestration with proxy, trainer, rollout, and `TelemetryActor`.
- gRPC proxy transport for clients that should not speak Monarch directly.
- Megatron Bridge training backend with Qwen3 0.6B LoRA on Modal GPUs.
- Megatron-Core smoke tests using `get_forward_backward_func()`.
- SGLang HTTP inference backend, including LoRA adapter loading.
- Modal multinode torchrun harness with NCCL and Qwen LoRA SFT smokes.
- Tensor-parallel and pipeline-parallel Qwen LoRA SFT smokes through Megatron
  Bridge.
- Code-SFT example modeled after Modal's StarCoder multinode guide.
- HF Trainer comparison path for checking Ganker/Megatron loss curves.

Current constraints:

- The Megatron Bridge SFT path supports tested DP-only, TP-only, and `TP=2,
  PP=2` smoke shapes, but larger model shapes still need dedicated validation.
- Pipeline parallelism requires enough logical batch to produce at least one
  microbatch per pipeline stage.
- The most tested real model is `Qwen/Qwen3-0.6B`.
- Local tests do not run real Megatron Bridge or SGLang; those run on Modal.
- The StarCoder-style SGLang eval wrapper exists, but the last smoke attempt
  needed better SGLang startup observability, so treat that path as still being
  hardened.

## Architecture

At the API level, users call a proxy:

```text
Python client
    |
    | ServiceClient / gRPC transport
    v
ProxyActor
    |---------------- TrainingActor -> training backend
    |---------------- RolloutActor  -> inference backend
    |---------------- TelemetryActor
```

For local development, Monarch can run all actors in one process or as a local
distributed mesh. For Modal runs, the trainer and inference processes can live
on separate Modal functions and communicate over Modal private networking or
through the exposed proxy API.

See the architecture notes for more detail:

- `architecture/overview.md`
- `architecture/local-testing.md`
- `architecture/distributed-modal.md`
- `architecture/sampling-sglang.md`

## Client API

The client never needs to call Monarch actor handles directly.

```python
from pathlib import Path

from ganker import ServiceClient
from ganker.contracts import AdamParams, Datum, ModelInput, SamplingParams, TensorData

with ServiceClient.local(Path("/tmp/ganker-artifacts")) as client:
    training = client.create_lora_training_client(
        base_model="Qwen/Qwen3-0.6B",
        rank=8,
    )

    datum = Datum(
        model_input=ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_ints([2, 3, 0]),
            "weights": TensorData.from_floats([0.0, 1.0, 1.0]),
        },
    )

    training.forward_backward([datum], loss_fn="cross_entropy")
    training.optim_step(params=AdamParams(learning_rate=1e-4))

    sampler = training.save_weights_and_get_sampling_client()
    response = sampler.sample_text(
        "Write one sentence about Monarch.",
        SamplingParams(max_tokens=16, temperature=0.7, top_p=0.9),
    )
    print(response.sequences[0].text)
```

For a local distributed actor mesh:

```python
from pathlib import Path

from ganker import ServiceClient

with ServiceClient.local_distributed(Path("/tmp/ganker-distributed")) as client:
    training = client.create_lora_training_client(
        base_model="Qwen/Qwen3-0.6B",
        rank=8,
    )
```

That starts local Monarch worker listeners for trainer and rollout, registers
their loopback endpoints, and has the controller attach to them.

## Development

Install and run with `uv`:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run pyright src examples modal_apps/starcoder_ganker \
  tests/test_code_sft_data.py \
  tests/test_sglang_backend.py
```

The default pytest suite is CPU-only. It does not require CUDA, Megatron Bridge,
SGLang, model weights, or checkpoints.

Useful focused tests:

```bash
uv run pytest tests/test_client.py tests/test_components.py
uv run pytest tests/test_distributed_torchrun.py
uv run pytest tests/test_sglang_backend.py
uv run pytest -m megatron_cpu
```

The full suite currently passes, though Monarch may print an ignored atexit
timeout after pytest has already reported success. A broad Pyright run over
`tests/modal_smoke/` is expected to report missing `torch` and Megatron imports
in the local CPU environment; those modules are executed inside Modal GPU
images.

## Backends

`fake`
: Deterministic CPU-only training and inference backends used by most tests.

`megatron`
: Import-isolated training backend. Locally it can be tested with fake runtime
objects and tiny Megatron-Core smokes. On Modal it can run Megatron Bridge with
real HF model weights and LoRA export.

`sglang`
: HTTP inference backend. It can connect to an existing SGLang server with
`SGLangBackendConfig(base_url=...)` or launch `python -m sglang.launch_server`
with `launch_server=True`.

## Modal Setup

Modal commands assume the project-specific environment:

```bash
source ~/.codex/modal.env
```

Use exact Modal regions for private networking. This repo defaults to
`us-east-1`.

## Modal Smokes

Megatron environment and training smokes:

```bash
uv run modal run modal_apps/megatron_smoke.py --mode env
uv run modal run modal_apps/megatron_smoke.py --mode pytest-cpu
uv run modal run modal_apps/megatron_smoke.py --mode megatron
uv run modal run modal_apps/megatron_smoke.py --mode ganker
```

SGLang smoke:

```bash
uv run modal run modal_apps/sglang_smoke.py --mode client
```

The `megatron` smoke runs a tiny synthetic GPT step through Megatron-Core's
`get_forward_backward_func()`. The `ganker` smoke runs the same kind of step
through the public client/proxy/training path.

## Qwen SFT Examples

Single-node Qwen SFT shape:

```bash
source ~/.codex/modal.env
GANKER_MODAL_GPU=A100 uv run modal deploy modal_apps/qwen_sft/infra.py
GANKER_MODAL_GPU=A100 uv run modal run modal_apps/qwen_sft/sft.py \
  --startup-timeout 900 \
  --sglang-startup-timeout 900
```

Multinode torchrun Qwen SFT:

```bash
source ~/.codex/modal.env
GANKER_QWEN_SFT_MULTINODE_NODES=2 \
uv run modal run modal_apps/qwen_sft_multinode/sft.py --mode torchrun-env

GANKER_QWEN_SFT_MULTINODE_NODES=2 \
uv run modal run modal_apps/qwen_sft_multinode/sft.py --mode nccl-smoke

GANKER_QWEN_SFT_MULTINODE_NODES=2 \
uv run modal run modal_apps/qwen_sft_multinode/sft.py \
  --mode qwen-lora-sft \
  --max-steps 1 \
  --sequence-length 32
```

Model-parallel Qwen SFT:

```bash
source ~/.codex/modal.env
GANKER_QWEN_SFT_MULTINODE_NODES=1 \
GANKER_QWEN_SFT_MULTINODE_GPU=H100:8 \
uv run modal run modal_apps/qwen_model_parallel_sft/sft.py
```

Pipeline-parallel Qwen SFT:

```bash
source ~/.codex/modal.env
GANKER_QWEN_SFT_MULTINODE_NODES=1 \
GANKER_QWEN_SFT_MULTINODE_GPU=H100:8 \
uv run modal run modal_apps/qwen_pipeline_parallel_sft/sft.py
```

The multinode Qwen path has been validated on 2 nodes with `H100:8`,
`world_size=16`, NCCL all-reduce, and a real Qwen3 0.6B Megatron Bridge LoRA
step. The model-parallel Qwen path has been validated on 1 node with `H100:8`,
`TP=2`, `PP=1`, `DP=4`, one real forward/backward/optimizer step, and HF/PEFT
LoRA adapter export. The pipeline-parallel Qwen path has been validated on 1
node with `H100:8`, `TP=2`, `PP=2`, `DP=2`, two Megatron microbatches, one real
forward/backward/optimizer step, and HF/PEFT LoRA adapter export.

HF Trainer comparison:

```bash
source ~/.codex/modal.env
GANKER_MODAL_GPU=A100 uv run modal run modal_apps/qwen_sft/compare_hf.py \
  --startup-timeout 900 \
  --dataset-size 256 \
  --max-steps 20 \
  --sequence-length 256
```

This materializes Alpaca-style JSONL, runs Ganker/Megatron Bridge, runs a
matching HF Trainer + PEFT LoRA baseline, then prints both loss curves and
agreement metrics.

## Code SFT Example

`modal_apps/starcoder_ganker/` mirrors the workflow shape of Modal's StarCoder
multinode training guide, but uses Ganker for training.

```bash
source ~/.codex/modal.env

uv run modal run modal_apps/starcoder_ganker/download_dataset.py \
  --max-examples 256

GANKER_STARCODER_NODES=2 \
uv run modal run modal_apps/starcoder_ganker/sft.py \
  --max-steps 10 \
  --sequence-length 512

uv run modal run modal_apps/starcoder_ganker/evaluate.py \
  --run-id meg-run-000001
```

Defaults:

- Dataset: `bigcode/the-stack-smol-xs`, so the example can run without gated
  BigCode access.
- Model: `Qwen/Qwen3-0.6B`.

For the exact gated StarCoderData source:

```bash
GANKER_STARCODER_DATASET_ID=bigcode/starcoderdata \
uv run modal run modal_apps/starcoder_ganker/download_dataset.py \
  --languages go,rust \
  --max-examples 256
```

For a cheaper debug training run:

```bash
uv run modal run modal_apps/starcoder_ganker/sft.py \
  --single-node \
  --max-steps 1 \
  --sequence-length 32
```

This one-step path has been validated with Qwen3 0.6B through Megatron Bridge
and writes a PEFT-compatible HF LoRA adapter.

## Repository Layout

```text
src/ganker/
  client.py              public client API
  actors.py              proxy, training, rollout, telemetry actors
  contracts.py           request/response and data contracts
  backends/              fake, Megatron, and SGLang backends
  distributed/           Monarch and torchrun helper contracts
  rpc/                   gRPC proxy transport

examples/sft/
  data.py                SFT datum encoding helpers
  real_data.py           Alpaca-style dataset materialization
  code_data.py           code dataset materialization

modal_apps/
  megatron_smoke.py      Megatron-Core and Ganker GPU smokes
  sglang_smoke.py        SGLang sampling smoke
  qwen_sft/              single-node Qwen SFT shape
  qwen_sft_multinode/    clustered torchrun Qwen SFT
  starcoder_ganker/      code-SFT workflow modeled after StarCoder
  distributed/           older distributed harness and compatibility path

architecture/            high-level architecture notes and diagrams
plans/                   RFCs for major design milestones
tests/                   CPU-only unit and integration tests
```

## Version Control

This repository uses `jj` for version control. Common commands:

```bash
jj status
jj diff
jj describe -m "message"
jj bookmark move main --to @
jj git push --bookmark main
```

If SSH-backed `jj` commands hang in Codex, check the loaded agent socket:

```bash
find /tmp -maxdepth 2 -type s | grep ssh
SSH_AUTH_SOCK=/tmp/ssh-.../agent... ssh-add -l
SSH_AUTH_SOCK=/tmp/ssh-.../agent... jj git push --bookmark main
```
