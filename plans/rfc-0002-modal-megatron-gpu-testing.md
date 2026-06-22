# RFC 0002: Modal GPU Megatron Test Harness

Status: Draft

## Summary

Add a Modal-backed GPU test path for real Megatron Bridge execution. Local tests should continue to cover the fake backend and CPU Megatron preflight. Modal should become the primary supported path for running real Megatron Bridge forward/backward, optimizer step, and checkpoint smoke tests.

This RFC builds on RFC 0001. RFC 0001 establishes the backend seam and local CPU preflight. This RFC defines what is needed to prove the real GPU runtime works remotely.

## Background

The current codebase has:

- `TrainingBackend` as the backend boundary.
- `MegatronTrainingBackend` as an import-isolated coordinator.
- CPU preflight tests under the `megatron_cpu` marker.
- No real Megatron GPU runtime yet.

The current Megatron adapter intentionally raises for real `forward_backward`, `optim_step`, and checkpoint writing because those operations need a GPU worker runtime.

Modal is a good fit for this path because it supports:

- Python app entrypoints with `modal run`.
- GPU-backed functions via `@app.function(gpu=...)`.
- custom images and registry-based images.
- running distributed-training entrypoints as subprocesses inside a GPU function.

References:

- Modal images: https://modal.com/docs/guide/images
- Modal apps and entrypoints: https://modal.com/docs/guide/apps
- Modal GPU functions: https://modal.com/docs/guide/gpu
- Megatron Bridge docs: https://docs.nvidia.com/nemo/megatron-bridge/latest/
- Megatron Bridge test guidance: https://docs.nvidia.com/nemo/megatron-bridge/nightly/skills/testing/SKILL.html

## Goals

- Add a Modal app that can run Ganker GPU smoke tests remotely.
- Keep local `uv run pytest` CPU-only.
- Keep Megatron Bridge and CUDA dependencies out of the default local environment.
- Verify Modal image health before running real training.
- Add a first real Megatron Bridge one-GPU smoke.
- Exercise the Ganker public client path against the Megatron backend on Modal.
- Make Modal the primary supported path for GPU Megatron tests.

## Non-Goals

- Do not implement multi-node Modal training in this RFC.
- Do not require GPU tests for default local development.
- Do not support RL losses in the Megatron backend.
- Do not make SGLang consume Megatron artifacts yet.
- Do not optimize training throughput.
- Do not make the Modal smoke use large public models or gated checkpoints initially.

## Proposed Test Lanes

```text
local default
  uv run pytest
  fake backend, CPU only

local CPU Megatron preflight
  uv run pytest -m megatron_cpu
  optional torch / Megatron Bridge import and config checks

Modal GPU image smoke
  modal run modal_apps/megatron_smoke.py --mode env
  verifies CUDA, torch, megatron.bridge, package versions

Modal GPU Megatron smoke
  modal run modal_apps/megatron_smoke.py --mode megatron
  runs one tiny real Megatron Bridge training step

Modal GPU Ganker smoke
  modal run modal_apps/megatron_smoke.py --mode ganker
  exercises ServiceClient -> ProxyActor -> TrainingActor -> MegatronTrainingBackend
```

## Modal App Shape

Add `modal_apps/megatron_smoke.py`.

Expected shape:

```python
import modal

app = modal.App("ganker-megatron-smoke")

image = (
    modal.Image.from_registry("<nemo-or-cuda-image>", add_python="3.12")
    # or Modal image construction with apt/uv installs
)

@app.function(gpu="A100", image=image, timeout=60 * 60)
def run_remote(mode: str = "env") -> dict:
    ...

@app.local_entrypoint()
def main(mode: str = "env"):
    print(run_remote.remote(mode))
```

Commands should use the project Modal credential convention:

```bash
source ~/.codex/modal.env
modal run modal_apps/megatron_smoke.py --mode env
modal run modal_apps/megatron_smoke.py --mode megatron
modal run modal_apps/megatron_smoke.py --mode ganker
```

## Image Strategy

Start with the lowest-risk image path:

1. Prefer an NVIDIA NeMo container that already includes Megatron Bridge, Megatron-Core, Transformer Engine, CUDA, and compatible PyTorch.
2. If the NeMo container is not directly usable through Modal, use a CUDA/PyTorch base image and install `megatron-bridge` with `uv`.
3. Keep Ganker installed from the repo into the image with `uv`.

The first implementation should prioritize reliability over image build speed.

## Runtime Strategy

Use an in-process runtime first for the single-GPU smoke. RFC 0003 defines the
runtime lifecycle in detail and makes Megatron-Core's `get_forward_backward_func()`
the primary implementation primitive for Ganker's `forward_backward(...)` API.

```text
MegatronTrainingBackend
  |
  v
InProcessMegatronRuntime
  |
  v
megatron.core.pipeline_parallel.get_forward_backward_func()
```

Rationale:

- `get_forward_backward_func()` is low-level enough to match the Tinker-shaped
  `forward_backward(...)` API.
- The first smoke should prove the same stateful lifecycle the Ganker API exposes:
  initialize, forward/backward, optimizer step, checkpoint/export.
- Keeping the runtime inside `TrainingActor` avoids inventing a worker protocol
  before the core Megatron adapter is proven.

A subprocess or `torchrun` runtime remains a likely fallback for multi-rank,
multi-GPU, or stronger process-isolation scenarios.

## Smoke Script

Add `scripts/megatron_bridge_smoke.py`.

Responsibilities:

- Create a tiny model/provider/config.
- Use synthetic token data.
- Run one forward/backward.
- Run one optimizer step.
- Save a minimal checkpoint or artifact payload.
- Print machine-readable JSON summary.
- Exit nonzero on failure.

Example output:

```json
{
  "ok": true,
  "loss": 1.23,
  "optimizer_step": 1,
  "checkpoint_path": "/tmp/ganker-megatron/...",
  "torch_version": "...",
  "cuda": true,
  "gpu_name": "..."
}
```

## Ganker Integration

Add a GPU-only pytest file, for example:

```text
tests/test_megatron_modal.py
```

Mark it:

```python
pytestmark = pytest.mark.megatron
```

Skip unless:

```text
GANKER_RUN_MEGATRON_TESTS=1
```

The Ganker smoke should call:

```python
with ServiceClient.local(
    tmp_path,
    training_backend="megatron",
    training_backend_config={...},
) as client:
    training = client.create_lora_training_client(...)
    training.forward_backward(...)
    training.optim_step(...)
    saved = training.save_weights()
```

Assertions:

- run is created
- loss is finite
- optimizer step increments
- artifact manifest exists
- telemetry records trainer activity

## Milestones

### M1: Modal Dependency and App Skeleton

- Add `modal` to the dev dependency group.
- Add `modal_apps/megatron_smoke.py`.
- Add an `env` mode that checks:
  - `nvidia-smi`
  - `torch.cuda.is_available()`
  - `import megatron.bridge`
  - versions for torch, CUDA, Megatron Bridge if available

### M2: Remote Pytest Runner

- Add a Modal function that runs:

```bash
uv run pytest -m megatron_cpu
```

- This proves the repo can be installed and tested inside the Modal image before real training.

### M3: Standalone Megatron Bridge Smoke

- Add `scripts/megatron_bridge_smoke.py`.
- Run it inside Modal GPU with an in-process Megatron Bridge/Core runtime.
- Keep it independent of Monarch and Ganker at first.
- Use a tiny synthetic setup.

### M4: In-Process Runtime Adapter

- Add `InProcessMegatronRuntime`.
- Wire it into `MegatronTrainingBackend` behind config.
- Translate `forward_backward`, `optim_step`, and `save_weights` into Megatron
  runtime calls inside the actor process.

### M5: Ganker End-to-End Modal Smoke

- Add GPU-marked Ganker test.
- Run:

```bash
GANKER_RUN_MEGATRON_TESTS=1 uv run pytest -m megatron
```

inside Modal.

- Exercise the public client through the proxy/actor/backend path.

### M6: Artifact Metadata for Rollout

- Ensure saved artifact manifests include:
  - artifact format
  - checkpoint path
  - base model
  - tuning mode
  - LoRA rank
  - TP/PP sizes
  - tokenizer/config paths
- Do not require SGLang loading yet.

## Open Questions

- Which exact NeMo/Megatron Bridge container tag should be the first supported Modal image?
- Should the tiny smoke use a Bridge recipe or direct `AutoBridge` provider construction?
- Can the smoke avoid Hugging Face network calls by using a local tiny config?
- Should `TorchrunMegatronRuntime` be stateless subprocess-per-call, or should it manage a long-lived worker process?
- How should Modal volumes be used for checkpoint artifacts and cache reuse?
- What GPU should be the default: A100, L40S, or H100?
- When should a subprocess or `torchrun` runtime be added for multi-rank tests?

## Risks

- Megatron Bridge dependencies may require a specific CUDA/PyTorch/Transformer Engine stack.
- Image build time may dominate iteration.
- In-process Megatron initialization may leave process-global distributed state
  that is hard to reset between runs.
- A later long-lived worker protocol may still be needed for multi-rank execution.
- Modal GPU availability and cold starts can make smoke tests slower than local tests.
- Tiny synthetic training may not catch multi-GPU or real model conversion failures.

## Recommended First Implementation Slice

Implement M1 and M2 first. Do not start with real Megatron training.

Order:

1. Add Modal app skeleton.
2. Prove GPU image health with `--mode env`.
3. Prove repo installation with `--mode pytest-cpu`.
4. Then add standalone Megatron Bridge smoke.
5. Then wire the Ganker backend through that runtime.

This keeps Modal, image dependencies, Megatron Bridge, and Ganker actor orchestration from all failing at the same time.
