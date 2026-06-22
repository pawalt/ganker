# RFC 0001: Megatron Bridge Training Backend

Status: Draft

## Summary

Add optional Megatron Bridge support behind the existing `TrainingBackend` boundary, while keeping fake backends as the default local development path. The first production-shaped milestone should prove a tiny SFT-style training loop on one GPU, save an artifact, and exercise it through the existing `ServiceClient -> ProxyActor -> TrainingActor` flow.

Megatron support should not make default unit tests depend on CUDA, Megatron Bridge, Megatron-Core, NGC containers, model weights, or distributed launchers.

## Background

The project currently has the right high-level seam:

```text
ServiceClient
  -> ProxyTransport
  -> ProxyActor
  -> TrainingActor
  -> TrainingComponent
  -> TrainingBackend
```

`FakeTrainingBackend` implements the backend contract for local tests. `MegatronTrainingBackend` is currently only an import-isolated placeholder.

Megatron Bridge is a PyTorch-native Megatron-Core bridge and training stack. It supports Hugging Face to Megatron conversion, Megatron to Hugging Face export, pretraining, SFT, and LoRA. It is launched in practice with distributed process launchers such as `torchrun` or NeMo-Run, and NVIDIA recommends the NeMo Framework container for the best supported environment.

Megatron Bridge's own testing guidance separates fast unit tests from GPU functional tests. Unit tests live under `tests/unit_tests/` and are explicitly documented as requiring no GPU. Functional tests are launch scripts under `tests/functional_tests/launch_scripts/{h100,gb200}/...`, run on GPU nodes, and are capped at small GPU counts. Model-support guidance follows the same split: provider/config/mapping unit tests are local, while HF <-> Megatron roundtrip functional tests are marked GPU-only and use toy models plus `torch.distributed.run`.

References:

- Megatron Bridge docs: https://docs.nvidia.com/nemo/megatron-bridge/latest/
- Megatron Bridge repository: https://github.com/NVIDIA-NeMo/Megatron-Bridge
- Recipe/launch docs: https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html
- Conversion details: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/docs/bridge-tech-details.md
- Megatron Bridge testing guidance: https://docs.nvidia.com/nemo/megatron-bridge/nightly/skills/testing/SKILL.html
- Megatron Bridge test/example patterns: https://docs.nvidia.com/nemo/megatron-bridge/latest/skills/adding-model-support/tests-and-examples.html

## Goals

- Keep the public Ganker API unchanged for basic training calls.
- Add a real `MegatronTrainingBackend` implementation for SFT-style training.
- Keep local fake backend tests fast and CPU-only.
- Add CPU-only local preflight tests for Megatron Bridge config, provider, data conversion, and adapter lifecycle.
- Add explicit GPU/Modal smoke tests for real Megatron execution.
- Save enough artifact metadata for later rollout loading and SGLang integration.
- Preserve heavy import isolation so users can develop the proxy/client/contracts without Megatron installed.

## Non-Goals

- Do not implement RL losses in Megatron Bridge initially.
- Do not make SGLang consume Megatron artifacts in the first milestone.
- Do not reproduce the full Tinker SDK type system.
- Do not require `uv run pytest` to launch distributed training.
- Do not claim CPU-only tests prove Megatron-Core forward/backward correctness.
- Do not support every model family on day one.

## Proposed Architecture

### Backend Selection

Add backend selection to `TrainingActor` construction:

```text
TrainingActor(artifact_root, backend_kind="fake", backend_config={...})
```

The actor should use `build_training_backend(...)` rather than constructing `FakeTrainingBackend` directly.

```text
fake:
  TrainingActor -> TrainingComponent -> FakeTrainingBackend

megatron:
  TrainingActor -> TrainingComponent -> MegatronTrainingBackend
```

### Megatron Backend Shape

`MegatronTrainingBackend` should implement the existing protocol:

```python
create_training_run(base_model, tuning_mode, lora_rank) -> TrainingRun
forward_backward(run_id, data, loss_fn, loss_fn_config) -> ForwardBackwardResult
optim_step(run_id, params) -> OptimStepResult
save_weights(run_id, kind) -> WeightArtifact
```

The backend should own per-run state:

- Megatron Bridge config/provider
- tokenizer/config references
- distributed process metadata
- optimizer/scheduler state
- current gradient version
- optimizer step
- checkpoint version
- artifact format and paths

### Distributed Lifecycle

Real Megatron should not run as a simple in-process object forever. Megatron Bridge examples assume distributed launch patterns, so the backend should use a coordinator/worker model:

```text
TrainingActor
  |
  v
MegatronTrainingBackend
  |
  +-- launches or connects to Megatron worker group
      |
      +-- torchrun / NeMo-Run / Modal GPU process group
```

First implementation can be one local GPU worker group. Later implementations can place the worker group on a separate Monarch mesh or Modal deployment.

CPU-only local testing should not attempt to run Megatron-Core training. It should stop at import/config/provider/data-conversion boundaries unless a specific Megatron Bridge API is documented and verified to support CPU execution. This mirrors Megatron Bridge's own split between no-GPU unit tests and GPU functional tests.

### Data Conversion

The Ganker API should continue accepting lightweight `Datum` objects:

```python
Datum(
    model_input=ModelInput.from_ints([...]),
    loss_fn_inputs={
        "target_tokens": TensorData.from_ints([...]),
        "weights": TensorData.from_floats([...]),
    },
)
```

The Megatron backend converts this into torch tensors with validated shapes:

- `model_input.token_ids` -> input token tensor
- `loss_fn_inputs["target_tokens"]` -> labels/targets
- `loss_fn_inputs["weights"]` -> loss mask or weights

Initial supported loss:

- `loss_fn="cross_entropy"`

Unsupported loss names should fail clearly.

### Artifact Format

Extend saved artifact metadata before relying on it for rollout:

```text
artifact_format: megatron | hf | lora_adapter | merged_hf
base_model
tuning_mode
lora_rank
checkpoint_version
optimizer_step
tensor_model_parallel_size
pipeline_model_parallel_size
tokenizer_path
config_path
payload_path
manifest_path
```

For the first milestone, raw Megatron checkpoint artifacts are enough. For SGLang rollout, the backend will likely need to export HF/safetensors or LoRA adapter artifacts using Bridge conversion.

## Milestones

### M1: Backend Configuration and Import Isolation

- Add `MegatronBackendConfig`.
- Add backend kind/config plumbing through local orchestration.
- Keep `megatron` imports inside `ganker.backends.megatron`.
- Add unit tests proving fake backend remains default.
- Add unit tests proving Megatron backend raises a clear unavailable error when dependencies are absent.

### M2: Tensor Conversion Layer

- Add conversion helpers from `list[Datum]` to torch tensors.
- Validate required loss inputs.
- Validate equal sequence lengths where required.
- Unit test conversion with monkeypatched or optional torch.
- Keep conversion independent from Megatron Bridge runtime.

### M3: CPU Megatron Bridge Preflight

- Add a `tests/megatron_cpu/` or marked `pytest -m megatron_cpu` suite.
- Skip Megatron Bridge import checks unless `megatron.bridge` is installed.
- Instantiate tiny Bridge recipe/provider/config objects where the API supports doing so without CUDA.
- Test Ganker config -> Megatron Bridge config translation.
- Test `Datum` -> torch CPU tensor conversion.
- Test backend lifecycle with mocked Bridge objects:
  - create run
  - forward/backward dispatch shape
  - optimizer-step dispatch shape
  - save artifact metadata
- Assert CPU preflight does not call CUDA APIs, launch `torchrun`, or require distributed process groups.

This milestone gives local confidence that the adapter tracks Megatron Bridge API drift without using a GPU. It does not validate Megatron-Core kernels, distributed optimizer behavior, checkpoint sharding, or real training.

### M4: Single-GPU Megatron Smoke Path

- Implement a minimal `MegatronTrainingBackend` using Megatron Bridge on one GPU.
- Start with a tiny architecture/config or smallest practical supported recipe.
- Support:
  - create run
  - one forward/backward
  - one Adam optimizer step
  - save checkpoint artifact
- Add `pytest -m megatron` smoke test.
- Skip unless `GANKER_RUN_MEGATRON_TESTS=1`.

### M5: Modal Test Harness

- Add Modal app/test entry point using the `peyton-agents` credentials environment.
- Use a NeMo/Megatron Bridge capable image.
- Run the single-GPU smoke test on GPU.
- Store artifacts in a temporary Modal volume or workspace path.
- Treat Modal as the primary supported GPU test path.

### M6: Artifact Export for Rollout

- Use Megatron Bridge export to produce rollout-consumable artifacts.
- Decide whether rollout receives:
  - full HF export,
  - LoRA adapter export,
  - merged HF export,
  - or raw Megatron shard metadata plus a future SGLang loader.
- Add integration test asserting artifact metadata is sufficient for `RolloutActor.refresh_weights`.

### M7: Multi-GPU/Parallelism Smoke

- Add an explicit 2-GPU test path.
- Validate tensor parallel size 2 with a tiny model.
- Check that artifact metadata records parallelism.
- Keep this out of default CI.

## Testing Strategy

### Default Tests

Run with:

```bash
uv run pytest
```

Default tests must stay CPU-only and should cover:

- backend config parsing
- import isolation
- Datum-to-tensor conversion, if torch is available
- CPU reference trainer behavior, if added
- fake backend behavior
- component request/response behavior
- public client behavior
- Monarch fake-backend integration

### CPU Megatron Preflight Tests

Run with:

```bash
uv run pytest -m megatron_cpu
```

These tests should run without CUDA. They may use optional Megatron Bridge imports if installed, but must skip cleanly when the package is unavailable.

They should cover:

- import isolation
- tiny config/provider construction if supported by the installed Bridge version
- Ganker -> Megatron config mapping
- `Datum` -> CPU tensor conversion
- mocked `MegatronTrainingBackend` lifecycle
- artifact metadata generation

They should not cover:

- real Megatron forward/backward
- CUDA kernels
- distributed optimizer behavior
- checkpoint sharding correctness
- SGLang compatibility

### Modal GPU Megatron Tests

Run with:

```bash
GANKER_RUN_MEGATRON_TESTS=1 uv run pytest -m megatron
```

These tests may require:

- NVIDIA GPU
- CUDA-compatible PyTorch
- Megatron Bridge
- Megatron-Core
- suitable container image
- Hugging Face credentials for gated models, if using gated checkpoints

Initial smoke assertions:

- create run succeeds
- forward/backward returns finite loss
- optimizer step increments step counters
- save weights creates a manifest and payload
- telemetry records trainer usage through the proxy path

### Modal Tests

Modal should run the GPU-marked tests in an environment that already contains the GPU stack. Use the existing project instruction to source Modal credentials from `~/.codex/modal.env` when interacting with Modal. Modal is the primary supported path for real Megatron execution tests.

## Open Questions

- Should the first real test use a Bridge recipe or direct `AutoBridge` provider construction?
- Which smallest model/config should be the supported smoke target?
- Which Megatron Bridge config/provider calls are stable and meaningful on CPU-only machines?
- Should Megatron workers be launched by `torchrun`, NeMo-Run, Modal process orchestration, or Monarch rank actors?
- What artifact format should be the first contract with rollout?
- Should `AdamParams` remain our public optimizer shape, or should Megatron config expose more scheduler/optimizer knobs separately?

## Risks

- Megatron Bridge dependency and container requirements may be too heavy for normal developer machines.
- CPU preflight tests can give false confidence if they drift away from the GPU backend lifecycle.
- Torch distributed launch can conflict with Monarch process management if both try to own rank lifecycle.
- Checkpoint export can dominate runtime even for small smoke tests.
- SGLang may not directly consume raw Megatron checkpoint shards, requiring HF export or adapter export before rollout.
- Supported model APIs in Megatron Bridge may shift, so the adapter needs tight version pinning in the GPU environment.

## Recommended First Implementation Slice

Build M1, M2, and M3 first. That gives local CPU coverage over import isolation, config translation, tensor conversion, and adapter lifecycle. Then add a Modal-only GPU smoke for real Megatron execution. This matches Megatron Bridge's own testing philosophy: cheap no-GPU unit/preflight tests first, GPU functional tests only where the actual distributed stack is required.
