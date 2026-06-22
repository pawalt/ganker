# RFC 0003: Megatron Runtime Lifecycle

Status: Draft

## Summary

Define the lifecycle contract for running Megatron Bridge/Megatron-Core behind the Ganker training API. The central decision is that the first real GPU runtime should be an in-process Megatron runtime inside `TrainingActor`, using Megatron-Core's `get_forward_backward_func()` as the implementation primitive for `TrainingClient.forward_backward(...)`.

This RFC refines RFC 0001 and RFC 0002. RFC 0001 defines the backend boundary. RFC 0002 defines Modal as the primary GPU test harness. This RFC defines what the Megatron runtime owns, how calls transition state, and where the Tinker-shaped API maps cleanly or awkwardly onto Megatron.

## Background

Ganker's public API is intentionally shaped like:

```text
TrainingClient.forward_backward(data, loss_fn, ...)
TrainingClient.optim_step(params)
TrainingClient.save_weights(...)
TrainingClient.save_weights_and_get_sampling_client(...)
```

Megatron has a matching low-level training primitive, but not a whole Ganker runtime object. Megatron-Core exposes `get_forward_backward_func()`, which returns the correct schedule for the current pipeline-parallel configuration. The returned function accepts a `forward_step_func`, `data_iterator`, `model`, `num_microbatches`, `seq_length`, `micro_batch_size`, and related schedule options. The `forward_step_func` returns model outputs plus a loss function callback.

That is low-level enough to implement Ganker's `forward_backward(...)` shape. The missing piece is lifecycle management around Megatron initialization, model ownership, optimizer ownership, pending gradients, checkpoint/export, and actor shutdown.

References:

- Megatron-Core pipeline schedules: https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.schedules.html
- Megatron Bridge RL integration example: https://docs.nvidia.com/nemo/megatron-bridge/latest/bridge-rl-integration.html
- RFC 0001: `plans/rfc-0001-megatron-bridge-support.md`
- RFC 0002: `plans/rfc-0002-modal-megatron-gpu-testing.md`

## Goals

- Make the `forward_backward` and `optim_step` semantics explicit.
- Keep the public client API independent from Monarch and Megatron.
- Put Megatron process/global state behind one runtime boundary.
- Prefer an in-process single-GPU runtime for the first Modal smoke.
- Keep fake and CPU preflight tests useful without CUDA or Megatron installed.
- Define failure behavior before real Megatron state enters the actor process.
- Define enough artifact/export semantics for later SGLang rollout refresh.

## Non-Goals

- Do not implement multi-node or multi-rank orchestration here.
- Do not require local CPU tests to execute real Megatron training.
- Do not expose Megatron or Monarch handles to the public client.
- Do not support arbitrary user-defined Python loss functions in the first runtime.
- Do not make SGLang consume Megatron artifacts in this RFC.
- Do not solve full run recovery from partial distributed failures yet.

## Ownership Model

```text
client code
  |
  v
ServiceClient / TrainingClient
  |
  v
ProxyTransport
  |
  v
ProxyActor
  |
  v
TrainingActor
  |
  v
MegatronTrainingBackend
  |
  +-- Run registry and per-run state
  +-- Per-run mutation lock
  +-- Artifact store
  |
  v
InProcessMegatronRuntime
  |
  +-- Megatron Bridge config/provider
  +-- Megatron-Core distributed initialization
  +-- model/model chunks
  +-- optimizer
  +-- scheduler
  +-- tokenizer/config references
  +-- get_forward_backward_func() schedule
```

The client never speaks Monarch. The proxy never speaks Megatron. `TrainingActor` owns the `MegatronTrainingBackend`, and the backend owns all runtime state needed to make Megatron calls deterministic.

For the first real runtime, a `TrainingActor` should host one active Megatron run. The fake backend can keep supporting many runs, but Megatron initialization uses process-global distributed state and model-parallel groups. Supporting multiple concurrent real Megatron runs in one actor should be a separate design.

## Runtime State Machine

```text
created
  |
  | create_training_run
  v
initializing
  |
  | Megatron init, model/provider, optimizer, scheduler
  v
ready
  |
  | forward_backward succeeds
  v
gradients_pending
  |
  | optim_step succeeds
  v
ready

ready
  |
  | save_weights / export
  v
checkpointing
  |
  v
ready

any state
  |
  | unrecoverable runtime failure
  v
failed

ready / failed
  |
  | actor shutdown
  v
closed
```

State transitions must be updated only after the runtime operation succeeds. Validation failures must leave the state unchanged.

## Public API Semantics

### `create_training_run`

Creates and initializes the Megatron runtime for a run.

Responsibilities:

- Validate `base_model`, `tuning_mode`, and LoRA settings.
- Build or load a Megatron Bridge provider.
- Initialize Megatron distributed state inside the actor process.
- Construct the model or model chunks.
- Construct optimizer and scheduler.
- Build the forward/backward schedule with `get_forward_backward_func()`.
- Return a `TrainingRun` only after the runtime reaches `ready`.

Initial restriction:

- One active real Megatron run per `TrainingActor`.

### `forward_backward`

Runs Megatron forward and backward for one Ganker batch.

Mapping:

```text
TrainingClient.forward_backward(...)
  |
  v
MegatronTrainingBackend.forward_backward(...)
  |
  v
datums_to_tensor_batch(...)
  |
  v
InProcessMegatronRuntime.forward_backward(...)
  |
  v
forward_backward = get_forward_backward_func()
forward_backward(
    forward_step_func=ganker_forward_step,
    data_iterator=microbatch_iterator,
    model=model_or_model_chunks,
    num_microbatches=num_microbatches,
    seq_length=sequence_length,
    micro_batch_size=micro_batch_size,
    forward_only=False,
)
```

`forward_backward(...)` must not call `optimizer.step()`. It accumulates gradients in the Megatron model/optimizer state. A successful call moves the run to `gradients_pending` and increments `gradient_version`.

Multiple `forward_backward(...)` calls before `optim_step(...)` may be supported as gradient accumulation, but the first implementation should make this explicit through config. If accumulation is not enabled, a second `forward_backward(...)` while gradients are pending should fail with a clear `InvalidRequestError`.

### `optim_step`

Consumes pending gradients and updates model weights.

Mapping:

```text
TrainingClient.optim_step(AdamParams(...))
  |
  v
Megatron optimizer.step()
Megatron scheduler.step(...)
zero / clear gradients according to Megatron optimizer behavior
```

`optim_step(...)` requires `gradients_pending`. Calling it in `ready` should fail clearly rather than silently stepping with no gradients.

Megatron optimizer hyperparameters are usually configured at optimizer construction time. The first runtime should treat `AdamParams` as follows:

- `learning_rate` may update optimizer parameter-group learning rates when Megatron exposes a safe path.
- `beta1`, `beta2`, `eps`, and `weight_decay` must match the run's initialized optimizer config.
- Unsupported per-step changes should fail clearly.

A successful optimizer step moves the run back to `ready`, increments `optimizer_step`, and increments `checkpoint_version`.

### `save_weights`

Writes a stable artifact for the current model weights.

The first runtime should allow `save_weights(...)` only from `ready`. If gradients are pending, the model weights have not yet changed, but the run is in a transient training state. Failing clearly is less surprising than silently saving pre-step weights.

Artifact metadata should include:

```text
artifact_format
backend
base_model
tuning_mode
lora_rank
gradient_version
optimizer_step
checkpoint_version
tensor_model_parallel_size
pipeline_model_parallel_size
micro_batch_size
global_batch_size
sequence_length
tokenizer_path
config_path
payload_path
manifest_path
```

Raw Megatron checkpoints are acceptable for the first real training smoke. SGLang rollout will likely need a later export to HF/safetensors, LoRA adapter, or merged HF format.

### `save_weights_and_get_sampling_client`

This is a client-side composition:

```text
save_weights
  |
  v
refresh_weights
  |
  v
SamplingClient
```

The training side must block until the artifact is fully written. The rollout side may reject unsupported Megatron artifact formats until SGLang export is implemented.

## Runtime Interface

The current `MegatronRuntime` protocol should evolve toward explicit lifecycle state:

```python
class MegatronRuntime(Protocol):
    def create_run(...) -> MegatronRunHandle:
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

    def save_weights(...) -> dict[str, Any]:
        ...

    def shutdown(self, *, handle: MegatronRunHandle) -> None:
        ...
```

`MegatronRunHandle` should become the only object that stores Megatron-owned runtime references:

```text
bridge/provider
megatron config container
model or model chunks
optimizer
scheduler
tokenizer
forward_backward schedule
distributed/context metadata
```

The backend should store Ganker-owned state:

```text
run_id
base_model
tuning_mode
lora_rank
state
gradient_version
optimizer_step
checkpoint_version
artifact metadata
```

## Loss and Batch Contract

The first supported loss should remain:

```text
loss_fn = "cross_entropy"
```

Required `Datum` fields:

```python
Datum(
    model_input=ModelInput.from_ints([...]),
    loss_fn_inputs={
        "target_tokens": TensorData.from_ints([...]),
        "weights": TensorData.from_floats([...]),
    },
)
```

The runtime adapter builds microbatches from these tensors. The Megatron `forward_step_func` should:

1. Pull the next microbatch.
2. Move tensors to the runtime device.
3. Call the Megatron model with `input_ids` and any required attention/position inputs.
4. Return outputs and a loss callback.
5. Compute cross entropy against `target_tokens`, applying `weights` as the loss mask.

Future losses such as DPO, PPO, GRPO, KL penalties, or custom reward losses should be added as named backend-supported loss functions, not arbitrary Python callables passed over the public client boundary.

## Concurrency Rules

Mutating calls for a run must be serialized:

```text
create_training_run
forward_backward
optim_step
save_weights
shutdown
```

The backend should use a per-run lock even if the current actor implementation appears sequential. This avoids accidental races if Monarch endpoint behavior changes or if async endpoints are added later.

Concurrent read-only telemetry is fine. Rollout sampling happens in `RolloutActor` and should not touch live Megatron training state.

## Failure Semantics

Validation errors:

- Return a typed request error.
- Do not touch Megatron runtime state.
- Do not update Ganker run state.

`forward_backward` runtime failure:

- Treat as unrecoverable for the initial implementation.
- Mark the run `failed`.
- Do not increment `gradient_version`.
- Require a new run or future checkpoint restore.

`optim_step` runtime failure:

- Treat as unrecoverable for the initial implementation.
- Mark the run `failed`.
- Do not increment `optimizer_step` or `checkpoint_version`.

`save_weights` failure:

- If model state was not modified, keep the run in `ready`.
- Do not increment `checkpoint_version`.
- Allow retry.

Actor shutdown:

- Call runtime `shutdown(...)` for any live handle.
- Best-effort cleanup is acceptable locally.
- Modal teardown can rely on process cleanup after the runtime gets a chance to flush artifacts.

## Testing Strategy

### Default Local Tests

Run with:

```bash
uv run pytest
```

Coverage:

- fake backend lifecycle
- public client contracts
- component behavior
- Monarch actor integration with fake backends
- import isolation
- state machine tests using fake runtime objects

These tests must not require CUDA, Megatron Bridge, Megatron-Core, SGLang, model weights, or Modal.

### CPU Megatron Preflight

Run with:

```bash
uv run pytest -m megatron_cpu
```

Coverage:

- config parsing
- optional Megatron Bridge import checks
- `Datum` to torch CPU tensor conversion
- mocked `InProcessMegatronRuntime` lifecycle
- failure semantics and state transitions

This lane should not claim to validate real Megatron forward/backward correctness.

### Modal Single-GPU Smoke

Run inside Modal with:

```bash
source ~/.codex/modal.env
modal run modal_apps/megatron_smoke.py --mode ganker
```

Coverage:

- initialize real Megatron Bridge runtime on one GPU
- run one `forward_backward(...)` through `get_forward_backward_func()`
- run one `optim_step(...)`
- save one artifact
- exercise the public `ServiceClient -> ProxyActor -> TrainingActor -> MegatronTrainingBackend` path

### Later Multi-Rank Smoke

Multi-GPU/multi-rank execution may require one of:

- Monarch rank actors
- `torchrun`
- NeMo-Run
- Modal process orchestration

That should be added only after the single-GPU in-process lifecycle is proven.

## Implementation Milestones

### M1: State Machine in Backend

- Add explicit Megatron run states.
- Add per-run mutation lock.
- Enforce one active real Megatron run per actor.
- Enforce invalid transitions:
  - `optim_step` before `forward_backward`
  - `save_weights` while gradients are pending
  - second `forward_backward` while gradients are pending when accumulation is disabled
- Unit test state transitions with a fake runtime.

### M2: In-Process Runtime Handle

- Expand `MegatronRunHandle` to hold model, optimizer, scheduler, config, tokenizer, and schedule references.
- Keep all heavy imports inside `ganker.backends.megatron`.
- Add `shutdown(...)`.
- Keep import isolation tests passing.

### M3: Forward Step Adapter

- Implement a first `cross_entropy` `forward_step_func`.
- Build a microbatch iterator from `MegatronTensorBatch`.
- Call `get_forward_backward_func()` from Megatron-Core.
- Return `ForwardBackwardOutput` with finite loss and metrics.
- Unit test the adapter with mocked model/runtime objects.

### M4: Optimizer Step Adapter

- Wire Megatron optimizer and scheduler calls.
- Validate `AdamParams` compatibility with initialized optimizer config.
- Return optimizer step and checkpoint version only after success.

### M5: Checkpoint and Artifact Adapter

- Save raw Megatron checkpoint or minimal runtime payload.
- Write Ganker artifact manifest through `FilesystemArtifactStore`.
- Ensure artifact metadata is sufficient for later rollout export.

### M6: Modal Ganker Smoke

- Update Modal smoke to use the in-process runtime first.
- Run one public-client training path on GPU.
- Keep subprocess/torchrun as fallback or later multi-rank work.

## Relation to RFC 0002

RFC 0002 originally proposed a subprocess runtime first. This RFC changes that ordering for the single-GPU path:

```text
first:  InProcessMegatronRuntime inside TrainingActor
later:  TorchrunMegatronRuntime or rank actor runtime for multi-GPU
```

The reason is that `get_forward_backward_func()` provides the low-level primitive needed for the Tinker-shaped API. A subprocess runtime may still be useful for multi-rank execution, isolation, or reproducing Megatron launch scripts, but it should not be the first implementation target for the single-GPU Modal smoke.

## Open Questions

- Which Bridge initialization path is most stable for tiny local smoke models: direct provider construction, `AutoBridge.from_hf_pretrained`, or a Bridge recipe?
- Can the first real smoke avoid Hugging Face network calls entirely with a local tiny config?
- Should gradient accumulation be enabled in the first runtime or rejected until explicit config exists?
- Should optimizer hyperparameters move from `optim_step(...)` into `create_training_run(...)` before real Megatron support?
- What is the first rollout-consumable export format: LoRA adapter, merged HF, or full HF/safetensors?
- How much runtime cleanup is possible after Megatron distributed initialization inside a long-lived actor process?
