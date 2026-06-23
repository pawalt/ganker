# RFC 0010: Model-Parallel SFT for Larger Models

Status: Draft

## Summary

Extend the Modal Megatron Bridge SFT path from data-parallel-only training to
model-parallel training. The first target is tensor parallelism (`TP > 1`,
`PP = 1`), followed by pipeline parallelism (`PP > 1`) once the data iterator,
microbatching, and loss handling are Megatron-native enough.

The goal is to make larger models fit by sharding one logical model across
multiple GPUs. Data parallelism can increase throughput, but it cannot make a
single oversized model fit. Tensor parallelism and pipeline parallelism are the
required primitives for that.

```text
world_size = n_nodes * gpus_per_node
model_parallel_size = tensor_parallel_size * pipeline_parallel_size
data_parallel_size = world_size / model_parallel_size

Example: 2 nodes * 8 GPUs = 16 ranks

TP=1 PP=1 -> DP=16   current path; full model on every GPU
TP=2 PP=1 -> DP=8    each model replica spans 2 GPUs
TP=4 PP=1 -> DP=4    each model replica spans 4 GPUs
TP=2 PP=2 -> DP=4    each model replica spans 4 GPUs split by layer and tensor
TP=8 PP=2 -> DP=1    one model replica spans all 16 GPUs
```

## Current State

The current distributed SFT implementation validates distributed shape in
`src/ganker/distributed/torchrun.py`, launches a Modal clustered `torchrun`
trainer, and uses `MegatronTrainingBackend` inside every rank.

Megatron Bridge already receives these fields:

```python
provider.tensor_model_parallel_size = config.tensor_model_parallel_size
provider.pipeline_model_parallel_size = config.pipeline_model_parallel_size
```

Megatron distributed initialization also already calls:

```python
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=tensor_parallel,
    pipeline_model_parallel_size=pipeline_parallel,
)
```

The hard stop is in the Ganker wrapper:

```python
distributed.require_dp_only()
```

That currently rejects:

- `tensor_model_parallel_size != 1`
- `pipeline_model_parallel_size != 1`
- `grad_accum_steps != 1`

The training loop also selects data with global rank/world size:

```python
select_data_parallel_item(..., data_parallel_rank=rank, data_parallel_size=world_size)
```

That is correct only for DP-only training. With TP/PP, all ranks in the same
model-parallel group must process the same microbatch.

## Goals

- Support `tensor_model_parallel_size > 1` for Qwen LoRA SFT on Modal.
- Keep Modal clustered `torchrun` as the primary GPU execution path.
- Keep the public Ganker API unchanged.
- Use Megatron's model-parallel groups for rank math, not global rank math.
- Support multiple data-parallel replicas of model-parallel groups.
- Preserve existing DP-only behavior and tests.
- Add CPU-only tests for config/rank/data scheduling helpers.
- Add GPU smoke tests that prove real TP forward/backward/save works.
- Define the PP work clearly enough that it can follow TP without redesign.

## Non-Goals

- Do not implement custom tensor-parallel collectives in Ganker.
- Do not implement custom pipeline activation send/recv in Ganker.
- Do not support arbitrary user-defined Python losses in the first TP/PP path.
- Do not make the first model-parallel milestone elastic or fault tolerant.
- Do not require local tests to import Megatron Bridge or run CUDA.
- Do not solve optimizer state checkpoint resume in this RFC.
- Do not add expert parallelism, context parallelism, or sequence parallelism as
  first-class Ganker knobs yet.

## Design Principles

Ganker should own:

- public request/response contracts;
- run lifecycle state;
- dataset-to-Datum conversion;
- high-level step orchestration;
- artifact metadata;
- Modal launch shape and validation.

Megatron should own:

- tensor-parallel collectives;
- pipeline stage scheduling;
- data-parallel gradient synchronization;
- model-parallel rank groups;
- forward/backward schedule selection.

The key rule is that Ganker submits one logical training operation, and every
Megatron rank enters the same Megatron schedule. Ganker should not schedule work
rank-by-rank.

## Rank Model

For model-parallel training, there are three useful rank spaces:

```text
global rank
  Unique PyTorch rank in [0, world_size).

model-parallel rank
  Rank within TP/PP groups. These ranks form one logical model replica.

data-parallel rank
  Rank of the model replica among replicas that train on different data.
```

Batch sharding must use data-parallel rank, not global rank.

Current behavior:

```text
rank 0 gets batch item A
rank 1 gets batch item B
rank 2 gets batch item C
...
```

Required TP behavior for `TP=2, PP=1`:

```text
global ranks 0,1 -> same model replica -> same batch item A
global ranks 2,3 -> same model replica -> same batch item B
global ranks 4,5 -> same model replica -> same batch item C
...
```

After Megatron initialization, the training entrypoint should get:

```python
from megatron.core import parallel_state

dp_rank = parallel_state.get_data_parallel_rank()
dp_size = parallel_state.get_data_parallel_world_size()
tp_rank = parallel_state.get_tensor_model_parallel_rank()
pp_rank = parallel_state.get_pipeline_model_parallel_rank()
```

The batch selector should use `dp_rank` and `dp_size`.

## Microbatching and Gradient Accumulation

The current runtime always calls Megatron schedule with:

```python
num_microbatches = 1
```

and the backend state machine rejects a second `forward_backward` before
`optim_step`.

For larger models, Ganker needs one of two accumulation models.

### Preferred Model: One Ganker Call, Many Megatron Microbatches

`TrainingClient.forward_backward(data=[...])` should accept enough datums for
all local microbatches in the current logical step. The backend builds a
microbatch iterator and calls:

```python
forward_backward(
    forward_step_func=...,
    data_iterator=microbatch_iterator,
    model=model_chunks,
    num_microbatches=grad_accum_steps,
    seq_length=sequence_length,
    micro_batch_size=micro_batch_size,
    forward_only=False,
)
```

Then `TrainingClient.optim_step(...)` consumes the accumulated gradients.

This keeps the existing lifecycle:

```text
ready -> forward_backward -> gradients_pending -> optim_step -> ready
```

### Alternative Model: Multiple Ganker Calls Per Optimizer Step

Allow:

```text
forward_backward(microbatch 1)
forward_backward(microbatch 2)
...
optim_step()
```

This maps naturally onto the public API, but it complicates distributed
coordination because every rank must make the same number of calls before step.
It also makes pipeline scheduling less efficient because Megatron sees many
one-microbatch schedules rather than one multi-microbatch schedule.

Decision: implement the preferred model first.

## Tensor Parallel Milestone

TP should be the first implementation target because it exercises model
sharding without requiring stage-specific pipeline loss behavior.

### Required Changes

1. Replace DP-only validation with model-parallel validation:

```text
world_size % (tp * pp) == 0
global_batch_size % (micro_batch_size * data_parallel_size) == 0
```

For milestone 1:

```text
tp > 1
pp = 1
grad_accum_steps = 1 initially
```

2. Add rank helpers in `src/ganker/distributed/torchrun.py` or a new helper:

```python
@dataclass(frozen=True)
class MegatronRankInfo:
    global_rank: int
    world_size: int
    data_parallel_rank: int
    data_parallel_size: int
    tensor_model_parallel_rank: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_rank: int
    pipeline_model_parallel_size: int
```

3. Update `run_qwen_lora_sft(...)` to select batches by Megatron DP rank.

4. Ensure all TP ranks in a model replica enter:

```text
create_training_run
forward_backward
optim_step
save_weights
close
```

with the same call order.

5. Preserve shared artifact behavior:

- all ranks participate in Bridge save/export if Bridge requires collectives;
- only a stable writer rank writes shared Modal Volume artifacts;
- non-writer ranks write to rank-local temp paths when needed.

For TP-only, writer rank should be:

```text
data_parallel_rank == 0
tensor_model_parallel_rank == 0
pipeline_model_parallel_rank == 0
```

or simply global rank 0 for the first TP smoke if Bridge export proves only one
rank writes HF adapter files.

6. Add tests:

- config validation for TP shapes;
- DP rank batch-selection behavior;
- CLI arg construction with TP fields;
- rejection of invalid `world_size % model_parallel_size`.

7. Add Modal smoke:

```bash
GANKER_QWEN_SFT_MULTINODE_NODES=1 \
GANKER_QWEN_SFT_MULTINODE_GPU=H100:8 \
uv run modal run modal_apps/qwen_sft_multinode/sft.py \
  --mode qwen-lora-sft \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1 \
  --micro-batch-size 1 \
  --global-batch-size 4 \
  --max-steps 1 \
  --sequence-length 128
```

Success criteria:

- Megatron initializes TP groups.
- Qwen loads through Bridge.
- `get_forward_backward_func()` runs.
- one optimizer step completes.
- LoRA adapter export succeeds.
- result JSON includes TP/DP rank metadata and one loss value.

## Pipeline Parallel Milestone

PP should follow TP because it adds stage-specific execution behavior.

The current forward step is simple:

```python
output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
return output_tensor, loss_func
```

That is enough for full-model or TP-only execution. With PP, only some ranks
own embeddings, some own middle layers, and the last pipeline stage owns final
loss computation. Megatron's schedule can handle the stage orchestration, but
Ganker's `forward_step_func` and data iterator must be compatible with that
schedule.

### Required Changes

1. Make the forward step pipeline-aware.

The forward step should branch on:

```python
parallel_state.is_pipeline_first_stage()
parallel_state.is_pipeline_last_stage()
```

Likely behavior:

- first stage consumes `tokens`, `position_ids`, and `attention_mask`;
- middle stages consume activations from the schedule;
- last stage computes shifted-token cross entropy with `labels` and `loss_mask`;
- non-last stages return output tensors without loss reduction.

2. Verify Bridge provider model chunks.

For PP, `provider.provide_distributed_model(...)` may return a list of model
chunks. The runtime must pass `model` in the shape Megatron schedule expects.

3. Increase `num_microbatches`.

PP with one microbatch is valid but inefficient and may expose more schedule
edge cases. The first PP smoke should use:

```text
grad_accum_steps >= pipeline_model_parallel_size
```

or at least more than one microbatch.

4. Preserve loss reporting.

Only last pipeline stage has the loss. The result writer should average loss
over data-parallel replicas and handle non-loss ranks cleanly.

5. Add Modal smoke:

```bash
GANKER_QWEN_SFT_MULTINODE_NODES=1 \
GANKER_QWEN_SFT_MULTINODE_GPU=H100:8 \
uv run modal run modal_apps/qwen_sft_multinode/sft.py \
  --mode qwen-lora-sft \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 2 \
  --micro-batch-size 1 \
  --global-batch-size 8 \
  --max-steps 1 \
  --sequence-length 128
```

Success criteria:

- Megatron initializes TP and PP groups.
- all pipeline stages enter the schedule;
- loss is emitted from last stage and reduced for reporting;
- optimizer step completes;
- artifact export succeeds.

## Serving Larger Models

Training TP/PP and serving TP are separate but related.

SGLang rollout does not need pipeline parallelism for the first larger-model
serving milestone. It should support tensor parallel serving:

```python
SGLangBackendConfig(
    tensor_parallel_size=2,
    launch_server=True,
    enable_lora=True,
)
```

The artifact contract can remain:

```text
base model path + HF/PEFT LoRA adapter path
```

For LoRA training, rollout serving should load the base HF model with SGLang TP
and then load the exported adapter.

## API and Config Surface

The public client API does not need new methods. Parallelism belongs in backend
configuration and Modal job config:

```python
MegatronBackendConfig(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    micro_batch_size=1,
    global_batch_size=4,
)
```

Modal CLIs should keep exposing:

```text
--tensor-model-parallel-size
--pipeline-model-parallel-size
--micro-batch-size
--global-batch-size
```

The README should stop saying the path is strictly DP-only after TP milestone 1
lands, but it should still document PP as experimental until the PP smoke is
green.

## Implementation Plan

### Phase 1: Validation and Rank Plumbing

- Replace `require_dp_only()` calls in Qwen and code-SFT paths with new
  `require_supported_model_parallel(...)` validation.
- Keep `PP=1` for this phase.
- Add unit tests for TP shape validation.
- Add a helper that can be tested without Megatron imports and another runtime
  helper that reads Megatron `parallel_state` when available.
- Add rank metadata to result JSON.

### Phase 2: TP Forward/Backward Smoke

- Update batch selection to use DP rank.
- Run Modal `TP=2, PP=1` one-step Qwen LoRA SFT.
- Fix checkpoint/export writer-rank assumptions if Bridge save behavior differs
  from DP-only.
- Add a documented command and expected result shape.

### Phase 3: Microbatch Iterator

- Teach the backend to pass multiple microbatches to Megatron schedule.
- Keep lifecycle as one Ganker `forward_backward` per optimizer step.
- Add CPU tests for datum-to-microbatch grouping.
- Add Modal smoke with `grad_accum_steps > 1`.

### Phase 4: PP Forward Step

- Make `_core_forward_step_func` or a Bridge-specific forward step compatible
  with PP.
- Add PP rank-aware loss reporting.
- Run `TP=2, PP=2` one-step Qwen LoRA SFT.

### Phase 5: Larger Model Proof

- Select a model that does not fit in the current DP-only shape but should fit
  with TP or TP+PP.
- Run a short LoRA SFT smoke.
- Serve with SGLang TP and load the exported adapter.

Candidate commands should be written into README only after a smoke has passed.

## Testing Strategy

CPU/local tests:

- distributed config math;
- rank-to-data-index mapping;
- CLI arg construction;
- microbatch grouping;
- fake runtime lifecycle with `grad_accum_steps > 1`;
- artifact writer-rank selection logic.

Modal GPU tests:

- `torchrun-env` still reports all ranks;
- `nccl-smoke` still passes;
- `qwen-lora-sft TP=2 PP=1` one step;
- `qwen-lora-sft TP=2 PP=1 grad_accum_steps>1` one step;
- `qwen-lora-sft TP=2 PP=2` one step;
- SGLang TP serving loads the exported LoRA adapter.

Regression tests:

- DP-only commands still work.
- Invalid `world_size % model_parallel_size` fails before launching work.
- Invalid `global_batch_size` divisibility fails before launching work.
- Non-writer ranks do not fail when Bridge export does not materialize
  adapter files locally.

## Open Questions

1. Does Bridge's LoRA HF adapter export require every TP/PP rank to call
   `save_hf_adapter`, or can only writer ranks call it safely?

2. Does Bridge expose a provider-native forward step for causal LM training that
   already handles PP cleanly, or should Ganker maintain the forward step?

3. How should optimizer state be saved for future resume support under TP/PP?

4. What is the first larger target model after Qwen3 0.6B? The answer should be
   chosen based on Modal GPU availability and expected memory footprint.

5. Should the public API expose an explicit gradient-accumulation concept, or is
   it enough to keep accumulation fully inside backend/job config?

## Acceptance Criteria

TP milestone accepted when:

- `TP=2, PP=1` Qwen LoRA SFT completes one Modal GPU step;
- the result JSON reports correct `tp`, `pp`, `dp`, and rank metadata;
- loss is finite;
- optimizer step and LoRA export succeed;
- DP-only tests and Modal smokes still pass.

PP milestone accepted when:

- `TP=2, PP=2` Qwen LoRA SFT completes one Modal GPU step;
- multiple microbatches run through Megatron schedule;
- loss reporting is correct on rank zero;
- optimizer step and LoRA export succeed.

Larger-model milestone accepted when:

- a model larger than the DP-only Qwen3 0.6B path trains for at least one
  forward/backward/optimizer step with model parallelism;
- the exported adapter can be loaded by SGLang TP serving and sampled from.
