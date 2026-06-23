# RFC 0009: Modal Multi-Node SFT with Megatron Bridge

Status: Milestone 1 implemented

## Summary

Extend the Qwen SFT path from a single Modal GPU worker to a Modal clustered
multi-node trainer. The controller and rollout/inference services remain normal
Modal functions, but the trainer becomes a gang-scheduled `torchrun` job that
initializes one Megatron process per GPU.

Milestone 1 is implemented in:

- `src/ganker/distributed/torchrun.py`
- `modal_apps/qwen_sft_multinode/infra.py`
- `modal_apps/qwen_sft_multinode/train_entry.py`
- `modal_apps/qwen_sft_multinode/sft.py`
- `modal_apps/qwen_sft_multinode/compare_hf.py`

The important design point is that multi-node forward/backward is not scheduled
by Ganker rank-by-rank. Ganker submits one logical training step. Every Megatron
rank must enter the same `get_forward_backward_func()` schedule at the same
time, and Megatron/PyTorch collectives move activations, gradients, and reduced
losses across ranks.

```text
client / script
    |
    | ServiceClient / SFT example
    v
+-----------------------------+
| Modal controller/job runner |
+--------------+--------------+
               |
               | one logical trainer command:
               | create, forward_backward, optim_step, save
               v
============================================================================
        Modal clustered trainer function, gang scheduled, RDMA enabled

 node 0                         node 1                         node N
+------------------+           +------------------+           +------------------+
| torchrun         |           | torchrun         |           | torchrun         |
| local ranks 0..7 |<--------->| local ranks 8..15|<--------->| local ranks ...  |
+--------+---------+  NCCL     +--------+---------+  NCCL     +--------+---------+
         |                              |                              |
         v                              v                              v
 Megatron process group          Megatron process group          Megatron process group
 TP / PP / DP groups             TP / PP / DP groups             TP / PP / DP groups

Artifacts:
  Modal Volume -> HF/PEFT LoRA adapter -> SGLang rollout refresh
```

## References

- Modal multi-node clusters documentation:
  https://modal.com/docs/guide/multi-node-training
- Modal multinode training guide example:
  https://github.com/modal-labs/multinode-training-guide/blob/main/benchmark/modal_train.py
- Megatron-Core pipeline schedule API:
  https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.pipeline_parallel.schedules.html
- RFC 0003: `plans/rfc-0003-megatron-runtime-lifecycle.md`
- RFC 0008: `plans/rfc-0008-modal-distributed-controller-orchestration.md`

## Current State

The implemented Qwen path is single-rank:

```text
modal_apps/qwen_sft/sft.py or compare_hf.py
  -> infra.run_sft_job / infra.run_training_job
  -> controller-local ServiceClient
  -> Monarch proxy
  -> one TrainingActor on one Modal GPU worker
  -> MegatronTrainingBackend(runtime_kind="bridge")
  -> InstalledMegatronBridgeRuntime
  -> get_forward_backward_func()
```

`bridge_training_config(...)` currently hard-codes:

```json
{
  "tensor_model_parallel_size": 1,
  "pipeline_model_parallel_size": 1,
  "micro_batch_size": 1,
  "global_batch_size": 1
}
```

This is enough for Qwen3 0.6B LoRA plumbing and the HF Trainer loss-curve
comparison, but it does not prove multi-node process-group construction,
distributed data loading, gradient synchronization, or multi-rank checkpoint
export.

## Goals

- Run Qwen LoRA SFT on a Modal clustered multi-node trainer.
- Keep Modal as the primary GPU test surface.
- Use `modal.experimental.clustered(..., rdma=True)` for gang-scheduled trainer
  nodes and RDMA-capable inter-node communication.
- Use `torchrun` inside the clustered function to launch one process per GPU.
- Let Megatron own distributed forward/backward scheduling after process-group
  initialization.
- Keep controller/proxy/rollout code outside the Megatron collective hot path.
- Keep SGLang rollout as a separate service consuming exported HF/PEFT adapter
  artifacts.
- Preserve CPU-only local tests by testing command construction, config
  validation, and fake process-group behavior without CUDA.
- Reuse the loss-curve comparison as a validation tool after the distributed
  path is stable.

## Non-Goals

- Do not implement custom all-reduce, activation passing, or pipeline schedule
  logic in Ganker.
- Do not use Monarch actor calls for every Megatron rank or every microbatch.
- Do not make the first milestone elastic. If any rank dies, the trainer job
  fails and the run is retried or marked failed.
- Do not support arbitrary user Python losses in the distributed trainer.
- Do not solve cross-cluster training or heterogeneous NIC placement.
- Do not require local unit tests to download Qwen, run Megatron, or start
  `torchrun`.

## Modal Cluster Mechanics

Modal multi-node clusters are a different primitive from plain `i6pn=True`
functions.

Plain i6pn functions are appropriate for:

- controller;
- proxy;
- SGLang rollout;
- single-node trainer workers;
- lightweight role-to-role Monarch calls.

Clustered functions are appropriate when a job needs:

- all trainer nodes scheduled together;
- fast direct communication between nodes;
- RDMA scale-out networking;
- a stable rank/IP list for `torchrun`.

Modal's clustered function call broadcasts the same function input to every
container. Each container gets a `cluster_info.rank`, a sorted list of peer IPs,
and only rank zero returns the result to the caller. The Modal docs also note
that clustered functions are GPU-only and, as of May 31, 2026, must request the
full number of devices per node. That means the production shape should request
full-node SKUs such as:

```python
@app.function(
    gpu="H100:8",
    timeout=60 * 60 * 24,
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=n_nodes, rdma=True)
def run_multinode_trainer(...):
    ...
```

The trainer function should then launch local GPU processes with:

```text
torchrun
  --nnodes=<n_nodes>
  --nproc-per-node=<gpus_per_node>
  --node-rank=<cluster_info.rank>
  --master-addr=<cluster_info.container_ips[0]>
  --master-port=<port>
  modal_apps/qwen_sft_multinode/train_entry.py
```

This mirrors the Modal guide's `torch.distributed.run` example. Modal handles
node co-scheduling and gives us the addresses; `torchrun` handles PyTorch rank
environment variables; Megatron consumes those variables when initializing
distributed state.

## Where Forward/Backward Is Scheduled

There are three scheduling layers. They should not be mixed up.

### 1. Modal Schedules Containers

Modal decides when a group of nodes starts. It does not know about model layers,
microbatches, tensors, losses, or optimizer steps.

```text
run_multinode_trainer.remote(config)
  -> Modal schedules all trainer nodes together
  -> each container runs the same Python function
```

### 2. Torchrun Schedules Processes and Ranks

Inside each Modal container, `torchrun` starts local GPU processes and assigns
global ranks:

```text
node 0: local_rank 0..7 -> global_rank 0..7
node 1: local_rank 0..7 -> global_rank 8..15
...
```

PyTorch distributed initializes:

```text
WORLD_SIZE = n_nodes * gpus_per_node
RANK       = global rank
LOCAL_RANK = GPU index on this node
MASTER_ADDR = cluster_info.container_ips[0]
MASTER_PORT = chosen port
```

### 3. Megatron Schedules Model Work

After every process initializes PyTorch distributed and Megatron model-parallel
groups, Megatron decides what each rank does.

```text
Ganker logical step:
  forward_backward(batch)

Every rank enters:
  get_forward_backward_func()(
      forward_step_func=...,
      data_iterator=...,
      model=...,
      num_microbatches=...,
      seq_length=...,
      micro_batch_size=...,
      forward_only=False,
  )
```

`get_forward_backward_func()` selects the correct schedule from Megatron's
current `parallel_state`:

- no pipeline parallelism: each rank runs local forward/backward and uses
  tensor/data-parallel collectives as configured;
- non-interleaved pipeline parallelism: Megatron runs a 1F1B schedule and
  sends activations forward and gradients backward between pipeline stages;
- interleaved pipeline parallelism: Megatron uses model chunks and an
  interleaved 1F1B schedule.

Ganker should not manually send activations or gradients. Ganker's job is to
turn public API calls into synchronized commands that cause every rank to enter
the Megatron schedule together.

## Parallelism Modes

The world size is decomposed as:

```text
world_size = DP * TP * PP * CP * EP
```

The first implementation should only expose:

```text
world_size = DP * TP * PP
```

Context parallelism and expert parallelism should wait.

### Data Parallelism

Each data-parallel replica owns a full copy of the model shard implied by
TP/PP. Different DP replicas consume different samples.

```text
DP group 0: sample batch shard A
DP group 1: sample batch shard B
...
backward -> gradient all-reduce / reduce-scatter across DP group
```

For Qwen3 0.6B LoRA, data parallelism is the most practical first multi-node
target because the model already fits on one GPU. It primarily proves
multi-node launch, dataset sharding, gradient synchronization, and checkpoint
export.

### Tensor Parallelism

Tensor parallelism splits layer math across ranks. Ranks in a TP group usually
consume the same tokens, then communicate partial results through Megatron's
tensor-parallel collectives.

This is useful for larger models that do not fit on one GPU or for throughput
on large layers, but it is more sensitive to network and kernel compatibility.
It is not the first required shape for Qwen3 0.6B.

### Pipeline Parallelism

Pipeline parallelism splits layers into stages. Stage 0 consumes tokens. Later
stages receive activations from the previous stage. The last stage computes
loss. Backward sends gradients in reverse.

For a non-interleaved 1F1B schedule with four pipeline stages:

```text
time ->

stage 0: F0 F1 F2 F3 B0 B1 B2 B3
stage 1:    F0 F1 F2 F3 B0 B1 B2 B3
stage 2:       F0 F1 F2 F3 B0 B1 B2 B3
stage 3:          F0 F1 F2 F3 B0 B1 B2 B3

F0 = forward for microbatch 0
B0 = backward for microbatch 0
```

Megatron handles the warmup, steady-state, and cooldown schedule. Ganker only
chooses `num_microbatches`, `micro_batch_size`, and `pipeline_model_parallel_size`.

## Ganker API Mapping

The public user-facing API can stay:

```python
summary = run_sft(
    client,
    base_model="Qwen/Qwen3-0.6B",
    dataset=batches,
    tuning="lora",
    lora_rank=8,
    learning_rate=1e-4,
    max_steps=20,
)
```

Internally, a distributed trainer must treat each call as a collective command.

```text
TrainingClient.forward_backward(batch)
  -> controller/trainer command: FORWARD_BACKWARD(step_id, batch_ref)
  -> rank 0 broadcasts command metadata
  -> every rank loads its local input shard
  -> every rank enters get_forward_backward_func()
  -> only loss-owning ranks produce losses
  -> rank 0 reduces/logs scalar loss
  -> response returns to Ganker client

TrainingClient.optim_step(...)
  -> command: OPTIM_STEP(step_id, lr)
  -> every rank calls optimizer.step()
  -> rank 0 returns optimizer/checkpoint version

TrainingClient.save_weights(...)
  -> command: SAVE_WEIGHTS(checkpoint_version)
  -> every rank participates if Bridge export/checkpoint requires it
  -> rank 0 writes manifest and commits Modal Volume
```

The dangerous anti-pattern is:

```text
controller -> call rank 0 forward_backward only
```

That will hang or fail for distributed Megatron because collectives require all
ranks to enter matching operations.

## Recommended Architecture

Use a staged design.

### Milestone 1: Clustered Whole-Job SFT

Run the entire SFT loop inside one clustered Modal call.

```text
modal_apps/qwen_sft_multinode/infra.py
  -> image, volumes, clustered trainer function

modal_apps/qwen_sft_multinode/train_entry.py
  -> torchrun child script
  -> initializes Megatron Bridge on each rank
  -> loads materialized SFT JSONL
  -> shards data by DP rank
  -> runs N SFT steps
  -> exports LoRA adapter
  -> writes rank-0 JSON result

modal_apps/qwen_sft_multinode/sft.py
  -> local entrypoint
  -> materialize dataset
  -> run clustered trainer
  -> optionally start/refresh SGLang with exported adapter
```

This does not preserve step-by-step remote `TrainingClient` calls, but it proves
the core hard part: multi-node Megatron process groups and SFT correctness.

The return value is rank 0's JSON:

```json
{
  "ok": true,
  "mode": "qwen-multinode-sft",
  "n_nodes": 2,
  "gpus_per_node": 8,
  "world_size": 16,
  "dp": 16,
  "tp": 1,
  "pp": 1,
  "steps": 20,
  "losses": [...],
  "artifact_format": "hf-lora-adapter"
}
```

This is the right first milestone because Modal clustered functions are
single-call training jobs by design, and Modal only returns rank zero's result.

### Milestone 2: Persistent Clustered Trainer Service

If we need Tinker-style interactive calls across a long-lived trainer cluster,
the clustered function becomes a persistent trainer service:

```text
controller
    |
    | TCP/gRPC/Monarch command to rank 0 trainer coordinator
    v
clustered trainer rank 0
    |
    | torch.distributed broadcast/object broadcast
    v
all Megatron ranks
```

Rank 0 owns a command loop:

```text
while alive:
  command = receive_from_controller()
  dist.broadcast_object_list([command], src=0)
  if command.type == FORWARD_BACKWARD:
      result = run_collective_forward_backward(command)
  elif command.type == OPTIM_STEP:
      result = run_collective_optim_step(command)
  elif command.type == SAVE_WEIGHTS:
      result = run_collective_save(command)
  elif command.type == SHUTDOWN:
      break
```

Non-zero ranks block in the same loop and receive commands from rank 0. This is
the only safe shape for interactive calls, because every rank reaches the same
collectives in the same order.

This milestone is more complex because it needs:

- controller-to-rank0 command transport;
- heartbeats;
- command idempotency;
- timeouts;
- shutdown;
- failure detection across ranks;
- backpressure so the controller never sends command `k+1` before all ranks
  finish command `k`.

### Milestone 3: Integrated Rollout Refresh

After the clustered trainer exports a PEFT LoRA adapter:

```text
rank 0 trainer
  -> write adapter files to Modal Volume
  -> commit volume
  -> return artifact manifest
  -> controller refreshes SGLang rollout
```

SGLang does not need to be in the clustered trainer group. It remains the same
separate rollout service from the single-node Qwen path.

## Data Loading and Loss Semantics

The current SFT helper eagerly materializes a JSONL file and builds Python
`Datum` batches in the controller process. That is not viable for large
multi-node training.

For Milestone 1:

- rank 0 materializes JSONL into a Modal Volume before launching the trainer;
- every trainer rank reads the same file;
- each data-parallel rank deterministically selects its shard;
- the first pipeline stage in each PP group owns token batches;
- later PP stages receive activations through Megatron, not through the dataset.

For DP-only Qwen3 0.6B LoRA:

```text
global_batch_size = micro_batch_size * data_parallel_size * grad_accum_steps
num_microbatches  = grad_accum_steps
```

For comparable validation against HF Trainer, use a deterministic fixed-order
dataset and log per-global-step losses. The exact scalar may differ because
multi-rank reduction semantics and bf16 kernels differ, but the curve should be
finite and broadly aligned.

## Configuration

New public knobs:

```text
--n-nodes
--gpus-per-node
--tensor-model-parallel-size
--pipeline-model-parallel-size
--micro-batch-size
--global-batch-size
--grad-accum-steps
--master-port
--rdma / --no-rdma
```

Validation:

```text
world_size = n_nodes * gpus_per_node
world_size % (tp * pp) == 0
dp = world_size // (tp * pp)
global_batch_size % (micro_batch_size * dp) == 0
grad_accum_steps = global_batch_size // (micro_batch_size * dp)
```

For the first Qwen3 0.6B multi-node run:

```text
n_nodes = 2
gpus_per_node = 8
tp = 1
pp = 1
dp = 16
micro_batch_size = 1
global_batch_size = 16
grad_accum_steps = 1
```

If Modal cluster access is unavailable or full-node H100 capacity is constrained,
the same code should accept `n_nodes=1` and `gpus_per_node=8` as a single-node
multi-process smoke.

## Artifact and Checkpointing

The first distributed artifact target remains:

```text
hf-lora-adapter
  adapter_config.json
  adapter_model.safetensors
```

Rank 0 should write the Ganker artifact manifest only after all ranks complete
the save/export barrier. If Bridge's LoRA export requires all ranks to
participate, every rank must enter save. If export can run on rank 0 after
gathering state, non-zero ranks should still enter a barrier before rank 0
commits the Modal Volume.

The artifact payload should add:

```json
{
  "distributed": true,
  "world_size": 16,
  "data_parallel_size": 16,
  "tensor_model_parallel_size": 1,
  "pipeline_model_parallel_size": 1,
  "n_nodes": 2,
  "gpus_per_node": 8
}
```

## Failure Model

Multi-node Megatron is a collective system. A single-rank failure invalidates
the current step.

Expected behavior:

- if any trainer rank exits non-zero, the clustered trainer call fails;
- rank 0 result is authoritative only when all ranks reached completion;
- the controller marks the training run failed unless Modal retries the whole
  clustered call from a known checkpoint;
- partial checkpoints are not promoted to artifact manifests;
- SGLang rollout is refreshed only from committed artifacts.

Modal's clustered documentation notes that rank zero determines the returned
result, while non-leader input failures are not automatically reflected as a
failed return. Ganker should therefore make rank zero monitor child `torchrun`
status and explicitly fail if any local or remote rank fails.

## Testing Plan

### CPU-Only Local Tests

- Validate distributed config arithmetic:
  - `world_size = n_nodes * gpus_per_node`;
  - `dp = world_size // (tp * pp)`;
  - invalid divisibility fails clearly.
- Unit test `torchrun` argument construction.
- Unit test environment mapping:
  - `MASTER_ADDR`;
  - `MASTER_PORT`;
  - `NNODES`;
  - `NODE_RANK`;
  - `NPROC_PER_NODE`.
- Unit test dataset sharding by DP rank.
- Unit test rank-0 result parsing and artifact manifest fields.
- Compile-test Modal app files without importing Megatron, CUDA, or SGLang.

### Modal Smoke Tests

1. `--mode torchrun-env`
   - clustered function starts;
   - each rank reports `RANK`, `WORLD_SIZE`, `LOCAL_RANK`;
   - no Megatron model load.

2. `--mode nccl-smoke`
   - run a tiny `torch.distributed` all-reduce across nodes;
   - verify finite result on rank 0.

3. `--mode qwen-lora-sft-multinode`
   - Qwen3 0.6B LoRA;
   - DP-only first;
   - small dataset;
   - 1-2 steps;
   - save adapter.

4. `--mode qwen-lora-sft-multinode-compare`
   - run the loss comparison against HF Trainer or the existing single-node
     Ganker result for a short deterministic dataset.

5. `--mode qwen-lora-sft-multinode-sglang`
   - export adapter;
   - start/refresh SGLang rollout;
   - sample with `SamplingClient`.

## Implementation Plan

1. Add `src/ganker/distributed/torchrun.py`:
   - config dataclasses;
   - validation;
   - torchrun argv builder;
   - rank/env helpers.

2. Add `modal_apps/qwen_sft_multinode/infra.py`:
   - reuse Bridge image strategy from `modal_apps/qwen_sft/infra.py`;
   - define `run_clustered_trainer`;
   - mount artifact and HF cache volumes;
   - request full-node GPU shape for clustered functions.

3. Add `modal_apps/qwen_sft_multinode/train_entry.py`:
   - torchrun child script;
   - initialize Megatron distributed and Bridge provider;
   - build LoRA model;
   - run fixed-step SFT loop;
   - write rank-0 result JSON.

4. Add `modal_apps/qwen_sft_multinode/sft.py`:
   - local entrypoint;
   - dataset materialization;
   - clustered trainer invocation;
   - optional SGLang refresh/sample.

5. Add tests:
   - CPU unit tests for config and torchrun argv;
   - compile tests for new Modal apps;
   - fake rank-result tests.

6. Add Modal validation commands to `AGENTS.md`, `README.md`, and
   `architecture/distributed-modal.md`.

7. Run Modal smoke sequence:
   - torchrun env;
   - NCCL all-reduce;
   - Qwen LoRA 1-step DP-only;
   - Qwen LoRA + SGLang sample.

## Open Questions

- Do we have Modal multi-node cluster access enabled in the `peyton-agents`
  environment?
answer: yes
- Which full-node GPU SKU should be the first target: `H100:8`, `H200:8`, or
  `B200:8`?
answer: h100:8
- Should Milestone 1 return only rank-0 JSON, or should it also write a
  structured run report into the artifact volume?
answer: it can just return json
- For LoRA export through Megatron Bridge, does rank 0 have enough state to
  write a complete PEFT adapter in DP-only mode, or does Bridge expect all ranks
  to participate in export?
answer: im not sure. you should look at this. its probably fine for only rank 0 to participate
- Should the first multi-node comparison use HF Trainer DP baseline, single-node
  Ganker baseline, or only finite/decreasing loss assertions?
answer: do a hf trainer dp baseline and a ganker baseline

## Recommendation

Implement Milestone 1 first: a whole-job clustered Qwen LoRA SFT run using
DP-only parallelism. That proves the Modal clustered launch, `torchrun`,
Megatron distributed initialization, data sharding, SFT loop, and adapter export
without adding an interactive trainer command protocol.

After that works, add the persistent clustered trainer service if we still need
step-by-step `TrainingClient.forward_backward(...)` calls from outside the
cluster. The persistent service is the right long-term shape for a Tinker-like
API, but it has substantially more failure and synchronization surface than a
whole-job clustered SFT milestone.
