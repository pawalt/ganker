# RFC 0007: Distributed Monarch Orchestration

Status: Draft

## Summary

Move Ganker from a single local Monarch proc mesh to a distributed Monarch deployment where proxy, training, rollout, and telemetry can run on separate proc meshes and, eventually, separate hosts.

The public client API should stay stable:

```text
ServiceClient / TrainingClient / SamplingClient
        |
        v
ProxyTransport
        |
        v
ProxyActor
```

The implementation change is underneath `ProxyTransport`: local tests can keep using `ServiceClient.local(...)`, while distributed runs use a deployment controller that starts or attaches to named Monarch host/proc meshes and exposes a proxy endpoint to the client.

## Current State

Today `start_local_monarch_mesh(...)` does everything on one local process mesh:

```text
this_host().spawn_procs(name="ganker_local")
        |
        +-- TrainingActor
        +-- RolloutActor
        +-- TelemetryActor
        +-- ProxyActor
```

This is useful for local tests, but it hides three production concerns:

- proxy/telemetry should not share lifecycle with GPU-heavy trainer processes;
- rollout and training will have different images, GPU requirements, and restart behavior;
- distributed Megatron needs collective calls across multiple trainer ranks, not `.choose(...)` to one actor.

## Goals

- Keep public callers off Monarch internals.
- Add a controller/orchestrator layer that owns deployment lifecycle.
- Allow proxy, telemetry, training, and rollout actors to live on separate Monarch proc meshes.
- Support a local distributed simulation without GPUs or Megatron.
- Support a Modal distributed smoke where at least proxy and training are not in the same proc mesh/container.
- Define the path from single-rank Bridge training to multi-rank Megatron training.
- Keep fake backend tests lightweight and deterministic.

## Non-Goals

- Do not replace Monarch with gRPC internally.
- Do not require every local test to start a distributed deployment.
- Do not solve autoscaling, queueing, or multi-tenant scheduling.
- Do not implement SGLang multi-node rollout in the first distributed milestone.
- Do not expose Megatron rank handles or Monarch actor handles in the user-facing API.

## Terminology

```text
client
  User code calling ServiceClient.

controller
  Deployment owner. Allocates/attaches host meshes, spawns proc meshes,
  wires actor handles, records deployment metadata, and shuts things down.
  The controller is not the ProxyActor.

proxy
  Client-facing request router. Owns no model state.

training coordinator
  Single actor that receives Ganker training requests from ProxyActor and
  coordinates one or more trainer worker ranks.

trainer workers
  Rank actors that own Megatron model chunks, distributed process groups,
  optimizer state, and checkpoint/export collectives.

rollout
  Inference actor or actor mesh. Eventually owns SGLang runtime state.
```

## Target Architecture

```text
                 +----------------------+
user code -----> | ServiceClient        |
                 | ProxyTransport       |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | ProxyActor           |  proxy proc mesh
                 +----+------------+----+
                      |            |
          +-----------+            +----------------+
          v                                         v
+--------------------------+             +----------------------+
| TrainingCoordinatorActor |             | RolloutActor         |
| trainer control mesh     |             | rollout proc mesh    |
+------------+-------------+             +----------+-----------+
             |                                      |
             v                                      v
  +----------------------+                 +------------------+
  | TrainerWorkerActor[] |                 | SGLang backend   |
  | trainer rank mesh    |                 | or fake backend  |
  +----------+-----------+                 +------------------+
             |
             v
  +----------------------+
  | Megatron Bridge/Core |
  | distributed ranks    |
  +----------------------+

+----------------------+
| TelemetryActor       | telemetry proc mesh
+----------------------+

+----------------------+
| Artifact store       | shared filesystem / Modal Volume / object store
+----------------------+
```

## Deployment Spec

Add a first-class deployment specification to `ganker.orchestration`.

Example shape:

```python
DistributedMeshSpec(
    artifact_root="/mnt/ganker-artifacts",
    transport="tcp",
    proxy=ProcMeshSpec(name="ganker_proxy", hosts=1, procs_per_host=1),
    telemetry=ProcMeshSpec(name="ganker_telemetry", hosts=1, procs_per_host=1),
    training=ProcMeshSpec(name="ganker_training", hosts=1, gpus_per_host=1),
    rollout=ProcMeshSpec(name="ganker_rollout", hosts=1, gpus_per_host=1),
    training_backend="megatron",
    training_backend_config={...},
    inference_backend="fake",
)
```

The first implementation can map this onto one physical host with separate proc meshes. The important contract is that actors are no longer spawned into the same `ganker_local` proc mesh. Modal can then map each proc mesh to separate containers or host allocations.

## Controller API

Add a controller object rather than growing `ServiceClient.local(...)`:

```python
deployment = start_distributed_monarch_deployment(spec)
client = deployment.client(timeout=120)
...
deployment.stop()
```

The deployment object should own:

- host mesh handles;
- proc mesh handles;
- actor handles;
- readiness futures;
- shutdown order;
- deployment metadata for debugging.

`ServiceClient.local(...)` remains a small local-test helper. Distributed callers either receive a `ServiceClient` from the deployment object or connect through a non-Monarch external `ProxyTransport`.

## Client Boundary

The client should still not speak Monarch directly.

For Python smoke tests inside the same controller process:

```text
DistributedGankerDeployment.client()
  -> ServiceClient(MonarchProxyTransport(proxy_actor))
```

For callers outside the Monarch controller process, add an external transport adapter:

```text
ServiceClient.connect(...)
  -> HttpProxyTransport / ModalProxyTransport
  -> ProxyActor
```

This keeps Monarch an internal orchestration layer while allowing a stable public API for users.

## Training Distribution

There are two separate milestones.

### Milestone A: Distributed Actor Placement

Keep a single-rank training backend, but place actors on separate proc meshes:

```text
proxy proc mesh      -> ProxyActor
telemetry proc mesh  -> TelemetryActor
training proc mesh   -> TrainingActor
rollout proc mesh    -> RolloutActor
```

This validates:

- deployment spec parsing;
- actor handle wiring across proc meshes;
- readiness and shutdown across proc meshes;
- shared artifact root;
- existing public `ServiceClient` flow.

It does not validate Megatron multi-rank behavior yet.

### Milestone B: Distributed Megatron Ranks

Replace direct `ProxyActor -> TrainingActor` routing with:

```text
ProxyActor
  -> TrainingCoordinatorActor
       -> TrainerWorkerActor rank 0
       -> TrainerWorkerActor rank 1
       -> ...
```

The coordinator is the only training actor that the proxy knows about. It exposes the current training endpoints:

```text
create_training_run
forward_backward
optim_step
save_weights
shutdown
```

Each endpoint becomes a collective operation across trainer workers:

```text
create_training_run
  coordinator validates request
  coordinator broadcasts init to all workers
  workers initialize Megatron distributed state
  rank 0 returns TrainingRun metadata

forward_backward
  coordinator sends batch or batch shards to workers
  all workers enter the same Megatron schedule
  rank 0 returns reduced loss and usage

optim_step
  all workers step optimizer
  rank 0 returns optimizer_step/checkpoint_version

save_weights
  all workers enter export/checkpoint collectives
  rank 0 writes artifact manifest/payload
  coordinator returns WeightArtifact
```

This avoids `.choose(...)` accidentally sending a collective operation to only one rank.

## Megatron Runtime Changes

The current `InstalledMegatronBridgeRuntime` assumes `WORLD_SIZE=1` and initializes distributed state inside one actor process. Distributed training needs a runtime config that comes from the trainer worker mesh:

```text
rank
world_size
local_rank
master_addr
master_port
tensor_model_parallel_size
pipeline_model_parallel_size
data_parallel_size
```

The worker runtime should not guess these values from process-local defaults. The coordinator should provide a rank assignment object to each worker.

Initial distributed constraints:

- one active run per trainer worker mesh;
- tensor parallel and pipeline parallel default to 1;
- data parallel can be the first multi-rank mode;
- rank 0 writes Ganker artifact metadata;
- all ranks participate in Bridge checkpoint/export calls when required.

## Artifact Store

Distributed actors need a shared artifact store.

Local simulation:

```text
temporary directory visible to all local processes
```

Modal:

```text
Modal Volume mounted at the same path in proxy/training/rollout containers
```

Later:

```text
object store-backed ArtifactStore
```

Artifact writes must remain atomic from the client perspective:

- checkpoint/export finishes before manifest write;
- manifest write happens on rank 0;
- rollout refresh reads only completed manifests;
- failed checkpoint attempts do not advance latest pointers.

## Modal Shape

Current `modal_apps/sft.py` runs the whole Ganker mesh inside one Modal function. The distributed milestone should add a new app rather than mutate the existing smoke:

```text
modal_apps/distributed_sft.py
```

Target stages:

1. `--mode fake-distributed`
   - CPU image.
   - separate proxy/training/rollout/telemetry proc meshes.
   - fake backends.
   - proves lifecycle and public API.

2. `--mode qwen-single-rank-distributed`
   - controlled Bridge image from RFC 0006.
   - proxy/telemetry separated from a single-rank training mesh.
   - runs Qwen full or LoRA one-step smoke.

3. `--mode qwen-data-parallel`
   - controlled Bridge image.
   - `TrainerWorkerActor[]` rank mesh.
   - data-parallel Megatron run across two GPUs.
   - one-step LoRA smoke first, full tuning second.

## Testing Plan

Unit tests:

- `DistributedMeshSpec` validation.
- controller spawn plan generation.
- coordinator state machine with fake worker handles.
- rank assignment generation.
- artifact write/read behavior under rank-0-only writes.

Local integration tests:

- separate local proc meshes with fake backends;
- public `ServiceClient` end-to-end flow;
- shutdown order and idempotent cleanup;
- failure injection where one worker fails and coordinator marks run failed.

Modal smoke tests:

- fake distributed actor placement;
- Qwen LoRA single-rank distributed placement;
- Qwen LoRA two-rank data parallel if the Modal GPU shape supports it.

Default `uv run pytest` must remain CPU-only and must not require Modal, CUDA, Megatron, SGLang, or model downloads.

## Rollout Distribution

Rollout should follow training only after the training coordinator design is stable.

First rollout target:

```text
ProxyActor -> RolloutActor on separate proc mesh -> fake backend
```

Second rollout target:

```text
ProxyActor -> RolloutActor on GPU proc mesh -> SGLang backend
```

The rollout actor should continue consuming `WeightArtifact` metadata. It should not know about trainer worker ranks.

## Failure Semantics

- Controller startup fails if any actor mesh does not become ready before timeout.
- Proxy should reject requests until training, rollout, and telemetry handles are ready.
- Coordinator marks the run failed if any trainer rank fails during a collective operation.
- `save_weights` failure leaves the previous artifact latest pointer intact.
- Shutdown should try actor-level backend cleanup before stopping proc meshes.
- Modal smoke should surface worker logs and deployment metadata in the returned payload.

## Implementation Plan

1. Add `ProcMeshSpec`, `DistributedMeshSpec`, and `DistributedGankerDeployment`.
2. Implement `start_distributed_monarch_deployment(...)` for separate proc meshes on the current host.
3. Add local integration tests with fake backends and separate proc mesh names.
4. Add `modal_apps/distributed_sft.py --mode fake-distributed`.
5. Split training into `TrainingCoordinatorActor` and fake `TrainerWorkerActor` mesh.
6. Add coordinator unit tests and failure injection.
7. Move Megatron Bridge runtime into trainer workers with explicit rank assignment.
8. Add Modal Qwen LoRA single-rank distributed smoke.
9. Add Modal Qwen LoRA two-rank data-parallel smoke.

## Acceptance Criteria

- Existing `ServiceClient.local(...)` tests still pass.
- A new local integration test proves proxy/training/rollout/telemetry are spawned on different proc meshes.
- Modal fake distributed smoke runs through the public client API.
- Modal Qwen LoRA smoke runs with proxy and training outside a single local mesh.
- Multi-rank training smoke runs one LoRA step and exports a PEFT adapter artifact.
- No user-facing API requires Monarch actor handles.

## Open Questions

- Which Monarch host acquisition path should be the stable Modal implementation: `hosts_from_config(...)`, explicit worker-loop containers, or a Modal-specific controller wrapper?
- Should the first multi-rank mode be pure data parallel, or should tensor parallel be validated immediately because Qwen/Megatron production will need it?
- Should external users call a Modal web endpoint, an HTTP proxy, or a Python-only Modal function wrapper for the first non-local API?
- How much of the trainer batch should the coordinator broadcast versus shard before reaching worker ranks?
- Do we need a separate artifact-store abstraction before distributed rollout, or is a shared Modal Volume enough for the next milestone?
