# RFC 0008: Modal Distributed Controller Orchestration

Status: Draft

## Summary

Ganker should move from a single Modal worker containing one local Monarch mesh
to a Modal-native distributed layout where the proxy, controller, trainer, and
inference server can live on separate physical machines.

The key change is making the controller a first-class service. The proxy remains
the client-facing gateway, but it should not own topology, readiness, artifact
versions, or failure policy. The controller owns those concerns.

The distributed control boundary should be gRPC between Modal containers. Monarch
can still be used inside a role when it helps local actor orchestration, but
cross-machine role communication should use explicit RPC contracts first.

```text
external client
        |
        | gRPC, through Modal tunnel or gateway
        v
+----------------+
| Proxy service  |
+--------+-------+
         |
         | private Modal networking, gRPC
         v
+---------------------+
| Controller service  |
+----+-----------+----+
     |           |
     |           +------------------+
     |                              |
     v                              v
+----------------------+     +----------------------+
| Trainer coordinator  |     | Inference service    |
| Megatron rank 0      |     | SGLang HTTP/gRPC     |
+----------+-----------+     +----------+-----------+
           |                            |
           | Megatron collectives       |
           v                            v
   trainer ranks                 SGLang worker(s)

Shared state:
  - Modal Dict or equivalent registry for discovery and heartbeats
  - Modal Volume for artifacts/checkpoints/adapters
```

## Current State

RFC 0007 added remote gRPC access to a single Modal-hosted Monarch mesh:

```text
remote client
  -> GrpcProxyTransport
  -> ProxyGrpcServer in one Modal worker
  -> local ProxyActor
  -> local TrainingActor / RolloutActor / TelemetryActor
```

That proves the external client boundary, but it still assumes all service roles
are in one process/container. The recent SGLang smoke also proves real GPU
inference through:

```text
ServiceClient.local(...)
  -> Monarch ProxyActor
  -> RolloutActor
  -> SGLangInferenceBackend
  -> local python -m sglang.launch_server
```

The missing layer is service orchestration across separate Modal machines.

## Goals

- Run proxy, controller, trainer, and inference as separately addressable Modal
  services or clustered functions.
- Keep user code on the existing `ServiceClient` / `TrainingClient` /
  `SamplingClient` API.
- Make the controller own run lifecycle, topology, readiness, artifact versions,
  and failure policy.
- Use gRPC for control-plane RPC between roles.
- Use Modal private networking for container-to-container traffic.
- Use Modal clustered GPU functions for multi-node trainer ranks.
- Use Modal Volumes for checkpoints, HF full exports, and LoRA adapters.
- Keep the local CPU suite independent of Modal, CUDA, Megatron, SGLang, and
  model downloads.
- Add a fake distributed smoke before a Megatron/SGLang GPU smoke.

## Non-Goals

- Do not replace Megatron/NCCL collectives with Ganker RPC.
- Do not expose Monarch actor handles to clients.
- Do not require SGLang or Megatron in local unit tests.
- Do not build a public multi-tenant serving platform in this milestone.
- Do not solve full autoscaling, scheduling economics, or long-running
  production SLAs yet.
- Do not require IPv4-only networking. Modal private networking is i6pn-based;
  store endpoint address metadata and let each role choose the reachable address
  family.

## Modal Assumptions

Modal gives us three primitives that matter here:

- `i6pn=True` for workspace-private container-to-container networking.
- `modal.experimental.clustered(...)` for gang-scheduled multi-node GPU groups.
- `modal.Dict`, `modal.Queue`, and `modal.Volume` for registry, signaling, and
  shared artifacts.

Modal cluster docs expose `cluster_info.rank`, `cluster_info.container_ips`, and
`cluster_info.container_ipv4_ips` for clustered containers. Ganker should store
all usable addresses in the registry rather than hard-coding IPv4 or IPv6 into
the service contract.

## Target Architecture

```text
                       public / developer boundary
============================================================================

client process
    |
    | ServiceClient.connect_grpc(...)
    v
+--------------------+
| Proxy gRPC API     |
| public Ganker API  |
+---------+----------+
          |
          | private role RPC
          v
============================================================================
                       Modal private service mesh

+--------------------------+
| Controller               |
| - topology registry      |
| - readiness barriers     |
| - artifact versions      |
| - failure policy         |
| - run state machine      |
+------+--------------+----+
       |              |
       |              +---------------------------+
       |                                          |
       v                                          v
+--------------------------+          +---------------------------+
| TrainerCoordinator       |          | InferenceCoordinator      |
| rank 0 of trainer group  |          | SGLang frontend/process   |
+-------------+------------+          +-------------+-------------+
              |                                     |
              | torch.distributed / Megatron        | SGLang native API
              v                                     v
+-------------+------------+          +-------------+-------------+
| TrainerWorker ranks      |          | SGLang server             |
| 1..N                     |          | /generate, /load_lora     |
+--------------------------+          +---------------------------+

Shared:
  Modal Dict      run registry, role endpoints, heartbeats
  Modal Volume    artifact payloads, exported HF/LoRA checkpoints
```

## Component Responsibilities

### Proxy Service

The proxy is the only client-facing service.

Responsibilities:

- Expose the existing Ganker gRPC API.
- Authenticate requests and set request IDs.
- Enforce deadlines and max message sizes.
- Route every operation to the controller.
- Optionally keep a short-lived cache of controller address metadata.
- Never own GPU state.
- Never own run lifecycle.
- Be restartable without losing training state.

The proxy should stay thin. It is a gateway, not the orchestrator.

### Controller Service

The controller is the control plane.

Responsibilities:

- Create and own `run_id` lifecycle state.
- Start or discover trainer and inference roles.
- Maintain a topology registry for each run.
- Implement readiness barriers before accepting training or sampling calls.
- Track current gradient, optimizer, checkpoint, and inference-refresh versions.
- Coordinate `save_weights -> export -> inference refresh`.
- Route requests to trainer or inference coordinators.
- Record telemetry or forward events to a telemetry service.
- Detect stale heartbeats and decide whether a run is failed, restartable, or
  needs manual intervention.

The controller should be the only service allowed to mutate run topology.

### Trainer Coordinator

The trainer coordinator is rank 0 of the training role.

Responsibilities:

- Own the public trainer RPC endpoint for the controller.
- Coordinate Megatron run creation across ranks.
- Translate controller RPCs into collective Megatron operations.
- Make `forward_backward`, `optim_step`, and `save_weights` collective.
- Export SGLang-compatible artifacts when requested.
- Register rank and readiness metadata with the controller registry.

Ranks other than 0 should not accept controller RPC directly unless needed for
diagnostics. Rank 0 fans out work using Megatron/NCCL/torch.distributed.

### Inference Service

The inference service owns SGLang lifecycle.

Responsibilities:

- Start SGLang with a base model or HF checkpoint.
- Expose private `/generate` and adapter-loading endpoints.
- Refresh from controller-selected artifact versions.
- Report loaded artifact version and health.
- Reject sampling if the required artifact version is not loaded.

The existing `SGLangHTTPRuntime` can be reused here, but in distributed mode it
should usually attach to an already-running private SGLang endpoint rather than
launching SGLang inside the same RolloutActor process.

### Telemetry

Telemetry can start simple:

- Controller records usage events into a Modal Dict-backed or Volume-backed log.
- Later, split telemetry into a separate service if event volume or retention
  requires it.

For this milestone, telemetry should not block trainer or inference progress.

## Control Plane RPCs

Keep the external Ganker API stable, but add internal role APIs.

### Proxy -> Controller

```text
CreateTrainingRun
ForwardBackward
OptimStep
SaveWeights
RefreshWeights
Sample
GetTelemetrySummary
DownloadArtifactFile
ShutdownRun
GetRunStatus
```

The proxy can reuse the current public protobuf messages for most calls. The
controller should return explicit run status and endpoint state for diagnostics.

### Controller -> TrainerCoordinator

```text
TrainerCreateRun(run_config) -> run_state
TrainerForwardBackward(run_id, batch) -> loss/metrics/usage
TrainerOptimStep(run_id, optimizer) -> optimizer_version/usage
TrainerSaveWeights(run_id, kind, export_format) -> WeightArtifact
TrainerGetStatus(run_id) -> trainer_status
TrainerShutdown(run_id)
```

For real Megatron, every mutating call must become a collective operation across
all ranks.

### Controller -> Inference

```text
InferenceLoadBaseModel(model_id)
InferenceRefreshWeights(run_id, artifact, expected_version)
InferenceSample(run_id, prompt, sampling_params, artifact_version)
InferenceGetStatus(run_id)
InferenceShutdown(run_id)
```

`InferenceSample` can initially proxy through the controller. Later, the
controller can issue a short-lived sampling route token so the proxy can call
the inference service directly for lower latency.

## Registry Contract

Use a registry object keyed by run and role.

```text
ganker:{deployment_id}:controller
ganker:{run_id}:proxy
ganker:{run_id}:trainer:rank0
ganker:{run_id}:trainer:rank1
ganker:{run_id}:inference:0
```

Each entry should be JSON-serializable:

```json
{
  "deployment_id": "default",
  "run_id": "run-000001",
  "role": "inference",
  "rank": 0,
  "protocol": "http",
  "addresses": [
    {"family": "ipv6", "host": "...", "port": 30000},
    {"family": "ipv4", "host": "...", "port": 30000}
  ],
  "status": "ready",
  "epoch": 4,
  "artifact_version": 7,
  "started_at": "2026-06-22T00:00:00Z",
  "last_heartbeat": "2026-06-22T00:00:05Z"
}
```

Rules:

- `epoch` increments whenever a role restarts.
- Callers must include the expected `epoch` on mutating calls.
- A stale epoch should fail fast.
- Heartbeat timeouts mark a role unhealthy.
- The controller is the writer for run-level topology. Roles may only update
  their own heartbeat/status fields.

## Run State Machine

```text
CREATING
  -> STARTING_TRAINER
  -> STARTING_INFERENCE
  -> READY
  -> TRAINING
  -> GRADIENTS_PENDING
  -> OPTIMIZING
  -> CHECKPOINTING
  -> REFRESHING_INFERENCE
  -> READY
  -> DRAINING
  -> STOPPED

Any state
  -> FAILED
```

Important invariants:

- `forward_backward` is accepted only when the trainer is ready.
- `optim_step` is accepted only after gradients are pending.
- `save_weights` is accepted only from a stable trainer state.
- `sample` may specify an artifact version.
- If `sample` has no artifact version, it uses the controller's current loaded
  inference artifact version.
- `refresh_weights` does not become visible until SGLang reports the expected
  artifact version loaded.

## Artifact Flow

```text
TrainerCoordinator.save_weights
        |
        | writes raw checkpoint and export payloads
        v
Modal Volume
        |
        | returns WeightArtifact
        v
Controller
        |
        | records checkpoint_version and artifact_format
        | tells inference to refresh expected version
        v
Inference service
        |
        | loads HF full checkpoint or LoRA adapter
        v
Controller marks inference_artifact_version ready
```

SGLang-compatible artifact formats remain:

```text
hf-full-safetensors
hf-lora-adapter
base_model payload for untuned base sampling
```

Raw Megatron artifacts are internal trainer artifacts. They should not be sent
to inference until exported.

## Request Flows

### Create Run

```text
client
  -> proxy.CreateTrainingRun
  -> controller.CreateTrainingRun
  -> controller starts trainer role
  -> controller starts inference role
  -> roles register endpoints and heartbeat
  -> controller reaches READY
  -> proxy returns TrainingRun
```

### Training Step

```text
client
  -> proxy.ForwardBackward
  -> controller.ForwardBackward
  -> trainer rank0 RPC
  -> Megatron collective forward/backward across ranks
  -> trainer returns loss/metrics/usage
  -> controller records state and telemetry
```

### Save And Refresh

```text
client
  -> proxy.SaveWeights
  -> controller.SaveWeights
  -> trainer rank0 collective checkpoint/export
  -> Modal Volume artifact
  -> controller.RefreshWeights
  -> inference loads artifact
  -> controller marks artifact ready
```

### Sampling

Initial path:

```text
client
  -> proxy.Sample
  -> controller.Sample
  -> inference.Sample
  -> SGLang /generate
  -> controller telemetry
  -> proxy response
```

Later low-latency path:

```text
client
  -> proxy.Sample
  -> proxy asks controller for route/artifact version
  -> proxy calls inference.Sample directly
  -> proxy reports usage to controller
```

Start with the initial path because it centralizes correctness.

## Failure Model

### Proxy Failure

Proxy failure should not affect run state. Clients reconnect to another proxy or
restart the proxy service.

### Controller Failure

Controller failure is serious because it owns topology and state. For the first
milestone, mark active runs interrupted and require explicit recovery. Later,
persist enough run state in the registry/volume to elect or restart a
controller.

### Trainer Rank Failure

If any trainer rank dies, Modal clustered execution may terminate the group.
Controller marks the run `FAILED` or `INTERRUPTED`. Retrying a trainer step is
safe only if the previous collective is known not to have committed state.

### Inference Failure

Inference can be restarted from the last controller-approved artifact version.
Sampling fails with `UNAVAILABLE` until the endpoint is healthy and the expected
artifact is loaded.

### Registry Staleness

Use heartbeats plus epochs. Stale endpoints must not accept mutating requests.

## Modal Deployment Shape

The first real implementation can live in a new app:

```text
modal_apps/distributed_mesh.py
```

Modes:

```text
fake-distributed
  Starts proxy, controller, fake trainer, fake inference as separate Modal
  containers with i6pn enabled.

sglang-distributed
  Starts proxy, controller, fake trainer, real SGLang inference service.

megatron-sglang-distributed
  Starts proxy, controller, clustered Megatron trainer, and SGLang inference.
```

The first mode should be CPU-only except for any Modal private networking
requirement. If clustered functions are needed, keep fake-distributed separate
from the trainer clustered path because Modal clustered functions require GPU
clusters.

## Internal Service Startup

A Modal local entrypoint should submit one orchestration function. That function
starts or references:

- a controller function;
- a proxy function with a Modal tunnel for external gRPC;
- a trainer role function;
- an inference role function.

The controller should not depend on local process globals. Every role should be
able to reconstruct state from:

- launch arguments;
- registry entries;
- Modal Volume artifact metadata.

## Security

- External traffic enters only through the proxy.
- Internal role RPCs should require a run-scoped bearer token or shared secret.
- Artifact download validates that the requested path is inside the configured
  artifact root.
- Internal endpoints should bind only to private Modal addresses where possible.
- Public tunnels are developer/test tools until a production ingress policy
  exists.

## Observability

Minimum required fields:

```text
request_id
run_id
role
rank
epoch
artifact_version
operation
duration_ms
status
error_type
input_tokens / output_tokens / training_steps / samples
```

Every role should log readiness and heartbeat transitions. The controller should
provide `GetRunStatus` with the current topology and last error.

## Testing Plan

### Unit Tests

- Registry key and entry serialization.
- Run state machine transitions.
- Controller routing with fake trainer/inference clients.
- Epoch mismatch rejection.
- Heartbeat timeout detection.
- Artifact version visibility rules.
- Proxy delegates to controller and does not mutate topology.

### Local Integration Tests

Use in-process fake RPC clients or loopback gRPC servers. Do not require Modal.

```text
client -> proxy server -> controller -> fake trainer/fake inference
```

### Modal Smokes

1. `fake-distributed`
   - Separate Modal containers.
   - gRPC over private network.
   - Fake trainer and fake inference.
   - Proves discovery, heartbeats, routing, and artifact download.

2. `sglang-distributed`
   - Fake trainer emits base-model or fake HF artifact metadata.
   - Real SGLang inference service on GPU.
   - Controller refreshes inference and samples through proxy.

3. `megatron-sglang-distributed`
   - Megatron Bridge/Core trainer on clustered GPU role.
   - Exports LoRA adapter.
   - SGLang loads adapter.
   - End-to-end SFT step followed by sample.

## Implementation Plan

1. Add controller contracts and service interfaces.
2. Add a `RunRegistry` abstraction with in-memory and Modal Dict backends.
3. Add controller state machine with fake trainer/inference clients.
4. Add gRPC services for controller, trainer coordinator, and inference.
5. Refactor proxy server to route to controller instead of directly owning a
   local Monarch mesh.
6. Add fake role servers and local loopback integration tests.
7. Add `modal_apps/distributed_mesh.py --mode fake-distributed`.
8. Add SGLang distributed inference role and Modal smoke.
9. Add trainer coordinator role for single-node Megatron.
10. Add clustered Megatron trainer role.
11. Add save/export/refresh orchestration from trainer to SGLang.
12. Add failure/heartbeat/epoch tests and served/manual tunnel mode.

## Acceptance Criteria

- User-facing `ServiceClient` API remains stable.
- Proxy and controller can run as separate processes.
- Controller owns run topology and state transitions.
- Fake trainer and fake inference can run on separate Modal containers.
- SGLang inference can run on a separate Modal GPU container and be reached over
  private networking.
- The controller can refresh SGLang to a selected artifact version.
- A Modal smoke proves `client -> proxy -> controller -> inference -> SGLang`.
- A later Modal smoke proves `client -> proxy -> controller -> trainer ->
  artifact -> inference -> sample`.
- Local tests remain CPU-only and do not import heavy ML packages by default.

## Open Questions

- Should the controller be a long-running Modal function, a Modal class, or a
  small gRPC server inside a served function?
- Should sampling initially route through the controller, or should the proxy get
  an inference route token and call inference directly?
- How much run state should be recoverable after controller restart in the first
  implementation?
- Should the trainer clustered function be launched per run, or should a warm
  trainer pool accept multiple runs?
- Should artifact refresh be synchronous from the client perspective, or should
  it return an operation ID that can be polled?

## References

- RFC 0007: Remote gRPC Access To A Modal-Hosted Monarch Mesh.
- Modal cluster networking: https://modal.com/docs/guide/private-networking
- Modal multi-node clusters: https://modal.com/docs/guide/multi-node-training
- Modal tunnels: https://modal.com/docs/guide/tunnels
- Modal multinode training guide example:
  https://github.com/modal-labs/multinode-training-guide/blob/main/benchmark/modal_train.py
