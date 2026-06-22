# RFC 0008: Modal Distributed Controller Orchestration

Status: Draft

## Summary

Ganker should move from a single Modal worker containing one local Monarch mesh
to a Modal-native distributed layout where the proxy, controller, trainer, and
inference server can live on separate physical machines.

The key change is making the controller a first-class service. The proxy remains
the client-facing gateway, but it should not own topology, readiness, artifact
versions, or failure policy. The controller owns those concerns.

The distributed control boundary inside Modal should be Monarch, not a second
internal protobuf/gRPC layer. External clients still use the existing Ganker
gRPC API at the proxy boundary, but proxy/controller/trainer orchestration
should happen through Monarch actor endpoints over i6pn. SGLang remains an HTTP
backend for sampling.

```text
external client
        |
        | gRPC, through Modal tunnel or gateway
        v
+----------------+
| Proxy service  |
+--------+-------+
         |
         | private Modal networking, Monarch actor call
         v
+---------------------+
| Controller service  |
+----+-----------+----+
     |           |
     |           +------------------+
     |                              |
     v                              v
+----------------------+     +----------------------+
| Trainer role         |     | Rollout role         |
| Megatron rank 0      |     | RolloutActor/SGLang  |
+----------+-----------+     +----------+-----------+
           |                            |
           | Megatron collectives       | HTTP /generate, /load_lora
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

- Run proxy, controller, trainer, and inference as separately addressable
  i6pn-enabled Modal functions.
- Keep user code on the existing `ServiceClient` / `TrainingClient` /
  `SamplingClient` API.
- Make the controller own run lifecycle, topology, readiness, artifact versions,
  and failure policy.
- Use Monarch actor calls for internal control-plane communication between
  proxy, controller, trainer, rollout, and telemetry roles.
- Use Modal private networking for container-to-container traffic, with every
  role pinned to the same exact Modal region.
- Use Modal clustered GPU functions only when gang scheduling is needed for
  multi-node trainer ranks.
- Use Modal Volumes for checkpoints, HF full exports, and LoRA adapters.
- Keep the local CPU suite independent of Modal, CUDA, Megatron, SGLang, and
  model downloads.
- Add a fake distributed smoke before a Megatron/SGLang GPU smoke.

## Non-Goals

- Do not replace Megatron/NCCL collectives with Ganker RPC.
- Do not expose Monarch actor handles to clients.
- Do not add internal protobuf/gRPC services for controller, trainer, rollout,
  or telemetry unless Monarch proves insufficient for a specific boundary.
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

All controller, proxy, trainer, and inference functions that communicate over
i6pn must be pinned to the same exact Modal region such as `us-east-1`. Broad
or meta regions such as `us-east` are not sufficient for this design because
they may not place containers on the same private-network connectivity plane.

`modal.experimental.clustered(...)` is not required merely to get i6pn. It is a
gang-scheduling tool for tightly coupled multi-node jobs. Single-node trainer,
SGLang inference, fake distributed roles, and the controller/proxy split should
use plain `@app.function(i6pn=True, region="us-east-1")`-style placement.

Each role should resolve and publish its own private IPv6 address through the
registry. The registry must store all usable addresses and the exact Modal
region, rather than hard-coding IPv4 or broad-region assumptions into the
service contract.

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
          | Monarch actor call, private i6pn
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
| Trainer role             |          | Rollout role              |
| TrainingActor/rank 0     |          | RolloutActor/SGLang client|
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
- Route every operation to the controller through a private Monarch actor
  handle.
- Optionally keep a short-lived cache of controller address metadata.
- Never own GPU state.
- Never own run lifecycle.
- Be restartable without losing training state.

The proxy should stay thin. It is a gateway, not the orchestrator.

### Controller Service

The controller is the control plane.

Responsibilities:

- Create and own `run_id` lifecycle state.
- Start or discover trainer and rollout/inference roles.
- Maintain a topology registry for each run.
- Implement readiness barriers before accepting training or sampling calls.
- Track current gradient, optimizer, checkpoint, and inference-refresh versions.
- Coordinate `save_weights -> export -> rollout refresh`.
- Route requests to trainer or rollout actors.
- Record telemetry or forward events to a telemetry service.
- Detect stale heartbeats and decide whether a run is failed, restartable, or
  needs manual intervention.

The controller should be the only service allowed to mutate run topology.

### Trainer Role

The trainer role is rank 0 of the training worker group.

Responsibilities:

- Own the trainer-side Monarch actor endpoint for the controller.
- Coordinate Megatron run creation across ranks.
- Translate controller actor calls into collective Megatron operations.
- Make `forward_backward`, `optim_step`, and `save_weights` collective.
- Export SGLang-compatible artifacts when requested.
- Register rank and readiness metadata with the controller registry.

Ranks other than 0 should not accept controller calls directly unless needed for
diagnostics. Rank 0 fans out work using Megatron/NCCL/torch.distributed.

### Rollout / Inference Service

The rollout/inference service owns rollout orchestration and SGLang lifecycle.

Responsibilities:

- Start SGLang with a base model or HF checkpoint.
- Expose or discover private SGLang `/generate` and adapter-loading endpoints.
- Refresh from controller-selected artifact versions.
- Report loaded artifact version and health.
- Reject sampling if the required artifact version is not loaded.

The existing `SGLangHTTPRuntime` can be reused for single-container smoke tests.
In distributed mode, `RolloutActor` should usually attach to an already-running
private SGLang endpoint rather than launching SGLang inside the same actor
process.

### Telemetry

Telemetry can start simple:

- Controller records usage events into a Modal Dict-backed or Volume-backed log.
- Later, split telemetry into a separate service if event volume or retention
  requires it.

For this milestone, telemetry should not block trainer or inference progress.

## Monarch Control Plane Calls

Keep the external Ganker API stable. Inside Modal, do not add a parallel
protobuf surface for controller/trainer/inference. Use Monarch actor endpoints
and the existing Python dataclass contracts from `ganker.contracts`.

### ProxyActor -> ControllerActor

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

The public gRPC proxy continues converting protobuf messages into Python
contracts at the edge. After that conversion, the proxy forwards typed Python
requests to the controller actor.

### ControllerActor -> TrainingActor

```text
create_training_run(CreateTrainingRunRequest) -> CreateTrainingRunResponse
forward_backward(ForwardBackwardRequest) -> ForwardBackwardResponse
optim_step(OptimStepRequest) -> OptimStepResponse
save_weights(SaveWeightsRequest) -> SaveWeightsResponse
get_status(run_id) -> trainer_status
shutdown(run_id)
```

For real Megatron, every mutating call must become a collective operation across
all ranks.

### ControllerActor -> RolloutActor

```text
load_base_model(model_id)
refresh_weights(RefreshWeightsRequest) -> RefreshWeightsResponse
get_status(run_id) -> rollout_status
shutdown(run_id)
```

For sampling, the preferred distributed shape is direct HTTP to SGLang. The
controller should return or record the selected SGLang HTTP route and artifact
version so `SamplingClient` can call the endpoint directly. The `RolloutActor`
still owns refresh/version correctness and can provide a local fake sampling
path for CPU tests.

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
- A direct sampling route is returned only after the controller has a ready
  SGLang endpoint and artifact version.
- `refresh_weights` does not become visible until SGLang reports the expected
  artifact version loaded.

## Artifact Flow

```text
TrainingActor.save_weights
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
        | tells RolloutActor to refresh expected version
        v
RolloutActor / SGLang service
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
  -> ControllerActor.create_training_run
  -> controller starts or discovers trainer role
  -> controller starts or discovers inference role
  -> roles publish i6pn Monarch/SGLang endpoints and heartbeat
  -> controller attach_to_workers for Monarch roles
  -> controller reaches READY
  -> proxy returns TrainingRun
```

### Training Step

```text
client
  -> proxy.ForwardBackward
  -> ControllerActor.forward_backward
  -> TrainingActor.forward_backward on trainer rank0
  -> Megatron collective forward/backward across ranks
  -> trainer returns loss/metrics/usage
  -> controller records state and telemetry
```

### Save And Refresh

```text
client
  -> proxy.SaveWeights
  -> ControllerActor.save_weights
  -> TrainingActor.save_weights collective checkpoint/export
  -> Modal Volume artifact
  -> ControllerActor.refresh_weights
  -> RolloutActor refreshes the SGLang endpoint
  -> controller marks artifact ready
```

### Sampling

Distributed path:

```text
TrainingClient.save_weights_and_get_sampling_client
  -> proxy.SaveWeights / RefreshWeights
  -> ControllerActor records ready SGLang route + artifact version
  -> returns SamplingClient configured with HTTP URL

SamplingClient.sample_text
  -> SGLang HTTP /generate
  -> local client response
```

Local fake/singleton tests can still route sampling through `ProxyActor` and
`RolloutActor` to avoid requiring SGLang or an HTTP server.

Telemetry for direct HTTP sampling should be reported separately after the
sampling call or collected from SGLang/server logs. It should not put the
controller back in the synchronous token path.

### Role Attachment

```text
trainer/rollout Modal function starts with i6pn=True and exact region pin
Monarch-managed role starts run_worker_loop_forever(address=tcp://[ipv6]:port)
role publishes endpoint metadata to Modal Dict/Queue
controller reads endpoint metadata
controller calls monarch.actor.attach_to_workers(...)
controller spawns or talks to role actors on the attached mesh
```

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
  containers with i6pn enabled and the same exact region pin.

sglang-distributed
  Starts proxy, controller, fake trainer, real SGLang inference service with
  all private-networked roles in the same exact region.

megatron-sglang-distributed
  Starts proxy, controller, Megatron trainer, and SGLang inference. Use a
  clustered trainer only when the trainer requires gang-scheduled multi-node
  placement.
```

The first mode should be CPU-only except for the Modal private networking
requirement. It should prove that plain i6pn-enabled functions in the same
exact region can publish addresses through the registry and reach one another
without `modal.experimental.clustered(...)`.

## Internal Service Startup

A Modal local entrypoint should submit one orchestration function. That function
starts or references:

- a controller function;
- a proxy function with a Modal tunnel for external gRPC;
- a trainer role function with `i6pn=True` and the deployment region pin;
- an inference role function with `i6pn=True` and the deployment region pin.

For Monarch-managed trainer workers, roles should follow the
`attach_to_workers` pattern:

```text
trainer role starts run_worker_loop_forever(...)
trainer role publishes tcp://[private-ipv6]:port to the registry
controller reads the trainer endpoints
controller calls enable_transport("tcp://[controller-private-ipv6]:controller-port")
controller calls attach_to_workers(...)
```

This keeps the controller as the orchestrator without requiring reverse
Monarch phone-home over a public tunnel.

The controller transport bind is not optional for the Modal distributed path.
Monarch attach pushes controller config to worker host agents during attach, so
the controller must advertise an address that workers can reach over i6pn. The
default hostname-based TCP transport is not sufficient inside Modal because the
container hostname resolves to local names such as `localhost`/`modal`, not a
peer-reachable private IPv6 address.

The controller should not depend on local process globals. Every role should be
able to reconstruct state from:

- launch arguments;
- registry entries;
- Modal Volume artifact metadata.

## Security

- External traffic enters only through the proxy.
- Internal Monarch worker listeners must bind only on private i6pn addresses.
- The first implementation can use Monarch's current `trust_all_connections`
  mode inside Modal private networking. Before broader deployment, add a real
  authentication story for worker attachment.
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
- Controller routing with fake trainer/rollout actor handles.
- Epoch mismatch rejection.
- Heartbeat timeout detection.
- Artifact version visibility rules.
- Proxy delegates to controller and does not mutate topology.

### Local Integration Tests

Use in-process fake clients, local Monarch process meshes, or loopback gRPC only
at the external proxy boundary. Do not require Modal.

```text
client -> proxy server -> ProxyActor -> ControllerActor -> fake trainer/fake rollout
```

### Modal Smokes

1. `fake-distributed`
   - Separate Modal containers.
   - Monarch `attach_to_workers` over i6pn private networking.
   - Controller calls `enable_transport("tcp://[controller-i6pn]:port")` before
     any other Monarch API.
   - Fake trainer and fake inference.
   - Proves discovery, heartbeats, routing, and artifact download.

2. `sglang-distributed`
   - Fake trainer emits base-model or fake HF artifact metadata.
   - Real SGLang inference service on GPU.
   - Controller refreshes inference and returns a direct SGLang HTTP route for
     sampling.

3. `megatron-sglang-distributed`
   - Megatron Bridge/Core trainer on clustered GPU role.
   - Exports LoRA adapter.
   - SGLang loads adapter.
   - End-to-end SFT step followed by sample.

## Implementation Plan

1. Add controller actor contracts using existing Python dataclasses.
2. Add a `RunRegistry` abstraction with in-memory and Modal Dict backends.
3. Add controller state machine with fake trainer/rollout actor handles.
4. Add Modal role functions that start Monarch worker listeners, publish i6pn
   endpoints, and stay alive.
5. Add controller-side explicit i6pn transport binding and
   `attach_to_workers` orchestration for the published role endpoints.
6. Refactor proxy server to route to `ControllerActor` instead of directly
   owning trainer/rollout topology.
7. Add fake role actors and local Monarch integration tests.
8. Add `modal_apps/distributed_mesh.py --mode fake-distributed`.
9. Add SGLang distributed rollout/inference role and Modal smoke.
10. Add training role for single-node Megatron.
11. Add clustered Megatron trainer role only when multi-node training needs
    gang scheduling.
12. Add save/export/refresh orchestration from trainer to SGLang.
13. Add failure/heartbeat/epoch tests and served/manual tunnel mode.

## Acceptance Criteria

- User-facing `ServiceClient` API remains stable.
- Proxy and controller can run as separate processes.
- Controller owns run topology and state transitions.
- Fake trainer and fake inference can run on separate Modal containers.
- A Modal smoke proves controller `attach_to_workers` against a private i6pn
  worker endpoint in the same exact region.
- SGLang inference can run on a separate Modal GPU container and be reached over
  private networking.
- The controller can refresh SGLang to a selected artifact version.
- A Modal smoke proves `client -> proxy -> controller -> refresh -> direct
  SamplingClient HTTP call to SGLang`.
- A later Modal smoke proves `client -> proxy -> controller -> trainer ->
  artifact -> SGLang route -> direct sample`.
- Local tests remain CPU-only and do not import heavy ML packages by default.

## Resolved Decisions

- The controller should be a long-lived CPU Modal function with a long input
  timeout. It owns deployment and discovery of the other roles.
- Use Modal i6pn private networking plus exact region pinning for private role
  connectivity. All controller, proxy, trainer, and inference roles that need
  to communicate over i6pn must use the same exact Modal region, for example
  `region="us-east-1"`.
- Do not use a public `modal.forward` tunnel as the Monarch worker rendezvous
  mechanism. Use the Monarch `attach_to_workers` style: trainer workers listen
  on private i6pn addresses, publish endpoints to the registry, and the
  controller attaches to those worker addresses.
- In Modal, initialize the controller's Monarch context with an explicit i6pn
  bind address before any other Monarch call; workers must be able to reach the
  controller's advertised transport during attach/config push.
- Do not create internal controller/trainer/rollout protobuf services for this
  milestone. Internal role orchestration should use Monarch actor endpoints and
  Python contracts; protobuf remains the external client/proxy boundary.
- `modal.experimental.clustered(...)` is not required for i6pn connectivity.
  Use it only when the trainer needs gang-scheduled multi-node placement.
- `SamplingClient` should receive an HTTP URL and call the Modal/SGLang HTTP
  server directly. This lets sampling use Modal's native HTTP serving and
  autoscaling path.
- The first implementation does not need controller recovery. If the controller
  dies, active runs can be marked interrupted or failed.
- The trainer role may stay warm and accept multiple runs in serial. It does
  not need to support multiple concurrent runs.
- Artifact refresh should return an operation ID that clients can poll.

## References

- RFC 0007: Remote gRPC Access To A Modal-Hosted Monarch Mesh.
- Modal cluster networking: https://modal.com/docs/guide/private-networking
- Modal multi-node clusters: https://modal.com/docs/guide/multi-node-training
- Modal tunnels: https://modal.com/docs/guide/tunnels
- Modal multinode training guide example:
  https://github.com/modal-labs/multinode-training-guide/blob/main/benchmark/modal_train.py
