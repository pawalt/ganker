# Ganker Singleton Architecture

Ganker is a local singleton prototype for a Tinker-style training API. The key boundary is between algorithm-level requests and infrastructure execution. In this version, PyTorch Monarch is the internal orchestration layer.

## Components

```text
client / training loop
        |
        | ServiceClient / TrainingClient / SamplingClient
        v
  +----------------+
  | ProxyTransport |
  +----------------+
        |
        | local: Monarch actor endpoint calls
        | remote: gRPC calls into ProxyGrpcServer
        v
  +-------------+
  | ProxyActor  |
  +-------------+
    |     |     |
    |     |     +------------------+
    |     |                        |
    v     v                        v
+---------------+          +----------------+          +----------------+
| TrainingActor |          | RolloutActor   |          | TelemetryActor |
+---------------+          +----------------+          +----------------+
        |                          |                          |
        v                          v                          v
+---------------------+    +----------------------+    +----------------+
| TrainingBackend     |    | InferenceBackend     |    | TelemetryLedger|
| fake now            |    | fake now             |    | in memory      |
| Megatron later      |    | sglang later         |    |                |
+---------------------+    +----------------------+    +----------------+
        |
        v
+--------------------------+
| FilesystemArtifactStore  |
| local Modal Volume stand |
+--------------------------+
        ^
        |
        +------ RolloutActor pulls latest artifacts
```

## Client Boundary

The user-facing client does not speak Monarch directly. It calls `ServiceClient`
`TrainingClient`, and `SamplingClient`, which build typed request dataclasses
and send them through a `ProxyTransport`.

```text
user code
  |
  v
ServiceClient.create_lora_training_client(...)
  |
  v
TrainingClient.forward_backward(...)
TrainingClient.optim_step(...)
TrainingClient.save_weights_and_get_sampling_client(...)
  |
  v
SamplingClient.sample(...)
  |
  v
ProxyTransport
```

For local development, `MonarchProxyTransport` wraps a `ProxyActor` handle. For
remote clients, `GrpcProxyTransport` calls `ProxyGrpcServer`, which owns the
local Monarch mesh privately and delegates to the same `ProxyActor`.

## Why Monarch Internally And gRPC Externally

Monarch gives this prototype the orchestration primitive we need for the internal mesh: actor processes, endpoint calls, futures, and later multi-process or multi-host placement. That replaces gRPC for trainer/proxy/rollout/telemetry communication inside the system.

gRPC is the external process boundary. `ServiceClient.connect_grpc(...)` keeps
remote callers off Monarch while giving the project explicit protobuf
contracts, generated stubs, deadlines, metadata, and status codes. The gRPC
server is an adapter; it should not replace the internal Monarch actor graph.

## Singleton Flow

```text
create_training_run
client -> ProxyTransport -> ProxyActor -> TrainingActor -> FakeTrainingBackend

forward_backward
client -> ProxyTransport -> ProxyActor -> TrainingActor -> FakeTrainingBackend
                         |
                         v
                  returns Usage
                         |
                         v
                  ProxyActor -> TelemetryActor

optim_step
client -> ProxyTransport -> ProxyActor -> TrainingActor -> FakeTrainingBackend
                         |
                         v
                  returns Usage
                         |
                         v
                  ProxyActor -> TelemetryActor

save_weights_and_get_sampling_client
client -> ProxyTransport -> ProxyActor -> TrainingActor -> FilesystemArtifactStore
client -> ProxyTransport -> ProxyActor -> RolloutActor -> FakeInferenceBackend

sample
client -> ProxyTransport -> ProxyActor -> RolloutActor -> FakeInferenceBackend
                                    -> FilesystemArtifactStore
                         |
                         v
                  returns Usage
                         |
                         v
                  ProxyActor -> TelemetryActor

download_artifact_file
client -> ProxyTransport -> local file read or ProxyGrpcServer -> FilesystemArtifactStore
```

## Production Shape

```text
Monarch controller
        |
        +-- proxy mesh
        +-- trainer mesh -> Megatron backend
        +-- rollout mesh -> sglang backend
        +-- telemetry mesh
```

The local singleton keeps all actors on a small local Monarch process mesh. Production can place trainer and rollout actors on separate meshes while preserving the same endpoint contracts.
