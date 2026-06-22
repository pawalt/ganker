# RFC 0007: Remote gRPC Access To A Modal-Hosted Monarch Mesh

Status: Draft

## Summary

The next milestone should not be multinode training yet. For now, assume Ganker runs on a single Modal node with one local Monarch mesh. The important change is that the client is no longer in the same Python process as the mesh. A remote client should be able to reach the proxy over TCP through a stable Ganker API.

That API should be gRPC, not a custom framed protocol. gRPC gives us explicit contracts, generated clients/servers, deadlines, structured errors, streaming options, and existing tooling while still using TCP underneath.

The public Python API remains:

```text
ServiceClient / TrainingClient / SamplingClient
        |
        v
ProxyTransport
```

The new work is a gRPC transport and a Modal-hosted gRPC proxy server:

```text
remote client process
        |
        | gRPC over TCP
        v
GrpcProxyTransport
        |
        v
ProxyGrpcServer  (inside Modal worker)
        |
        | Monarch actor calls
        v
ProxyActor -> TrainingActor / RolloutActor / TelemetryActor
```

This gets us out of the current all-local shape without requiring multi-host Monarch placement or distributed Megatron ranks.

## Current State

Today `ServiceClient.local(...)` owns the mesh and the client gets a direct in-process Monarch actor handle:

```text
client process
  |
  v
ServiceClient.local(...)
  |
  v
start_local_monarch_mesh(...)
  |
  v
this_host().spawn_procs(name="ganker_local")
  |
  +-- TrainingActor
  +-- RolloutActor
  +-- TelemetryActor
  +-- ProxyActor
  |
  v
MonarchProxyTransport(proxy_actor_handle)
```

That is good for local tests, but it does not prove that a separate client can call into a running mesh. It also makes the Modal examples look more local than the production shape we want.

## Goals

- Run the Ganker actor mesh inside one Modal function/container.
- Expose the `ProxyActor` through a gRPC server that lives inside that Modal worker.
- Add a `GrpcProxyTransport` so `ServiceClient` can talk to the proxy without Monarch actor handles.
- Make the remote API contract explicit in protobuf.
- Keep local unit tests CPU-only and independent of Modal, CUDA, Megatron, SGLang, and model downloads.
- Add lightweight integration tests for the gRPC transport using fake backends.
- Add a Modal smoke where a separate client connects over gRPC to the server and runs the existing public SFT flow.

## Non-Goals

- Do not implement multi-node Megatron yet.
- Do not split proxy/training/rollout/telemetry across Modal containers yet.
- Do not require users to speak Monarch.
- Do not require an HTTP/JSON compatibility API in this milestone.
- Do not solve public internet exposure policy in this RFC. The first smoke can assume the gRPC endpoint is reachable from the test client environment.

## Why gRPC

The custom frame idea was attractive only because it was small. It is the wrong long-term boundary for this project.

gRPC is a better fit because:

- contracts live in `.proto` files instead of informal JSON shapes;
- generated clients and servers keep transport code boring;
- deadlines, cancellation, metadata, status codes, and max message sizes are already handled;
- unary RPCs cover the current API, and server/client streaming is available later;
- grpcurl and reflection can make smoke debugging easier;
- gRPC over TCP satisfies the remote mesh access requirement without exposing Monarch internals.

Ganker still owns the application-level service contract. It should not own a wire protocol.

## Target Architecture

```text
                outside mesh boundary
remote python client
        |
        | ServiceClient.connect_grpc(target)
        v
+----------------------+
| GrpcProxyTransport   |
+----------+-----------+
           |
           | gRPC unary calls over TCP
           v
========================================================
Modal worker / single node

+----------------------+
| ProxyGrpcServer      |
| - listens on gRPC    |
| - validates protobuf |
| - calls ProxyActor   |
+----------+-----------+
           |
           | MonarchProxyTransport or direct actor calls
           v
+----------------------+
| ProxyActor           |
+----+------------+----+
     |            |
     v            v
+-------------+  +-------------+  +----------------+
|TrainingActor|  |RolloutActor |  |TelemetryActor  |
+------+------+  +------+------+  +--------+-------+
       |                |                  |
       v                v                  v
 Megatron/fake      SGLang/fake      telemetry ledger

Shared artifact root: local path or Modal Volume
```

The internal Monarch mesh can still be created with `this_host().spawn_procs(name="ganker_local")` for the first implementation. The milestone is remote reachability, not actor placement.

## Protobuf Contract

Add protobuf definitions under:

```text
proto/ganker/v1/proxy.proto
src/ganker/rpc/generated/
```

The service should mirror the existing `ProxyTransport` methods:

```proto
syntax = "proto3";

package ganker.v1;

service GankerProxy {
  rpc CreateTrainingRun(CreateTrainingRunRequest) returns (CreateTrainingRunResponse);
  rpc ForwardBackward(ForwardBackwardRequest) returns (ForwardBackwardResponse);
  rpc OptimStep(OptimStepRequest) returns (OptimStepResponse);
  rpc SaveWeights(SaveWeightsRequest) returns (SaveWeightsResponse);
  rpc RefreshWeights(RefreshWeightsRequest) returns (RefreshWeightsResponse);
  rpc Sample(SampleRequest) returns (SampleResponse);
  rpc GetTelemetrySummary(GetTelemetrySummaryRequest) returns (GetTelemetrySummaryResponse);
}
```

The proto messages should be direct equivalents of the current dataclasses in `src/ganker/contracts.py`:

```text
RequestContext
ModelInput
TensorData
Datum
SamplingParams
SampledSequence
ForwardBackwardOutput
AdamParams
Usage
UsageEvent
UsageBySource
TelemetrySummary
WeightArtifact
TrainingRun
```

Enums should be protobuf enums, not raw integers:

```proto
enum TuningMode {
  TUNING_MODE_UNSPECIFIED = 0;
  TUNING_MODE_LORA = 1;
  TUNING_MODE_FULL = 2;
}

enum ArtifactKind {
  ARTIFACT_KIND_UNSPECIFIED = 0;
  ARTIFACT_KIND_FULL = 1;
  ARTIFACT_KIND_DELTA = 2;
}
```

Use generated protobuf classes only at the RPC boundary. Inside components, actors, and backends, keep using the existing Python dataclasses.

## Conversion Layer

Add explicit conversion helpers rather than passing protobuf objects through the whole codebase:

```text
ganker.rpc.conversion
  create_training_run_request_from_proto(...)
  create_training_run_response_to_proto(...)
  forward_backward_request_from_proto(...)
  forward_backward_response_to_proto(...)
  ...
```

The conversion layer should be mechanical and unit-tested. This keeps the rest of the system independent of gRPC and avoids letting generated classes leak into training, rollout, telemetry, or backend code.

## Client API

Add a connection constructor without changing training-loop code:

```python
client = ServiceClient.connect_grpc("127.0.0.1:38211", timeout=120)
training = client.create_training_client(
    base_model="Qwen/Qwen3-0.6B",
    tuning="lora",
    rank=8,
)
```

Implementation:

```text
ServiceClient.connect_grpc(...)
  -> grpc.insecure_channel(...) or grpc.secure_channel(...)
  -> generated GankerProxyStub
  -> GrpcProxyTransport(stub, timeout)
  -> ServiceClient(_transport=transport)
```

`ServiceClient.local(...)` remains unchanged for local tests and fast development.

## Server API

Add a server object that owns a local mesh and a gRPC listener:

```python
server = start_grpc_proxy_server(
    bind="0.0.0.0:0",
    artifact_root=Path("/tmp/ganker-artifacts"),
    training_backend="megatron",
    training_backend_config={...},
    inference_backend="fake",
)
print(server.bound_address)
server.serve_forever()
```

Server responsibilities:

- start the local Monarch mesh;
- keep the proxy actor handle private;
- accept gRPC calls;
- convert protobuf requests into existing dataclasses;
- route to the matching proxy endpoint;
- convert dataclass responses into protobuf responses;
- map known Ganker exceptions to gRPC status codes;
- shut down the mesh when the server stops.

The first implementation should use synchronous `grpc.server(...)` because the public `ProxyTransport` API is synchronous today. We can switch the server internals to `grpc.aio` later if concurrent streaming or async cancellation becomes important.

## Error Semantics

Map known exceptions to stable gRPC status codes:

```text
InvalidRequestError      -> INVALID_ARGUMENT
unknown run/artifact     -> NOT_FOUND
timeout                  -> DEADLINE_EXCEEDED
backend unavailable      -> UNAVAILABLE
unexpected exception     -> INTERNAL
```

The server should include a short public error message. Full tracebacks stay in logs.

Client-side `GrpcProxyTransport` should translate gRPC failures back into Ganker exceptions where the mapping is unambiguous. Otherwise it should raise a transport error that includes the gRPC code and details.

## Modal Shape

Add a new Modal app rather than changing `modal_apps/sft.py`:

```text
modal_apps/remote_mesh.py
```

Initial modes:

1. `--mode grpc-smoke-fake`
   - Starts a local Monarch mesh in a Modal worker.
   - Starts `ProxyGrpcServer` on localhost.
   - Starts a separate client inside the same Modal function and connects over gRPC.
   - Uses fake backends.
   - Proves the remote transport path without GPU dependencies.

2. `--mode grpc-smoke-qwen-lora`
   - Uses the controlled Bridge image from RFC 0006.
   - Starts the same gRPC server.
   - Starts a separate client inside the same Modal function and connects over gRPC.
   - Runs one Qwen LoRA SFT step.
   - Exports a PEFT adapter safetensors artifact.

3. `--mode serve`
   - Starts the gRPC server and reports host/port metadata.
   - Keeps the Modal worker alive for manual remote testing.
   - This mode can assume the caller has a way to reach the Modal worker over TCP, such as a Modal-supported tunnel, private network, or in-cluster client.

The first two modes are enough to prove the Ganker gRPC boundary. They do not require Modal multinode clustering.

## Dependencies

Use `uv` and lock all dependency changes.

Runtime dependencies:

```text
grpcio
protobuf
```

Development/codegen dependencies:

```text
grpcio-tools
```

Generated files should be committed so normal users do not need `grpcio-tools` at runtime. A regeneration command should live in project docs or a small script:

```text
uv run python -m grpc_tools.protoc \
  -I proto \
  --python_out=src \
  --grpc_python_out=src \
  proto/ganker/v1/proxy.proto
```

## Local Testing

Unit tests:

- protobuf/dataclass conversion for every message;
- `GrpcProxyTransport` maps each method to the generated stub method;
- server dispatch maps each gRPC method to the correct proxy endpoint;
- known exceptions map to expected gRPC status codes;
- gRPC status errors map back to client-side Ganker errors.

Integration tests:

- start a local Monarch mesh and gRPC server on `127.0.0.1:0`;
- connect with `ServiceClient.connect_grpc(...)`;
- run the same flow as `test_full_singleton_flow_through_public_client`;
- assert telemetry, artifacts, and sample output match the in-process transport test.

The default `uv run pytest` can include this gRPC integration test because it uses fake backends only.

## Artifact Store

Single-node Modal can keep using a filesystem path:

```text
/tmp/ganker-artifacts
```

For any served/manual mode that should outlive one function invocation, mount a Modal Volume at a stable path:

```text
/mnt/ganker-artifacts
```

The gRPC transport does not change artifact semantics. `save_weights` still returns a `WeightArtifact` containing manifest and payload paths meaningful to the server environment. A later RFC should define downloadable artifacts for clients outside the Modal filesystem.

## Security

The first milestone can be private/test-only, but the gRPC server should still avoid obvious footguns:

- no pickle over the gRPC boundary;
- optional bearer token in gRPC metadata;
- maximum receive/send message sizes;
- per-request deadline;
- structured error responses without arbitrary tracebacks by default.

Production authentication, TLS, and public endpoint policy are out of scope for this milestone.

## Later: Separate Actor Meshes

Once remote gRPC access works, we can split actors across proc meshes on the same host:

```text
proxy proc mesh      -> ProxyActor
telemetry proc mesh  -> TelemetryActor
training proc mesh   -> TrainingActor
rollout proc mesh    -> RolloutActor
```

That validates lifecycle separation without changing the remote gRPC client.

## Later: Multi-Rank Megatron

Multirank training should be a separate milestone after gRPC access works.

The eventual shape is:

```text
ProxyActor
  -> TrainingCoordinatorActor
       -> TrainerWorkerActor rank 0
       -> TrainerWorkerActor rank 1
       -> ...
```

The coordinator must make `create_training_run`, `forward_backward`, `optim_step`, and `save_weights` collective operations across all trainer ranks. That work should not block the single-node remote gRPC milestone.

## Implementation Plan

1. Add `proto/ganker/v1/proxy.proto`.
2. Add generated protobuf/gRPC Python modules under `src/ganker/rpc/generated/`.
3. Add `ganker.rpc.conversion` for dataclass/protobuf mapping.
4. Add `GrpcProxyTransport`.
5. Add `ProxyGrpcServer` that wraps a private local Monarch mesh.
6. Add `ServiceClient.connect_grpc(...)`.
7. Add fake-backend local gRPC integration tests.
8. Add `modal_apps/remote_mesh.py --mode grpc-smoke-fake`.
9. Add `modal_apps/remote_mesh.py --mode grpc-smoke-qwen-lora`.
10. Add optional bearer-token metadata support.
11. Add served/manual mode once the smoke path is stable.

## Acceptance Criteria

- Existing `ServiceClient.local(...)` behavior is unchanged.
- A committed `.proto` defines the remote proxy service.
- Generated gRPC code is checked in or regenerated by a documented `uv` command.
- A local gRPC integration test drives the full public client flow through `ServiceClient.connect_grpc(...)`.
- Modal fake gRPC smoke passes without GPU dependencies.
- Modal Qwen LoRA gRPC smoke passes and exports `hf-lora-adapter`.
- No user-facing API exposes Monarch actor handles.
- No gRPC payload uses pickle.

## Open Questions

- Should the first implementation expose gRPC reflection for easier `grpcurl` debugging?
- Should served Modal mode use raw gRPC reachability, a Modal tunnel, or a thin Modal web endpoint that forwards to the same gRPC server?
- Should artifacts be downloadable through a gRPC endpoint, or should artifact download wait for an object-store-backed artifact layer?
- Do we need streaming RPCs soon, or are unary request/response calls enough for the current training API?
