# RFC 0007: Remote TCP Access To A Modal-Hosted Monarch Mesh

Status: Draft

## Summary

The next milestone should not be multinode training yet. For now, assume Ganker runs on a single Modal node with one local Monarch mesh. The important change is that the client is no longer in the same Python process as the mesh. A remote client should be able to reach the proxy over TCP through a stable Ganker transport.

The public API remains:

```text
ServiceClient / TrainingClient / SamplingClient
        |
        v
ProxyTransport
```

The new work is a TCP transport and a Modal-hosted TCP proxy server:

```text
remote client process
        |
        | framed TCP request/response
        v
TcpProxyTransport
        |
        v
ProxyTcpServer  (inside Modal worker)
        |
        | Monarch actor calls
        v
ProxyActor -> TrainingActor / RolloutActor / TelemetryActor
```

This gets us out of the current all-local shape without requiring multi-host Monarch placement, external gRPC, or distributed Megatron ranks.

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
- Expose the `ProxyActor` through a TCP server that lives inside that Modal worker.
- Add a `TcpProxyTransport` so `ServiceClient` can talk to the proxy without Monarch actor handles.
- Keep request/response contracts as the existing dataclasses.
- Keep local unit tests CPU-only and independent of Modal, CUDA, Megatron, SGLang, and model downloads.
- Add lightweight integration tests for the TCP transport using fake backends.
- Add a Modal smoke where a separate client connects over TCP to the server and runs the existing public SFT flow.

## Non-Goals

- Do not implement multi-node Megatron yet.
- Do not split proxy/training/rollout/telemetry across Modal containers yet.
- Do not require users to speak Monarch.
- Do not introduce gRPC unless we later need a standardized external wire protocol.
- Do not solve public internet exposure policy in this RFC. The first smoke can assume the TCP endpoint is reachable from the test client environment.

## Target Architecture

```text
                outside mesh boundary
remote python client
        |
        | ServiceClient.connect_tcp(host, port)
        v
+----------------------+
| TcpProxyTransport    |
+----------+-----------+
           |
           | length-prefixed TCP frames
           v
========================================================
Modal worker / single node

+----------------------+
| ProxyTcpServer       |
| - listens on TCP     |
| - decodes requests   |
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

## TCP Wire Contract

Use a small framed protocol over TCP:

```text
uint32_be frame_length
json payload bytes
```

Initial payload shape:

```json
{
  "request_id": "req-123",
  "method": "forward_backward",
  "body": {
    "...": "serialized request dataclass"
  }
}
```

Response shape:

```json
{
  "request_id": "req-123",
  "ok": true,
  "body": {
    "...": "serialized response dataclass"
  }
}
```

Error response:

```json
{
  "request_id": "req-123",
  "ok": false,
  "error_type": "InvalidRequestError",
  "message": "..."
}
```

JSON is enough for the first milestone because the existing contract payloads are small and dataclass-shaped. We can move to msgpack or protobuf only if payload size or multi-language clients require it.

## Serialization

Add explicit serialization helpers for the request/response dataclasses:

```text
ganker.wire
  encode_request(method, dataclass) -> dict
  decode_request(method, dict) -> dataclass
  encode_response(method, dataclass) -> dict
  decode_response(method, dict) -> dataclass
```

The serializer should be explicit rather than blindly pickling objects over the network. Pickle is acceptable inside Monarch actor internals, but the remote TCP boundary should be inspectable and stable.

The first supported methods should match `ProxyTransport`:

```text
create_training_run
forward_backward
optim_step
save_weights
refresh_weights
sample
get_telemetry_summary
```

## Client API

Add a connection constructor without changing training-loop code:

```python
client = ServiceClient.connect_tcp("127.0.0.1", 38211, timeout=120)
training = client.create_training_client(
    base_model="Qwen/Qwen3-0.6B",
    tuning="lora",
    rank=8,
)
```

Implementation:

```text
ServiceClient.connect_tcp(...)
  -> TcpProxyTransport(host, port, timeout)
  -> ServiceClient(_transport=transport)
```

`ServiceClient.local(...)` remains unchanged for local tests and fast development.

## Server API

Add a server object that owns a local mesh and a TCP listener:

```python
server = start_tcp_proxy_server(
    host="0.0.0.0",
    port=0,
    artifact_root=Path("/tmp/ganker-artifacts"),
    training_backend="megatron",
    training_backend_config={...},
    inference_backend="fake",
)
print(server.bound_host, server.bound_port)
server.serve_forever()
```

Server responsibilities:

- start the local Monarch mesh;
- keep the proxy actor handle private;
- accept TCP connections;
- decode one framed request at a time;
- route to the matching proxy endpoint;
- encode the typed response;
- return typed errors;
- shut down the mesh when the server stops.

Concurrency can be simple initially: one request per connection, or sequential requests on a connection guarded by a per-connection loop. Parallel request handling should wait until the stateful training lifecycle needs it.

## Modal Shape

Add a new Modal app rather than changing `modal_apps/sft.py`:

```text
modal_apps/remote_mesh.py
```

Initial modes:

1. `--mode tcp-smoke-fake`
   - Starts a local Monarch mesh in a Modal worker.
   - Starts `ProxyTcpServer` on localhost.
   - Starts a separate client inside the same Modal function and connects over TCP.
   - Uses fake backends.
   - Proves the remote transport path without GPU dependencies.

2. `--mode tcp-smoke-qwen-lora`
   - Uses the controlled Bridge image from RFC 0006.
   - Starts the same TCP server.
   - Starts a separate client inside the same Modal function and connects over TCP.
   - Runs one Qwen LoRA SFT step.
   - Exports a PEFT adapter safetensors artifact.

3. `--mode serve`
   - Starts the TCP server and reports host/port metadata.
   - Keeps the Modal worker alive for manual remote testing.
   - This mode can assume the caller has a way to reach the Modal worker over TCP, such as a Modal-supported tunnel, private network, or in-cluster client.

The first two modes are enough to prove the Ganker TCP boundary. They do not require Modal multinode clustering.

## Local Testing

Unit tests:

- frame encode/decode;
- request/response dataclass serialization;
- `TcpProxyTransport` maps each method to a framed request;
- server dispatch maps each method to the correct proxy endpoint;
- remote errors become appropriate client exceptions.

Integration tests:

- start a local Monarch mesh and TCP server on `127.0.0.1:0`;
- connect with `ServiceClient.connect_tcp(...)`;
- run the same flow as `test_full_singleton_flow_through_public_client`;
- assert telemetry, artifacts, and sample output match the in-process transport test.

The default `uv run pytest` can include this TCP integration test because it uses fake backends only.

## Artifact Store

Single-node Modal can keep using a filesystem path:

```text
/tmp/ganker-artifacts
```

For any served/manual mode that should outlive one function invocation, mount a Modal Volume at a stable path:

```text
/mnt/ganker-artifacts
```

The TCP transport does not change artifact semantics. `save_weights` still returns a `WeightArtifact` containing manifest and payload paths meaningful to the server environment. A later RFC should define downloadable artifacts for clients outside the Modal filesystem.

## Security

The first milestone can be private/test-only, but the TCP server should still avoid obvious footguns:

- no pickle over the TCP boundary;
- optional shared bearer token in request metadata;
- maximum frame size;
- per-request timeout;
- structured error responses without arbitrary tracebacks by default.

Production authentication, TLS, and public endpoint policy are out of scope for this milestone.

## Later: Separate Actor Meshes

Once remote TCP access works, we can split actors across proc meshes on the same host:

```text
proxy proc mesh      -> ProxyActor
telemetry proc mesh  -> TelemetryActor
training proc mesh   -> TrainingActor
rollout proc mesh    -> RolloutActor
```

That validates lifecycle separation without changing the remote TCP client.

## Later: Multi-Rank Megatron

Multirank training should be a separate milestone after TCP access works.

The eventual shape is:

```text
ProxyActor
  -> TrainingCoordinatorActor
       -> TrainerWorkerActor rank 0
       -> TrainerWorkerActor rank 1
       -> ...
```

The coordinator must make `create_training_run`, `forward_backward`, `optim_step`, and `save_weights` collective operations across all trainer ranks. That work should not block the single-node remote TCP milestone.

## Implementation Plan

1. Add `ganker.wire` serialization helpers for current `ProxyTransport` methods.
2. Add TCP frame read/write helpers with frame-size limits.
3. Add `TcpProxyTransport`.
4. Add `ProxyTcpServer` that wraps a private local Monarch mesh.
5. Add `ServiceClient.connect_tcp(...)`.
6. Add fake-backend local TCP integration tests.
7. Add `modal_apps/remote_mesh.py --mode tcp-smoke-fake`.
8. Add `modal_apps/remote_mesh.py --mode tcp-smoke-qwen-lora`.
9. Add optional bearer-token support.
10. Add served/manual mode once the smoke path is stable.

## Acceptance Criteria

- Existing `ServiceClient.local(...)` behavior is unchanged.
- A local TCP integration test drives the full public client flow through `ServiceClient.connect_tcp(...)`.
- Modal fake TCP smoke passes without GPU dependencies.
- Modal Qwen LoRA TCP smoke passes and exports `hf-lora-adapter`.
- No user-facing API exposes Monarch actor handles.
- No TCP payload uses pickle.

## Open Questions

- Should the first TCP wire format be JSON only, or JSON headers plus binary tensor payloads for future larger batches?
- Should served Modal mode use raw TCP reachability, a Modal tunnel, or a thin HTTP wrapper around the same `ProxyTransport`?
- Should artifacts be downloadable through the TCP server, or should artifact download wait for an object-store-backed artifact layer?
- Do we need streaming/bidirectional requests soon, or are request/response frames enough for the current training API?
