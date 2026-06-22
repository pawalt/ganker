# Local Testing Orchestration

Most local tests do not start ports or gRPC servers. They start a Monarch process mesh and spawn actors into that mesh. The remote-boundary integration tests additionally bind a loopback gRPC server on `127.0.0.1:0` and still use fake backends only. User-facing tests call `ServiceClient`, not Monarch actor handles.

## Harness

`ganker.orchestration.start_local_monarch_mesh(tmp_path)` does this:

```text
enable_transport("tcp")
        |
        v
this_host().spawn_procs(name="ganker-local")
        |
        +-- spawn("training", TrainingActor, artifact_root)
        +-- spawn("rollout", RolloutActor, artifact_root)
        +-- spawn("telemetry", TelemetryActor)
        +-- spawn("proxy", ProxyActor, training, rollout, telemetry)
        |
        v
wait for actor.initialized
```

The public client uses `MonarchProxyTransport` internally:

```text
ServiceClient.local(tmp_path)
        |
        v
start_local_monarch_mesh(tmp_path)
        |
        v
MonarchProxyTransport(mesh.proxy)
        |
        v
TrainingClient / SamplingClient methods
```

## Unit vs Integration Tests

```text
unit tests
  contracts       plain dataclasses
  artifacts       filesystem store
  backends        fake training/inference
  sglang backend  injected runtime/client, no SGLang server
  components      pure request/response behavior
  client          fake ProxyTransport
  telemetry       in-memory event ledger

integration test
  real Monarch process mesh
  public ServiceClient
  real actor endpoint calls
  fake local backends
  temporary filesystem artifact root

grpc integration test
  real Monarch process mesh
  local ProxyGrpcServer on 127.0.0.1:0
  ServiceClient.connect_grpc(...)
  fake local backends
  artifact download over gRPC
```

This keeps component behavior easy to test without Monarch while still verifying the actual actor orchestration path.

## Shutdown

`LocalMonarchMesh.stop()` first calls explicit actor shutdown endpoints, then stops the spawned actors and process mesh:

```text
TrainingActor.shutdown()  -> TrainingComponent.shutdown() -> backend.close()
RolloutActor.shutdown()   -> RolloutComponent.shutdown()  -> backend.close() if present
TelemetryActor.shutdown()
ProxyActor.shutdown()

ProxyActor.stop()
TrainingActor.stop()
RolloutActor.stop()
TelemetryActor.stop()
ProcMesh.stop()
```

This gives stateful backends, including the Megatron runtime adapter, a chance
to flush or release runtime handles before Monarch tears down the process mesh.

The harness does not call `this_host().shutdown()` because `this_host()` returns a reference to the current host, not an owned host allocation.
