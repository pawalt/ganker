# Local Testing Orchestration

Local tests do not start ports or gRPC servers. They start a Monarch process mesh and spawn actors into that mesh. User-facing tests call `ServiceClient`, not Monarch actor handles.

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
  components      pure request/response behavior
  client          fake ProxyTransport
  telemetry       in-memory event ledger

integration test
  real Monarch process mesh
  public ServiceClient
  real actor endpoint calls
  fake local backends
  temporary filesystem artifact root
```

This keeps component behavior easy to test without Monarch while still verifying the actual actor orchestration path.

## Shutdown

`LocalMonarchMesh.stop()` stops the spawned actors and process mesh:

```text
ProxyActor.stop()
TrainingActor.stop()
RolloutActor.stop()
TelemetryActor.stop()
ProcMesh.stop()
```

The harness does not call `this_host().shutdown()` because `this_host()` returns a reference to the current host, not an owned host allocation.
