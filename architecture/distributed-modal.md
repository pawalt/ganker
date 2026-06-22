# Modal Distributed Orchestration

This document describes the current distributed smoke path. It is still a fake
trainer/fake rollout implementation, but it uses the same Monarch
`attach_to_workers` pattern intended for real Modal deployment.

## Shape

```text
external client
      |
      | public API: ServiceClient / gRPC proxy boundary
      v
+-------------+
| ProxyActor  |
+-------------+
      |
      | private Monarch actor call
      v
+-----------------+
| ControllerActor |
+-----------------+
    |      |       |
    |      |       +----------------+
    |      |                        |
    v      v                        v
+---------+----------------+   +--------------+
| trainer worker function  |   | rollout      |
| i6pn=True                |   | worker fn    |
| region="us-east-1"       |   | i6pn=True    |
+--------------------------+   +--------------+
    |                              |
    v                              v
TrainingActor                 RolloutActor
Megatron backend later         SGLang HTTP backend later
```

The external client should not speak Monarch. Internal Modal roles should not
add controller/trainer/rollout gRPC services. gRPC remains the public proxy
boundary; Monarch actors are the internal orchestration boundary.

## Rendezvous

Each worker role starts a Monarch worker listener and publishes endpoint
metadata to the run registry:

```text
worker container starts
  -> resolves i6pn.modal.local
  -> run_worker_loop_forever("tcp://[worker-i6pn]:26600")
  -> registry.put("ganker:{deployment}:{run}:{role}:{rank}", endpoint)
  -> waits for controller attach
```

The controller then attaches to those private endpoints:

```text
controller container starts
  -> resolves i6pn.modal.local
  -> enable_transport("tcp://[controller-i6pn]:26610")
  -> reads worker endpoints from registry
  -> attach_to_workers(["tcp://[worker-i6pn]:26600"])
  -> spawn TrainingActor / RolloutActor on attached hosts
  -> spawn ControllerActor / ControllerProxyActor locally
```

The explicit controller bind is required. Monarch attach pushes controller
configuration to remote host agents, so workers need a controller address they
can reach over i6pn. Modal container hostnames such as `modal` or `localhost`
are not sufficient for cross-container Monarch attach.

## Placement

All roles that communicate over i6pn must use:

```python
@app.function(i6pn=True, region="us-east-1")
```

Use an exact region such as `us-east-1`. Do not use a broad region such as
`us-east` for roles that need private networking with each other.
`modal.experimental.clustered(...)` is only needed later for gang-scheduled
multi-node training; it is not required for i6pn connectivity.

## Smoke Tests

Plain private IPv6 TCP:

```bash
source ~/.codex/modal.env
uv run modal run modal_apps/distributed_mesh.py --mode tcp-smoke --port 26620
```

Monarch attach plus fake training flow:

```bash
source ~/.codex/modal.env
uv run modal run modal_apps/distributed_mesh.py \
  --mode fake-distributed \
  --port 26600 \
  --controller-port 26610
```

The `fake-distributed` smoke verifies:

```text
Modal controller function
  -> worker endpoint discovery through Modal Dict
  -> Monarch attach_to_workers over private i6pn
  -> ControllerProxyActor
  -> ControllerActor
  -> TrainingActor on trainer worker
  -> RolloutActor on rollout worker
```
