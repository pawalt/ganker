# Modal Distributed Orchestration

This document describes the current distributed smoke paths. The cheap modes use
fake backends, and the GPU modes use the same Monarch `attach_to_workers`
pattern with Megatron Bridge training and optional SGLang rollout inference.

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
Fake or Megatron backend       Fake or SGLang HTTP backend
    |                              ^
    v                              |
Modal Volume: /vol/ganker-artifacts+
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

## Artifacts

Distributed SFT cannot use container-local `/tmp` artifacts. Trainer and
rollout run in separate Modal containers, so saved weights must live on a
shared Volume mounted at the same path in all roles:

```text
trainer save_weights
  -> write /vol/ganker-artifacts/weights/...
  -> artifact_volume.commit()

rollout refresh_weights
  -> artifact_volume.reload()
  -> read or accept /vol/ganker-artifacts/weights/...

rollout sample
  -> use already-loaded artifact state
```

The current Modal smoke uses `ganker-distributed-artifacts` mounted at
`/vol/ganker-artifacts`. This mirrors the later Megatron-to-SGLang path where
the trainer exports a checkpoint or adapter and the rollout service reloads it.
Reload happens before `refresh_weights`, not before every `sample`, because an
SGLang process can keep adapter files open after loading them and Modal rejects
Volume reloads while files are open.

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

The Modal implementation is split into:

```text
modal_apps/distributed/infra.py
  -> Modal app, images, volumes, i6pn worker roles, Monarch attach
  -> generic run_cpu_distributed_job / run_bridge_distributed_job
  -> deployable with modal deploy

modal_apps/distributed/sft_job.py
  -> Tinker-style ServiceClient / TrainingClient / SamplingClient code
  -> job-specific dataset, tokenizer, training, refresh, sampling
  -> modal run entrypoint for smokes and SFT jobs

modal_apps/distributed_mesh.py
  -> compatibility wrapper for older commands
```

Deploy the infra independently:

```bash
source ~/.codex/modal.env
uv run modal deploy modal_apps/distributed/infra.py
```

Plain private IPv6 TCP:

```bash
source ~/.codex/modal.env
uv run modal run modal_apps/distributed/sft_job.py --mode tcp-smoke --port 26620
```

Monarch attach plus fake training flow:

```bash
source ~/.codex/modal.env
uv run modal run modal_apps/distributed/sft_job.py \
  --mode fake-distributed \
  --port 26600 \
  --controller-port 26610
```

Full toy SFT job through the distributed topology:

```bash
source ~/.codex/modal.env
uv run modal run modal_apps/distributed/sft_job.py \
  --mode sft-distributed \
  --port 26600 \
  --controller-port 26610
```

Real Qwen3 0.6B LoRA SFT through Megatron Bridge:

```bash
source ~/.codex/modal.env
GANKER_MODAL_GPU=A100 uv run modal run modal_apps/distributed/sft_job.py \
  --mode qwen-bridge-sft-distributed \
  --port 26600 \
  --controller-port 26610 \
  --startup-timeout 900 \
  --tuning lora \
  --lora-rank 8 \
  --max-steps 1 \
  --sequence-length 32 \
  --micro-batch-size 1
```

Real Qwen3 0.6B LoRA SFT through Megatron Bridge plus SGLang rollout:

```bash
source ~/.codex/modal.env
GANKER_MODAL_GPU=A100 uv run modal run modal_apps/distributed/sft_job.py \
  --mode qwen-bridge-sglang-distributed \
  --port 26600 \
  --controller-port 26610 \
  --startup-timeout 900 \
  --sglang-startup-timeout 900 \
  --tuning lora \
  --lora-rank 8 \
  --max-steps 1 \
  --sequence-length 32 \
  --micro-batch-size 1
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

The `sft-distributed` smoke additionally verifies:

```text
examples.sft.run_sft(...)
  -> 4 toy SFT steps
  -> save LoRA artifact on Modal Volume
  -> rollout refresh from saved artifact
  -> sample through SamplingClient
  -> telemetry records trainer and rollout usage
```

The `qwen-bridge-sft-distributed` smoke replaces the fake trainer worker with
a GPU Modal function running the controlled Megatron Bridge image:

```text
controller function, Bridge image, CPU
  -> attach trainer worker, Bridge image, GPU
  -> attach rollout worker, slim image, CPU fake rollout
  -> HFAutoTokenizerAdapter(Qwen/Qwen3-0.6B)
  -> TrainingActor(training_backend="megatron", runtime_kind="bridge")
  -> load Qwen HF weights through Megatron Bridge
  -> apply LoRA
  -> forward_backward + optim_step
  -> export hf-lora-adapter safetensors to Modal Volume
  -> rollout refresh + sample through SamplingClient
```

The `qwen-bridge-sglang-distributed` smoke keeps that trainer path and replaces
the fake rollout worker with a GPU Modal function running the SGLang image:

```text
controller function, Bridge image, CPU
  -> attach trainer worker, Bridge image, GPU
  -> attach rollout worker, SGLang image, GPU
  -> HFAutoTokenizerAdapter(Qwen/Qwen3-0.6B)
  -> TrainingActor(training_backend="megatron", runtime_kind="bridge")
  -> export hf-lora-adapter safetensors to Modal Volume
  -> RolloutActor(inference_backend="sglang")
  -> launch python -m sglang.launch_server --model-path Qwen/Qwen3-0.6B
  -> load /vol/ganker-artifacts/.../adapter_model.safetensors
  -> sample text through SamplingClient.sample_text(...)
```
