# Project Instructions

## Version Control

- Use `jj` for version control inspection and change management.
- Do not use destructive VCS commands unless explicitly requested.

## Python Environment

- Use `uv` for dependency management.
- Keep dependencies in `pyproject.toml` and `uv.lock`.
- Do not install Python packages at the system level.
- Run tests with:

```bash
uv run pytest
```

## Architecture Direction

- Internal orchestration uses PyTorch Monarch actors, not gRPC.
- External remote clients use gRPC through `ServiceClient.connect_grpc(...)` and `GrpcProxyTransport`; keep Monarch handles private to the server process.
- Do not add internal controller/trainer/rollout protobuf or gRPC services for distributed Modal orchestration unless explicitly requested. Use Monarch `attach_to_workers` and actor endpoints internally; protobuf belongs at the external proxy boundary.
- Public callers should use `ServiceClient` / `TrainingClient`; do not expose Monarch `.choose(...).get()` calls in user-facing APIs.
- New external proxy adapters should implement `ProxyTransport`.
- Keep request/response contracts as plain Python dataclasses in `ganker.contracts`; convert to protobuf only at the `ganker.rpc` boundary.
- Emulate Tinker's outer API shape where practical (`Datum`, `ModelInput`, `TensorData`, `TrainingClient`, `SamplingClient`), but keep local payload implementations lightweight unless a real integration requires the full SDK type system.
- The local singleton mesh is spawned through `ganker.orchestration.start_local_monarch_mesh`.
- Use `TelemetryActor` for usage/events/tracking. Do not call this component `BillingActor`; pricing and billing policy are out of scope for this layer.

## Backend Boundaries

- Production training is expected to use Megatron behind `TrainingBackend`.
- Production inference is expected to use `sglang` behind `InferenceBackend`.
- Local development and tests must keep working without Megatron, `sglang`, CUDA, model weights, or checkpoints.
- Heavy backend imports must stay isolated in their adapter modules.
- The SGLang inference backend talks to SGLang over HTTP and must remain import-isolated; importing `ganker.backends.sglang` must not import `sglang`.
- SGLang local tests should inject `SGLangRuntime` or `SGLangHTTPClient` fakes and assert `/generate`, `/load_lora_adapter`, and artifact contract behavior without starting a model server.
- SGLang-compatible artifacts are HF safetensors full checkpoints, HF/PEFT LoRA adapters, or base-model payloads. Raw Megatron checkpoint artifacts should fail clearly until an export step produces one of those formats.
- The Megatron backend owns a stateful runtime lifecycle: one active run per actor, `forward_backward` moves to pending gradients, `optim_step` consumes pending gradients, and `save_weights` is allowed only from ready state.
- `runtime_kind="core"` is the Modal smoke path and uses Megatron-Core directly. `runtime_kind="bridge"` remains the default for future Bridge/HF conversion work.
- Test Megatron lifecycle behavior with injected runtimes; do not require real Megatron imports for lifecycle unit tests.

## Testing Expectations

- Unit test pure components and fake backends directly.
- Integration test orchestration through real Monarch actor endpoints.
- Local integration tests should use fake backends and temporary filesystem artifact roots. gRPC integration tests may bind `127.0.0.1:0`, but must not require Modal, CUDA, Megatron, SGLang, or model downloads.
- CPU Megatron preflight tests should use `pytest -m megatron_cpu`; they may skip optional torch/Megatron Bridge checks when those packages are absent.
- Real Megatron execution smoke tests should run through Modal/GPU infrastructure, not the default local suite. Use `source ~/.codex/modal.env` and `modal run modal_apps/megatron_smoke.py --mode megatron`.
- Remote mesh smoke tests live in `modal_apps/remote_mesh.py`. Use `modal run modal_apps/remote_mesh.py --mode grpc-smoke-fake` for fake backends, `--mode grpc-smoke-qwen-lora` for Bridge/GPU, and `--mode serve` for a singleton gRPC server behind a Modal tunnel.
- Modal smoke implementation lives in `tests/modal_smoke/` as importable test helpers. `scripts/megatron_bridge_smoke.py` is only a compatibility CLI wrapper, and `modal_apps/megatron_smoke.py` should call the importable helpers directly.
- SGLang backend tests live in `tests/test_sglang_backend.py`; run them with `uv run pytest tests/test_sglang_backend.py` and do not require SGLang, CUDA, or model downloads.
- Real SGLang execution smoke tests should run through Modal/GPU infrastructure. Use `source ~/.codex/modal.env` and `modal run modal_apps/sglang_smoke.py --mode client`.
- Distributed Modal roles that communicate over i6pn must all use `i6pn=True` and the same exact Modal region, such as `region="us-east-1"`. Do not use broad/meta regions such as `us-east` for i6pn-connected controller, proxy, trainer, or inference roles.
- Distributed Monarch attach across Modal containers needs a reachable controller transport too. Call `enable_transport("tcp://[controller-i6pn]:port")` before any other Monarch API in the controller container, then `attach_to_workers(...)` against worker endpoints like `tcp://[worker-i6pn]:26600`.
- The clean Qwen SFT example lives in `modal_apps/qwen_sft/`: `infra.py` deploys only the Qwen3 0.6B Megatron Bridge trainer plus SGLang rollout infra, `sft.py` contains the single Tinker-style SFT flow, and `compare_hf.py` compares a longer Alpaca-style SFT loss curve against a Hugging Face Trainer/PEFT baseline. Deploy infra with `GANKER_MODAL_GPU=A100 uv run modal deploy modal_apps/qwen_sft/infra.py`; run the full rollout example with `GANKER_MODAL_GPU=A100 uv run modal run modal_apps/qwen_sft/sft.py --startup-timeout 900 --sglang-startup-timeout 900`; run the loss comparison with `GANKER_MODAL_GPU=A100 uv run modal run modal_apps/qwen_sft/compare_hf.py --startup-timeout 900 --dataset-size 256 --max-steps 20 --sequence-length 256`.
- Distributed Modal infra lives in `modal_apps/distributed/infra.py`; SFT job logic lives in `modal_apps/distributed/sft_job.py`; `modal_apps/distributed_mesh.py` is only a compatibility wrapper. Deploy infra with `source ~/.codex/modal.env` and `uv run modal deploy modal_apps/distributed/infra.py`.
- Distributed Modal smoke tests can run with `uv run modal run modal_apps/distributed/sft_job.py --mode tcp-smoke --port 26620` to verify private i6pn TCP, then `uv run modal run modal_apps/distributed/sft_job.py --mode fake-distributed --port 26600 --controller-port 26610` to verify Monarch `attach_to_workers`.
- Full distributed SFT smoke runs with `uv run modal run modal_apps/distributed/sft_job.py --mode sft-distributed --port 26600 --controller-port 26610`. It uses a Modal Volume mounted at `/vol/ganker-artifacts`; trainer actors must commit after `save_weights`, and rollout actors must reload during `refresh_weights` before loading saved artifacts. Do not reload the artifact Volume on every SGLang sample because loaded adapters can keep files open.
- Real distributed Qwen/Megatron Bridge SFT runs with `GANKER_MODAL_GPU=A100 uv run modal run modal_apps/distributed/sft_job.py --mode qwen-bridge-sft-distributed --port 26600 --controller-port 26610 --startup-timeout 900 --tuning lora --lora-rank 8 --max-steps 1 --sequence-length 32 --micro-batch-size 1`. This uses a GPU Bridge trainer worker and a CPU fake rollout worker over Monarch/i6pn.
- Real distributed Qwen/Megatron Bridge SFT with SGLang rollout runs with `GANKER_MODAL_GPU=A100 uv run modal run modal_apps/distributed/sft_job.py --mode qwen-bridge-sglang-distributed --port 26600 --controller-port 26610 --startup-timeout 900 --sglang-startup-timeout 900 --tuning lora --lora-rank 8 --max-steps 1 --sequence-length 32 --micro-batch-size 1`. This needs separate GPU trainer and SGLang rollout workers, loads the exported HF/PEFT LoRA adapter into SGLang, and samples through `SamplingClient.sample_text(...)`.
- Multi-node Qwen SFT lives in `modal_apps/qwen_sft_multinode/`. It uses Modal clustered functions plus `torchrun`, not the Monarch/i6pn role mesh. Cluster size is fixed at import time with `GANKER_QWEN_SFT_MULTINODE_NODES`, and the first supported training shape is DP-only Qwen3 0.6B LoRA with `GANKER_QWEN_SFT_MULTINODE_GPU=H100:8`. Run light clustered checks with `source ~/.codex/modal.env` and `GANKER_QWEN_SFT_MULTINODE_NODES=2 uv run modal run modal_apps/qwen_sft_multinode/sft.py --mode torchrun-env`, then `--mode nccl-smoke`, then `--mode qwen-lora-sft --max-steps 1 --sequence-length 32`. Run the longer comparison with `GANKER_QWEN_SFT_MULTINODE_NODES=2 uv run modal run modal_apps/qwen_sft_multinode/compare_hf.py --dataset-size 256 --max-steps 20 --sequence-length 256`; it compares multinode Ganker/Megatron, HF Trainer DDP, and a single-node Ganker/Megatron baseline from the same Modal app.
- Keep tests lightweight enough to run with `uv run pytest` on a CPU-only development machine.

## gRPC Codegen

- Source protobufs live under `proto/ganker/rpc/v1`.
- Generated Python modules live under `src/ganker/rpc/v1` and are checked in.
- Regenerate with:

```bash
uv run python -m grpc_tools.protoc -I proto --python_out=src --grpc_python_out=src proto/ganker/rpc/v1/proxy.proto
```
