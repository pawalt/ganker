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
- Public callers should use `ServiceClient` / `TrainingClient`; do not expose Monarch `.choose(...).get()` calls in user-facing APIs.
- New external proxy adapters should implement `ProxyTransport`.
- Keep request/response contracts as plain Python dataclasses in `ganker.contracts` unless there is a concrete need for an external wire format.
- Emulate Tinker's outer API shape where practical (`Datum`, `ModelInput`, `TensorData`, `TrainingClient`, `SamplingClient`), but keep local payload implementations lightweight unless a real integration requires the full SDK type system.
- The local singleton mesh is spawned through `ganker.orchestration.start_local_monarch_mesh`.
- Use `TelemetryActor` for usage/events/tracking. Do not call this component `BillingActor`; pricing and billing policy are out of scope for this layer.

## Backend Boundaries

- Production training is expected to use Megatron behind `TrainingBackend`.
- Production inference is expected to use `sglang` behind `InferenceBackend`.
- Local development and tests must keep working without Megatron, `sglang`, CUDA, model weights, or checkpoints.
- Heavy backend imports must stay isolated in their adapter modules.

## Testing Expectations

- Unit test pure components and fake backends directly.
- Integration test orchestration through real Monarch actor endpoints.
- Local integration tests should use fake backends and temporary filesystem artifact roots.
- CPU Megatron preflight tests should use `pytest -m megatron_cpu`; they may skip optional torch/Megatron Bridge checks when those packages are absent.
- Real Megatron execution tests should use the `megatron` marker and run through Modal/GPU infrastructure, not the default local suite.
- Keep tests lightweight enough to run with `uv run pytest` on a CPU-only development machine.
