# SGLang Sampling Architecture

Ganker keeps sampling behind the existing `SamplingClient -> ProxyActor ->
RolloutActor` path. The client does not speak SGLang directly; SGLang is an
implementation detail of `InferenceBackend`.

## RPC Flow

```text
client process
  |
  | SamplingClient.sample(...) / sample_text(...)
  v
ServiceClient transport
  |
  | local: Monarch endpoint
  | remote: gRPC proxy endpoint
  v
ProxyActor
  |
  v
RolloutActor
  |
  v
SGLangInferenceBackend
  |
  +-- FilesystemArtifactStore.latest(run_id)
  +-- read artifact payload JSON
  +-- SGLangRuntime.refresh_weights(...)
  +-- SGLangRuntime.sample(...)
  v
SGLang HTTP server
```

## Artifact Contract

`SGLangInferenceBackend` accepts these artifact payloads:

```text
base model:
  {"base_model": "Qwen/Qwen3-0.6B"}

full HF checkpoint:
  {"artifact_format": "hf-full-safetensors",
   "hf_checkpoint_path": "/checkpoints/full"}

LoRA adapter:
  {"artifact_format": "hf-lora-adapter",
   "base_model": "Qwen/Qwen3-0.6B",
   "hf_adapter_path": "/checkpoints/adapter"}
```

Raw Megatron artifacts such as `megatron-core-torch-state-dict` are rejected at
refresh time. They must first be exported to a full HF safetensors checkpoint or
a PEFT-compatible LoRA adapter.

## HTTP Runtime

```text
refresh full checkpoint
  SGLangHTTPRuntime
    |
    +-- attach to configured base_url, or launch:
    |   python -m sglang.launch_server --model-path <checkpoint>
    |
    +-- later sample:
        POST /generate

refresh LoRA adapter
  SGLangHTTPRuntime
    |
    +-- ensure base model server exists
    +-- POST /load_lora_adapter
    +-- later sample:
        POST /generate with lora_path=<adapter-name>
```

The native `/generate` request carries either `text` or `input_ids` plus:

```text
sampling_params:
  max_new_tokens
  temperature
  top_p
  stop
return_logprob/logprob_start_len when configured
```

## Local Testing

Local tests do not import or start SGLang. They inject one of:

```text
SGLangRuntime fake
  tests backend refresh/sample mapping and usage accounting

SGLangHTTPClient fake
  tests /load_lora_adapter and /generate payload construction
```

Run:

```bash
uv run pytest tests/test_sglang_backend.py
```

This keeps the default suite CPU-only while still testing the contracts that a
real SGLang server will see on Modal.

## GPU Smoke

The real GPU smoke lives in `modal_apps/sglang_smoke.py`:

```bash
source ~/.codex/modal.env
modal run modal_apps/sglang_smoke.py --mode client
```

`--mode client` exercises:

```text
ServiceClient.local(...)
  -> Monarch ProxyActor
  -> RolloutActor
  -> SGLangInferenceBackend
  -> python -m sglang.launch_server
  -> /generate
```

`--mode backend` skips Monarch and calls `SGLangInferenceBackend` directly. It
is useful for isolating SGLang server or payload issues from actor
orchestration issues.
