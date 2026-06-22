# RFC 0004: Small-Dataset SFT Path

Status: Draft

## Summary

Define the path from the current Ganker/Megatron prototype to running supervised fine-tuning on a small dataset. The first milestone should prove the complete Ganker control path on real data:

```text
JSONL SFT dataset
  -> tokenizer / preprocessor
  -> Datum batches
  -> ServiceClient / TrainingClient
  -> ProxyActor
  -> TrainingActor
  -> MegatronTrainingBackend
  -> Megatron-Core forward/backward
  -> optimizer steps
  -> checkpoint artifact
```

The first implementation should intentionally be a plumbing SFT milestone, not a production fine-tuning system. It can train the current tiny Megatron-Core model on a toy dataset to validate data flow, batching, loss masking, training loop behavior, telemetry, and checkpointing. A later milestone should replace the random tiny model with a pretrained Hugging Face model loaded through Megatron Bridge.

This RFC builds on:

- RFC 0001: `plans/rfc-0001-megatron-bridge-support.md`
- RFC 0002: `plans/rfc-0002-modal-megatron-gpu-testing.md`
- RFC 0003: `plans/rfc-0003-megatron-runtime-lifecycle.md`

## Background

The current prototype already has:

- A public client path: `ServiceClient -> TrainingClient`.
- Monarch orchestration: `ProxyActor -> TrainingActor -> TelemetryActor`.
- A training lifecycle: `create_training_run`, `forward_backward`, `optim_step`, `save_weights`.
- A Modal GPU path that runs real Megatron-Core `get_forward_backward_func()`.
- A tiny in-process Megatron-Core runtime under `runtime_kind="core"`.
- Import-isolated backend tests and Modal smoke tests.

The current prototype does not yet have:

- A dataset format or loader.
- Tokenizer integration.
- Prompt/completion loss masking.
- A reusable SFT training loop.
- Gradient accumulation across multiple `forward_backward` calls.
- Real pretrained model loading in the active GPU runtime.
- Real LoRA adapter insertion.
- A user-facing model export path for SGLang rollout.

## Goals

- Add a small, testable SFT data path that converts JSONL examples into `Datum` batches.
- Keep `Datum` token-level for now; do not put raw text into the core training contract.
- Add a reusable SFT loop that drives the existing `TrainingClient` API.
- Validate the SFT loop locally with fake backends and small unit tests.
- Validate the real training path on Modal with the tiny Megatron-Core runtime.
- Produce checkpoint artifacts and machine-readable run summaries.
- Keep default local tests CPU-only and lightweight.
- Make the distinction clear between plumbing SFT and meaningful pretrained-model SFT.

## Non-Goals

- Do not implement a production trainer CLI in the first milestone.
- Do not require local CUDA, Megatron, SGLang, or Hugging Face model downloads.
- Do not make the first milestone depend on Megatron Bridge.
- Do not support multi-node or multi-GPU SFT initially.
- Do not implement RL losses or sampling-in-the-loop training here.
- Do not make SGLang consume the resulting Megatron artifacts yet.
- Do not promise real LoRA support until adapter insertion is implemented in the Megatron runtime.

## Milestone Shape

### M1: Plumbing SFT

Use the current tiny Megatron-Core runtime and a small JSONL dataset. This proves the Ganker API, Monarch orchestration, data conversion, training loop, and checkpointing.

Properties:

- Runs on Modal GPU.
- Uses `runtime_kind="core"`.
- Builds a tiny random GPT model from config.
- Uses a tokenizer or deterministic toy tokenizer.
- Trains on a tiny dataset for a small number of steps.
- Saves a final checkpoint.
- Reports losses and step counts.

This milestone answers: can Ganker run an SFT-shaped training job end to end?

It does not answer: can Ganker fine-tune a real pretrained model yet?

### M2: Pretrained Small-Model SFT

Use Megatron Bridge to load a small pretrained Hugging Face causal LM, then run the same SFT loop.

Properties:

- Runs on Modal GPU.
- Uses `runtime_kind="bridge"` or a new Bridge-backed runtime implementation.
- Loads a real tokenizer and model config.
- Supports either full fine-tuning first or real LoRA once adapter insertion is implemented.
- Saves a Megatron checkpoint or Bridge-managed checkpoint.

This milestone answers: can Ganker perform meaningful small-model SFT?

### M3: Rollout-Compatible Export

Export final weights into a format that the rollout backend can consume.

Properties:

- Produces a clearly typed artifact: Megatron checkpoint, HF checkpoint, LoRA adapter, or merged model.
- Defines `refresh_weights` behavior for non-fake rollout backends.
- Connects to SGLang only after the training artifact contract is stable.

This milestone answers: can fine-tuned weights be served?

## Dataset Contract

Start with JSONL:

```json
{"prompt": "Question: 2+2?\nAnswer:", "completion": " 4"}
{"prompt": "Question: capital of France?\nAnswer:", "completion": " Paris"}
```

Required fields:

- `prompt`: text shown to the model as context.
- `completion`: text to train the model to produce.

Optional future fields:

- `metadata`: arbitrary JSON metadata.
- `weight`: per-example scalar weight.
- `messages`: chat-style input that can be rendered into prompt/completion.

The first dataset loader should reject malformed records with line numbers and clear errors.

## Tokenization And Loss Masking

The loader should convert each JSONL example into token-level fields:

```text
prompt_tokens      = tokenizer(prompt)
completion_tokens  = tokenizer(completion)
input_ids          = prompt_tokens + completion_tokens
target_tokens      = input_ids shifted left by one
weights            = 0.0 for prompt positions, 1.0 for completion positions
```

The exact mask boundary needs care because the model predicts token `i + 1` from position `i`. The first implementation should include unit tests with tiny deterministic token sequences so the expected `target_tokens` and `weights` are obvious.

Current `Datum` already has enough shape for this:

```python
Datum(
    model_input=ModelInput.from_ints(input_ids),
    loss_fn_inputs={
        "target_tokens": TensorData.from_ints(target_tokens),
        "weights": TensorData.from_floats(weights),
    },
)
```

Do not add raw text fields to `Datum` in the first milestone. Keep raw text in SFT preprocessing helpers.

## Batching Contract

The first batching implementation should:

- truncate examples longer than `sequence_length`;
- optionally drop examples that do not contain any completion loss tokens after truncation;
- pad examples shorter than `sequence_length`;
- produce fixed-length batches because `datums_to_tensor_batch` currently requires equal token lengths;
- set `target_tokens` for padded positions to `0`;
- set `weights` for padded positions to `0.0`.

Initial config:

```python
SFTDataConfig(
    sequence_length=128,
    batch_size=1,
    drop_overlong=False,
    shuffle=True,
    seed=1234,
)
```

`batch_size=1` is sufficient for the first Modal smoke. Larger batches can work once memory and sequence length are tuned.

## Training Loop Contract

Add a reusable helper that drives the public API:

```python
def run_sft(
    client: ServiceClient,
    *,
    base_model: str,
    dataset: Iterable[list[Datum]],
    tuning: TuningMode | Literal["lora", "full"],
    lora_rank: int,
    learning_rate: float,
    max_steps: int,
    save_every: int,
) -> SFTRunSummary:
    ...
```

The first loop can be simple:

```text
create training run
for each batch:
  forward_backward(batch)
  optim_step(learning_rate)
  record loss
  optionally save_weights
save final weights
return summary
```

The summary should be JSON-serializable:

```python
SFTRunSummary(
    ok=True,
    run_id="...",
    steps=20,
    losses=[...],
    final_loss=...,
    artifact_path="...",
)
```

This helper should live outside the actor implementation at first. It is orchestration code over the public API, not a new actor endpoint.

## Gradient Accumulation

The current Megatron backend allows one `forward_backward` followed by one `optim_step`:

```text
READY -> forward_backward -> GRADIENTS_PENDING -> optim_step -> READY
```

That is enough for M1 with `batch_size=1`.

For realistic SFT, add explicit gradient accumulation:

```text
READY
  -> forward_backward microbatch 1
  -> forward_backward microbatch 2
  -> ...
  -> optim_step
  -> READY
```

Backend changes:

- Add `gradient_accumulation_steps` or `max_pending_microbatches` to `MegatronBackendConfig`.
- Track `pending_microbatches` in run state.
- Allow repeated `forward_backward` while gradients are pending until the configured accumulation limit.
- Require `optim_step` after at least one pending microbatch.
- Report usage per microbatch and training step separately.

Do not make accumulation implicit. If accumulation is disabled, the existing strict state machine should remain.

## Model Runtime Strategy

### M1 Runtime

Use `InProcessMegatronCoreRuntime`.

Restrictions:

- `tensor_model_parallel_size=1`
- `pipeline_model_parallel_size=1`
- small sequence length
- small hidden size/layer count
- full fine-tuning semantics even if the public call uses `create_lora_training_client`

The first implementation should either:

- use `create_training_client(..., tuning="full", rank=0)` for M1; or
- explicitly document that `runtime_kind="core"` ignores LoRA until adapter support exists.

Prefer full tuning for M1 to avoid pretending that LoRA is implemented.

### M2 Runtime

Finish a Bridge-backed runtime that can:

- load an HF tokenizer/model/config;
- construct a Megatron provider/model;
- initialize optimizer and scheduler;
- call `get_forward_backward_func()` through the same runtime lifecycle;
- save a Bridge/Megatron checkpoint.

LoRA should be added only once the runtime can actually insert/train adapters.

## Modal Entry Point

Add an SFT mode to the existing Modal smoke harness or a dedicated app:

```bash
source ~/.codex/modal.env
modal run modal_apps/sft.py --mode toy-sft
```

or:

```bash
modal run modal_apps/megatron_smoke.py --mode sft-toy
```

The first version should prefer a dedicated app if it makes the commands clearer.

Expected result:

```json
{
  "ok": true,
  "mode": "sft-toy",
  "runtime_kind": "core",
  "steps": 10,
  "losses": [4.9, 4.7, 4.5],
  "final_loss": 4.5,
  "artifact_path": "/tmp/ganker-sft/weights/..."
}
```

Modal resources:

- Use the existing `peyton-agents` environment.
- Use the existing repo mount pattern.
- Keep the tiny dataset in the repository for M1.
- Use Modal Volumes later for larger datasets, model caches, and checkpoints.
- Add Hugging Face secrets only for M2 if the selected model requires them.

## Local Test Plan

Default local tests must remain CPU-only and should not require real Megatron.

Add tests for:

- JSONL parsing with valid and invalid records.
- prompt/completion tokenization boundaries.
- shifted `target_tokens`.
- prompt-token and padding-token loss masks.
- truncation behavior.
- batching equal-length datums.
- SFT loop behavior using fake `TrainingClient` or fake backend.
- summary JSON serialization.

Useful local commands:

```bash
uv run pytest
uv run pytest tests/test_sft_data.py tests/test_sft_loop.py
```

## Modal Test Plan

M1 Modal tests:

```bash
source ~/.codex/modal.env
modal run modal_apps/sft.py --mode env
modal run modal_apps/sft.py --mode toy-sft
```

Assertions:

- dataset loads;
- at least one optimizer step completes;
- all reported losses are finite;
- final checkpoint artifact exists;
- telemetry records input tokens and training steps;
- run summary is JSON-serializable without local torch installed.

M2 Modal tests:

```bash
modal run modal_apps/sft.py --mode hf-small-sft --base-model <small-public-model>
```

Assertions:

- tokenizer loads;
- model loads through Bridge;
- one or more SFT steps complete;
- checkpoint can be written;
- artifact metadata identifies model source, tokenizer source, and checkpoint format.

## Proposed Files

Initial M1 files:

```text
examples/tiny_sft.jsonl
src/ganker/sft/__init__.py
src/ganker/sft/data.py
src/ganker/sft/loop.py
tests/test_sft_data.py
tests/test_sft_loop.py
modal_apps/sft.py
```

Rationale:

- Dataset and SFT loop code are reusable product-facing helpers, so they can live under `src/ganker/sft`.
- Modal-only execution glue should stay in `modal_apps/`.
- Modal smoke test internals should stay in `tests/modal_smoke/`.

If the helpers feel too experimental during M1, start them under `examples/sft/` and promote to `src/ganker/sft` once the API stabilizes.

## API Surface

Keep the core public API stable:

```python
client = ServiceClient.local(...)
training = client.create_training_client(base_model="...", tuning="full", rank=0)
training.forward_backward(batch)
training.optim_step(learning_rate=...)
training.save_weights()
```

Add optional convenience APIs only after the helper stabilizes:

```python
from ganker.sft import load_jsonl_sft_dataset, run_sft
```

Do not add a new actor endpoint like `run_sft` yet. The actor API should continue to expose low-level Tinker-shaped training primitives.

## Implementation Plan

1. Add SFT dataclasses and JSONL parser.
2. Add a deterministic toy tokenizer for unit tests.
3. Add optional Hugging Face tokenizer support behind an optional dependency or Modal-only dependency.
4. Add prompt/completion to `Datum` conversion.
5. Add padding/truncation/batching.
6. Add the public-API SFT loop helper.
7. Add fake-client/fake-backend unit tests for the SFT loop.
8. Add `examples/tiny_sft.jsonl`.
9. Add `modal_apps/sft.py` with `env` and `toy-sft` modes.
10. Run `toy-sft` on Modal against `runtime_kind="core"`.
11. Add gradient accumulation support in the Megatron backend.
12. Finish Bridge-backed pretrained model loading for `hf-small-sft`.
13. Add checkpoint/export metadata needed for rollout refresh.

## Risks

- Loss masking is easy to get subtly wrong because target tokens are shifted.
- A tiny random model can validate plumbing while still producing noisy losses.
- Hugging Face tokenizer/model compatibility with Megatron Bridge may dominate M2.
- Real LoRA support may require a different insertion path than the current tiny Megatron-Core runtime.
- Checkpoint artifacts may not be directly usable by SGLang without a later export step.
- Modal image size and cold start time may grow once HF/Megatron Bridge dependencies are added.

## Open Questions

- Should M1 use a deterministic toy tokenizer, a real HF tokenizer, or both?
- Which small public HF model should be the first Bridge SFT target?
- Should full fine-tuning be the first meaningful M2 path, with LoRA added later?
- Should `run_sft` live under `src/ganker/sft` immediately or start in `examples/sft`?
- What artifact format should be considered the first rollout-compatible target: HF full checkpoint, LoRA adapter, or Megatron checkpoint?
- How much of dataset packing should be implemented before real pretrained-model SFT?

## Recommendation

Implement M1 first with the current tiny Megatron-Core runtime and a toy JSONL dataset. Keep the SFT loop outside the actors and drive the existing public training API. Once the data path, loss masks, loop, Modal execution, and artifacts are stable, move to M2 and make Megatron Bridge load a real small pretrained model.

