# RFC 0005: Qwen3 0.6B Megatron Bridge SFT

Status: Implemented, updated by RFC 0006

## Summary

Add a Modal-backed `hf-small-sft` path that runs a small supervised fine-tuning job on `Qwen/Qwen3-0.6B` through Megatron Bridge. This extends RFC 0004 from toy Megatron-Core SFT into a real pretrained-model path.

The implemented path is:

```text
examples/tiny_sft.jsonl
  -> Qwen HF tokenizer
  -> Datum batches
  -> ServiceClient / TrainingClient
  -> ProxyActor
  -> TrainingActor
  -> MegatronTrainingBackend(runtime_kind="bridge")
  -> Megatron Bridge AutoBridge
  -> Megatron-Core forward/backward
  -> optimizer step
  -> full HF checkpoint export
```

## Decisions

- Use `Qwen/Qwen3-0.6B` as the first Bridge-backed model.
- Use full fine-tuning first. LoRA is added in RFC 0006.
- Use the NVIDIA NeMo Framework container as the first working Bridge image. RFC 0006 replaces this with a controlled NGC PyTorch + pinned Bridge source image.
- Keep SFT data and loop helpers under `examples/sft`, not `src/ganker`.
- Export HF full checkpoints first.
- Skip dataset packing; fixed-length padded batches are enough for the small dataset path.

## Image Strategy

Plain pip installation of `megatron-bridge` on a slim image was not reliable:

- `megatron-bridge` metadata pulled `megatron-core[dev]`, which pulled the `emerging-optimizers` dependency-confusion stub.
- Transformer Engine wheels did not match the default torch/CUDA combination and attempted source builds.

The working image is:

```text
nvcr.io/nvidia/nemo:25.09.02
```

This container provides compatible torch, Megatron-Core, Megatron Bridge, and Transformer Engine packages. The Modal app allows overriding it with:

```bash
GANKER_MODAL_BRIDGE_IMAGE=<image> modal run modal_apps/sft.py --mode hf-small-sft
```

## Runtime Notes

Bridge model construction uses:

```python
auto_bridge = AutoBridge.from_hf_pretrained(base_model, trust_remote_code=True)
provider = auto_bridge.to_megatron_provider(load_weights=False)
model = auto_bridge.to_megatron_model(
    load_weights=True,
    hf_path=base_model,
    ddp_config=DistributedDataParallelConfig(),
    wrap_with_ddp=True,
)
```

The DDP wrapper is required because the Qwen Megatron layers expect Megatron main-gradient buffers during backward.

The runtime uses Megatron-Core's `get_forward_backward_func()` for the actual `TrainingClient.forward_backward(...)` implementation, preserving the same low-level API shape as the toy runtime.

## Checkpoint Export

Bridge's built-in `save_hf_pretrained(...)` and `save_hf_weights(...)` hit a Qwen tied-weight safetensors error for `model.embed_tokens.weight` and `lm_head.weight`.

The implemented export uses `bridge.export_hf_weights(...)` and writes:

```text
config.json
tokenizer files
generation_config.json when available
pytorch_model.bin
```

The `.bin` format tolerated shared/tied tensors and was valid for this milestone. RFC 0006 replaces this with safetensors export that clones tied tensors before writing.

## Command

```bash
source ~/.codex/modal.env
modal run modal_apps/sft.py --mode hf-small-sft --max-steps 1 --sequence-length 32
```

Expected output includes:

```json
{
  "ok": true,
  "mode": "hf-small-sft",
  "base_model": "Qwen/Qwen3-0.6B",
  "runtime_kind": "bridge",
  "steps": 1,
  "hf_checkpoint_path": "/tmp/ganker-artifacts/hf-full/..."
}
```

## Follow-Ups

- Add Modal Volumes for HF cache and output checkpoints.
- Add resume support.
- Add a small eval/sample check after checkpoint export.
- Add LoRA once adapter insertion is implemented and tested.
- Move from one-GPU smoke to a more production-like multi-GPU setup only after the single-GPU lifecycle stays stable.
