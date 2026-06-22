# RFC 0006: Controlled Megatron Bridge Image, LoRA, and Safetensors Export

Status: Implemented

## Summary

Replace the NeMo Framework image dependency in `modal_apps/sft.py` with a controlled Megatron Bridge runtime image, add real LoRA training support to the Bridge backend, and export rollout-consumable safetensors artifacts.

The target command shape is:

```bash
source ~/.codex/modal.env
modal run modal_apps/sft.py --mode hf-small-sft --tuning full --max-steps 1 --sequence-length 32
modal run modal_apps/sft.py --mode hf-small-sft --tuning lora --lora-rank 8 --max-steps 1 --sequence-length 32
```

## Motivation

RFC 0005 proved Qwen3 0.6B training through Megatron Bridge, but it depended on `nvcr.io/nvidia/nemo:25.09.02` and exported `pytorch_model.bin` to avoid a Qwen tied-weight safetensors failure. That was useful for proof of life but too opaque for production work:

- dependency versions were inherited from a large NeMo image;
- plain PyPI installation did not resolve the same dependency tree as Bridge CI;
- LoRA was accepted by the public API but ignored by the Bridge runtime;
- full-checkpoint export did not use safetensors.

## Decisions

- Build the Modal Bridge image from `nvcr.io/nvidia/pytorch:26.02-py3`, not the NeMo Framework image.
- Clone `NVIDIA-NeMo/Megatron-Bridge` at `v0.4.2` by default.
- Install Bridge with its own `uv.lock`, matching the Bridge Dockerfile dependency strategy.
- Keep image knobs overrideable:
  - `GANKER_MODAL_BRIDGE_BASE_IMAGE`
  - `GANKER_MEGATRON_BRIDGE_REPO`
  - `GANKER_MEGATRON_BRIDGE_REF`
  - `GANKER_MEGATRON_BRIDGE_UV_VERSION`
  - `GANKER_MODAL_TORCHMONARCH_VERSION`
- Apply Bridge `LoRA` through `provider.register_pre_wrap_hook(...)`.
- Optimize only trainable parameters for LoRA runs.
- Export full tuning as HF safetensors.
- Export LoRA as PEFT-compatible `adapter_config.json` plus `adapter_model.safetensors`.

## Dependency Tree Notes

The failed slim-image install was a resolver problem, not just a missing package problem. Bridge's `pyproject.toml` depends on `megatron-core[dev,mlm]`, which introduces packages such as `emerging-optimizers`, Transformer Engine, Mamba, and CUDA-oriented extensions. The Bridge lock file resolves `emerging-optimizers` from NVIDIA's GitHub source and pins Transformer Engine against the expected NGC PyTorch base.

The controlled image follows Bridge's own Dockerfile pattern:

```text
nvcr.io/nvidia/pytorch:26.02-py3
  -> install uv
  -> uv venv /opt/venv --system-site-packages
  -> git clone Megatron-Bridge@v0.4.2 --recurse-submodules
  -> uv sync --frozen --only-group build
  -> uv sync --frozen --no-dev --no-install-package transformer-engine
  -> install pinned torchmonarch wheel
```

This still uses an NVIDIA PyTorch image because CUDA, NCCL, PyTorch, Transformer Engine, and compiled extensions need to line up. The important change is that Ganker controls the Bridge source tag and installs from Bridge's lockfile rather than inheriting an entire NeMo Framework distribution. Frozen sync is used so Modal builds consume the upstream lock without attempting to rewrite it. Transformer Engine is intentionally supplied by the NGC PyTorch base instead of source-built from Bridge's git pin, because rebuilding it inside each Modal image is too slow for iteration and duplicates the compatibility work already done by the base image.

## Runtime Contract

`MegatronTrainingBackend.create_training_run(...)` now passes `tuning_mode` and `lora_rank` into the installed Bridge runtime as behavior, not just metadata.

Full tuning:

```text
AutoBridge.from_hf_pretrained(base_model)
  -> to_megatron_provider(load_weights=True, hf_path=base_model)
  -> configure TP/PP/sequence/batch/dtype
  -> provide_distributed_model(wrap_with_ddp=True)
  -> Adam(all trainable params)
  -> save HF full safetensors
```

LoRA:

```text
AutoBridge.from_hf_pretrained(base_model)
  -> to_megatron_provider(load_weights=True, hf_path=base_model)
  -> LoRA(target_modules=[linear_qkv, linear_proj, linear_fc1, linear_fc2], dim=rank, alpha=2*rank)
  -> provider.register_pre_wrap_hook(lora)
  -> provide_distributed_model(wrap_with_ddp=True)
  -> Adam(adapter params only)
  -> save HF PEFT adapter safetensors
```

## Artifact Contract

Full tuning payload fields include:

```json
{
  "artifact_format": "hf-full-safetensors",
  "hf_checkpoint_path": "...",
  "hf_weight_format": "safetensors",
  "hf_weights_path": ".../model.safetensors",
  "hf_weight_files": ["..."],
  "hf_weight_count": 311
}
```

LoRA payload fields include:

```json
{
  "artifact_format": "hf-lora-adapter",
  "hf_adapter_path": "...",
  "hf_adapter_config_path": ".../adapter_config.json",
  "hf_adapter_weights_path": ".../adapter_model.safetensors",
  "hf_weight_format": "safetensors"
}
```

## Safetensors Tied-Weight Fix

Qwen exposes tied embedding/output weights. Some safetensors writers reject shared tensor storage because reloading can duplicate or drop aliasing. Ganker's full export now converts Bridge's streamed HF weights into detached, CPU, contiguous clones before writing safetensors shards. This produces valid HF parameter names without shared storage in the in-memory state dict.

This is intentionally explicit instead of falling back to `pytorch_model.bin`. It trades a small amount of extra checkpoint disk for a rollout-friendly artifact format.

## Follow-Ups

- Cache `/opt/Megatron-Bridge` and HF model downloads with Modal volumes or an image prebuild once the dependency set stabilizes.
- Add a post-export load check with `AutoModelForCausalLM.from_pretrained(...)` for full checkpoints and `peft.PeftModel.from_pretrained(...)` for adapters.
- Add configurable LoRA target modules after the Qwen path is stable.
- Decide whether merged-LoRA export is needed for rollout engines that do not load PEFT adapters directly.
