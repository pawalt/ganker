# Qwen Model-Parallel SFT

This is the clean smoke example for tensor-parallel Qwen LoRA SFT on Modal. It
uses the shared `qwen_sft_multinode` infra, but the entrypoint only exposes the
training shape:

- Megatron Bridge trainer
- Qwen HF checkpoint import
- LoRA tuning
- `TP=2`, `PP=1`
- one logical optimizer step by default
- HF/PEFT LoRA adapter export

Run:

```bash
source ~/.codex/modal.env
GANKER_QWEN_SFT_MULTINODE_NODES=1 \
GANKER_QWEN_SFT_MULTINODE_GPU=H100:8 \
uv run modal run modal_apps/qwen_model_parallel_sft/sft.py
```

The default command requests one Modal clustered node with eight H100 GPUs.
Modal currently requires the full GPU slice for clustered functions. With
`TP=2`, this gives `DP=4` and a default global batch size of `4`.

To try gradient accumulation, set a larger global batch size that remains
divisible by `micro_batch_size * DP`, for example:

```bash
source ~/.codex/modal.env
GANKER_QWEN_SFT_MULTINODE_NODES=1 \
GANKER_QWEN_SFT_MULTINODE_GPU=H100:8 \
uv run modal run modal_apps/qwen_model_parallel_sft/sft.py \
  --global-batch-size 8
```
