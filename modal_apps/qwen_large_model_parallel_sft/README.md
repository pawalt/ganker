# Qwen Large Model-Parallel SFT

This is the clean smoke example for a larger Qwen LoRA SFT run that requires
both tensor and pipeline parallelism:

- Megatron Bridge trainer
- Qwen HF checkpoint import
- LoRA tuning
- default model: `Qwen/Qwen3-32B`
- required shape: `TP=8`, `PP=2`
- two Modal clustered nodes with eight H100 GPUs each by default
- one logical optimizer step by default
- HF/PEFT LoRA adapter export

Run:

```bash
source ~/.codex/modal.env
GANKER_QWEN_SFT_MULTINODE_NODES=2 \
GANKER_QWEN_SFT_MULTINODE_GPU=H100:8 \
uv run modal run modal_apps/qwen_large_model_parallel_sft/sft.py
```

With the default two-node shape, `world_size=16`, `model_parallel_size=16`, and
`DP=1`. The default global batch size is `2`, which gives the two Megatron
microbatches required by `PP=2`.

To try another large model, override the model before launching:

```bash
source ~/.codex/modal.env
GANKER_QWEN_LARGE_SFT_MODEL=Qwen/Qwen2.5-72B \
GANKER_QWEN_SFT_MULTINODE_NODES=2 \
GANKER_QWEN_SFT_MULTINODE_GPU=H100:8 \
uv run modal run modal_apps/qwen_large_model_parallel_sft/sft.py
```

This is intentionally a smoke example. It proves model load, distributed
forward/backward, optimizer step, and adapter export. Use a longer dataset,
higher sequence length, and more steps only after the one-step smoke is green.
