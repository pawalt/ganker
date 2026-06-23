# StarCoderData-Style Ganker Example

This example mirrors the workflow shape of Modal's StarCoder multinode training
guide, but runs training through Ganker:

1. Select a small set of StarCoderData parquet shards.
2. Materialize prompt/completion JSONL into a Modal Volume.
3. Launch clustered `torchrun` on Modal.
4. Train LoRA with Ganker's Megatron Bridge backend.
5. Sample from the saved LoRA artifact with Ganker's SGLang backend.

The default dataset is `bigcode/the-stack-smol-xs`, a small public Stack subset,
so the example can run without accepting gated BigCode terms. To use the exact
upstream StarCoderData source, set
`GANKER_STARCODER_DATASET_ID=bigcode/starcoderdata` and provide an HF token with
access.

The default model is `Qwen/Qwen3-0.6B` rather than StarCoder/Llama because the
repository already validates that model through Megatron Bridge. Override it
with `GANKER_STARCODER_MODEL` or `--base-model`.

## Commands

```bash
source ~/.codex/modal.env

uv run modal run modal_apps/starcoder_ganker/download_dataset.py \
  --max-examples 256

GANKER_STARCODER_NODES=2 \
uv run modal run modal_apps/starcoder_ganker/sft.py \
  --max-steps 10 \
  --sequence-length 512

uv run modal run modal_apps/starcoder_ganker/evaluate.py \
  --run-id meg-run-000001
```

For a smaller debug run, add `--single-node` to `sft.py`. To reuse an existing
SGLang server instead of launching one inside the eval function, pass
`--sglang-base-url http://host:port` to `evaluate.py`.
