"""Sample from a Ganker-trained code SFT checkpoint with SGLang.

Run:

    source ~/.codex/modal.env
    uv run modal run modal_apps/starcoder_ganker/evaluate.py --run-id meg-run-000001
"""

from __future__ import annotations

import json

from modal_apps.starcoder_ganker import common, infra


app = infra.app


@app.local_entrypoint()
def main(
    run_id: str,
    checkpoint_version: int = -1,
    base_model: str = common.MODEL,
    prompt: str = "",
    sglang_base_url: str = "",
    max_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    port: int = 30000,
    startup_timeout: int = 900,
    context_length: int = 4096,
    mem_fraction_static: float = 0.75,
) -> None:
    prompts = [prompt] if prompt else None
    result = infra.run_sglang_eval.remote(
        run_id=run_id,
        prompts=prompts,
        checkpoint_version=checkpoint_version,
        base_model=base_model,
        sglang_base_url=sglang_base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        port=port,
        startup_timeout=startup_timeout,
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
