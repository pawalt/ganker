"""Materialize a small StarCoderData-style JSONL dataset into a Modal volume.

Run:

    source ~/.codex/modal.env
    uv run modal run modal_apps/starcoder_ganker/download_dataset.py
"""

from __future__ import annotations

import json

from modal_apps.starcoder_ganker import common, infra


app = infra.app


@app.local_entrypoint()
def main(
    dataset_path: str = str(common.DEFAULT_DATASET_PATH),
    dataset_id: str = common.DATASET_ID,
    languages: str = ",".join(common.DEFAULT_LANGUAGES),
    max_files_per_language: int = 1,
    max_examples: int = 256,
    min_chars: int = 16,
    max_chars: int = 12_000,
    seed: int = 1234,
    content_column: str = "content",
    prompt_column: str = "prompt",
    completion_column: str = "completion",
    shuffle_buffer: int = 10_000,
    trust_remote_code: bool = True,
    clear: bool = False,
) -> None:
    if clear:
        print(json.dumps(infra.clear_dataset_volume.remote(), indent=2, sort_keys=True))

    result = infra.prepare_starcoder_dataset.remote(
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        languages=common.csv_list(languages),
        max_files_per_language=max_files_per_language,
        max_examples=max_examples,
        min_chars=min_chars,
        max_chars=max_chars,
        seed=seed,
        content_column=content_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        shuffle_buffer=shuffle_buffer,
        trust_remote_code=trust_remote_code,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
