from __future__ import annotations

from collections.abc import Iterable, Mapping
import importlib
import json
from pathlib import Path
import random
from typing import Any

from .data import SFTExample


DEFAULT_REAL_DATASET = "tatsu-lab/alpaca"
DEFAULT_REAL_DATASET_SPLIT = "train"
DEFAULT_REAL_DATASET_FORMAT = "alpaca"


def alpaca_record_to_sft_example(record: Mapping[str, Any]) -> SFTExample:
    instruction = _required_text(record, "instruction")
    output = _required_text(record, "output")
    input_text = _optional_text(record, "input")

    prompt_parts = [
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        "",
        "### Instruction:",
        instruction.strip(),
    ]
    if input_text.strip():
        prompt_parts.extend(["", "### Input:", input_text.strip()])
    prompt_parts.extend(["", "### Response:", ""])

    return SFTExample(
        prompt="\n".join(prompt_parts),
        completion=output.strip(),
        metadata={
            "dataset_format": "alpaca",
        },
    )


def write_sft_jsonl(examples: Iterable[SFTExample], path: str | Path) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "prompt": example.prompt,
                        "completion": example.completion,
                        "weight": example.weight,
                        "metadata": example.metadata,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
    if count == 0:
        raise ValueError("cannot write empty SFT dataset")
    return count


def load_hf_sft_examples(
    *,
    dataset_name: str = DEFAULT_REAL_DATASET,
    split: str = DEFAULT_REAL_DATASET_SPLIT,
    dataset_format: str = DEFAULT_REAL_DATASET_FORMAT,
    max_examples: int = 256,
    seed: int = 1234,
) -> list[SFTExample]:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    if dataset_format != "alpaca":
        raise ValueError(f"unsupported dataset_format={dataset_format!r}")

    datasets = importlib.import_module("datasets")
    dataset = datasets.load_dataset(dataset_name, split=split)
    records = list(dataset)
    random.Random(seed).shuffle(records)
    examples: list[SFTExample] = []
    for record in records:
        try:
            example = alpaca_record_to_sft_example(record)
        except ValueError:
            continue
        examples.append(example)
        if len(examples) >= max_examples:
            break
    if not examples:
        raise ValueError(f"{dataset_name}:{split} produced no examples")
    return examples


def materialize_hf_sft_jsonl(
    path: str | Path,
    *,
    dataset_name: str = DEFAULT_REAL_DATASET,
    split: str = DEFAULT_REAL_DATASET_SPLIT,
    dataset_format: str = DEFAULT_REAL_DATASET_FORMAT,
    max_examples: int = 256,
    seed: int = 1234,
) -> dict[str, Any]:
    examples = load_hf_sft_examples(
        dataset_name=dataset_name,
        split=split,
        dataset_format=dataset_format,
        max_examples=max_examples,
        seed=seed,
    )
    count = write_sft_jsonl(examples, path)
    return {
        "dataset_name": dataset_name,
        "dataset_split": split,
        "dataset_format": dataset_format,
        "dataset_examples": count,
        "dataset_seed": seed,
        "dataset_path": str(path),
    }


def _required_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"alpaca record must contain non-empty string field {key!r}")
    return value


def _optional_text(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"alpaca record field {key!r} must be a string when present")
    return value
