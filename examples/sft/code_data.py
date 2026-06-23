from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import importlib
from pathlib import Path
from typing import Any

from .data import SFTExample
from .real_data import write_sft_jsonl


DEFAULT_CODE_DATASET = "bigcode/the-stack-smol-xs"
STARCODER_DATASET = "bigcode/starcoderdata"
DEFAULT_CODE_LANGUAGES: tuple[str, ...] = ()

STARCODER_EXCLUDED_PATH_FRAGMENTS = (
    "jupyter-scripts-dedup-filtered",
    "jupyter-structured-clean-dedup",
    "github-issues-filtered-structured",
    "git-commits-cleaned",
)


def select_starcoder_parquet_files(
    paths: Iterable[str],
    *,
    languages: Sequence[str] | None = DEFAULT_CODE_LANGUAGES,
    max_files_per_language: int = 1,
) -> list[str]:
    """Select a small, deterministic StarCoderData shard set."""

    if max_files_per_language <= 0:
        raise ValueError("max_files_per_language must be positive")

    language_filter = {
        _normalize_language(language) for language in languages or () if language.strip()
    }
    selected: list[str] = []
    counts: dict[str, int] = {}
    for path in sorted(str(path) for path in paths):
        if not path.endswith(".parquet"):
            continue
        if any(fragment in path for fragment in STARCODER_EXCLUDED_PATH_FRAGMENTS):
            continue

        language = _top_level_dir(path)
        if language_filter and language not in language_filter:
            continue
        if counts.get(language, 0) >= max_files_per_language:
            continue

        selected.append(path)
        counts[language] = counts.get(language, 0) + 1

    if not selected:
        requested = ", ".join(sorted(language_filter)) or "<any>"
        raise ValueError(f"no StarCoderData parquet files selected for languages={requested}")
    return selected


def code_record_to_sft_example(
    record: Mapping[str, Any],
    *,
    language: str = "code",
    content_column: str = "content",
    prompt_column: str = "prompt",
    completion_column: str = "completion",
    path_columns: Sequence[str] = (
        "max_stars_repo_path",
        "path",
        "filepath",
        "file_path",
        "repo_name",
    ),
    min_chars: int = 16,
    max_chars: int = 12_000,
) -> SFTExample | None:
    if min_chars < 0:
        raise ValueError("min_chars cannot be negative")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    source_path = _first_text_field(record, path_columns)
    prompt_value = record.get(prompt_column)
    completion_value = record.get(completion_column)
    if isinstance(prompt_value, str) and isinstance(completion_value, str):
        prompt = prompt_value.rstrip() + "\n"
        content_value = completion_value
    else:
        content_value = record.get(content_column)
        if not isinstance(content_value, str):
            return None
        language_label = _language_label(language)
        location = f" for `{source_path}`" if source_path else ""
        prompt = f"Write an idiomatic {language_label} source file{location}.\n\n"

    content = content_value.strip()
    if len(content) < min_chars:
        return None
    if len(content) > max_chars:
        content = content[:max_chars].rstrip()

    return SFTExample(
        prompt=prompt,
        completion=content,
        metadata={
            "dataset_format": "code_completion",
            "language": language,
            "source_path": source_path,
            "truncated": len(content_value.strip()) > len(content),
        },
    )


def materialize_hf_code_sft_jsonl(
    path: str | Path,
    *,
    dataset_name: str = DEFAULT_CODE_DATASET,
    data_files: str | Sequence[str] | None = None,
    split: str = "train",
    language: str = "code",
    allowed_languages: Sequence[str] | None = None,
    content_column: str = "content",
    prompt_column: str = "prompt",
    completion_column: str = "completion",
    max_examples: int = 256,
    min_chars: int = 16,
    max_chars: int = 12_000,
    seed: int = 1234,
    shuffle_buffer: int = 10_000,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")

    datasets = importlib.import_module("datasets")
    load_kwargs: dict[str, Any] = {
        "split": split,
        "streaming": True,
        "trust_remote_code": trust_remote_code,
    }
    if data_files:
        load_kwargs["data_files"] = data_files
    dataset = datasets.load_dataset(dataset_name, **load_kwargs)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)

    examples: list[SFTExample] = []
    language_filter = {
        _normalize_language(value) for value in allowed_languages or () if value.strip()
    }
    scanned = 0
    for record in dataset:
        scanned += 1
        if not isinstance(record, Mapping):
            continue
        found_language = _record_language(record)
        if language_filter and found_language and _normalize_language(found_language) not in language_filter:
            continue
        record_language = found_language or language
        example = code_record_to_sft_example(
            record,
            language=record_language,
            content_column=content_column,
            prompt_column=prompt_column,
            completion_column=completion_column,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if example is None:
            continue
        examples.append(example)
        if len(examples) >= max_examples:
            break

    count = write_sft_jsonl(examples, path)
    return {
        "dataset_name": dataset_name,
        "dataset_split": split,
        "dataset_format": "code_completion",
        "data_files": [data_files] if isinstance(data_files, str) else list(data_files or []),
        "language": language,
        "allowed_languages": list(allowed_languages or []),
        "dataset_examples": count,
        "scanned_examples": scanned,
        "dataset_seed": seed,
        "dataset_path": str(path),
        "content_column": content_column,
        "prompt_column": prompt_column,
        "completion_column": completion_column,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "trust_remote_code": trust_remote_code,
    }


def _normalize_language(value: str) -> str:
    return value.strip().lower()


def _top_level_dir(path: str) -> str:
    return path.strip("/").split("/", 1)[0].lower()


def _language_label(language: str) -> str:
    values = [item.strip() for item in language.split(",") if item.strip()]
    if not values:
        return "code"
    known = {
        "go": "Go",
        "golang": "Go",
        "rust": "Rust",
        "python": "Python",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "java": "Java",
        "cpp": "C++",
        "c++": "C++",
        "c": "C",
    }
    labels = [known.get(item.lower(), item) for item in values]
    if len(labels) == 1:
        return labels[0]
    return "/".join(labels)


def _first_text_field(record: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _record_language(record: Mapping[str, Any]) -> str:
    return _first_text_field(record, ("lang", "language", "programming_language"))
