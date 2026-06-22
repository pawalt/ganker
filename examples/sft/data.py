from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Protocol

from ganker.contracts import Datum, ModelInput, TensorData


class Tokenizer(Protocol):
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int

    def encode(self, text: str) -> list[int]:
        ...


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    completion: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SFTDataConfig:
    sequence_length: int = 128
    batch_size: int = 1
    drop_overlong: bool = False
    shuffle: bool = True
    seed: int = 1234

    def validate(self) -> None:
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class ToyTokenizer:
    """Deterministic byte-level tokenizer for small plumbing tests.

    This tokenizer is intentionally not reversible or production-shaped. It
    keeps IDs inside the tiny Megatron smoke model's vocab and reserves 0/1/2
    for padding, BOS, and EOS.
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, vocab_size: int = 128):
        if vocab_size < 4:
            raise ValueError("vocab_size must be at least 4")
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        bucket_count = self.vocab_size - 3
        return [3 + (byte % bucket_count) for byte in text.encode("utf-8")]


class HFAutoTokenizerAdapter:
    """Adapter from Hugging Face tokenizers to the example SFT tokenizer protocol."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("Hugging Face tokenizer must define eos_token_id")
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(tokenizer.pad_token_id or eos_token_id)
        self.bos_token_id = int(tokenizer.bos_token_id or eos_token_id)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "HFAutoTokenizerAdapter":
        AutoTokenizer = importlib.import_module("transformers").AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        return [
            int(token)
            for token in self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        ]


def load_jsonl_examples(path: str | Path) -> list[SFTExample]:
    examples: list[SFTExample] = []
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: record must be a JSON object")

        prompt = record.get("prompt")
        completion = record.get("completion")
        if not isinstance(prompt, str):
            raise ValueError(f"{path}:{line_number}: prompt must be a string")
        if not isinstance(completion, str):
            raise ValueError(f"{path}:{line_number}: completion must be a string")

        weight = record.get("weight", 1.0)
        if not isinstance(weight, int | float):
            raise ValueError(f"{path}:{line_number}: weight must be numeric")
        if float(weight) < 0:
            raise ValueError(f"{path}:{line_number}: weight must be non-negative")

        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}:{line_number}: metadata must be an object")

        examples.append(
            SFTExample(
                prompt=prompt,
                completion=completion,
                weight=float(weight),
                metadata=metadata,
            )
        )
    if not examples:
        raise ValueError(f"{path}: no SFT examples found")
    return examples


def encode_sft_example(
    example: SFTExample,
    *,
    tokenizer: Tokenizer,
    config: SFTDataConfig,
) -> Datum | None:
    config.validate()
    prompt_tokens = tokenizer.encode(example.prompt)
    completion_tokens = tokenizer.encode(example.completion)
    if not completion_tokens:
        raise ValueError("completion must produce at least one token")

    input_ids = (
        [tokenizer.bos_token_id]
        + prompt_tokens
        + completion_tokens
        + [tokenizer.eos_token_id]
    )
    if len(input_ids) > config.sequence_length and config.drop_overlong:
        return None

    target_tokens = input_ids[1:] + [tokenizer.pad_token_id]
    first_completion_loss_index = len(prompt_tokens)
    end_completion_loss_index = len(prompt_tokens) + len(completion_tokens) + 1
    weights = [
        example.weight if first_completion_loss_index <= index < end_completion_loss_index else 0.0
        for index in range(len(input_ids))
    ]
    weights[-1] = 0.0

    input_ids = input_ids[: config.sequence_length]
    target_tokens = target_tokens[: config.sequence_length]
    weights = weights[: config.sequence_length]

    pad_count = config.sequence_length - len(input_ids)
    if pad_count > 0:
        input_ids.extend([tokenizer.pad_token_id] * pad_count)
        target_tokens.extend([tokenizer.pad_token_id] * pad_count)
        weights.extend([0.0] * pad_count)

    if not any(weight > 0 for weight in weights):
        return None

    return Datum(
        model_input=ModelInput.from_ints(input_ids),
        loss_fn_inputs={
            "target_tokens": TensorData.from_ints(target_tokens),
            "weights": TensorData.from_floats(weights),
        },
    )


def batch_datums(datums: Iterable[Datum], *, batch_size: int) -> list[list[Datum]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    batches: list[list[Datum]] = []
    batch: list[Datum] = []
    for datum in datums:
        batch.append(datum)
        if len(batch) == batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    if not batches:
        raise ValueError("no SFT datums available after filtering")
    return batches


def load_jsonl_sft_batches(
    path: str | Path,
    *,
    tokenizer: Tokenizer,
    config: SFTDataConfig,
) -> list[list[Datum]]:
    examples = load_jsonl_examples(path)
    if config.shuffle:
        examples = list(examples)
        random.Random(config.seed).shuffle(examples)

    datums = [
        datum
        for example in examples
        if (datum := encode_sft_example(example, tokenizer=tokenizer, config=config)) is not None
    ]
    return batch_datums(datums, batch_size=config.batch_size)
