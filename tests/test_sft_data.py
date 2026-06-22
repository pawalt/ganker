import json
from pathlib import Path

import pytest

from examples.sft.data import (
    HFAutoTokenizerAdapter,
    SFTDataConfig,
    SFTExample,
    ToyTokenizer,
    batch_datums,
    encode_sft_example,
    load_jsonl_examples,
    load_jsonl_sft_batches,
)
from examples.sft.real_data import alpaca_record_to_sft_example, write_sft_jsonl


class FixedTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, mapping: dict[str, list[int]]):
        self.mapping = mapping

    def encode(self, text: str) -> list[int]:
        return list(self.mapping[text])


def test_load_jsonl_examples_validates_records(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"prompt": 123, "completion": "x"}) + "\n")

    with pytest.raises(ValueError, match="prompt must be a string"):
        load_jsonl_examples(path)


def test_hf_tokenizer_adapter_uses_eos_for_missing_bos_and_pad():
    class FakeHFTokenizer:
        eos_token_id = 99
        bos_token_id = None
        pad_token_id = None

        def encode(self, text, *, add_special_tokens):
            assert text == "abc"
            assert add_special_tokens is False
            return [10, 11, 12]

    adapter = HFAutoTokenizerAdapter(FakeHFTokenizer())

    assert adapter.eos_token_id == 99
    assert adapter.bos_token_id == 99
    assert adapter.pad_token_id == 99
    assert adapter.encode("abc") == [10, 11, 12]


def test_encode_sft_example_shifts_targets_and_masks_completion_boundary():
    tokenizer = FixedTokenizer({"p": [10, 11], "c": [20, 21]})
    datum = encode_sft_example(
        SFTExample(prompt="p", completion="c"),
        tokenizer=tokenizer,
        config=SFTDataConfig(sequence_length=8, shuffle=False),
    )

    assert datum is not None
    assert datum.model_input.token_ids == [1, 10, 11, 20, 21, 2, 0, 0]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [10, 11, 20, 21, 2, 0, 0, 0]
    assert datum.loss_fn_inputs["weights"].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_encode_sft_example_truncates_and_keeps_only_remaining_loss_tokens():
    tokenizer = FixedTokenizer({"prompt": [10, 11, 12], "completion": [20, 21]})
    datum = encode_sft_example(
        SFTExample(prompt="prompt", completion="completion"),
        tokenizer=tokenizer,
        config=SFTDataConfig(sequence_length=5, shuffle=False),
    )

    assert datum is not None
    assert datum.model_input.token_ids == [1, 10, 11, 12, 20]
    assert datum.loss_fn_inputs["target_tokens"].tolist() == [10, 11, 12, 20, 21]
    assert datum.loss_fn_inputs["weights"].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_encode_sft_example_can_drop_overlong_records():
    tokenizer = FixedTokenizer({"prompt": [10, 11, 12], "completion": [20]})
    datum = encode_sft_example(
        SFTExample(prompt="prompt", completion="completion"),
        tokenizer=tokenizer,
        config=SFTDataConfig(sequence_length=4, drop_overlong=True, shuffle=False),
    )

    assert datum is None


def test_load_jsonl_sft_batches_uses_toy_tokenizer_and_fixed_lengths(tmp_path: Path):
    path = tmp_path / "tiny.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "A:", "completion": " B"}),
                json.dumps({"prompt": "C:", "completion": " D"}),
                json.dumps({"prompt": "E:", "completion": " F"}),
            ]
        )
        + "\n"
    )

    batches = load_jsonl_sft_batches(
        path,
        tokenizer=ToyTokenizer(vocab_size=64),
        config=SFTDataConfig(sequence_length=12, batch_size=2, shuffle=False),
    )

    assert [len(batch) for batch in batches] == [2, 1]
    assert all(len(datum.model_input.token_ids) == 12 for batch in batches for datum in batch)


def test_batch_datums_rejects_empty_iterable():
    with pytest.raises(ValueError, match="no SFT datums"):
        batch_datums([], batch_size=1)


def test_alpaca_record_to_sft_example_formats_instruction_input_and_response():
    example = alpaca_record_to_sft_example(
        {
            "instruction": "Classify the sentiment.",
            "input": "I liked the movie.",
            "output": "Positive",
        }
    )

    assert "### Instruction:\nClassify the sentiment." in example.prompt
    assert "### Input:\nI liked the movie." in example.prompt
    assert example.prompt.endswith("### Response:\n")
    assert example.completion == "Positive"
    assert example.metadata == {"dataset_format": "alpaca"}


def test_write_sft_jsonl_round_trips_through_existing_loader(tmp_path: Path):
    path = tmp_path / "alpaca.jsonl"

    count = write_sft_jsonl(
        [
            SFTExample(
                prompt="Prompt",
                completion="Completion",
                weight=0.5,
                metadata={"source": "test"},
            )
        ],
        path,
    )

    assert count == 1
    [loaded] = load_jsonl_examples(path)
    assert loaded.prompt == "Prompt"
    assert loaded.completion == "Completion"
    assert loaded.weight == 0.5
    assert loaded.metadata == {"source": "test"}
