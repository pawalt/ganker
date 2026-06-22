"""Small supervised fine-tuning helpers for example workflows."""

from .data import (
    HFAutoTokenizerAdapter,
    SFTDataConfig,
    SFTExample,
    ToyTokenizer,
    batch_datums,
    encode_sft_example,
    load_jsonl_examples,
    load_jsonl_sft_batches,
)
from .loop import SFTRunSummary, run_sft
from .real_data import (
    DEFAULT_REAL_DATASET,
    DEFAULT_REAL_DATASET_FORMAT,
    DEFAULT_REAL_DATASET_SPLIT,
    alpaca_record_to_sft_example,
    materialize_hf_sft_jsonl,
    write_sft_jsonl,
)

__all__ = [
    "HFAutoTokenizerAdapter",
    "DEFAULT_REAL_DATASET",
    "DEFAULT_REAL_DATASET_FORMAT",
    "DEFAULT_REAL_DATASET_SPLIT",
    "SFTDataConfig",
    "SFTExample",
    "SFTRunSummary",
    "ToyTokenizer",
    "alpaca_record_to_sft_example",
    "batch_datums",
    "encode_sft_example",
    "load_jsonl_examples",
    "load_jsonl_sft_batches",
    "materialize_hf_sft_jsonl",
    "run_sft",
    "write_sft_jsonl",
]
