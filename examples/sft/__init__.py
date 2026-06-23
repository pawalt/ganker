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
from .code_data import (
    DEFAULT_CODE_DATASET,
    DEFAULT_CODE_LANGUAGES,
    code_record_to_sft_example,
    materialize_hf_code_sft_jsonl,
    select_starcoder_parquet_files,
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
    "DEFAULT_CODE_DATASET",
    "DEFAULT_CODE_LANGUAGES",
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
    "code_record_to_sft_example",
    "encode_sft_example",
    "load_jsonl_examples",
    "load_jsonl_sft_batches",
    "materialize_hf_code_sft_jsonl",
    "materialize_hf_sft_jsonl",
    "run_sft",
    "select_starcoder_parquet_files",
    "write_sft_jsonl",
]
