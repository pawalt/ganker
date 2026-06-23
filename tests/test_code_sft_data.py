from examples.sft import (
    code_record_to_sft_example,
    select_starcoder_parquet_files,
)


def test_select_starcoder_parquet_files_filters_and_caps_languages():
    selected = select_starcoder_parquet_files(
        [
            "go/train-00002.parquet",
            "go/train-00001.parquet",
            "rust/train-00001.parquet",
            "python/train-00001.parquet",
            "jupyter-scripts-dedup-filtered/train-00001.parquet",
            "rust/readme.txt",
        ],
        languages=["go", "rust"],
        max_files_per_language=1,
    )

    assert selected == ["go/train-00001.parquet", "rust/train-00001.parquet"]


def test_code_record_to_sft_example_formats_prompt_and_metadata():
    example = code_record_to_sft_example(
        {
            "content": "package main\n\nfunc main() {}\n",
            "max_stars_repo_path": "cmd/demo/main.go",
        },
        language="go",
        min_chars=1,
    )

    assert example is not None
    assert "Go source file" in example.prompt
    assert "`cmd/demo/main.go`" in example.prompt
    assert example.completion == "package main\n\nfunc main() {}"
    assert example.metadata["language"] == "go"
    assert example.metadata["source_path"] == "cmd/demo/main.go"
    assert example.metadata["truncated"] is False


def test_code_record_to_sft_example_supports_prompt_completion_rows():
    example = code_record_to_sft_example(
        {
            "prompt": "def add(a, b):",
            "completion": "\n    return a + b",
            "task_id": "add",
        },
        language="python",
        min_chars=1,
    )

    assert example is not None
    assert example.prompt == "def add(a, b):\n"
    assert example.completion == "return a + b"
    assert example.metadata["language"] == "python"


def test_code_record_to_sft_example_skips_tiny_or_missing_content():
    assert code_record_to_sft_example({"content": "x"}, min_chars=2) is None
    assert code_record_to_sft_example({"body": "package main"}, min_chars=1) is None


def test_code_record_to_sft_example_truncates_large_content():
    example = code_record_to_sft_example(
        {"content": "0123456789", "path": "src/lib.rs"},
        language="rust",
        min_chars=1,
        max_chars=4,
    )

    assert example is not None
    assert example.completion == "0123"
    assert example.metadata["truncated"] is True
