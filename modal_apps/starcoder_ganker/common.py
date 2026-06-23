"""Shared constants for the StarCoderData-style Ganker example."""

from __future__ import annotations

import os
from pathlib import Path


APP_NAME = os.getenv("GANKER_STARCODER_APP", "ganker-starcoder-code-sft")
DATASET_ID = os.getenv("GANKER_STARCODER_DATASET_ID", "bigcode/the-stack-smol-xs")
DATASET_VOLUME_NAME = os.getenv(
    "GANKER_STARCODER_DATASET_VOLUME",
    f"{DATASET_ID.replace('/', '-')}-ganker-jsonl",
)
DATASET_MOUNT = Path(os.getenv("GANKER_STARCODER_DATASET_MOUNT", "/vol/ganker-code-data"))
DEFAULT_DATASET_PATH = DATASET_MOUNT / "starcoder_go_rust_sft.jsonl"
DEFAULT_LANGUAGES: tuple[str, ...] = ()
DEFAULT_STARCODER_LANGUAGES = ("go", "rust")

MODEL = os.getenv("GANKER_STARCODER_MODEL", "Qwen/Qwen3-0.6B")
RUN_PREFIX = os.getenv("GANKER_STARCODER_RUN_PREFIX", "starcoder-ganker")

SGLANG_IMAGE = os.getenv("GANKER_STARCODER_SGLANG_IMAGE", "lmsysorg/sglang:v0.5.12")
SGLANG_GPU = os.getenv("GANKER_STARCODER_EVAL_GPU", os.getenv("GANKER_MODAL_GPU", "L40S"))


DEFAULT_EVAL_PROMPTS = [
    """
// Fib takes a number n and returns the nth Fibonacci number using
// the efficient iterative algorithm.
func Fib(n int) int {""",
    """
// CheckRotation takes two strings and returns true if one is a rotation of the other.
func CheckRotation(s1 string, s2 string) bool {""",
    """
/// Returns the nth Fibonacci number using the efficient iterative algorithm.
fn fib(n: u32) -> u32 {""",
    """
/// Returns true if one string is a rotation of the other.
fn check_rotation(s1: &str, s2: &str) -> bool {""",
]


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
