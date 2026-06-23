import json
import importlib
import subprocess
import sys

from modal_smoke.cli import run_from_argv


def test_megatron_smoke_env_mode_is_importable_api():
    payload = run_from_argv(["--mode", "env"])

    assert payload["ok"] is True
    assert payload["mode"] == "env"
    assert "python" in payload
    assert "packages" in payload


def test_megatron_smoke_env_mode_returns_json():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/megatron_bridge_smoke.py",
            "--mode",
            "env",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["ok"] is True
    assert payload["mode"] == "env"
    assert "python" in payload
    assert "packages" in payload


def test_modal_smoke_modules_import_without_heavy_ml_modules():
    sys.modules.pop("modal_smoke.cli", None)
    sys.modules.pop("modal_smoke.megatron_core_smoke", None)
    sys.modules.pop("modal_smoke.ganker_smoke", None)
    sys.modules.pop("megatron", None)
    sys.modules.pop("torch", None)

    importlib.import_module("modal_smoke.cli")

    assert "megatron" not in sys.modules
    assert "torch" not in sys.modules


def test_modal_smoke_app_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/megatron_smoke.py",
        ],
        check=True,
    )


def test_sft_modal_app_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/sft.py",
        ],
        check=True,
    )


def test_sglang_modal_app_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/sglang_smoke.py",
        ],
        check=True,
    )


def test_distributed_mesh_modal_app_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/distributed_mesh.py",
        ],
        check=True,
    )


def test_distributed_modal_infra_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/distributed/infra.py",
        ],
        check=True,
    )


def test_distributed_sft_job_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/distributed/sft_job.py",
        ],
        check=True,
    )


def test_qwen_sft_modal_infra_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/qwen_sft/infra.py",
        ],
        check=True,
    )


def test_qwen_sft_modal_example_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/qwen_sft/sft.py",
        ],
        check=True,
    )


def test_qwen_sft_hf_comparison_example_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/qwen_sft/compare_hf.py",
        ],
        check=True,
    )


def test_qwen_sft_multinode_infra_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/qwen_sft_multinode/infra.py",
        ],
        check=True,
    )


def test_qwen_sft_multinode_train_entry_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/qwen_sft_multinode/train_entry.py",
        ],
        check=True,
    )


def test_qwen_sft_multinode_example_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/qwen_sft_multinode/sft.py",
        ],
        check=True,
    )


def test_qwen_sft_multinode_hf_comparison_compiles():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "py_compile",
            "modal_apps/qwen_sft_multinode/compare_hf.py",
        ],
        check=True,
    )


def test_starcoder_ganker_modal_files_compile():
    for path in [
        "modal_apps/starcoder_ganker/common.py",
        "modal_apps/starcoder_ganker/infra.py",
        "modal_apps/starcoder_ganker/download_dataset.py",
        "modal_apps/starcoder_ganker/sft.py",
        "modal_apps/starcoder_ganker/evaluate.py",
    ]:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "py_compile",
                path,
            ],
            check=True,
        )
