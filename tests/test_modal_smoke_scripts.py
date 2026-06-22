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
