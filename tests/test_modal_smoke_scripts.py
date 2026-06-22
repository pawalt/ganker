import json
import subprocess
import sys


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
