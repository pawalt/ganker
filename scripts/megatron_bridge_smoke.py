#!/usr/bin/env python
"""Compatibility wrapper for the importable Modal smoke test package."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT / "tests", PROJECT_ROOT / "src", PROJECT_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from modal_smoke.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
