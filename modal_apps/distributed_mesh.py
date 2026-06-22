"""Compatibility entrypoint for the distributed Modal mesh.

New code lives in:

- `modal_apps/distributed/infra.py` for deployable Modal/Monarch infra
- `modal_apps/distributed/sft_job.py` for Tinker-style training logic

Existing commands such as:

    uv run modal run modal_apps/distributed_mesh.py --mode qwen-bridge-sglang-distributed

still work through the imported local entrypoint below.
"""

from __future__ import annotations

from modal_apps.distributed.infra import app
from modal_apps.distributed.sft_job import main


__all__ = ["app", "main"]
