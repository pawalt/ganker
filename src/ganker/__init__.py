"""Local singleton Tinker-style Monarch prototype."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ganker.client import SamplingClient, ServiceClient, TrainingClient

__all__ = ["SamplingClient", "ServiceClient", "TrainingClient", "__version__"]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name in {"SamplingClient", "ServiceClient", "TrainingClient"}:
        from ganker.client import SamplingClient, ServiceClient, TrainingClient

        return {
            "SamplingClient": SamplingClient,
            "ServiceClient": ServiceClient,
            "TrainingClient": TrainingClient,
        }[name]
    raise AttributeError(f"module 'ganker' has no attribute {name!r}")
