"""Local singleton Tinker-style Monarch prototype."""

from ganker.client import SamplingClient, ServiceClient, TrainingClient

__all__ = ["SamplingClient", "ServiceClient", "TrainingClient", "__version__"]

__version__ = "0.1.0"
