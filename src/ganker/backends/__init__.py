"""Backend interfaces and local fake implementations."""

from ganker.backends.base import (
    ForwardBackwardResult,
    InferenceBackend,
    OptimStepResult,
    SampleResult,
    TrainingBackend,
)
from ganker.backends.fake import FakeInferenceBackend, FakeTrainingBackend

__all__ = [
    "FakeInferenceBackend",
    "FakeTrainingBackend",
    "ForwardBackwardResult",
    "InferenceBackend",
    "OptimStepResult",
    "SampleResult",
    "TrainingBackend",
]
