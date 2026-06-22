"""Proxy transports used by public clients."""

from __future__ import annotations

from typing import Any, Protocol

from ganker.contracts import (
    CreateTrainingRunRequest,
    CreateTrainingRunResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    GetTelemetrySummaryRequest,
    GetTelemetrySummaryResponse,
    OptimStepRequest,
    OptimStepResponse,
    RefreshWeightsRequest,
    RefreshWeightsResponse,
    SampleRequest,
    SampleResponse,
    SaveWeightsRequest,
    SaveWeightsResponse,
)


class ProxyTransport(Protocol):
    """Synchronous proxy API used by `ServiceClient`."""

    def create_training_run(self, request: CreateTrainingRunRequest) -> CreateTrainingRunResponse:
        ...

    def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        ...

    def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        ...

    def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        ...

    def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        ...

    def sample(self, request: SampleRequest) -> SampleResponse:
        ...

    def get_telemetry_summary(
        self,
        request: GetTelemetrySummaryRequest,
    ) -> GetTelemetrySummaryResponse:
        ...


class MonarchProxyTransport:
    """Proxy transport backed by a Monarch `ProxyActor` handle."""

    def __init__(self, proxy_actor: Any, *, timeout: float = 20):
        self._proxy = proxy_actor
        self._timeout = timeout

    def create_training_run(self, request: CreateTrainingRunRequest) -> CreateTrainingRunResponse:
        return self._proxy.create_training_run.choose(request).get(timeout=self._timeout)

    def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        return self._proxy.forward_backward.choose(request).get(timeout=self._timeout)

    def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        return self._proxy.optim_step.choose(request).get(timeout=self._timeout)

    def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        return self._proxy.save_weights.choose(request).get(timeout=self._timeout)

    def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        return self._proxy.refresh_weights.choose(request).get(timeout=self._timeout)

    def sample(self, request: SampleRequest) -> SampleResponse:
        return self._proxy.sample.choose(request).get(timeout=self._timeout)

    def get_telemetry_summary(
        self,
        request: GetTelemetrySummaryRequest,
    ) -> GetTelemetrySummaryResponse:
        return self._proxy.get_telemetry_summary.choose(request).get(timeout=self._timeout)
