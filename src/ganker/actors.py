"""Monarch actors for the singleton Tinker-style system."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from monarch.actor import Actor, endpoint

from ganker.backends.factory import build_inference_backend, build_training_backend
from ganker.components import RolloutComponent, TelemetryComponent, TelemetryLedger, TrainingComponent
from ganker.contracts import (
    CreateTrainingRunRequest,
    CreateTrainingRunResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    GetTelemetrySummaryRequest,
    GetTelemetrySummaryResponse,
    OptimStepRequest,
    OptimStepResponse,
    RecordTelemetryRequest,
    RecordTelemetryResponse,
    RefreshWeightsRequest,
    RefreshWeightsResponse,
    RequestContext,
    SampleRequest,
    SampleResponse,
    SaveWeightsRequest,
    SaveWeightsResponse,
    UsageEvent,
)


class TrainingActor(Actor):
    """Owns singleton training state and delegates execution to a backend."""

    def __init__(
        self,
        artifact_root: str,
        backend_kind: str = "fake",
        backend_config: dict[str, Any] | None = None,
    ):
        backend = build_training_backend(
            backend_kind,
            Path(artifact_root),
            config=backend_config,
        )
        self._component = TrainingComponent(backend)

    @endpoint
    def create_training_run(self, request: CreateTrainingRunRequest) -> CreateTrainingRunResponse:
        return self._component.create_training_run(request)

    @endpoint
    def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        return self._component.forward_backward(request)

    @endpoint
    def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        return self._component.optim_step(request)

    @endpoint
    def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        return self._component.save_weights(request)

    @endpoint
    def shutdown(self) -> None:
        self._component.shutdown()


class RolloutActor(Actor):
    """Owns rollout state and delegates sample calls to an inference backend."""

    def __init__(
        self,
        artifact_root: str,
        backend_kind: str = "fake",
        backend_config: dict[str, Any] | None = None,
    ):
        backend = build_inference_backend(
            backend_kind,
            Path(artifact_root),
            config=backend_config,
        )
        self._component = RolloutComponent(backend)

    @endpoint
    def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        return self._component.refresh_weights(request)

    @endpoint
    def sample(self, request: SampleRequest) -> SampleResponse:
        return self._component.sample(request)

    @endpoint
    def shutdown(self) -> None:
        self._component.shutdown()


class TelemetryActor(Actor):
    """Records general usage and event telemetry for local development."""

    def __init__(self):
        self._component = TelemetryComponent(TelemetryLedger())

    @endpoint
    def record(self, request: RecordTelemetryRequest) -> RecordTelemetryResponse:
        return self._component.record(request)

    @endpoint
    def get_summary(self, request: GetTelemetrySummaryRequest) -> GetTelemetrySummaryResponse:
        return self._component.get_summary(request)

    @endpoint
    def shutdown(self) -> None:
        return None


class ControllerActor(Actor):
    """Owns run topology and routes requests to trainer, rollout, and telemetry."""

    def __init__(self, training: Any, rollout: Any, telemetry: Any):
        self._training = training
        self._rollout = rollout
        self._telemetry = telemetry

    @endpoint
    async def create_training_run(self, request: CreateTrainingRunRequest) -> CreateTrainingRunResponse:
        return await self._training.create_training_run.choose(request)

    @endpoint
    async def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        response = await self._training.forward_backward.choose(request)
        await self._record_usage(response.request_id, response.run_id, "trainer", response.usage)
        return response

    @endpoint
    async def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        response = await self._training.optim_step.choose(request)
        await self._record_usage(response.request_id, response.run_id, "trainer", response.usage)
        return response

    @endpoint
    async def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        return await self._training.save_weights.choose(request)

    @endpoint
    async def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        return await self._rollout.refresh_weights.choose(request)

    @endpoint
    async def sample(self, request: SampleRequest) -> SampleResponse:
        response = await self._rollout.sample.choose(request)
        await self._record_usage(response.request_id, response.run_id, "rollout", response.usage)
        return response

    @endpoint
    async def get_telemetry_summary(
        self,
        request: GetTelemetrySummaryRequest,
    ) -> GetTelemetrySummaryResponse:
        return await self._telemetry.get_summary.choose(request)

    @endpoint
    async def shutdown(self) -> None:
        return None

    async def _record_usage(self, request_id: str, run_id: str, event_source: str, usage):
        if not usage.has_activity():
            return
        request = RecordTelemetryRequest(
            context=RequestContext(request_id=request_id),
            event=UsageEvent(
                request_id=request_id,
                run_id=run_id,
                event_source=event_source,
                usage=usage,
            ),
        )
        await self._telemetry.record.choose(request)


class ControllerProxyActor(Actor):
    """Client-facing actor that delegates every operation to a ControllerActor."""

    def __init__(self, controller: Any):
        self._controller = controller

    @endpoint
    async def create_training_run(self, request: CreateTrainingRunRequest) -> CreateTrainingRunResponse:
        return await self._controller.create_training_run.choose(request)

    @endpoint
    async def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        return await self._controller.forward_backward.choose(request)

    @endpoint
    async def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        return await self._controller.optim_step.choose(request)

    @endpoint
    async def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        return await self._controller.save_weights.choose(request)

    @endpoint
    async def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        return await self._controller.refresh_weights.choose(request)

    @endpoint
    async def sample(self, request: SampleRequest) -> SampleResponse:
        return await self._controller.sample.choose(request)

    @endpoint
    async def get_telemetry_summary(
        self,
        request: GetTelemetrySummaryRequest,
    ) -> GetTelemetrySummaryResponse:
        return await self._controller.get_telemetry_summary.choose(request)

    @endpoint
    async def shutdown(self) -> None:
        return None


class ProxyActor(Actor):
    """Client-facing actor that routes requests to trainer, rollout, and telemetry."""

    def __init__(self, training: Any, rollout: Any, telemetry: Any):
        self._training = training
        self._rollout = rollout
        self._telemetry = telemetry

    @endpoint
    async def create_training_run(self, request: CreateTrainingRunRequest) -> CreateTrainingRunResponse:
        return await self._training.create_training_run.choose(request)

    @endpoint
    async def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        response = await self._training.forward_backward.choose(request)
        await self._record_usage(response.request_id, response.run_id, "trainer", response.usage)
        return response

    @endpoint
    async def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        response = await self._training.optim_step.choose(request)
        await self._record_usage(response.request_id, response.run_id, "trainer", response.usage)
        return response

    @endpoint
    async def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        return await self._training.save_weights.choose(request)

    @endpoint
    async def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        return await self._rollout.refresh_weights.choose(request)

    @endpoint
    async def sample(self, request: SampleRequest) -> SampleResponse:
        response = await self._rollout.sample.choose(request)
        await self._record_usage(response.request_id, response.run_id, "rollout", response.usage)
        return response

    @endpoint
    async def get_telemetry_summary(
        self,
        request: GetTelemetrySummaryRequest,
    ) -> GetTelemetrySummaryResponse:
        return await self._telemetry.get_summary.choose(request)

    @endpoint
    async def shutdown(self) -> None:
        return None

    async def _record_usage(self, request_id: str, run_id: str, event_source: str, usage):
        if not usage.has_activity():
            return
        request = RecordTelemetryRequest(
            context=RequestContext(request_id=request_id),
            event=UsageEvent(
                request_id=request_id,
                run_id=run_id,
                event_source=event_source,
                usage=usage,
            ),
        )
        await self._telemetry.record.choose(request)
