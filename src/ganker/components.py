"""Pure component implementations behind Monarch actor endpoints."""

from __future__ import annotations

import uuid

from ganker.backends.base import InferenceBackend, TrainingBackend
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
    SampleRequest,
    SampleResponse,
    SaveWeightsRequest,
    SaveWeightsResponse,
    TelemetrySummary,
    UsageBySource,
    UsageEvent,
)
from ganker.errors import InvalidRequestError


def request_id_from(request_id: str) -> str:
    return request_id or uuid.uuid4().hex


class TrainingComponent:
    def __init__(self, backend: TrainingBackend):
        self._backend = backend

    def create_training_run(self, request: CreateTrainingRunRequest) -> CreateTrainingRunResponse:
        request_id = request_id_from(request.context.request_id)
        run = self._backend.create_training_run(
            base_model=request.base_model,
            tuning_mode=request.tuning_mode,
            lora_rank=request.lora_rank,
        )
        return CreateTrainingRunResponse(request_id=request_id, run=run)

    def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        request_id = request_id_from(request.context.request_id)
        result = self._backend.forward_backward(
            run_id=request.run_id,
            data=request.data,
            loss_fn=request.loss_fn,
            loss_fn_config=request.loss_fn_config,
        )
        return ForwardBackwardResponse(
            request_id=request_id,
            run_id=result.run_id,
            output=result.output,
            gradient_version=result.gradient_version,
            usage=result.usage,
        )

    def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        request_id = request_id_from(request.context.request_id)
        result = self._backend.optim_step(
            run_id=request.run_id,
            params=request.optimizer,
        )
        return OptimStepResponse(
            request_id=request_id,
            run_id=result.run_id,
            optimizer_step=result.optimizer_step,
            checkpoint_version=result.checkpoint_version,
            usage=result.usage,
        )

    def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        request_id = request_id_from(request.context.request_id)
        artifact = self._backend.save_weights(run_id=request.run_id, kind=request.kind)
        return SaveWeightsResponse(request_id=request_id, artifact=artifact)


class RolloutComponent:
    def __init__(self, backend: InferenceBackend):
        self._backend = backend

    def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        request_id = request_id_from(request.context.request_id)
        artifact = self._backend.refresh_weights(
            run_id=request.run_id,
            artifact=request.artifact,
        )
        return RefreshWeightsResponse(request_id=request_id, artifact=artifact)

    def sample(self, request: SampleRequest) -> SampleResponse:
        request_id = request_id_from(request.context.request_id)
        result = self._backend.sample(
            run_id=request.run_id,
            prompt=request.prompt,
            sampling_params=request.sampling_params,
            num_samples=request.num_samples,
        )
        return SampleResponse(
            request_id=request_id,
            run_id=result.run_id,
            sequences=result.sequences,
            artifact=result.artifact,
            usage=result.usage,
        )


class TelemetryLedger:
    """In-memory event ledger for local singleton development."""

    def __init__(self):
        self._events: list[UsageEvent] = []

    @property
    def event_count(self) -> int:
        return len(self._events)

    def record(self, event: UsageEvent) -> int:
        if not event.request_id:
            raise InvalidRequestError("event.request_id is required")
        if not event.run_id:
            raise InvalidRequestError("event.run_id is required")
        if not event.event_source:
            raise InvalidRequestError("event.event_source is required")

        self._events.append(event)
        return len(self._events)

    def summary(self, run_id: str) -> TelemetrySummary:
        if not run_id:
            raise InvalidRequestError("run_id is required")

        summary = TelemetrySummary(run_id=run_id)
        by_source: dict[str, UsageBySource] = {}
        for event in self._events:
            if event.run_id != run_id:
                continue
            summary.event_count += 1
            summary.total.add(event.usage)
            source = by_source.setdefault(
                event.event_source,
                UsageBySource(event_source=event.event_source),
            )
            source.usage.add(event.usage)
            source.event_count += 1

        summary.by_source = [by_source[name] for name in sorted(by_source)]
        return summary


class TelemetryComponent:
    def __init__(self, ledger: TelemetryLedger):
        self._ledger = ledger

    def record(self, request: RecordTelemetryRequest) -> RecordTelemetryResponse:
        request_id = request_id_from(request.context.request_id)
        event = request.event
        if not event.request_id:
            event = UsageEvent(
                request_id=request_id,
                run_id=event.run_id,
                event_source=event.event_source,
                usage=event.usage,
            )
        count = self._ledger.record(event)
        return RecordTelemetryResponse(request_id=request_id, event_count=count)

    def get_summary(self, request: GetTelemetrySummaryRequest) -> GetTelemetrySummaryResponse:
        request_id = request_id_from(request.context.request_id)
        summary = self._ledger.summary(request.run_id)
        return GetTelemetrySummaryResponse(request_id=request_id, summary=summary)
