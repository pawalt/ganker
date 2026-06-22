"""Proxy transports used by public clients."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ganker.contracts import (
    CreateTrainingRunRequest,
    CreateTrainingRunResponse,
    DownloadArtifactFileRequest,
    DownloadArtifactFileResponse,
    ArtifactFileKind,
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
from ganker.errors import InvalidRequestError, NotFoundError


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

    def download_artifact_file(
        self,
        request: DownloadArtifactFileRequest,
    ) -> DownloadArtifactFileResponse:
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

    def download_artifact_file(
        self,
        request: DownloadArtifactFileRequest,
    ) -> DownloadArtifactFileResponse:
        if request.file_kind == ArtifactFileKind.MANIFEST:
            path = request.artifact.manifest_path
        elif request.file_kind == ArtifactFileKind.PAYLOAD:
            path = request.artifact.payload_path
        else:
            raise InvalidRequestError(f"unsupported artifact file kind: {request.file_kind}")

        artifact_path = Path(path)
        if not artifact_path.exists():
            raise NotFoundError(f"artifact file is missing: {artifact_path}")
        return DownloadArtifactFileResponse(
            request_id=request.context.request_id,
            artifact=request.artifact,
            file_kind=request.file_kind,
            path=str(artifact_path),
            contents=artifact_path.read_bytes(),
        )
