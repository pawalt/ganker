"""gRPC-backed implementation of the public proxy transport."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import grpc

from ganker.contracts import (
    CreateTrainingRunRequest,
    CreateTrainingRunResponse,
    DownloadArtifactFileRequest,
    DownloadArtifactFileResponse,
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
from ganker.errors import BackendUnavailableError, GankerError, InvalidRequestError, NotFoundError
from ganker.rpc import conversion as conv
from ganker.rpc.v1 import proxy_pb2_grpc


class GrpcTransportError(GankerError):
    """Raised when a gRPC failure has no narrower domain mapping."""

    def __init__(self, code: grpc.StatusCode, details: str):
        super().__init__(f"{code.name}: {details}")
        self.code = code
        self.details = details


class GrpcProxyTransport:
    """Synchronous `ProxyTransport` backed by a generated gRPC stub."""

    def __init__(
        self,
        stub: proxy_pb2_grpc.GankerProxyStub,
        *,
        timeout: float = 20,
        metadata: Iterable[tuple[str, str]] = (),
        channel: grpc.Channel | None = None,
    ):
        self._stub = stub
        self._timeout = timeout
        self._metadata = tuple(metadata)
        self._channel = channel

    @classmethod
    def connect(
        cls,
        target: str,
        *,
        timeout: float = 20,
        bearer_token: str | None = None,
        options: Iterable[tuple[str, Any]] = (),
    ) -> "GrpcProxyTransport":
        channel = grpc.insecure_channel(target, options=tuple(options))
        metadata: tuple[tuple[str, str], ...] = ()
        if bearer_token:
            metadata = (("authorization", f"Bearer {bearer_token}"),)
        return cls(
            proxy_pb2_grpc.GankerProxyStub(channel),
            timeout=timeout,
            metadata=metadata,
            channel=channel,
        )

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def create_training_run(self, request: CreateTrainingRunRequest) -> CreateTrainingRunResponse:
        response = self._call(
            self._stub.CreateTrainingRun,
            conv.create_training_run_request_to_proto(request),
        )
        return conv.create_training_run_response_from_proto(response)

    def forward_backward(self, request: ForwardBackwardRequest) -> ForwardBackwardResponse:
        response = self._call(
            self._stub.ForwardBackward,
            conv.forward_backward_request_to_proto(request),
        )
        return conv.forward_backward_response_from_proto(response)

    def optim_step(self, request: OptimStepRequest) -> OptimStepResponse:
        response = self._call(
            self._stub.OptimStep,
            conv.optim_step_request_to_proto(request),
        )
        return conv.optim_step_response_from_proto(response)

    def save_weights(self, request: SaveWeightsRequest) -> SaveWeightsResponse:
        response = self._call(
            self._stub.SaveWeights,
            conv.save_weights_request_to_proto(request),
        )
        return conv.save_weights_response_from_proto(response)

    def refresh_weights(self, request: RefreshWeightsRequest) -> RefreshWeightsResponse:
        response = self._call(
            self._stub.RefreshWeights,
            conv.refresh_weights_request_to_proto(request),
        )
        return conv.refresh_weights_response_from_proto(response)

    def sample(self, request: SampleRequest) -> SampleResponse:
        response = self._call(
            self._stub.Sample,
            conv.sample_request_to_proto(request),
        )
        return conv.sample_response_from_proto(response)

    def get_telemetry_summary(
        self,
        request: GetTelemetrySummaryRequest,
    ) -> GetTelemetrySummaryResponse:
        response = self._call(
            self._stub.GetTelemetrySummary,
            conv.get_telemetry_summary_request_to_proto(request),
        )
        return conv.get_telemetry_summary_response_from_proto(response)

    def download_artifact_file(
        self,
        request: DownloadArtifactFileRequest,
    ) -> DownloadArtifactFileResponse:
        response = self._call(
            self._stub.DownloadArtifactFile,
            conv.download_artifact_file_request_to_proto(request),
        )
        return conv.download_artifact_file_response_from_proto(response)

    def _call(self, method, request):
        try:
            return method(
                request,
                timeout=self._timeout,
                metadata=self._metadata,
            )
        except grpc.RpcError as exc:
            raise _map_rpc_error(exc) from exc


def _map_rpc_error(exc: grpc.RpcError) -> GankerError:
    code = exc.code()
    details = exc.details() or ""
    if code == grpc.StatusCode.INVALID_ARGUMENT:
        return InvalidRequestError(details)
    if code == grpc.StatusCode.NOT_FOUND:
        return NotFoundError(details)
    if code == grpc.StatusCode.UNAVAILABLE:
        return BackendUnavailableError(details)
    return GrpcTransportError(code, details)
