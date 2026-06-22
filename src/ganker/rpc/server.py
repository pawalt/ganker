"""gRPC server wrapper for a private local Monarch mesh."""

from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
import re
import uuid

import grpc

from ganker.config import MegatronBackendConfig
from ganker.contracts import (
    ArtifactFileKind,
    DownloadArtifactFileResponse,
)
from ganker.errors import BackendUnavailableError, GankerError, InvalidRequestError, NotFoundError
from ganker.orchestration import LocalMonarchMesh, start_local_monarch_mesh
from ganker.rpc import conversion as conv
from ganker.rpc.v1 import proxy_pb2_grpc
from ganker.transport import MonarchProxyTransport, ProxyTransport


_DEFAULT_GRPC_OPTIONS = (
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
)


@dataclass
class ProxyGrpcServer:
    """Owns the gRPC server and the private local Monarch mesh behind it."""

    grpc_server: grpc.Server
    mesh: LocalMonarchMesh
    bound_address: str

    def serve_forever(self) -> None:
        self.grpc_server.wait_for_termination()

    def stop(self, grace: float | None = 1) -> None:
        self.grpc_server.stop(grace).wait()
        self.mesh.stop()

    def __enter__(self) -> "ProxyGrpcServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


class GankerProxyServicer(proxy_pb2_grpc.GankerProxyServicer):
    """Generated gRPC servicer that delegates to a `ProxyTransport`."""

    def __init__(
        self,
        transport: ProxyTransport,
        *,
        artifact_root: Path,
        bearer_token: str | None = None,
    ):
        self._transport = transport
        self._artifact_root = Path(artifact_root)
        self._bearer_token = bearer_token

    def CreateTrainingRun(self, request, context):
        return self._handle(
            context,
            lambda: conv.create_training_run_response_to_proto(
                self._transport.create_training_run(
                    conv.create_training_run_request_from_proto(request)
                )
            ),
        )

    def ForwardBackward(self, request, context):
        return self._handle(
            context,
            lambda: conv.forward_backward_response_to_proto(
                self._transport.forward_backward(
                    conv.forward_backward_request_from_proto(request)
                )
            ),
        )

    def OptimStep(self, request, context):
        return self._handle(
            context,
            lambda: conv.optim_step_response_to_proto(
                self._transport.optim_step(conv.optim_step_request_from_proto(request))
            ),
        )

    def SaveWeights(self, request, context):
        return self._handle(
            context,
            lambda: conv.save_weights_response_to_proto(
                self._transport.save_weights(conv.save_weights_request_from_proto(request))
            ),
        )

    def RefreshWeights(self, request, context):
        return self._handle(
            context,
            lambda: conv.refresh_weights_response_to_proto(
                self._transport.refresh_weights(
                    conv.refresh_weights_request_from_proto(request)
                )
            ),
        )

    def Sample(self, request, context):
        return self._handle(
            context,
            lambda: conv.sample_response_to_proto(
                self._transport.sample(conv.sample_request_from_proto(request))
            ),
        )

    def GetTelemetrySummary(self, request, context):
        return self._handle(
            context,
            lambda: conv.get_telemetry_summary_response_to_proto(
                self._transport.get_telemetry_summary(
                    conv.get_telemetry_summary_request_from_proto(request)
                )
            ),
        )

    def DownloadArtifactFile(self, request, context):
        return self._handle(
            context,
            lambda: conv.download_artifact_file_response_to_proto(
                self._download_artifact_file(
                    conv.download_artifact_file_request_from_proto(request)
                )
            ),
        )

    def _handle(self, context, fn):
        self._check_auth(context)
        try:
            return fn()
        except InvalidRequestError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        except NotFoundError as exc:
            context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        except BackendUnavailableError as exc:
            context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
        except TimeoutError as exc:
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(exc))
        except GankerError as exc:
            context.abort(grpc.StatusCode.INTERNAL, str(exc))
        except Exception as exc:
            code, message = _map_wrapped_exception(exc)
            context.abort(code, message)

    def _check_auth(self, context) -> None:
        if not self._bearer_token:
            return
        metadata = dict(context.invocation_metadata())
        expected = f"Bearer {self._bearer_token}"
        if metadata.get("authorization") != expected:
            context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid bearer token")

    def _download_artifact_file(self, request) -> DownloadArtifactFileResponse:
        if request.file_kind == ArtifactFileKind.MANIFEST:
            path = request.artifact.manifest_path
        elif request.file_kind == ArtifactFileKind.PAYLOAD:
            path = request.artifact.payload_path
        else:
            raise InvalidRequestError(f"unsupported artifact file kind: {request.file_kind}")

        resolved_root = self._artifact_root.resolve()
        artifact_path = Path(path)
        try:
            resolved_path = artifact_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise NotFoundError(f"artifact file is missing: {artifact_path}") from exc

        if not resolved_path.is_relative_to(resolved_root):
            raise InvalidRequestError("artifact path is outside the configured artifact root")

        return DownloadArtifactFileResponse(
            request_id=request.context.request_id or uuid.uuid4().hex,
            artifact=request.artifact,
            file_kind=request.file_kind,
            path=str(resolved_path),
            contents=resolved_path.read_bytes(),
        )


def start_grpc_proxy_server(
    *,
    bind: str = "127.0.0.1:0",
    artifact_root: Path,
    monarch_transport: str = "tcp",
    training_backend: str = "fake",
    inference_backend: str = "fake",
    training_backend_config: dict | MegatronBackendConfig | None = None,
    inference_backend_config: dict | None = None,
    timeout: float = 20,
    bearer_token: str | None = None,
    max_workers: int = 8,
) -> ProxyGrpcServer:
    """Start a local Monarch mesh behind a gRPC listener."""

    mesh = start_local_monarch_mesh(
        artifact_root,
        transport=monarch_transport,
        training_backend=training_backend,
        inference_backend=inference_backend,
        training_backend_config=training_backend_config,
        inference_backend_config=inference_backend_config,
    )
    grpc_server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=_DEFAULT_GRPC_OPTIONS,
    )
    servicer = GankerProxyServicer(
        MonarchProxyTransport(mesh.proxy, timeout=timeout),
        artifact_root=artifact_root,
        bearer_token=bearer_token,
    )
    proxy_pb2_grpc.add_GankerProxyServicer_to_server(servicer, grpc_server)
    port = grpc_server.add_insecure_port(bind)
    if port == 0:
        mesh.stop()
        raise BackendUnavailableError(f"failed to bind gRPC server to {bind}")
    grpc_server.start()
    host = bind.rsplit(":", 1)[0] or "127.0.0.1"
    return ProxyGrpcServer(
        grpc_server=grpc_server,
        mesh=mesh,
        bound_address=f"{host}:{port}",
    )


def _map_wrapped_exception(exc: Exception) -> tuple[grpc.StatusCode, str]:
    text = f"{exc!r}\n{exc}"
    match = re.search(
        r"(InvalidRequestError|NotFoundError|BackendUnavailableError)\('([^']*)'\)",
        text,
    )
    if match is None:
        return grpc.StatusCode.INTERNAL, "unexpected server error"

    error_name, message = match.groups()
    if error_name == "InvalidRequestError":
        return grpc.StatusCode.INVALID_ARGUMENT, message
    if error_name == "NotFoundError":
        return grpc.StatusCode.NOT_FOUND, message
    if error_name == "BackendUnavailableError":
        return grpc.StatusCode.UNAVAILABLE, message
    return grpc.StatusCode.INTERNAL, message
