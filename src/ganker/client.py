"""Public client API for talking to the proxy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from ganker.contracts import (
    AdamParams,
    ArtifactFileKind,
    ArtifactKind,
    CreateTrainingRunRequest,
    Datum,
    DownloadArtifactFileRequest,
    DownloadArtifactFileResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    GetTelemetrySummaryRequest,
    GetTelemetrySummaryResponse,
    ModelInput,
    OptimStepRequest,
    OptimStepResponse,
    RefreshWeightsRequest,
    RefreshWeightsResponse,
    RequestContext,
    SampleRequest,
    SampleResponse,
    SamplingParams,
    SaveWeightsRequest,
    SaveWeightsResponse,
    TrainingRun,
    TuningMode,
    WeightArtifact,
)
from ganker.config import MegatronBackendConfig
from ganker.orchestration import LocalMonarchMesh, start_local_monarch_mesh
from ganker.transport import MonarchProxyTransport, ProxyTransport


def _tuning_mode(value: TuningMode | Literal["lora", "full"]) -> TuningMode:
    if isinstance(value, TuningMode):
        return value
    return {
        "lora": TuningMode.LORA,
        "full": TuningMode.FULL,
    }[value]


def _model_input(value: ModelInput | Sequence[int]) -> ModelInput:
    if isinstance(value, ModelInput):
        return value
    return ModelInput.from_ints(value)


def _datum_list(value: Datum | Sequence[Datum]) -> list[Datum]:
    if isinstance(value, Datum):
        return [value]
    return list(value)


@dataclass
class ServiceClient:
    """Client-facing API for a Ganker proxy.

    The client owns a `ProxyTransport`; local development uses a Monarch-backed
    transport, but callers never need to speak Monarch directly.
    """

    _transport: ProxyTransport
    _owned_mesh: LocalMonarchMesh | None = None

    @classmethod
    def local(
        cls,
        artifact_root: Path,
        *,
        monarch_transport: str = "tcp",
        training_backend: str = "fake",
        inference_backend: str = "fake",
        training_backend_config: dict | MegatronBackendConfig | None = None,
        inference_backend_config: dict | None = None,
        timeout: float = 20,
    ) -> "ServiceClient":
        mesh = start_local_monarch_mesh(
            artifact_root,
            transport=monarch_transport,
            training_backend=training_backend,
            inference_backend=inference_backend,
            training_backend_config=training_backend_config,
            inference_backend_config=inference_backend_config,
        )
        return cls(
            _transport=MonarchProxyTransport(mesh.proxy, timeout=timeout),
            _owned_mesh=mesh,
        )

    @classmethod
    def connect_grpc(
        cls,
        target: str,
        *,
        timeout: float = 20,
        bearer_token: str | None = None,
    ) -> "ServiceClient":
        from ganker.rpc.client import GrpcProxyTransport

        return cls(
            _transport=GrpcProxyTransport.connect(
                target,
                timeout=timeout,
                bearer_token=bearer_token,
            )
        )

    def close(self) -> None:
        transport_close = getattr(self._transport, "close", None)
        if transport_close is not None:
            transport_close()
        if self._owned_mesh is not None:
            self._owned_mesh.stop()
            self._owned_mesh = None

    def __enter__(self) -> "ServiceClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def create_lora_training_client(
        self,
        *,
        base_model: str,
        rank: int = 32,
        request_id: str = "",
    ) -> "TrainingClient":
        return self.create_training_client(
            base_model=base_model,
            tuning=TuningMode.LORA,
            rank=rank,
            request_id=request_id,
        )

    def create_training_client(
        self,
        *,
        base_model: str,
        tuning: TuningMode | Literal["lora", "full"] = TuningMode.LORA,
        rank: int = 32,
        request_id: str = "",
    ) -> "TrainingClient":
        response = self._transport.create_training_run(
            CreateTrainingRunRequest(
                context=RequestContext(request_id),
                base_model=base_model,
                tuning_mode=_tuning_mode(tuning),
                lora_rank=rank,
            )
        )
        return TrainingClient(service=self, run=response.run)

    def create_sampling_client(
        self,
        *,
        base_model: str,
        request_id: str = "",
    ) -> "SamplingClient":
        """Create a local base-model sampler.

        The local fake backend models this by creating a run and snapshotting
        its initial weights. Production can map this method to an sglang base
        model deployment without changing callers.
        """

        training = self.create_training_client(
            base_model=base_model,
            tuning=TuningMode.FULL,
            rank=0,
            request_id=request_id,
        )
        return training.save_weights_and_get_sampling_client(
            kind=ArtifactKind.FULL,
            request_id=request_id,
        )

    def get_telemetry_summary(
        self,
        run_id: str,
        *,
        request_id: str = "",
    ) -> GetTelemetrySummaryResponse:
        return self._transport.get_telemetry_summary(
            GetTelemetrySummaryRequest(
                context=RequestContext(request_id),
                run_id=run_id,
            )
        )

    def download_artifact_file(
        self,
        artifact: WeightArtifact,
        *,
        file_kind: ArtifactFileKind | Literal["manifest", "payload"] = ArtifactFileKind.PAYLOAD,
        request_id: str = "",
    ) -> DownloadArtifactFileResponse:
        if isinstance(file_kind, str):
            file_kind = {
                "manifest": ArtifactFileKind.MANIFEST,
                "payload": ArtifactFileKind.PAYLOAD,
            }[file_kind]
        return self._transport.download_artifact_file(
            DownloadArtifactFileRequest(
                context=RequestContext(request_id),
                artifact=artifact,
                file_kind=file_kind,
            )
        )


@dataclass(frozen=True)
class TrainingClient:
    """Run-scoped client for training operations."""

    service: ServiceClient
    run: TrainingRun

    @property
    def run_id(self) -> str:
        return self.run.run_id

    def forward_backward(
        self,
        data: Datum | Sequence[Datum],
        *,
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, float] | None = None,
        request_id: str = "",
    ) -> ForwardBackwardResponse:
        return self.service._transport.forward_backward(
            ForwardBackwardRequest(
                context=RequestContext(request_id),
                run_id=self.run_id,
                data=_datum_list(data),
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config or {},
            )
        )

    def optim_step(
        self,
        *,
        params: AdamParams | None = None,
        learning_rate: float | None = None,
        request_id: str = "",
    ) -> OptimStepResponse:
        if params is None:
            if learning_rate is None:
                raise ValueError("params or learning_rate is required")
            params = AdamParams(learning_rate=learning_rate)
        return self.service._transport.optim_step(
            OptimStepRequest(
                context=RequestContext(request_id),
                run_id=self.run_id,
                optimizer=params,
            )
        )

    def save_weights(
        self,
        *,
        kind: ArtifactKind = ArtifactKind.DELTA,
        request_id: str = "",
    ) -> SaveWeightsResponse:
        return self.service._transport.save_weights(
            SaveWeightsRequest(
                context=RequestContext(request_id),
                run_id=self.run_id,
                kind=kind,
            )
        )

    def refresh_weights(
        self,
        artifact: WeightArtifact | None = None,
        *,
        request_id: str = "",
    ) -> RefreshWeightsResponse:
        return self.service._transport.refresh_weights(
            RefreshWeightsRequest(
                context=RequestContext(request_id),
                run_id=self.run_id,
                artifact=artifact,
            )
        )

    def save_weights_for_sampler(
        self,
        *,
        kind: ArtifactKind = ArtifactKind.DELTA,
        request_id: str = "",
    ) -> SaveWeightsResponse:
        return self.save_weights(kind=kind, request_id=request_id)

    def save_weights_and_get_sampling_client(
        self,
        *,
        kind: ArtifactKind = ArtifactKind.DELTA,
        request_id: str = "",
    ) -> "SamplingClient":
        saved = self.save_weights_for_sampler(kind=kind, request_id=request_id)
        refreshed = self.refresh_weights(saved.artifact, request_id=request_id)
        return SamplingClient(
            service=self.service,
            run=self.run,
            artifact=refreshed.artifact,
        )

    def get_telemetry_summary(
        self,
        *,
        request_id: str = "",
    ) -> GetTelemetrySummaryResponse:
        return self.service.get_telemetry_summary(self.run_id, request_id=request_id)


@dataclass(frozen=True)
class SamplingClient:
    """Run-scoped client for sampling from a saved weight artifact."""

    service: ServiceClient
    run: TrainingRun
    artifact: WeightArtifact

    @property
    def run_id(self) -> str:
        return self.run.run_id

    @property
    def model_path(self) -> str:
        return self.artifact.manifest_path

    def sample(
        self,
        prompt: ModelInput | Sequence[int],
        sampling_params: SamplingParams | None = None,
        *,
        num_samples: int = 1,
        request_id: str = "",
    ) -> SampleResponse:
        return self.service._transport.sample(
            SampleRequest(
                context=RequestContext(request_id),
                run_id=self.run_id,
                prompt=_model_input(prompt),
                sampling_params=sampling_params or SamplingParams(),
                num_samples=num_samples,
            )
        )

    def get_telemetry_summary(
        self,
        *,
        request_id: str = "",
    ) -> GetTelemetrySummaryResponse:
        return self.service.get_telemetry_summary(self.run_id, request_id=request_id)
