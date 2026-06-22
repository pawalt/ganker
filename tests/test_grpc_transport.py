from pathlib import Path

from ganker.client import ServiceClient
from ganker.contracts import (
    AdamParams,
    ArtifactFileKind,
    Datum,
    ModelInput,
    SamplingParams,
    TensorData,
)
from ganker.errors import InvalidRequestError
from ganker.rpc.client import GrpcTransportError
from ganker.rpc.server import start_grpc_proxy_server


def test_full_singleton_flow_through_grpc_client(tmp_path: Path):
    with start_grpc_proxy_server(artifact_root=tmp_path) as server:
        client = ServiceClient.connect_grpc(server.bound_address)
        try:
            training = client.create_lora_training_client(
                base_model="Qwen/Qwen3-8B",
                rank=32,
                request_id="req-create",
            )

            fb = training.forward_backward(
                Datum(
                    model_input=ModelInput.from_ints([10, 11, 12]),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_ints([11, 12, 13]),
                        "weights": TensorData.from_floats([1.0, 1.0, 1.0]),
                    },
                ),
                loss_fn="cross_entropy",
                request_id="req-fb",
            )
            step = training.optim_step(
                params=AdamParams(learning_rate=1e-4),
                request_id="req-step",
            )
            sampling = training.save_weights_and_get_sampling_client(request_id="req-sampler")
            sample = sampling.sample(
                ModelInput.from_ints([100, 101]),
                SamplingParams(max_tokens=2),
                request_id="req-sample",
            )
            summary = sampling.get_telemetry_summary(request_id="req-summary")
            payload = client.download_artifact_file(
                sampling.artifact,
                file_kind=ArtifactFileKind.PAYLOAD,
                request_id="req-download",
            )
        finally:
            client.close()

        assert training.run.run_id == "run-000001"
        assert fb.request_id == "req-fb"
        assert fb.usage.input_tokens == 3
        assert step.optimizer_step == 1
        assert step.checkpoint_version == 1
        assert sampling.artifact.checkpoint_version == 1
        assert sample.sequences[0].tokens == [103, 104]
        assert payload.request_id == "req-download"
        assert payload.file_kind == ArtifactFileKind.PAYLOAD
        assert b'"backend": "fake"' in payload.contents

        assert summary.summary.event_count == 3
        assert summary.summary.total.input_tokens == 5
        assert summary.summary.total.output_tokens == 2
        assert summary.summary.total.training_steps == 1
        assert summary.summary.total.samples == 1


def test_grpc_server_maps_invalid_requests_to_domain_errors(tmp_path: Path):
    with start_grpc_proxy_server(artifact_root=tmp_path) as server:
        client = ServiceClient.connect_grpc(server.bound_address)
        try:
            try:
                client.create_lora_training_client(
                    base_model="",
                    rank=32,
                    request_id="req-create",
                )
            except InvalidRequestError as exc:
                assert "base_model is required" in str(exc)
            else:
                raise AssertionError("expected InvalidRequestError")
        finally:
            client.close()


def test_grpc_server_can_require_bearer_token(tmp_path: Path):
    with start_grpc_proxy_server(artifact_root=tmp_path, bearer_token="secret") as server:
        unauthenticated = ServiceClient.connect_grpc(server.bound_address)
        try:
            try:
                unauthenticated.create_lora_training_client(
                    base_model="Qwen/Qwen3-8B",
                    rank=32,
                )
            except GrpcTransportError as exc:
                assert exc.code.name == "UNAUTHENTICATED"
            else:
                raise AssertionError("expected GrpcTransportError")
        finally:
            unauthenticated.close()

        authenticated = ServiceClient.connect_grpc(server.bound_address, bearer_token="secret")
        try:
            training = authenticated.create_lora_training_client(
                base_model="Qwen/Qwen3-8B",
                rank=32,
            )
            assert training.run_id == "run-000001"
        finally:
            authenticated.close()
