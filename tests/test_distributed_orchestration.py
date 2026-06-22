from pathlib import Path

from ganker.client import ServiceClient
from ganker.contracts import AdamParams, Datum, ModelInput, SamplingParams, TensorData


def test_local_distributed_flow_uses_controller_path(tmp_path: Path):
    with ServiceClient.local_distributed(tmp_path, timeout=30) as client:
        training = client.create_lora_training_client(
            base_model="Qwen/Qwen3-8B",
            rank=8,
            request_id="req-create",
        )
        fb = training.forward_backward(
            Datum(
                model_input=ModelInput.from_ints([1, 2, 3]),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_ints([2, 3, 4]),
                    "weights": TensorData.from_floats([1.0, 1.0, 1.0]),
                },
            ),
            request_id="req-fb",
        )
        step = training.optim_step(
            params=AdamParams(learning_rate=1e-4),
            request_id="req-step",
        )
        sampling = training.save_weights_and_get_sampling_client(request_id="req-save")
        sample = sampling.sample(
            ModelInput.from_ints([10, 11]),
            SamplingParams(max_tokens=2),
            request_id="req-sample",
        )
        summary = sampling.get_telemetry_summary(request_id="req-summary")

        assert training.run_id == "run-000001"
        assert fb.loss > 0
        assert step.optimizer_step == 1
        assert sample.sequences[0].tokens == [13, 14]
        assert summary.summary.event_count == 3
        assert summary.summary.total.training_steps == 1
        assert [(item.event_source, item.event_count) for item in summary.summary.by_source] == [
            ("rollout", 1),
            ("trainer", 2),
        ]
