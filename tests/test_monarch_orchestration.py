from pathlib import Path

from ganker.client import ServiceClient
from ganker.contracts import (
    AdamParams,
    Datum,
    ModelInput,
    SamplingParams,
    TensorData,
)


def test_full_singleton_flow_through_public_client(tmp_path: Path):
    with ServiceClient.local(tmp_path) as client:
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

        assert training.run.run_id == "run-000001"
        assert fb.request_id == "req-fb"
        assert fb.usage.input_tokens == 3
        assert step.optimizer_step == 1
        assert step.checkpoint_version == 1
        assert sampling.artifact.checkpoint_version == 1
        assert sample.sequences[0].tokens == [103, 104]

        assert summary.summary.event_count == 3
        assert summary.summary.total.input_tokens == 5
        assert summary.summary.total.output_tokens == 2
        assert summary.summary.total.training_steps == 1
        assert summary.summary.total.samples == 1
        assert [(item.event_source, item.event_count) for item in summary.summary.by_source] == [
            ("rollout", 1),
            ("trainer", 2),
        ]
