from pathlib import Path

from examples.sft import SFTDataConfig, ToyTokenizer, load_jsonl_sft_batches, run_sft
from ganker.client import SamplingClient, ServiceClient, TrainingClient
from ganker.contracts import (
    ArtifactKind,
    ModelInput,
    SamplingParams,
    TrainingRun,
    TuningMode,
    WeightArtifact,
)


def test_local_distributed_sft_runs_full_training_and_sampling_flow(tmp_path: Path):
    batches = load_jsonl_sft_batches(
        Path("examples/tiny_sft.jsonl"),
        tokenizer=ToyTokenizer(vocab_size=64),
        config=SFTDataConfig(
            sequence_length=64,
            batch_size=1,
            shuffle=False,
        ),
    )

    with ServiceClient.local_distributed(tmp_path, timeout=30) as client:
        summary = run_sft(
            client,
            base_model="local/tiny-sft",
            dataset=batches,
            tuning="lora",
            lora_rank=4,
            learning_rate=1e-4,
            max_steps=3,
            save_every=2,
        )

        artifact = WeightArtifact(
            run_id=summary.run_id,
            checkpoint_version=summary.checkpoint_version,
            kind=ArtifactKind.DELTA,
            manifest_path=summary.manifest_path,
            payload_path=summary.artifact_path,
        )
        run = TrainingRun(
            run_id=summary.run_id,
            base_model=summary.base_model,
            tuning_mode=TuningMode.LORA,
            lora_rank=4,
            checkpoint_version=summary.checkpoint_version,
        )
        training = TrainingClient(service=client, run=run)
        refreshed = training.refresh_weights(
            artifact,
            request_id="distributed-sft-refresh",
        )
        sampler = SamplingClient(service=client, run=run, artifact=refreshed.artifact)
        sample = sampler.sample(
            ModelInput.from_ints([7, 8]),
            SamplingParams(max_tokens=3),
            request_id="distributed-sft-sample",
        )
        telemetry = sampler.get_telemetry_summary(request_id="distributed-sft-telemetry")

    assert summary.ok is True
    assert summary.steps == 3
    assert summary.optimizer_step == 3
    assert summary.checkpoint_version == 3
    assert Path(summary.artifact_path).exists()
    assert sample.artifact.checkpoint_version == 3
    assert sample.sequences[0].tokens == [12, 13, 14]
    assert telemetry.summary.total.training_steps == 3
    assert telemetry.summary.total.samples == 1
    assert telemetry.summary.total.output_tokens == 3
