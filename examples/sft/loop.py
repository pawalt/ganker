from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

from ganker.contracts import ArtifactKind, Datum, TuningMode


@dataclass(frozen=True)
class SFTRunSummary:
    ok: bool
    run_id: str
    base_model: str
    tuning: str
    steps: int
    losses: list[float]
    final_loss: float | None
    optimizer_step: int
    checkpoint_version: int
    artifact_path: str
    manifest_path: str
    input_tokens: int
    training_steps: int

    def to_dict(self) -> dict:
        return asdict(self)


def run_sft(
    client,
    *,
    base_model: str,
    dataset: Iterable[list[Datum]],
    tuning: TuningMode | Literal["lora", "full"] = "full",
    lora_rank: int = 0,
    learning_rate: float = 1e-4,
    max_steps: int = 10,
    save_every: int = 0,
) -> SFTRunSummary:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if save_every < 0:
        raise ValueError("save_every cannot be negative")

    training = client.create_training_client(
        base_model=base_model,
        tuning=tuning,
        rank=lora_rank,
    )
    losses: list[float] = []
    input_tokens = 0
    training_steps = 0
    optimizer_step = 0
    checkpoint_version = training.run.checkpoint_version
    artifact_kind = ArtifactKind.DELTA if _tuning_name(tuning) == "lora" else ArtifactKind.FULL
    saved = None

    for batch in dataset:
        if len(losses) >= max_steps:
            break
        fb = training.forward_backward(batch, loss_fn="cross_entropy")
        step = training.optim_step(learning_rate=learning_rate)
        losses.append(float(fb.loss))
        input_tokens += int(fb.usage.input_tokens)
        training_steps += int(step.usage.training_steps)
        optimizer_step = int(step.optimizer_step)
        checkpoint_version = int(step.checkpoint_version)
        if save_every and optimizer_step % save_every == 0:
            saved = training.save_weights(kind=artifact_kind)

    if not losses:
        raise ValueError("dataset produced no SFT steps")
    if saved is None:
        saved = training.save_weights(kind=artifact_kind)

    return SFTRunSummary(
        ok=True,
        run_id=training.run_id,
        base_model=base_model,
        tuning=_tuning_name(tuning),
        steps=len(losses),
        losses=losses,
        final_loss=losses[-1],
        optimizer_step=optimizer_step,
        checkpoint_version=checkpoint_version,
        artifact_path=saved.artifact.payload_path,
        manifest_path=saved.artifact.manifest_path,
        input_tokens=input_tokens,
        training_steps=training_steps,
    )


def _tuning_name(tuning: TuningMode | Literal["lora", "full"]) -> str:
    if isinstance(tuning, TuningMode):
        return tuning.name.lower()
    return tuning
