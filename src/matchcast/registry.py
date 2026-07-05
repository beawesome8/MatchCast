"""Model registry: the durable record of every trained model, and
which one (if any) is the current champion.

This file is deliberately dumb — it does not decide whether a model
SHOULD be promoted (that's Phase 3's job). It only records decisions
already made and answers "what's the champion right now?"

model_bytes is stored directly in the row (not just a file path)
because model_versions written on GitHub Actions runners live on
ephemeral disks that vanish when the job ends. Storing the trained
model's bytes in Postgres means the registry entry is self-contained
regardless of which machine trained it.
"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from matchcast.models import ModelVersion
from matchcast.train import TrainingResult


def register_model(
    session: Session,
    result: TrainingResult,
    model_bytes: bytes,
    status: str,
    rejection_reason: str | None = None,
) -> ModelVersion:
    if status not in {"champion", "rejected", "retired"}:
        raise ValueError(f"invalid status: {status!r}")
    if status == "rejected" and not rejection_reason:
        raise ValueError("rejected models must have a rejection_reason")

    entry = ModelVersion(
        model_path=result.model_path,
        model_bytes=model_bytes,
        data_hash=result.data_hash,
        n_train=result.n_train,
        n_holdout=result.n_holdout,
        train_brier=result.train_brier,
        holdout_brier=result.holdout_brier,
        holdout_log_loss=result.holdout_log_loss,
        beats_random_baseline=result.beats_random_baseline,
        status=status,
        rejection_reason=rejection_reason,
    )
    session.add(entry)
    session.flush()
    return entry


def promote_to_champion(session: Session, new_champion: ModelVersion) -> None:
    """Retire the current champion (if any) and promote new_champion."""
    current = get_current_champion(session)
    if current is not None and current.id != new_champion.id:
        current.status = "retired"
    new_champion.status = "champion"
    session.flush()


def get_current_champion(session: Session) -> ModelVersion | None:
    return session.execute(
        select(ModelVersion).where(ModelVersion.status == "champion")
    ).scalar_one_or_none()


def list_history(session: Session) -> list[ModelVersion]:
    return list(
        session.execute(select(ModelVersion).order_by(ModelVersion.created_at)).scalars().all()
    )