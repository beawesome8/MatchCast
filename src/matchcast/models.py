"""ORM models — core tables plus the model registry."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_team_id: Mapped[int] = mapped_column(unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))
    tla: Mapped[str | None] = mapped_column(String(3))


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_match_id: Mapped[int] = mapped_column(unique=True, index=True)
    tournament_id: Mapped[str] = mapped_column(String(20), index=True, default="WC2026")
    stage: Mapped[str | None] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(20), index=True)
    kickoff_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    home_goals: Mapped[int | None]
    away_goals: Mapped[int | None]
    winner: Mapped[str | None] = mapped_column(String(12))
    last_synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IngestionQuarantine(Base):
    __tablename__ = "ingestion_quarantine"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    source: Mapped[str] = mapped_column(String(50))
    reason: Mapped[str] = mapped_column(Text)
    payload: Mapped[str] = mapped_column(Text)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model_path: Mapped[str] = mapped_column(String(255))
    model_bytes: Mapped[bytes] = mapped_column(LargeBinary)
    data_hash: Mapped[str] = mapped_column(String(32))
    n_train: Mapped[int]
    n_holdout: Mapped[int]
    train_brier: Mapped[float]
    holdout_brier: Mapped[float]
    holdout_log_loss: Mapped[float]
    beats_random_baseline: Mapped[bool | None]
    status: Mapped[str] = mapped_column(String(20), index=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text)