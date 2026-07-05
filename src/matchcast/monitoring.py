"""Monitoring: aggregate real, scored predictions into a performance
summary — Brier score, hit rate, calibration, and basic pipeline
health signals.

Every number here comes from predictions_log rows that were logged
BEFORE kickoff (serving.py) and scored AFTER the match finished
(scoring.py) — this is the verifiable track record the whole project
is built around, never a backtest.
"""

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from matchcast.models import IngestionQuarantine, ModelVersion, PredictionLog

CALIBRATION_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


@dataclass
class CalibrationBin:
    range_low: float
    range_high: float
    n_predictions: int
    avg_predicted_probability: float | None
    observed_frequency: float | None


@dataclass
class ModelVersionSummary:
    model_version_id: int
    status: str
    n_scored_predictions: int
    mean_brier_score: float


@dataclass
class PerformanceSummary:
    n_predictions_logged: int
    n_predictions_scored: int
    n_predictions_pending: int
    overall_brier_score: float | None
    hit_rate: float | None
    calibration: list[CalibrationBin]
    by_model_version: list[ModelVersionSummary]
    n_quarantined_batches: int
    latest_model_version_id: int | None
    latest_model_trained_at: str | None


def _predicted_class_and_prob(log: PredictionLog) -> tuple[str, float]:
    probs = {"HOME_TEAM": log.prob_home, "DRAW": log.prob_draw, "AWAY_TEAM": log.prob_away}
    predicted_class = max(probs, key=probs.get)
    return predicted_class, probs[predicted_class]


def get_performance_summary(session: Session) -> PerformanceSummary:
    all_logs = session.execute(select(PredictionLog)).scalars().all()
    scored_logs = [log for log in all_logs if log.actual_outcome is not None]

    n_logged = len(all_logs)
    n_scored = len(scored_logs)

    # (log, predicted_class, predicted_probability) — computed once,
    # reused by every summary below instead of recomputed repeatedly.
    scored = [(log, *_predicted_class_and_prob(log)) for log in scored_logs]

    overall_brier = None
    hit_rate = None
    if scored:
        overall_brier = sum(log.brier_score for log, _, _ in scored) / n_scored
        hits = sum(1 for log, cls, _ in scored if cls == log.actual_outcome)
        hit_rate = hits / n_scored

    calibration = []
    for low, high in CALIBRATION_BINS:
        bucket = [(log, cls, prob) for log, cls, prob in scored if low <= prob < high]
        if bucket:
            avg_pred = sum(prob for _, _, prob in bucket) / len(bucket)
            hits_in_bucket = sum(1 for log, cls, _ in bucket if cls == log.actual_outcome)
            observed = hits_in_bucket / len(bucket)
        else:
            avg_pred = None
            observed = None
        calibration.append(
            CalibrationBin(low, min(high, 1.0), len(bucket), avg_pred, observed)
        )

    by_version: dict[int, list[PredictionLog]] = {}
    for log, _, _ in scored:
        by_version.setdefault(log.model_version_id, []).append(log)

    model_versions = {
        mv.id: mv for mv in session.execute(select(ModelVersion)).scalars().all()
    }

    by_model_version = []
    for version_id, logs in sorted(by_version.items()):
        model_version = model_versions.get(version_id)
        status = model_version.status if model_version is not None else "unknown"
        mean_brier = sum(log.brier_score for log in logs) / len(logs)
        by_model_version.append(
            ModelVersionSummary(
                model_version_id=version_id,
                status=status,
                n_scored_predictions=len(logs),
                mean_brier_score=mean_brier,
            )
        )

    n_quarantined = session.execute(
        select(func.count()).select_from(IngestionQuarantine)
    ).scalar_one()

    # Order by created_at, then by id as a tiebreaker — two models
    # trained in rapid succession can land on the identical timestamp
    # at whatever resolution the database stores; id is monotonically
    # increasing and can't tie, so it's a safe secondary sort key.
    latest_model = session.execute(
        select(ModelVersion)
        .order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc())
        .limit(1)
    ).scalar_one_or_none()

    return PerformanceSummary(
        n_predictions_logged=n_logged,
        n_predictions_scored=n_scored,
        n_predictions_pending=n_logged - n_scored,
        overall_brier_score=overall_brier,
        hit_rate=hit_rate,
        calibration=calibration,
        by_model_version=by_model_version,
        n_quarantined_batches=n_quarantined,
        latest_model_version_id=latest_model.id if latest_model else None,
        latest_model_trained_at=(
            latest_model.created_at.isoformat() if latest_model else None
        ),
    )