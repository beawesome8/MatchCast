"""Monitoring: aggregate real, scored predictions into a performance
summary — Brier score, hit rate, calibration, and basic pipeline
health signals — plus a match-by-match prediction history.

Every number here comes from predictions_log rows that were logged
BEFORE kickoff (serving.py) and scored AFTER the match finished
(scoring.py) — this is the verifiable track record the whole project
is built around, never a backtest.
"""

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from matchcast.models import IngestionQuarantine, Match, ModelVersion, PredictionLog, Team

CALIBRATION_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]

# The public "Prediction record" list starts from the Quarterfinals
# onward, at the project owner's request — Round of 16 predictions
# logged earlier in development are excluded from that specific list.
# This does NOT affect overall_brier_score / hit_rate / calibration
# below, which still include every scored prediction from the whole
# tournament — that distinction is called out in the UI, not hidden.
HISTORY_TRACK_RECORD_START = datetime(2026, 7, 9, tzinfo=UTC)


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
    first_model_holdout_brier: float | None
    current_champion_holdout_brier: float | None
    brier_improvement: float | None  # first - current; positive = improved


@dataclass
class PredictionHistoryEntry:
    match_id: int
    home_team_name: str
    away_team_name: str
    stage: str | None
    kickoff_utc: str
    prob_home: float
    prob_draw: float
    prob_away: float
    predicted_outcome: str
    actual_outcome: str
    correct: bool
    brier_score: float
    model_version_id: int


def _predicted_class_and_prob(log: PredictionLog) -> tuple[str, float]:
    probs = {"HOME_TEAM": log.prob_home, "DRAW": log.prob_draw, "AWAY_TEAM": log.prob_away}
    predicted_class = max(probs, key=probs.get)
    return predicted_class, probs[predicted_class]


def get_performance_summary(session: Session) -> PerformanceSummary:
    all_logs = session.execute(select(PredictionLog)).scalars().all()
    scored_logs = [log for log in all_logs if log.actual_outcome is not None]

    n_logged = len(all_logs)
    n_scored = len(scored_logs)

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
        calibration.append(CalibrationBin(low, min(high, 1.0), len(bucket), avg_pred, observed))

    by_version: dict[int, list[PredictionLog]] = {}
    for log, _, _ in scored:
        by_version.setdefault(log.model_version_id, []).append(log)

    model_versions = {mv.id: mv for mv in session.execute(select(ModelVersion)).scalars().all()}

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

    first_model = session.execute(
        select(ModelVersion)
        .order_by(ModelVersion.created_at.asc(), ModelVersion.id.asc())
        .limit(1)
    ).scalar_one_or_none()

    current_champion = session.execute(
        select(ModelVersion).where(ModelVersion.status == "champion")
    ).scalar_one_or_none()

    first_brier = first_model.holdout_brier if first_model else None
    current_brier = current_champion.holdout_brier if current_champion else None
    brier_improvement = (
        first_brier - current_brier
        if first_brier is not None and current_brier is not None
        else None
    )

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
        latest_model_trained_at=(latest_model.created_at.isoformat() if latest_model else None),
        first_model_holdout_brier=first_brier,
        current_champion_holdout_brier=current_brier,
        brier_improvement=brier_improvement,
    )


def get_prediction_history(session: Session, limit: int = 50) -> list[PredictionHistoryEntry]:
    """The most recent SCORED prediction per match, for matches that
    kicked off on or after HISTORY_TRACK_RECORD_START, most recent
    match first.

    A single match can accumulate several logged predictions if the
    champion retrained multiple times before kickoff (each retrain is
    a genuinely distinct data point, kept in predictions_log). This
    view intentionally shows only the LATEST one per match — what the
    final champion actually predicted — since that's what a reader
    means by "what did MatchCast call this game."
    """
    logs = (
        session.execute(
            select(PredictionLog)
            .where(PredictionLog.actual_outcome.is_not(None))
            .order_by(PredictionLog.created_at.desc())
        )
        .scalars()
        .all()
    )
    if not logs:
        return []

    # Keep only the first (= most recent, since we sorted desc) log
    # seen per match_id.
    latest_per_match: dict[int, PredictionLog] = {}
    for log in logs:
        if log.match_id not in latest_per_match:
            latest_per_match[log.match_id] = log
    logs = list(latest_per_match.values())[:limit]

    match_ids = {log.match_id for log in logs}
    matches = {
        m.id: m
        for m in session.execute(select(Match).where(Match.id.in_(match_ids))).scalars()
    }

    logs = [
        log
        for log in logs
        if (m := matches.get(log.match_id)) is not None
        and m.kickoff_utc >= HISTORY_TRACK_RECORD_START
    ]
    if not logs:
        return []

    team_ids = {tid for m in matches.values() for tid in (m.home_team_id, m.away_team_id)}
    teams = {
        t.id: t.name
        for t in session.execute(select(Team).where(Team.id.in_(team_ids))).scalars()
    }

    entries = []
    for log in logs:
        match = matches[log.match_id]
        predicted_class, _ = _predicted_class_and_prob(log)
        entries.append(
            PredictionHistoryEntry(
                match_id=log.match_id,
                home_team_name=teams.get(match.home_team_id, "Unknown"),
                away_team_name=teams.get(match.away_team_id, "Unknown"),
                stage=match.stage,
                kickoff_utc=match.kickoff_utc.isoformat(),
                prob_home=log.prob_home,
                prob_draw=log.prob_draw,
                prob_away=log.prob_away,
                predicted_outcome=predicted_class,
                actual_outcome=log.actual_outcome,
                correct=(predicted_class == log.actual_outcome),
                brier_score=log.brier_score,
                model_version_id=log.model_version_id,
            )
        )
    return entries