"""Prediction serving: load the current champion, score upcoming
matches, and log every prediction before returning it.

No FastAPI code lives here — api.py is a thin HTTP wrapper around
get_upcoming_predictions() below, so this logic is testable with a
plain database session, no HTTP client required.

Loading uses xgboost's core Booster API, not the sklearn XGBClassifier
wrapper. The sklearn wrapper's load_model() calls scikit-learn's
is_classifier(), which has a known incompatibility with the
tag-inspection system introduced in scikit-learn 1.6 — it crashes on
load (not on train, which is why the training pipeline never hit
this). The Booster API does pure inference with no sklearn dependency
at all, sidestepping the issue entirely.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb
from sqlalchemy import select
from sqlalchemy.orm import Session

from matchcast.features import build_upcoming_features
from matchcast.models import PredictionLog, Team
from matchcast.registry import get_current_champion
from matchcast.train import FEATURE_COLUMNS, OUTCOME_LABELS


@dataclass
class MatchPrediction:
    match_id: int
    source_match_id: int
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    stage: str | None
    kickoff_utc: str
    prob_home: float
    prob_draw: float
    prob_away: float
    model_version_id: int


def load_model_from_bytes(model_bytes: bytes) -> xgb.Booster:
    """Load via the core Booster API (not XGBClassifier) to avoid a
    known XGBoost/scikit-learn incompatibility on the sklearn load path."""
    booster = xgb.Booster()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(model_bytes)
        temp_path = f.name
    try:
        booster.load_model(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)
    return booster


def get_upcoming_predictions(session: Session) -> list[MatchPrediction]:
    champion = get_current_champion(session)
    if champion is None:
        raise ValueError("no champion model exists yet — run the training pipeline first")
    if champion.model_bytes is None:
        raise ValueError(
            f"champion model_version_id={champion.id} has no stored bytes "
            "(trained before model_bytes was added to the registry)"
        )

    booster = load_model_from_bytes(champion.model_bytes)

    upcoming = build_upcoming_features(session)
    if not upcoming:
        return []

    team_ids = {r["home_team_id"] for r in upcoming} | {r["away_team_id"] for r in upcoming}
    teams = {
        t.id: t.name for t in session.execute(select(Team).where(Team.id.in_(team_ids))).scalars()
    }

    x = np.array([[row[col] for col in FEATURE_COLUMNS] for row in upcoming])
    dmatrix = xgb.DMatrix(x)
    # Trained with objective="multi:softprob", so predict() returns
    # shape (n_samples, n_classes) probabilities directly.
    probs = booster.predict(dmatrix)

    predictions = []
    for row, prob_row in zip(upcoming, probs, strict=True):
        label_probs = dict(zip(OUTCOME_LABELS, prob_row, strict=True))
        prediction = MatchPrediction(
            match_id=row["match_id"],
            source_match_id=row["source_match_id"],
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            home_team_name=teams.get(row["home_team_id"], "Unknown"),
            away_team_name=teams.get(row["away_team_id"], "Unknown"),
            stage=row["stage"],
            kickoff_utc=row["kickoff_utc"].isoformat(),
            prob_home=float(label_probs["HOME_TEAM"]),
            prob_draw=float(label_probs["DRAW"]),
            prob_away=float(label_probs["AWAY_TEAM"]),
            model_version_id=champion.id,
        )
        predictions.append(prediction)
        _log_prediction(session, prediction)

    session.commit()
    return predictions


def _log_prediction(session: Session, prediction: MatchPrediction) -> None:
    """Idempotent: re-serving the same match under the same champion
    does not create a duplicate log row. A NEW champion predicting the
    same match DOES get its own row — that's a genuinely new, distinct
    prediction worth recording."""
    existing = session.execute(
        select(PredictionLog).where(
            PredictionLog.match_id == prediction.match_id,
            PredictionLog.model_version_id == prediction.model_version_id,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return

    session.add(
        PredictionLog(
            match_id=prediction.match_id,
            model_version_id=prediction.model_version_id,
            prob_home=prediction.prob_home,
            prob_draw=prediction.prob_draw,
            prob_away=prediction.prob_away,
        )
    )