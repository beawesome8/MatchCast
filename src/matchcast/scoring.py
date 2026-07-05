"""Prediction scoring: once a match finishes, look up its real result
and fill in the actual_outcome and brier_score for every logged
prediction of that match.

This is what turns predictions_log from a write-only ledger into
verifiable evidence — the "here's what we predicted, here's what
actually happened, here's the model's real accuracy" story this whole
project is built around.

Reuses train.py's _multiclass_brier (the same formula the promotion
gate uses to compare champion vs challenger) rather than re-deriving
the Brier calculation a second time — one true formula for "how good
was this probability," same principle as _pre_match_features.
"""

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from matchcast.models import Match, PredictionLog
from matchcast.train import OUTCOME_LABELS, _multiclass_brier


def score_pending_predictions(session: Session) -> dict:
    """Find every prediction log row for a now-FINISHED match that
    hasn't been scored yet, compute its actual outcome and Brier
    score, and commit the update. Safe to run repeatedly — already
    scored rows are excluded from the query, never touched twice."""
    pending = (
        session.execute(select(PredictionLog).where(PredictionLog.actual_outcome.is_(None)))
        .scalars()
        .all()
    )

    scored = 0
    still_pending = 0

    for log in pending:
        match = session.get(Match, log.match_id)
        if match is None or match.status != "FINISHED" or match.winner is None:
            still_pending += 1
            continue

        prob_by_label = {
            "HOME_TEAM": log.prob_home,
            "DRAW": log.prob_draw,
            "AWAY_TEAM": log.prob_away,
        }
        probs = np.array([[prob_by_label[label] for label in OUTCOME_LABELS]])
        y_true = np.array([OUTCOME_LABELS.index(match.winner)])

        log.actual_outcome = match.winner
        log.brier_score = _multiclass_brier(y_true, probs)
        scored += 1

    session.commit()
    return {"scored": scored, "still_pending": still_pending}


if __name__ == "__main__":
    from matchcast.db import get_session_factory

    with get_session_factory()() as session:
        print(score_pending_predictions(session))