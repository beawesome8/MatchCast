"""Model training: features -> XGBoost -> versioned artifact.

This file trains and evaluates a challenger model. It does NOT decide
whether to promote it over the current champion — that's Phase 3's
job (the promotion gate). Keeping these separate means each piece is
independently testable: this file's tests never need a "champion" to
exist, and the gate's tests never need to retrain a real model.

Split strategy: chronological, not random. We train on the first N%
of matches by kickoff time and evaluate on the rest. A random split
would leak future information into training (the model could learn
from a Round of 16 result while predicting an earlier group match),
which silently inflates every metric we'd report.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

FEATURE_COLUMNS = ["elo_diff", "home_form_ppg", "away_form_ppg", "goal_diff_diff", "is_knockout"]
OUTCOME_LABELS = ["AWAY_TEAM", "DRAW", "HOME_TEAM"]  # sorted order used for label encoding

# 3-class uniform baseline: (2/3)^2 + 2*(1/3)^2 = 0.667. Any model that
# can't beat this is worse than knowing nothing about the two teams.
RANDOM_GUESS_BRIER = 2.0 / 3.0

ARTIFACT_DIR = Path("artifacts")


@dataclass
class TrainingResult:
    model_path: str
    data_hash: str
    n_train: int
    n_holdout: int
    train_brier: float
    holdout_brier: float
    holdout_log_loss: float
    beats_random_baseline: bool | None
    trained_at: str


def _hash_rows(rows: list[dict]) -> str:
    """Content hash of the training data, so any model artifact can be
    traced back to exactly what it was trained on — a core registry
    requirement, not a nice-to-have."""
    canonical = json.dumps(rows, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _to_frame(rows: list[dict]) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.DataFrame(rows)
    label_index = {label: i for i, label in enumerate(OUTCOME_LABELS)}
    y = df["outcome"].map(label_index).to_numpy()
    x = df[FEATURE_COLUMNS].to_numpy()
    return x, y


def chronological_split(
    rows: list[dict], train_fraction: float = 0.8
) -> tuple[list[dict], list[dict]]:
    """rows must already be in chronological order (build_feature_table
    guarantees this, since it walks matches by kickoff_utc)."""
    split_at = max(1, int(len(rows) * train_fraction))
    return rows[:split_at], rows[split_at:]


def train_model(
    rows: list[dict], train_fraction: float = 0.8
) -> tuple[xgb.XGBClassifier, TrainingResult]:
    if len(rows) < 10:
        raise ValueError(f"need at least 10 rows to train, got {len(rows)}")

    train_rows, holdout_rows = chronological_split(rows, train_fraction)
    x_train, y_train = _to_frame(train_rows)

    # XGBoost's multiclass classifier requires every class to appear in
    # the training fold. With a chronological split on a small dataset,
    # it's entirely possible the training window has zero examples of
    # one outcome (e.g. no draws yet, early in a tournament). Without
    # this check, that produces a cryptic internal XGBoost error instead
    # of an actionable one.
    missing = set(range(len(OUTCOME_LABELS))) - set(y_train.tolist())
    if missing:
        missing_names = [OUTCOME_LABELS[i] for i in sorted(missing)]
        raise ValueError(
            f"training fold is missing example(s) of: {missing_names}. "
            "The chronological split produced a training window with no "
            "examples of this outcome — likely too little data yet, or "
            "an unlucky split point. Try a larger dataset or a different "
            "train_fraction."
        )

    model = xgb.XGBClassifier(
        n_estimators=25,
        max_depth=2,
        learning_rate=0.05,
        reg_lambda=2.0,
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
    )
    model.fit(x_train, y_train)

    train_probs = model.predict_proba(x_train)
    train_brier = _multiclass_brier(y_train, train_probs)

    if holdout_rows:
        x_holdout, y_holdout = _to_frame(holdout_rows)
        holdout_probs = model.predict_proba(x_holdout)
        holdout_brier = _multiclass_brier(y_holdout, holdout_probs)
        holdout_ll = log_loss(y_holdout, holdout_probs, labels=[0, 1, 2])
        beats_baseline = holdout_brier < RANDOM_GUESS_BRIER
    else:
        holdout_brier = float("nan")
        holdout_ll = float("nan")
        beats_baseline = None

    ARTIFACT_DIR.mkdir(exist_ok=True)
    data_hash = _hash_rows(rows)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_path = ARTIFACT_DIR / f"model_{timestamp}_{data_hash}.json"
    model.save_model(model_path)

    result = TrainingResult(
        model_path=str(model_path),
        data_hash=data_hash,
        n_train=len(train_rows),
        n_holdout=len(holdout_rows),
        train_brier=train_brier,
        holdout_brier=holdout_brier,
        holdout_log_loss=holdout_ll,
        beats_random_baseline=beats_baseline,
        trained_at=timestamp,
    )
    return model, result


def _multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multiclass Brier score: mean squared distance between predicted
    probability vectors and one-hot true outcomes. Lower is better,
    0 is a perfect prediction. This is THE metric the promotion gate
    (Phase 3) will use to compare challenger vs champion."""
    one_hot = np.eye(len(OUTCOME_LABELS))[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


if __name__ == "__main__":
    from matchcast.db import get_session_factory
    from matchcast.features import build_feature_table

    with get_session_factory()() as session:
        rows = build_feature_table(session)
    _, result = train_model(rows)
    print(json.dumps(asdict(result), indent=2))