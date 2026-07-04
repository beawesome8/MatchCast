"""Model training tests.

Uses tiny hand-built row lists (not the real database) so every
number is predictable and hand-checkable — Brier score and split
logic are exactly the kind of math that looks right when subtly
wrong, so exact-value assertions matter more here than almost
anywhere else in the codebase.
"""

import numpy as np

from matchcast.train import (
    RANDOM_GUESS_BRIER,
    _hash_rows,
    _multiclass_brier,
    chronological_split,
    train_model,
)


def _row(outcome: str, elo_diff: float = 0.0) -> dict:
    return {
        "match_id": 0,
        "source_match_id": 0,
        "elo_diff": elo_diff,
        "home_form_ppg": 1.0,
        "away_form_ppg": 1.0,
        "goal_diff_diff": 0.0,
        "is_knockout": 0,
        "outcome": outcome,
    }


def test_chronological_split_preserves_order_and_sizes():
    rows = [_row("HOME_TEAM") for _ in range(10)]
    train, holdout = chronological_split(rows, train_fraction=0.8)
    assert len(train) == 8
    assert len(holdout) == 2


def test_chronological_split_never_reorders_rows():
    rows = [_row("HOME_TEAM", elo_diff=i) for i in range(10)]
    train, holdout = chronological_split(rows, train_fraction=0.7)
    assert [r["elo_diff"] for r in train] == [0, 1, 2, 3, 4, 5, 6]
    assert [r["elo_diff"] for r in holdout] == [7, 8, 9]


def test_multiclass_brier_is_zero_for_perfect_prediction():
    y_true = np.array([2])  # HOME_TEAM (labels sorted: AWAY=0, DRAW=1, HOME=2)
    probs = np.array([[0.0, 0.0, 1.0]])
    assert _multiclass_brier(y_true, probs) == 0.0


def test_multiclass_brier_matches_random_guess_baseline():
    y_true = np.array([0, 1, 2])
    uniform_probs = np.full((3, 3), 1 / 3)
    score = _multiclass_brier(y_true, uniform_probs)
    assert abs(score - RANDOM_GUESS_BRIER) < 1e-9


def test_multiclass_brier_penalizes_confident_wrong_predictions():
    y_true = np.array([2])
    confident_wrong = np.array([[1.0, 0.0, 0.0]])
    uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])
    assert _multiclass_brier(y_true, confident_wrong) > _multiclass_brier(y_true, uniform)


def test_hash_rows_is_deterministic():
    rows = [_row("HOME_TEAM"), _row("DRAW")]
    assert _hash_rows(rows) == _hash_rows(rows)


def test_hash_rows_differs_for_different_data():
    rows_a = [_row("HOME_TEAM")]
    rows_b = [_row("AWAY_TEAM")]
    assert _hash_rows(rows_a) != _hash_rows(rows_b)


def test_train_model_raises_on_too_little_data():
    rows = [_row("HOME_TEAM") for _ in range(5)]
    try:
        train_model(rows)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "at least 10 rows" in str(exc)


def test_train_model_raises_clear_error_when_a_class_is_missing_from_training_fold():
    # 8 HOME_TEAM, then 8 AWAY_TEAM, then 4 DRAW, in that order.
    # train_fraction=0.8 of 20 rows takes the first 16 — all HOME+AWAY,
    # zero draws. This should trigger our explicit guard, not XGBoost's
    # cryptic internal error.
    rows = [_row("HOME_TEAM", elo_diff=300) for _ in range(8)]
    rows += [_row("AWAY_TEAM", elo_diff=-300) for _ in range(8)]
    rows += [_row("DRAW", elo_diff=0) for _ in range(4)]
    try:
        train_model(rows, train_fraction=0.8)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "DRAW" in str(exc)


def test_train_model_produces_artifact_and_beats_baseline_on_easy_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Strongly separable synthetic data, interleaved so every class
    # appears throughout — including within the first 80% that becomes
    # the training fold. A model that can't beat random on data this
    # easy would be truly broken.
    rows = []
    for _ in range(30):
        rows.append(_row("HOME_TEAM", elo_diff=300))
        rows.append(_row("AWAY_TEAM", elo_diff=-300))
        rows.append(_row("DRAW", elo_diff=0))

    _, result = train_model(rows, train_fraction=0.8)

    assert result.beats_random_baseline is True
    assert result.n_train == 72
    assert result.n_holdout == 18
    from pathlib import Path
    assert Path(result.model_path).exists()