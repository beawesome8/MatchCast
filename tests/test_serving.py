"""Serving layer tests.

Trains a small real XGBoost model in-memory as the "champion" rather
than mocking prediction — this exercises the exact Booster load/predict
path used in production, including the temp-file round-trip, so a
regression in that mechanism (like the sklearn-wrapper incompatibility
we hit) would be caught here, not just discovered live.
"""

from datetime import UTC, datetime, timedelta

import numpy as np
import xgboost as xgb

from matchcast.models import Match, ModelVersion, PredictionLog, Team
from matchcast.serving import get_upcoming_predictions, load_model_from_bytes

START = datetime(2026, 6, 1, tzinfo=UTC)


def _train_tiny_model() -> bytes:
    """A real, trained, tiny XGBoost model — same shape as production
    (5 features, 3 classes), saved and reloaded as bytes."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(30, 5))
    y = rng.integers(0, 3, size=30)

    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": 3, "max_depth": 2},
        xgb.DMatrix(x, label=y),
        num_boost_round=5,
    )
    return booster.save_raw(raw_format="json")


def _seed_champion(session, model_bytes: bytes | None = None) -> ModelVersion:
    champion = ModelVersion(
        model_path="artifacts/fake.json",
        model_bytes=model_bytes if model_bytes is not None else _train_tiny_model(),
        data_hash="abc",
        n_train=20,
        n_holdout=10,
        train_brier=0.4,
        holdout_brier=0.5,
        holdout_log_loss=0.9,
        beats_random_baseline=True,
        status="champion",
        rejection_reason=None,
    )
    session.add(champion)
    session.flush()
    return champion


def _seed_upcoming_match(session, home_name="Home", away_name="Away") -> tuple[Team, Team, Match]:
    home = Team(source_team_id=100, name=home_name)
    away = Team(source_team_id=200, name=away_name)
    session.add_all([home, away])
    session.flush()

    match = Match(
        source_match_id=9001,
        tournament_id="WC2026",
        stage="GROUP_STAGE",
        status="SCHEDULED",
        kickoff_utc=START + timedelta(days=1),
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=None,
        away_goals=None,
        winner=None,
    )
    session.add(match)
    session.flush()
    return home, away, match


def test_load_model_from_bytes_returns_working_booster():
    model_bytes = _train_tiny_model()
    booster = load_model_from_bytes(model_bytes)

    x = np.zeros((1, 5))
    probs = booster.predict(xgb.DMatrix(x))
    assert probs.shape == (1, 3)
    assert abs(probs.sum() - 1.0) < 1e-5


def test_raises_when_no_champion_exists(session_factory):
    with session_factory() as s:
        try:
            get_upcoming_predictions(s)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "no champion model exists" in str(exc)


def test_raises_when_champion_has_no_stored_bytes(session_factory):
    with session_factory() as s:
        champion = ModelVersion(
            model_path="artifacts/legacy.json",
            model_bytes=None,
            data_hash="abc",
            n_train=20,
            n_holdout=10,
            train_brier=0.4,
            holdout_brier=0.5,
            holdout_log_loss=0.9,
            beats_random_baseline=True,
            status="champion",
        )
        s.add(champion)
        s.commit()

        try:
            get_upcoming_predictions(s)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "no stored bytes" in str(exc)


def test_returns_empty_list_when_no_upcoming_matches(session_factory):
    with session_factory() as s:
        _seed_champion(s)
        s.commit()

        predictions = get_upcoming_predictions(s)
        assert predictions == []


def test_predicts_upcoming_match_with_team_names_and_valid_probabilities(session_factory):
    with session_factory() as s:
        _seed_champion(s)
        _seed_upcoming_match(s, home_name="Brazil", away_name="Norway")
        s.commit()

        predictions = get_upcoming_predictions(s)
        assert len(predictions) == 1

        p = predictions[0]
        assert p.home_team_name == "Brazil"
        assert p.away_team_name == "Norway"
        assert abs((p.prob_home + p.prob_draw + p.prob_away) - 1.0) < 1e-5
        assert 0.0 <= p.prob_home <= 1.0


def test_predictions_are_logged_to_database(session_factory):
    with session_factory() as s:
        _seed_champion(s)
        _seed_upcoming_match(s)
        s.commit()

        get_upcoming_predictions(s)

        logs = s.query(PredictionLog).all()
        assert len(logs) == 1
        assert logs[0].match_id is not None


def test_serving_same_match_twice_does_not_duplicate_log(session_factory):
    with session_factory() as s:
        _seed_champion(s)
        _seed_upcoming_match(s)
        s.commit()

        get_upcoming_predictions(s)
        get_upcoming_predictions(s)  # served again, e.g. a second API call

        logs = s.query(PredictionLog).all()
        assert len(logs) == 1  # NOT 2 — this is the idempotency guarantee