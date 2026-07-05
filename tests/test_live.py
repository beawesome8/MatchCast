"""Live update tests.

All math-only — no network, no live match required. Hand-picked
scores/times chosen so results are checkable by intuition: a big lead
late in the match should be near-certain; a tied score at full time
should be a near-certain draw.
"""

from datetime import UTC, datetime

from matchcast.live import (
    compute_live_probabilities,
    get_latest_prediction,
    get_live_update,
)
from matchcast.models import Match, PredictionLog, Team

START = datetime(2026, 7, 1, tzinfo=UTC)


def test_probabilities_always_sum_to_one():
    result = compute_live_probabilities(
        prob_home=0.5, prob_draw=0.3, prob_away=0.2,
        home_score=1, away_score=1, minutes_elapsed=60,
    )
    assert abs(result.prob_home + result.prob_draw + result.prob_away - 1.0) < 1e-9


def test_ordering_preserved_at_kickoff():
    # Not an exact round-trip (see module docstring) — but a team
    # favored pre-match should still be favored at minute 0, 0-0.
    result = compute_live_probabilities(
        prob_home=0.6, prob_draw=0.25, prob_away=0.15,
        home_score=0, away_score=0, minutes_elapsed=0,
    )
    assert result.prob_home > result.prob_away


def test_big_late_lead_is_near_certain_win():
    result = compute_live_probabilities(
        prob_home=0.4, prob_draw=0.3, prob_away=0.3,
        home_score=3, away_score=0, minutes_elapsed=85,
    )
    assert result.prob_home > 0.95


def test_tied_score_at_full_time_is_certain_draw():
    result = compute_live_probabilities(
        prob_home=0.5, prob_draw=0.3, prob_away=0.2,
        home_score=1, away_score=1, minutes_elapsed=90,
    )
    assert result.prob_draw > 0.99


def test_leading_team_at_full_time_is_certain_winner():
    result = compute_live_probabilities(
        prob_home=0.3, prob_draw=0.3, prob_away=0.4,
        home_score=2, away_score=1, minutes_elapsed=90,
    )
    assert result.prob_home > 0.99


def test_more_time_remaining_gives_trailing_team_more_hope():
    early = compute_live_probabilities(
        prob_home=0.5, prob_draw=0.3, prob_away=0.2,
        home_score=0, away_score=1, minutes_elapsed=10,
    )
    late = compute_live_probabilities(
        prob_home=0.5, prob_draw=0.3, prob_away=0.2,
        home_score=0, away_score=1, minutes_elapsed=80,
    )
    # Home is trailing 0-1 in both cases; with more time left (minute
    # 10 vs minute 80), home's comeback chances should be higher.
    assert early.prob_home > late.prob_home


def test_get_latest_prediction_returns_none_when_no_prediction_logged(session_factory):
    with session_factory() as s:
        assert get_latest_prediction(s, match_id=999) is None


def _seed_match_and_prediction(session, prob_home=0.5, prob_draw=0.3, prob_away=0.2):
    home = Team(source_team_id=1, name="Home")
    away = Team(source_team_id=2, name="Away")
    session.add_all([home, away])
    session.flush()

    match = Match(
        source_match_id=9001,
        tournament_id="WC2026",
        stage="GROUP_STAGE",
        status="IN_PLAY",
        kickoff_utc=START,
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=None,
        away_goals=None,
        winner=None,
    )
    session.add(match)
    session.flush()

    log = PredictionLog(
        match_id=match.id, model_version_id=1,
        prob_home=prob_home, prob_draw=prob_draw, prob_away=prob_away,
    )
    session.add(log)
    session.flush()
    return match, log


def test_get_latest_prediction_returns_the_logged_row(session_factory):
    with session_factory() as s:
        match, log = _seed_match_and_prediction(s)
        s.commit()

        found = get_latest_prediction(s, match.id)
        assert found.id == log.id


def test_get_live_update_combines_logged_prediction_with_current_score(session_factory):
    with session_factory() as s:
        match, _ = _seed_match_and_prediction(s, prob_home=0.6, prob_draw=0.25, prob_away=0.15)
        s.commit()

        result = get_live_update(s, match.id, home_score=2, away_score=0, minutes_elapsed=80)
        assert result.prob_home > 0.9


def test_get_live_update_raises_when_no_prediction_exists(session_factory):
    with session_factory() as s:
        try:
            get_live_update(s, match_id=999, home_score=0, away_score=0, minutes_elapsed=10)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "no logged pre-match prediction" in str(exc)