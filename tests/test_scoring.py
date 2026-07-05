"""Prediction scoring tests.

Hand-picked probabilities and known outcomes so every Brier score is
checkable by hand, not just plausible-looking.
"""

from datetime import UTC, datetime

from matchcast.models import Match, PredictionLog, Team
from matchcast.scoring import score_pending_predictions

START = datetime(2026, 7, 1, tzinfo=UTC)


def _seed_match(session, status="FINISHED", winner="HOME_TEAM"):
    home = Team(source_team_id=1, name="Home")
    away = Team(source_team_id=2, name="Away")
    session.add_all([home, away])
    session.flush()

    match = Match(
        source_match_id=9001,
        tournament_id="WC2026",
        stage="GROUP_STAGE",
        status=status,
        kickoff_utc=START,
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=2 if status == "FINISHED" else None,
        away_goals=0 if status == "FINISHED" else None,
        winner=winner if status == "FINISHED" else None,
    )
    session.add(match)
    session.flush()
    return match


def _seed_prediction(session, match, prob_home, prob_draw, prob_away):
    log = PredictionLog(
        match_id=match.id,
        model_version_id=1,
        prob_home=prob_home,
        prob_draw=prob_draw,
        prob_away=prob_away,
    )
    session.add(log)
    session.flush()
    return log


def test_scores_prediction_for_finished_match(session_factory):
    with session_factory() as s:
        match = _seed_match(s, status="FINISHED", winner="HOME_TEAM")
        log = _seed_prediction(s, match, prob_home=1.0, prob_draw=0.0, prob_away=0.0)
        s.commit()

        result = score_pending_predictions(s)

        assert result == {"scored": 1, "still_pending": 0}
        s.refresh(log)
        assert log.actual_outcome == "HOME_TEAM"
        assert abs(log.brier_score - 0.0) < 1e-9  # perfect prediction


def test_leaves_prediction_for_unfinished_match_unscored(session_factory):
    with session_factory() as s:
        match = _seed_match(s, status="SCHEDULED")
        log = _seed_prediction(s, match, prob_home=0.5, prob_draw=0.3, prob_away=0.2)
        s.commit()

        result = score_pending_predictions(s)

        assert result == {"scored": 0, "still_pending": 1}
        s.refresh(log)
        assert log.actual_outcome is None
        assert log.brier_score is None


def test_confident_wrong_prediction_gets_max_brier_score(session_factory):
    with session_factory() as s:
        match = _seed_match(s, status="FINISHED", winner="AWAY_TEAM")
        log = _seed_prediction(s, match, prob_home=1.0, prob_draw=0.0, prob_away=0.0)
        s.commit()

        score_pending_predictions(s)

        s.refresh(log)
        # Fully confident in the wrong outcome: max possible Brier is 2.0
        assert abs(log.brier_score - 2.0) < 1e-9


def test_scoring_is_idempotent(session_factory):
    with session_factory() as s:
        match = _seed_match(s, status="FINISHED", winner="DRAW")
        _seed_prediction(s, match, prob_home=0.3, prob_draw=0.4, prob_away=0.3)
        s.commit()

        first = score_pending_predictions(s)
        second = score_pending_predictions(s)

        assert first == {"scored": 1, "still_pending": 0}
        assert second == {"scored": 0, "still_pending": 0}  # already scored, not re-counted


def test_missing_match_is_treated_as_still_pending(session_factory):
    with session_factory() as s:
        # A prediction log row whose match is somehow gone — should not
        # crash, just remain unscored.
        log = PredictionLog(
            match_id=99999, model_version_id=1, prob_home=0.5, prob_draw=0.3, prob_away=0.2
        )
        s.add(log)
        s.commit()

        result = score_pending_predictions(s)
        assert result == {"scored": 0, "still_pending": 1}