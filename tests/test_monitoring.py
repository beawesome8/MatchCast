"""Monitoring tests.

Hand-picked predictions/outcomes so every aggregate number (Brier
average, hit rate, calibration bucket) is checkable by hand.
"""

from datetime import UTC, datetime

from matchcast.models import IngestionQuarantine, Match, ModelVersion, PredictionLog, Team
from matchcast.monitoring import get_performance_summary, get_prediction_history

START = datetime(2026, 7, 1, tzinfo=UTC)


def _seed_model_version(session, status="champion", holdout_brier=0.5) -> ModelVersion:
    mv = ModelVersion(
        model_path="artifacts/fake.json",
        model_bytes=b"fake",
        data_hash="abc",
        n_train=10,
        n_holdout=5,
        train_brier=0.4,
        holdout_brier=holdout_brier,
        holdout_log_loss=0.9,
        beats_random_baseline=True,
        status=status,
    )
    session.add(mv)
    session.flush()
    return mv


def _seed_match(session, source_id: int, kickoff_utc=None) -> Match:
    home = Team(source_team_id=source_id * 10 + 1, name=f"Home{source_id}")
    away = Team(source_team_id=source_id * 10 + 2, name=f"Away{source_id}")
    session.add_all([home, away])
    session.flush()
    match = Match(
        source_match_id=source_id,
        tournament_id="WC2026",
        stage="GROUP_STAGE",
        status="FINISHED",
        kickoff_utc=kickoff_utc or START,
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=1,
        away_goals=0,
        winner="HOME_TEAM",
    )
    session.add(match)
    session.flush()
    return match


def _seed_scored_prediction(
    session, match, model_version, prob_home, prob_draw, prob_away, actual_outcome, brier_score
) -> PredictionLog:
    log = PredictionLog(
        match_id=match.id,
        model_version_id=model_version.id,
        prob_home=prob_home,
        prob_draw=prob_draw,
        prob_away=prob_away,
        actual_outcome=actual_outcome,
        brier_score=brier_score,
    )
    session.add(log)
    session.flush()
    return log


def test_empty_database_returns_safe_defaults(session_factory):
    with session_factory() as s:
        summary = get_performance_summary(s)

        assert summary.n_predictions_logged == 0
        assert summary.n_predictions_scored == 0
        assert summary.overall_brier_score is None
        assert summary.hit_rate is None
        assert summary.by_model_version == []
        assert summary.latest_model_version_id is None


def test_pending_predictions_are_counted_but_not_scored(session_factory):
    with session_factory() as s:
        mv = _seed_model_version(s)
        match = _seed_match(s, source_id=1)
        log = PredictionLog(
            match_id=match.id, model_version_id=mv.id,
            prob_home=0.5, prob_draw=0.3, prob_away=0.2,
        )
        s.add(log)
        s.commit()

        summary = get_performance_summary(s)

        assert summary.n_predictions_logged == 1
        assert summary.n_predictions_scored == 0
        assert summary.n_predictions_pending == 1
        assert summary.overall_brier_score is None


def test_overall_brier_is_mean_of_scored_predictions(session_factory):
    with session_factory() as s:
        mv = _seed_model_version(s)
        match_a = _seed_match(s, source_id=1)
        match_b = _seed_match(s, source_id=2)
        _seed_scored_prediction(
            s, match_a, mv, 1.0, 0.0, 0.0, actual_outcome="HOME_TEAM", brier_score=0.0
        )
        _seed_scored_prediction(
            s, match_b, mv, 0.0, 0.0, 1.0, actual_outcome="HOME_TEAM", brier_score=2.0
        )
        s.commit()

        summary = get_performance_summary(s)

        assert summary.n_predictions_scored == 2
        assert abs(summary.overall_brier_score - 1.0) < 1e-9  # mean of 0.0 and 2.0


def test_hit_rate_counts_correct_top_predictions(session_factory):
    with session_factory() as s:
        mv = _seed_model_version(s)
        match_a = _seed_match(s, source_id=1)
        match_b = _seed_match(s, source_id=2)
        _seed_scored_prediction(
            s, match_a, mv, 0.7, 0.2, 0.1, actual_outcome="HOME_TEAM", brier_score=0.1
        )
        _seed_scored_prediction(
            s, match_b, mv, 0.7, 0.2, 0.1, actual_outcome="AWAY_TEAM", brier_score=1.0
        )
        s.commit()

        summary = get_performance_summary(s)

        assert abs(summary.hit_rate - 0.5) < 1e-9  # 1 correct out of 2


def test_calibration_buckets_predictions_by_confidence(session_factory):
    with session_factory() as s:
        mv = _seed_model_version(s)
        match = _seed_match(s, source_id=1)
        _seed_scored_prediction(
            s, match, mv, 0.9, 0.05, 0.05, actual_outcome="HOME_TEAM", brier_score=0.02
        )
        s.commit()

        summary = get_performance_summary(s)
        top_bucket = summary.calibration[-1]  # [0.8, 1.0] is the last bin

        assert top_bucket.n_predictions == 1
        assert abs(top_bucket.avg_predicted_probability - 0.9) < 1e-9
        assert abs(top_bucket.observed_frequency - 1.0) < 1e-9


def test_by_model_version_groups_correctly(session_factory):
    with session_factory() as s:
        mv1 = _seed_model_version(s, status="retired", holdout_brier=0.6)
        mv2 = _seed_model_version(s, status="champion", holdout_brier=0.5)
        match_a = _seed_match(s, source_id=1)
        match_b = _seed_match(s, source_id=2)
        _seed_scored_prediction(
            s, match_a, mv1, 0.6, 0.2, 0.2, actual_outcome="HOME_TEAM", brier_score=0.3
        )
        _seed_scored_prediction(
            s, match_b, mv2, 0.6, 0.2, 0.2, actual_outcome="HOME_TEAM", brier_score=0.1
        )
        s.commit()

        summary = get_performance_summary(s)

        assert len(summary.by_model_version) == 2
        by_id = {v.model_version_id: v for v in summary.by_model_version}
        assert by_id[mv1.id].status == "retired"
        assert by_id[mv1.id].mean_brier_score == 0.3
        assert by_id[mv2.id].status == "champion"
        assert by_id[mv2.id].mean_brier_score == 0.1


def test_quarantine_count_is_reported(session_factory):
    with session_factory() as s:
        s.add(IngestionQuarantine(source="test", reason="bad data", payload="{}"))
        s.add(IngestionQuarantine(source="test", reason="bad data", payload="{}"))
        s.commit()

        summary = get_performance_summary(s)
        assert summary.n_quarantined_batches == 2


def test_latest_model_version_is_reported(session_factory):
    with session_factory() as s:
        _seed_model_version(s, status="retired")
        latest = _seed_model_version(s, status="champion")
        s.commit()

        summary = get_performance_summary(s)
        assert summary.latest_model_version_id == latest.id
        assert summary.latest_model_trained_at is not None
        
def test_prediction_history_empty_when_nothing_scored(session_factory):
    with session_factory() as s:
        assert get_prediction_history(s) == []


def test_prediction_history_flags_correct_and_incorrect(session_factory):
    with session_factory() as s:
        mv = _seed_model_version(s)
        late = datetime(2026, 7, 10, tzinfo=UTC)
        match_a = _seed_match(s, source_id=1, kickoff_utc=late)
        match_b = _seed_match(s, source_id=2, kickoff_utc=late)
        _seed_scored_prediction(
            s, match_a, mv, 0.7, 0.2, 0.1, actual_outcome="HOME_TEAM", brier_score=0.1
        )
        _seed_scored_prediction(
            s, match_b, mv, 0.1, 0.2, 0.7, actual_outcome="HOME_TEAM", brier_score=1.5
        )
        s.commit()

        history = get_prediction_history(s)
        by_match = {h.match_id: h for h in history}
        assert by_match[match_a.id].correct is True
        assert by_match[match_b.id].correct is False


def test_prediction_history_includes_team_names(session_factory):
    with session_factory() as s:
        mv = _seed_model_version(s)
        match = _seed_match(s, source_id=1, kickoff_utc=datetime(2026, 7, 10, tzinfo=UTC))
        _seed_scored_prediction(
            s, match, mv, 0.8, 0.1, 0.1, actual_outcome="HOME_TEAM", brier_score=0.05
        )
        s.commit()

        history = get_prediction_history(s)
        assert history[0].home_team_name == "Home1"
        assert history[0].away_team_name == "Away1"
        
def test_prediction_history_excludes_matches_before_july_9(session_factory):
    with session_factory() as s:
        mv = _seed_model_version(s)
        early_match = _seed_match(s, source_id=1)  # kickoff is START = July 1
        _seed_scored_prediction(
            s, early_match, mv, 0.7, 0.2, 0.1, actual_outcome="HOME_TEAM", brier_score=0.1
        )
        s.commit()

        assert get_prediction_history(s) == []


def test_prediction_history_includes_matches_on_or_after_july_9(session_factory):
    with session_factory() as s:
        mv = _seed_model_version(s)
        home = Team(source_team_id=501, name="LateHome")
        away = Team(source_team_id=502, name="LateAway")
        s.add_all([home, away])
        s.flush()
        late_match = Match(
            source_match_id=501,
            tournament_id="WC2026",
            stage="QUARTER_FINALS",
            status="FINISHED",
            kickoff_utc=datetime(2026, 7, 9, 20, 0, tzinfo=UTC),
            home_team_id=home.id,
            away_team_id=away.id,
            home_goals=1,
            away_goals=0,
            winner="HOME_TEAM",
        )
        s.add(late_match)
        s.flush()
        _seed_scored_prediction(
            s, late_match, mv, 0.7, 0.2, 0.1, actual_outcome="HOME_TEAM", brier_score=0.1
        )
        s.commit()

        history = get_prediction_history(s)
        assert len(history) == 1
        assert history[0].home_team_name == "LateHome"


def test_performance_summary_reports_first_and_current_model_brier(session_factory):
    with session_factory() as s:
        _seed_model_version(s, status="retired", holdout_brier=0.62)
        _seed_model_version(s, status="champion", holdout_brier=0.55)
        s.commit()

        summary = get_performance_summary(s)
        assert summary.first_model_holdout_brier == 0.62
        assert summary.current_champion_holdout_brier == 0.55
        assert abs(summary.brier_improvement - 0.07) < 1e-9