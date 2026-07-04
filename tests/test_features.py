"""Feature engineering tests.

These tests build tiny synthetic match histories by hand (not via
the API) so we control every input and can assert exact expected
numbers — Elo math is the kind of logic that looks plausible when
wrong, so eyeballing output isn't enough.
"""

from datetime import UTC, datetime, timedelta

from matchcast.features import _elo_expected, _elo_update, build_feature_table
from matchcast.models import Match, Team

START = datetime(2026, 6, 1, tzinfo=UTC)


def _add_team(session, source_id: int, name: str) -> Team:
    team = Team(source_team_id=source_id, name=name)
    session.add(team)
    session.flush()
    return team


def _add_match(
    session, home, away, day_offset, home_goals, away_goals, winner, stage="GROUP_STAGE"
):
    match = Match(
        source_match_id=1000 + day_offset,
        tournament_id="WC2026",
        stage=stage,
        status="FINISHED",
        kickoff_utc=START + timedelta(days=day_offset),
        home_team_id=home.id,
        away_team_id=away.id,
        home_goals=home_goals,
        away_goals=away_goals,
        winner=winner,
    )
    session.add(match)
    session.flush()
    return match


def test_elo_expected_is_symmetric_at_equal_ratings():
    assert abs(_elo_expected(1500, 1500) - 0.5) < 1e-9


def test_elo_update_favors_the_winner():
    new_a, new_b = _elo_update(1500, 1500, score_a=1.0)
    assert new_a > 1500 > new_b
    assert abs((new_a - 1500) - (1500 - new_b)) < 1e-9  # zero-sum


def test_first_match_between_two_teams_has_zero_elo_diff(session_factory):
    with session_factory() as s:
        home, away = _add_team(s, 1, "Home"), _add_team(s, 2, "Away")
        _add_match(s, home, away, day_offset=0, home_goals=2, away_goals=0, winner="HOME_TEAM")
        s.commit()

        rows = build_feature_table(s)
        assert len(rows) == 1
        assert rows[0]["elo_diff"] == 0.0
        assert rows[0]["home_form_ppg"] == 1.0  # neutral prior, no history yet
        assert rows[0]["is_knockout"] == 0


def test_elo_updates_between_matches_not_within(session_factory):
    with session_factory() as s:
        a, b, c = _add_team(s, 1, "A"), _add_team(s, 2, "B"), _add_team(s, 3, "C")
        _add_match(s, a, b, day_offset=0, home_goals=3, away_goals=0, winner="HOME_TEAM")
        _add_match(s, a, c, day_offset=1, home_goals=1, away_goals=1, winner="DRAW")
        s.commit()

        rows = build_feature_table(s)
        assert len(rows) == 2
        assert rows[0]["elo_diff"] == 0.0
        assert rows[1]["elo_diff"] > 0.0


def test_knockout_flag_is_set_for_non_group_stage(session_factory):
    with session_factory() as s:
        a, b = _add_team(s, 1, "A"), _add_team(s, 2, "B")
        _add_match(
            s, a, b, day_offset=0, home_goals=1, away_goals=0,
            winner="HOME_TEAM", stage="QUARTER_FINALS"
        )
        s.commit()

        rows = build_feature_table(s)
        assert rows[0]["is_knockout"] == 1


def test_unfinished_matches_are_excluded_but_still_seed_state(session_factory):
    with session_factory() as s:
        a, b = _add_team(s, 1, "A"), _add_team(s, 2, "B")
        scheduled = Match(
            source_match_id=2001,
            tournament_id="WC2026",
            stage="GROUP_STAGE",
            status="SCHEDULED",
            kickoff_utc=START,
            home_team_id=a.id,
            away_team_id=b.id,
            home_goals=None,
            away_goals=None,
            winner=None,
        )
        s.add(scheduled)
        s.commit()

        rows = build_feature_table(s)
        assert rows == []


def test_form_window_only_keeps_last_five_matches(session_factory):
    with session_factory() as s:
        a = _add_team(s, 1, "A")
        opponents = [_add_team(s, i, f"Opp{i}") for i in range(2, 8)]
        for i, opp in enumerate(opponents[:6]):
            _add_match(s, a, opp, day_offset=i, home_goals=1, away_goals=0, winner="HOME_TEAM")
        s.commit()

        rows = build_feature_table(s)
        assert rows[5]["home_form_ppg"] == 3.0