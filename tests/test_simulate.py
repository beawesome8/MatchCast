"""Tournament simulation tests.

Uses a tiny, fully hand-built bracket (not real WC2026 data) so every
outcome is either deterministic or checkable by hand — with 10,000
trials involved, a subtly wrong probability calculation could easily
look "plausible" without being correct.
"""

import random
from datetime import UTC, datetime, timedelta

import numpy as np
import xgboost as xgb

from matchcast.models import Match, ModelVersion, Team
from matchcast.simulate import (
    BRACKET_FEEDS,
    FIFA_MATCH_OFFSET,
    ROUND_OF_16,
    _resolve_knockout_winner,
    run_simulation,
)

START = datetime(2026, 7, 1, tzinfo=UTC)


def _train_tiny_model() -> bytes:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(30, 5))
    y = rng.integers(0, 3, size=30)
    booster = xgb.train(
        {"objective": "multi:softprob", "num_class": 3, "max_depth": 2},
        xgb.DMatrix(x, label=y),
        num_boost_round=5,
    )
    return booster.save_raw(raw_format="json")


def _seed_champion(session) -> ModelVersion:
    champion = ModelVersion(
        model_path="artifacts/fake.json",
        model_bytes=_train_tiny_model(),
        data_hash="abc",
        n_train=20,
        n_holdout=10,
        train_brier=0.4,
        holdout_brier=0.5,
        holdout_log_loss=0.9,
        beats_random_baseline=True,
        status="champion",
    )
    session.add(champion)
    session.flush()
    return champion


def _seed_full_bracket(
    session, finished_r16: bool = False
) -> dict[int, Team]:
    """Seeds 16 teams and all 8 Round of 16 matches (plus empty slots
    for QF/SF/Final, which simulate.py resolves at runtime, not from
    the database, so they don't need to be created here).

    Returns a dict mapping each ROUND_OF_16 FIFA match number to the
    HOME team of that match — the numbering callers actually need,
    not an arbitrary team-slot index.
    """
    team_pool = [Team(source_team_id=i, name=f"Team{i}") for i in range(1, 17)]
    session.add_all(team_pool)
    session.flush()

    home_team_by_match: dict[int, Team] = {}
    for i, fifa_number in enumerate(ROUND_OF_16):
        home, away = team_pool[i * 2], team_pool[i * 2 + 1]
        match = Match(
            source_match_id=fifa_number + FIFA_MATCH_OFFSET,
            tournament_id="WC2026",
            stage="ROUND_OF_16",
            status="FINISHED" if finished_r16 else "SCHEDULED",
            kickoff_utc=START + timedelta(days=i),
            home_team_id=home.id,
            away_team_id=away.id,
            home_goals=1 if finished_r16 else None,
            away_goals=0 if finished_r16 else None,
            winner="HOME_TEAM" if finished_r16 else None,
        )
        session.add(match)
        home_team_by_match[fifa_number] = home
    session.commit()
    return home_team_by_match

def test_draw_redistribution_favors_stronger_side():
    rng = random.Random(1)
    # Home is much stronger (0.7 vs 0.1); after redistributing the 0.2
    # draw probability proportionally, home should win the large
    # majority of samples.
    home_wins = sum(
        1 for _ in range(1000) if _resolve_knockout_winner(rng, 0.7, 0.2, 0.1) == "home"
    )
    assert home_wins > 800


def test_draw_redistribution_is_fair_when_teams_are_equal():
    rng = random.Random(2)
    home_wins = sum(
        1 for _ in range(2000) if _resolve_knockout_winner(rng, 0.4, 0.2, 0.4) == "home"
    )
    # Equal teams: should land close to 50/50, not exactly (randomness),
    # but nowhere near a systematic bias in either direction.
    assert 900 < home_wins < 1100


def test_bracket_feeds_match_fifa_published_topology():
    # Locks in the exact sourced topology so an accidental edit can't
    # silently scramble which matches feed which.
    assert BRACKET_FEEDS[97] == (89, 90)
    assert BRACKET_FEEDS[98] == (93, 94)
    assert BRACKET_FEEDS[99] == (91, 92)
    assert BRACKET_FEEDS[100] == (95, 96)
    assert BRACKET_FEEDS[101] == (97, 98)
    assert BRACKET_FEEDS[102] == (99, 100)
    assert BRACKET_FEEDS[104] == (101, 102)


def test_simulation_raises_with_no_champion(session_factory):
    with session_factory() as s:
        try:
            run_simulation(s, n_trials=10)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "no champion" in str(exc)


def test_simulation_probabilities_sum_correctly(session_factory):
    with session_factory() as s:
        _seed_champion(s)
        _seed_full_bracket(s, finished_r16=False)
        s.commit()

        result = run_simulation(s, n_trials=500, seed=1)

        assert result.n_trials == 500
        assert abs(sum(result.win_probabilities.values()) - 1.0) < 1e-9
        assert abs(sum(result.final_appearance_probabilities.values()) - 2.0) < 1e-9


def test_simulation_is_deterministic_given_a_seed(session_factory):
    with session_factory() as s:
        _seed_champion(s)
        _seed_full_bracket(s, finished_r16=False)
        s.commit()

        result_a = run_simulation(s, n_trials=200, seed=99)
        result_b = run_simulation(s, n_trials=200, seed=99)

        assert result_a.win_probabilities == result_b.win_probabilities


def test_finished_round_of_16_matches_are_never_re_simulated(session_factory):
    with session_factory() as s:
        _seed_champion(s)
        home_teams = _seed_full_bracket(s, finished_r16=True)  # every R16 match decided
        s.commit()

        result = run_simulation(s, n_trials=300, seed=5)

        # Every R16 match was seeded as a HOME_TEAM win, so only the 8
        # home-side teams can possibly reach the final — the 8 away-side
        # teams should have exactly zero appearances.
        home_side_names = {team.name for team in home_teams.values()}
        for name in result.final_appearance_probabilities:
            assert name in home_side_names