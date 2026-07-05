"""Monte Carlo tournament simulation: given the current bracket state,
estimate each remaining team's probability of winning the World Cup.

Bracket topology (Round of 16 through Final) is sourced from FIFA's
official published knockout schedule (fifa.com/tournaments/mens/
worldcup/canadamexicousa2026/articles/knockout-stage-match-schedule-
bracket), not guessed. FIFA's match numbering is offset by a constant
from football-data.org's source_match_id — verified against three real
matches already in this database (91->537377, 92->537378, 97->537383).

Match 103 (third-place playoff) is intentionally excluded: it has no
bearing on who wins the tournament.

Rather than re-running the model inside every one of N trials, each
UNIQUE possible matchup is predicted once and cached; trials then
cheaply sample from that cache. The randomness lives in which team
advances each round, not in whether a probability was computed fresh —
this is still a genuine Monte Carlo simulation, just an efficient one.

Knockout matches can't end in a draw at the final whistle (extra time
and penalties resolve it). Per DESIGN.md, draw probability is
redistributed to the two teams proportional to their raw win
probabilities before a winner is sampled.
"""

import random
from dataclasses import dataclass

import numpy as np
import xgboost as xgb
from sqlalchemy.orm import Session

from matchcast.features import TeamState, _pre_match_features, get_current_team_states
from matchcast.models import Match, Team
from matchcast.registry import get_current_champion
from matchcast.serving import load_model_from_bytes
from matchcast.train import FEATURE_COLUMNS, OUTCOME_LABELS

# FIFA match number -> source_match_id offset. Verified against three
# real matches already in the database (see module docstring).
FIFA_MATCH_OFFSET = 537286

ROUND_OF_16 = [89, 90, 91, 92, 93, 94, 95, 96]

# FIFA match number -> (match feeding the "home" slot, match feeding
# the "away" slot). Sourced from fifa.com's published bracket.
BRACKET_FEEDS = {
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
    101: (97, 98),
    102: (99, 100),
    104: (101, 102),
}
FINAL_MATCH = 104


@dataclass
class SimulationResult:
    n_trials: int
    win_probabilities: dict[str, float]
    final_appearance_probabilities: dict[str, float]


def _match_probs(
    booster: xgb.Booster, home_state: TeamState, away_state: TeamState
) -> tuple[float, float, float]:
    features = _pre_match_features(home_state, away_state, is_knockout=1)
    x = np.array([[features[col] for col in FEATURE_COLUMNS]])
    probs = booster.predict(xgb.DMatrix(x))[0]
    label_probs = dict(zip(OUTCOME_LABELS, probs, strict=True))
    return (
        float(label_probs["HOME_TEAM"]),
        float(label_probs["DRAW"]),
        float(label_probs["AWAY_TEAM"]),
    )


def _resolve_knockout_winner(
    rng: random.Random, prob_home: float, prob_draw: float, prob_away: float
) -> str:
    """Redistribute draw probability proportionally, then sample.
    Returns "home" or "away"."""
    win_total = prob_home + prob_away
    if win_total <= 0:
        return "home" if rng.random() < 0.5 else "away"
    adjusted_home = prob_home + prob_draw * (prob_home / win_total)
    return "home" if rng.random() < adjusted_home else "away"


def run_simulation(session: Session, n_trials: int = 10_000, seed: int = 42) -> SimulationResult:
    champion = get_current_champion(session)
    if champion is None or champion.model_bytes is None:
        raise ValueError("no champion model with stored bytes available for simulation")
    booster = load_model_from_bytes(champion.model_bytes)

    team_states = get_current_team_states(session)
    team_names = {t.id: t.name for t in session.query(Team).all()}

    matches_by_fifa_number: dict[int, Match | None] = {}
    for fifa_number in [*ROUND_OF_16, *BRACKET_FEEDS.keys()]:
        matches_by_fifa_number[fifa_number] = (
            session.query(Match)
            .filter_by(source_match_id=fifa_number + FIFA_MATCH_OFFSET)
            .one_or_none()
        )

    prob_cache: dict[tuple[int, int], tuple[float, float, float]] = {}

    def get_probs(home_id: int, away_id: int) -> tuple[float, float, float]:
        key = (home_id, away_id)
        if key not in prob_cache:
            home_state = team_states.setdefault(home_id, TeamState())
            away_state = team_states.setdefault(away_id, TeamState())
            prob_cache[key] = _match_probs(booster, home_state, away_state)
        return prob_cache[key]

    rng = random.Random(seed)
    trophy_counts: dict[int, int] = {}
    finalist_counts: dict[int, int] = {}

    for _ in range(n_trials):
        winner_of: dict[int, int] = {}

        for fifa_number in ROUND_OF_16:
            match = matches_by_fifa_number[fifa_number]
            if match is None:
                raise ValueError(
                    f"Round of 16 match {fifa_number} not found — has the "
                    "tournament progressed past this point unexpectedly?"
                )
            if match.status == "FINISHED" and match.winner is not None:
                winner_of[fifa_number] = (
                    match.home_team_id if match.winner == "HOME_TEAM" else match.away_team_id
                )
                continue
            prob_home, prob_draw, prob_away = get_probs(match.home_team_id, match.away_team_id)
            outcome = _resolve_knockout_winner(rng, prob_home, prob_draw, prob_away)
            winner_of[fifa_number] = match.home_team_id if outcome == "home" else match.away_team_id

        for fifa_number, (feed_home, feed_away) in BRACKET_FEEDS.items():
            match = matches_by_fifa_number.get(fifa_number)
            if match is not None and match.status == "FINISHED" and match.winner is not None:
                winner_of[fifa_number] = (
                    match.home_team_id if match.winner == "HOME_TEAM" else match.away_team_id
                )
                continue

            home_id = winner_of[feed_home]
            away_id = winner_of[feed_away]
            prob_home, prob_draw, prob_away = get_probs(home_id, away_id)
            outcome = _resolve_knockout_winner(rng, prob_home, prob_draw, prob_away)
            winner_of[fifa_number] = home_id if outcome == "home" else away_id

        trophy_counts[winner_of[FINAL_MATCH]] = trophy_counts.get(winner_of[FINAL_MATCH], 0) + 1
        for finalist_id in (winner_of[101], winner_of[102]):
            finalist_counts[finalist_id] = finalist_counts.get(finalist_id, 0) + 1

    win_probabilities = {
        team_names.get(team_id, f"Team {team_id}"): count / n_trials
        for team_id, count in sorted(trophy_counts.items(), key=lambda kv: -kv[1])
    }
    final_appearance_probabilities = {
        team_names.get(team_id, f"Team {team_id}"): count / n_trials
        for team_id, count in sorted(finalist_counts.items(), key=lambda kv: -kv[1])
    }

    return SimulationResult(
        n_trials=n_trials,
        win_probabilities=win_probabilities,
        final_appearance_probabilities=final_appearance_probabilities,
    )


if __name__ == "__main__":
    import json
    from dataclasses import asdict

    from matchcast.db import get_session_factory

    with get_session_factory()() as session:
        result = run_simulation(session)
    print(json.dumps(asdict(result), indent=2))