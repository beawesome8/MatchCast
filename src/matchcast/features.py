"""Feature engineering: turn raw match history into model inputs.

Design decision (documented limitation): Elo ratings are computed
self-consistently from matches already in our database, seeded at a
neutral 1500 for every team at the point they first appear. This
means Elo differentials early in the tournament carry little signal
(everyone starts equal) and sharpen as more matches are played. A
production version would seed from a real pre-tournament Elo snapshot
(e.g. eloratings.net) or FIFA rankings; we traded that off explicitly
given the compressed tournament timeline, rather than silently.

All features for a given match use only data strictly before that
match's kickoff — no leakage from the match's own result or from
matches that happen later.

Two public functions share one feature formula (_pre_match_features)
to guarantee training and serving can never compute features
differently — a mismatch there (train/serve skew) is a classic,
easy-to-miss source of silently bad predictions:

  build_feature_table(session)     -> training rows (FINISHED matches)
  build_upcoming_features(session) -> prediction rows (not-yet-played)

The two functions deliberately do NOT share their iteration loop, only
the math. build_feature_table is already running in production against
a live champion model; keeping its control flow untouched avoids
risking a regression there for the sake of avoiding a few duplicated
lines of straightforward bookkeeping in build_upcoming_features.
"""

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from matchcast.models import Match

BASE_ELO = 1500.0
K_FACTOR = 30.0
FORM_WINDOW = 5  # matches


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _elo_update(rating_a: float, rating_b: float, score_a: float) -> tuple[float, float]:
    """score_a: 1.0 win, 0.5 draw, 0.0 loss (from a's perspective)."""
    expected_a = _elo_expected(rating_a, rating_b)
    new_a = rating_a + K_FACTOR * (score_a - expected_a)
    new_b = rating_b + K_FACTOR * ((1 - score_a) - (1 - expected_a))
    return new_a, new_b


def _match_score(match: Match, team_id: int) -> float | None:
    """1.0/0.5/0.0 for team_id's result in this match, or None if unresolved."""
    if match.winner is None:
        return None
    if match.winner == "DRAW":
        return 0.5
    is_home = team_id == match.home_team_id
    if match.winner == "HOME_TEAM":
        return 1.0 if is_home else 0.0
    if match.winner == "AWAY_TEAM":
        return 0.0 if is_home else 1.0
    return None


@dataclass
class TeamState:
    elo: float = BASE_ELO
    recent_points: list = field(default_factory=list)
    recent_goal_diff: list = field(default_factory=list)


def _pre_match_features(home: TeamState, away: TeamState, is_knockout: int) -> dict:
    """The one true feature formula. Both training and prediction call
    this — never compute these values any other way, anywhere."""
    home_form_ppg = (
        sum(home.recent_points) / len(home.recent_points) if home.recent_points else 1.0
    )
    away_form_ppg = (
        sum(away.recent_points) / len(away.recent_points) if away.recent_points else 1.0
    )
    home_gd_avg = (
        sum(home.recent_goal_diff) / len(home.recent_goal_diff)
        if home.recent_goal_diff
        else 0.0
    )
    away_gd_avg = (
        sum(away.recent_goal_diff) / len(away.recent_goal_diff)
        if away.recent_goal_diff
        else 0.0
    )
    return {
        "elo_diff": home.elo - away.elo,
        "home_form_ppg": home_form_ppg,
        "away_form_ppg": away_form_ppg,
        "goal_diff_diff": home_gd_avg - away_gd_avg,
        "is_knockout": is_knockout,
    }


def _apply_result(home: TeamState, away: TeamState, match: Match) -> None:
    """Update both teams' rolling state after a resolved match. Mutates
    in place. Called only for matches with a known result."""
    home_score = _match_score(match, match.home_team_id)
    if home_score is None:
        return

    home.elo, away.elo = _elo_update(home.elo, away.elo, home_score)

    home_points = 3.0 if home_score == 1.0 else (1.0 if home_score == 0.5 else 0.0)
    away_points = 3.0 if home_score == 0.0 else (1.0 if home_score == 0.5 else 0.0)
    home.recent_points = (home.recent_points + [home_points])[-FORM_WINDOW:]
    away.recent_points = (away.recent_points + [away_points])[-FORM_WINDOW:]

    gd = (match.home_goals or 0) - (match.away_goals or 0)
    home.recent_goal_diff = (home.recent_goal_diff + [gd])[-FORM_WINDOW:]
    away.recent_goal_diff = (away.recent_goal_diff + [-gd])[-FORM_WINDOW:]


def _all_matches_chronological(session: Session, tournament_id: str) -> list[Match]:
    return (
        session.execute(
            select(Match)
            .where(Match.tournament_id == tournament_id)
            .order_by(Match.kickoff_utc)
        )
        .scalars()
        .all()
    )


def build_feature_table(session: Session, tournament_id: str = "WC2026") -> list[dict]:
    """Walk all matches in kickoff order, computing pre-match features
    for each, then updating each team's rolling state with the result.

    Returns one row per FINISHED match, suitable for training.
    """
    matches = _all_matches_chronological(session, tournament_id)
    state: dict[int, TeamState] = {}

    def get_state(team_id: int) -> TeamState:
        return state.setdefault(team_id, TeamState())

    rows: list[dict] = []

    for match in matches:
        home = get_state(match.home_team_id)
        away = get_state(match.away_team_id)
        is_knockout = 0 if match.stage == "GROUP_STAGE" else 1
        features = _pre_match_features(home, away, is_knockout)

        if match.status == "FINISHED" and match.winner is not None:
            rows.append({
                "match_id": match.id,
                "source_match_id": match.source_match_id,
                "outcome": match.winner,
                **features,
            })

        _apply_result(home, away, match)

    return rows


def build_upcoming_features(session: Session, tournament_id: str = "WC2026") -> list[dict]:
    """Same walk as build_feature_table, but collects rows for matches
    that have NOT been played yet, using the current accumulated state
    of both teams (i.e. exactly what a model would see if asked to
    predict this match right now).

    Uses the same underlying state-update mechanics as build_feature_table
    but does not share its loop, by design — see module docstring.
    """
    matches = _all_matches_chronological(session, tournament_id)
    state: dict[int, TeamState] = {}

    def get_state(team_id: int) -> TeamState:
        return state.setdefault(team_id, TeamState())

    upcoming: list[dict] = []

    for match in matches:
        home = get_state(match.home_team_id)
        away = get_state(match.away_team_id)
        is_knockout = 0 if match.stage == "GROUP_STAGE" else 1
        features = _pre_match_features(home, away, is_knockout)

        if match.status != "FINISHED":
            upcoming.append({
                "match_id": match.id,
                "source_match_id": match.source_match_id,
                "home_team_id": match.home_team_id,
                "away_team_id": match.away_team_id,
                "stage": match.stage,
                "status": match.status,
                "kickoff_utc": match.kickoff_utc,
                **features,
            })

        _apply_result(home, away, match)

    return upcoming

def get_current_team_states(
    session: Session, tournament_id: str = "WC2026"
) -> dict[int, TeamState]:
    """Every team's current Elo/form state, as of the most recently
    ingested data. Used by simulate.py to score hypothetical future
    matchups (e.g. a quarterfinal between two teams not yet paired).

    Reuses the same walk-and-update mechanics as the other two
    functions, but returns only the final state, not per-match rows.
    """
    matches = _all_matches_chronological(session, tournament_id)
    state: dict[int, TeamState] = {}

    def get_state(team_id: int) -> TeamState:
        return state.setdefault(team_id, TeamState())

    for match in matches:
        home = get_state(match.home_team_id)
        away = get_state(match.away_team_id)
        _apply_result(home, away, match)

    return state