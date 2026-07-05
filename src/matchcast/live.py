"""Live in-match win-probability updates.

Combines a match's PRE-MATCH prediction (the one already logged in
predictions_log before kickoff — never recomputed with hindsight) with
the current score and time elapsed, to produce an updated in-match
probability.

Model: each team's REMAINING goals are treated as independent Poisson
processes (the standard textbook model for football scoring). The
difference of two independent Poisson variables follows a Skellam
distribution, which lets us compute P(home win) / P(draw) / P(away
win) for the rest of the match given how many goals each side is
still expected to score.

Two documented approximations, not hidden assumptions:

1. Our model predicts win/draw/loss, never goal counts, so there is no
   trained "expected goals" number to draw on. We assume a fixed total
   expected-goals baseline for a full match (2.6, a common rough figure
   for international football) and split it between the two teams in
   proportion to their PRE-MATCH win shares (ignoring the draw
   component, since draws don't indicate which team is stronger).
   This means compute_live_probabilities at minute 0 will NOT exactly
   reproduce the original three probabilities — only their relative
   ordering and rough proportions. It is a reasonable in-game prior,
   not a faithful inverse of the original prediction.

2. football-data.org's free tier delivers delayed scores, not
   real-time — this module's math is instantaneous, but any live score
   it's fed will lag the real match by some seconds to low minutes.
   Fetching that live score is a separate, later integration step,
   deliberately not built here — the math is fully testable without a
   live match; the live fetch is not, and needs a real in-progress game
   to verify.

3. Extra time and penalties for tied knockout matches are NOT modeled
   separately here — this reports the probability of ending regulation
   in each state (home/draw/away), consistent with the rest of the
   app's three-class framing. Resolving a live draw to a shootout
   winner is out of scope.
"""

from dataclasses import dataclass

from scipy.stats import skellam
from sqlalchemy import select
from sqlalchemy.orm import Session

from matchcast.models import PredictionLog

DEFAULT_TOTAL_EXPECTED_GOALS = 2.6
DEFAULT_TOTAL_MINUTES = 90


@dataclass
class LiveProbabilities:
    prob_home: float
    prob_draw: float
    prob_away: float
    minutes_elapsed: float
    remaining_fraction: float


def _expected_goal_split(
    prob_home: float, prob_away: float, total_expected_goals: float
) -> tuple[float, float]:
    """Split a total expected-goals baseline between the two teams,
    proportional to their pre-match win shares (draw excluded — it
    doesn't indicate which side is stronger)."""
    win_total = prob_home + prob_away
    home_share = prob_home / win_total if win_total > 0 else 0.5
    return total_expected_goals * home_share, total_expected_goals * (1 - home_share)


def compute_live_probabilities(
    prob_home: float,
    prob_draw: float,
    prob_away: float,
    home_score: int,
    away_score: int,
    minutes_elapsed: float,
    total_minutes: int = DEFAULT_TOTAL_MINUTES,
    total_expected_goals: float = DEFAULT_TOTAL_EXPECTED_GOALS,
) -> LiveProbabilities:
    minutes_elapsed = max(0.0, min(minutes_elapsed, total_minutes))
    remaining_fraction = max(0.0, (total_minutes - minutes_elapsed) / total_minutes)

    home_xg_total, away_xg_total = _expected_goal_split(prob_home, prob_away, total_expected_goals)
    mu_home = home_xg_total * remaining_fraction
    mu_away = away_xg_total * remaining_fraction

    current_margin = home_score - away_score

    # With zero time remaining, the current score IS the final score —
    # no distribution needed, and scipy's skellam degenerates to nan
    # when both Poisson parameters are exactly zero, so this must be
    # handled explicitly rather than passed through to skellam below.
    if remaining_fraction <= 0.0:
        if current_margin > 0:
            return LiveProbabilities(1.0, 0.0, 0.0, minutes_elapsed, remaining_fraction)
        if current_margin < 0:
            return LiveProbabilities(0.0, 0.0, 1.0, minutes_elapsed, remaining_fraction)
        return LiveProbabilities(0.0, 1.0, 0.0, minutes_elapsed, remaining_fraction)
    
    # Skellam(mu_home, mu_away) models (remaining home goals - remaining
    # away goals). The FINAL margin is current_margin + that difference,
    # so we ask: what's the chance the remaining swing pushes the final
    # margin positive / zero / negative?
    p_draw = skellam.pmf(-current_margin, mu_home, mu_away)
    p_home = skellam.sf(-current_margin, mu_home, mu_away)
    p_away = skellam.cdf(-current_margin - 1, mu_home, mu_away)

    # Exact identity (sf + pmf + cdf-of-one-less = 1) should hold, but
    # normalize defensively against floating-point drift.
    total = p_home + p_draw + p_away
    if total > 0:
        p_home, p_draw, p_away = p_home / total, p_draw / total, p_away / total

    return LiveProbabilities(
        prob_home=float(p_home),
        prob_draw=float(p_draw),
        prob_away=float(p_away),
        minutes_elapsed=minutes_elapsed,
        remaining_fraction=remaining_fraction,
    )


def get_latest_prediction(session: Session, match_id: int) -> PredictionLog | None:
    """The most recently logged pre-match prediction for this match —
    if the champion has changed since it was first predicted, the
    newest logged row is the one to build a live update on."""
    return session.execute(
        select(PredictionLog)
        .where(PredictionLog.match_id == match_id)
        .order_by(PredictionLog.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def get_live_update(
    session: Session,
    match_id: int,
    home_score: int,
    away_score: int,
    minutes_elapsed: float,
) -> LiveProbabilities:
    """Look up the logged pre-match prediction for this match and
    update it with a current score/time snapshot. Score and time are
    passed in explicitly — fetching them from a real live match is a
    separate integration step, not built here (see module docstring)."""
    prediction = get_latest_prediction(session, match_id)
    if prediction is None:
        raise ValueError(f"no logged pre-match prediction found for match_id={match_id}")

    return compute_live_probabilities(
        prob_home=prediction.prob_home,
        prob_draw=prediction.prob_draw,
        prob_away=prediction.prob_away,
        home_score=home_score,
        away_score=away_score,
        minutes_elapsed=minutes_elapsed,
    )
    