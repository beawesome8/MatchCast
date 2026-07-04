"""Data validation: the gate between the outside world and our database.

Everything the API returns is treated as untrusted until it passes
this schema. `strict=True` means the batch must have exactly these
columns — an upstream API change fails loudly here instead of
corrupting training data silently three phases later.

Validation runs with lazy=True so a failing batch reports *all* its
problems at once, not just the first one.
"""

import pandera as pa
from pandera import Check, Column

# Statuses football-data.org can return for a match.
ALLOWED_STATUSES = [
    "SCHEDULED",
    "TIMED",
    "IN_PLAY",
    "PAUSED",
    "FINISHED",
    "POSTPONED",
    "SUSPENDED",
    "CANCELLED",
    "AWARDED",
]


def _finished_matches_have_scores(df):
    return (df["status"] != "FINISHED") | (df["home_goals"].notna() & df["away_goals"].notna())


def _teams_are_distinct(df):
    return df["home_team_source_id"] != df["away_team_source_id"]


MATCH_BATCH_SCHEMA = pa.DataFrameSchema(
    columns={
        "source_match_id": Column(int, Check.gt(0), unique=True),
        "tournament_id": Column(str),
        "stage": Column(str, nullable=True),
        "status": Column(str, Check.isin(ALLOWED_STATUSES)),
        "home_team_source_id": Column(int, Check.gt(0)),
        "away_team_source_id": Column(int, Check.gt(0)),
        "home_team_name": Column(str),
        "away_team_name": Column(str),
        "home_team_tla": Column(str, nullable=True),
        "away_team_tla": Column(str, nullable=True),
        "home_goals": Column("Int64", Check.ge(0), nullable=True),
        "away_goals": Column("Int64", Check.ge(0), nullable=True),
        "winner": Column(str, nullable=True),
        "kickoff_utc": Column(nullable=False),
    },
    checks=[
        Check(_finished_matches_have_scores, error="FINISHED matches must have both scores"),
        Check(_teams_are_distinct, error="a team cannot play itself"),
    ],
    strict=True,
    coerce=True,
)