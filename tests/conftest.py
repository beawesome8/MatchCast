"""Shared test fixtures.

The in-memory SQLite database with StaticPool gives every test a real
(but throwaway) database: same SQLAlchemy code paths as Postgres, zero
setup, gone when the test ends. CI needs no database service for this.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from matchcast.db import get_session_factory, init_db


@pytest.fixture()
def session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    init_db(engine)
    return get_session_factory(engine)


def api_match(
    match_id: int,
    home_id: int,
    away_id: int,
    status: str = "FINISHED",
    home_goals=None,
    away_goals=None,
    winner=None,
):
    """Build a payload shaped like football-data.org's /matches items."""
    return {
        "id": match_id,
        "utcDate": "2026-06-30T18:00:00Z",
        "status": status,
        "stage": "GROUP_STAGE",
        "homeTeam": {"id": home_id, "name": f"Team {home_id}", "tla": f"T{home_id:02d}"},
        "awayTeam": {"id": away_id, "name": f"Team {away_id}", "tla": f"T{away_id:02d}"},
        "score": {"winner": winner, "fullTime": {"home": home_goals, "away": away_goals}},
    }