from sqlalchemy import func, select
from tests.conftest import api_match

from matchcast.ingest import run_ingest
from matchcast.models import IngestionQuarantine, Match, Team


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def get_competition_matches(self, competition="WC"):
        return self.payload


GOOD_PAYLOAD = [
    api_match(101, 1, 2, home_goals=2, away_goals=1, winner="HOME_TEAM"),
    api_match(102, 3, 4, home_goals=0, away_goals=0, winner="DRAW"),
    api_match(103, 1, 3, status="TIMED"),
]


def _tbd_match(match_id: int) -> dict:
    """A future knockout slot where teams aren't determined yet."""
    return {
        "id": match_id,
        "utcDate": "2026-07-09T18:00:00Z",
        "status": "TIMED",
        "stage": "QUARTER_FINALS",
        "homeTeam": {"id": None, "name": None, "tla": None},
        "awayTeam": {"id": None, "name": None, "tla": None},
        "score": {"winner": None, "fullTime": {"home": None, "away": None}},
    }


def _count(session_factory, model):
    with session_factory() as s:
        return s.execute(select(func.count()).select_from(model)).scalar_one()


def test_ingest_loads_matches_and_teams(session_factory):
    summary = run_ingest(session_factory, FakeClient(GOOD_PAYLOAD))
    assert summary == {"fetched": 3, "loaded": 3, "skipped": 0, "quarantined": False}
    assert _count(session_factory, Match) == 3
    assert _count(session_factory, Team) == 4


def test_ingest_is_idempotent(session_factory):
    run_ingest(session_factory, FakeClient(GOOD_PAYLOAD))
    run_ingest(session_factory, FakeClient(GOOD_PAYLOAD))
    assert _count(session_factory, Match) == 3
    assert _count(session_factory, Team) == 4


def test_rerun_updates_result_in_place(session_factory):
    run_ingest(session_factory, FakeClient(GOOD_PAYLOAD))
    finished = [api_match(103, 1, 3, home_goals=1, away_goals=0, winner="HOME_TEAM")]
    run_ingest(session_factory, FakeClient(finished))
    with session_factory() as s:
        match = s.query(Match).filter_by(source_match_id=103).one()
        assert match.status == "FINISHED"
        assert (match.home_goals, match.away_goals) == (1, 0)


def test_bad_batch_is_quarantined_and_not_loaded(session_factory):
    bad = [api_match(201, 5, 6, status="FINISHED")]  # finished but no scores
    summary = run_ingest(session_factory, FakeClient(bad))
    assert summary["quarantined"] is True
    assert _count(session_factory, Match) == 0
    assert _count(session_factory, IngestionQuarantine) == 1


def test_tbd_knockout_slots_are_skipped_not_quarantined(session_factory):
    payload = GOOD_PAYLOAD + [_tbd_match(901), _tbd_match(902)]
    summary = run_ingest(session_factory, FakeClient(payload))
    assert summary == {"fetched": 5, "loaded": 3, "skipped": 2, "quarantined": False}
    assert _count(session_factory, Match) == 3  # only the 3 real matches


def test_all_tbd_batch_returns_early_without_touching_db(session_factory):
    payload = [_tbd_match(901), _tbd_match(902)]
    summary = run_ingest(session_factory, FakeClient(payload))
    assert summary == {"fetched": 2, "loaded": 0, "skipped": 2, "quarantined": False}
    assert _count(session_factory, Match) == 0