"""Client tests run against httpx.MockTransport — no network, no token,
no flakiness. CI must never depend on a third-party API being up."""

import httpx
from tests.conftest import api_match

from matchcast.clients.football_data import FootballDataClient


def _mock_transport(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v4/competitions/WC/matches"
        assert "X-Auth-Token" in request.headers
        return httpx.Response(200, json={"matches": payload})

    return httpx.MockTransport(handler)


def test_client_parses_matches():
    payload = [api_match(1, 10, 20, home_goals=2, away_goals=0, winner="HOME_TEAM")]
    with FootballDataClient(token="test", transport=_mock_transport(payload)) as client:
        matches = client.get_competition_matches("WC")
    assert len(matches) == 1
    assert matches[0]["id"] == 1


def test_throttle_sleeps_when_calls_are_too_close():
    client = FootballDataClient(token="test", calls_per_minute=6)  # min interval = 10s
    sleeps: list[float] = []
    fake_now = iter([0.0, 3.0, 3.0])  # first call at t=0, second at t=3

    client._throttle(clock=lambda: next(fake_now), sleep=sleeps.append)
    client._throttle(clock=lambda: next(fake_now), sleep=sleeps.append)

    assert len(sleeps) == 1
    assert abs(sleeps[0] - 7.0) < 1e-9  # waited the remaining 7s of the 10s window
    client.close()


def test_throttle_does_not_sleep_when_interval_has_passed():
    client = FootballDataClient(token="test", calls_per_minute=6)
    sleeps: list[float] = []
    fake_now = iter([0.0, 60.0, 60.0])

    client._throttle(clock=lambda: next(fake_now), sleep=sleeps.append)
    client._throttle(clock=lambda: next(fake_now), sleep=sleeps.append)

    assert sleeps == []
    client.close()