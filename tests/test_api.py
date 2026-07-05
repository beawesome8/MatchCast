"""API tests using FastAPI's TestClient — calls the app in-process,
no real HTTP server or network socket involved. We patch the session
factory the API uses so tests run against the same throwaway SQLite
database as everything else, never a real one.
"""

from fastapi.testclient import TestClient
from tests.test_serving import _seed_champion, _seed_upcoming_match

import matchcast.api as api_module


def _client(session_factory, monkeypatch):
    monkeypatch.setattr(api_module, "get_session_factory", lambda: session_factory)
    return TestClient(api_module.app)


def test_health_reports_ok_with_no_champion(session_factory, monkeypatch):
    client = _client(session_factory, monkeypatch)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["champion_model_version_id"] is None


def test_health_reports_champion_details(session_factory, monkeypatch):
    with session_factory() as s:
        _seed_champion(s)
        s.commit()

    client = _client(session_factory, monkeypatch)
    response = client.get("/health")
    body = response.json()
    assert body["champion_model_version_id"] is not None
    assert body["champion_holdout_brier"] == 0.5


def test_predictions_upcoming_returns_503_with_no_champion(session_factory, monkeypatch):
    client = _client(session_factory, monkeypatch)
    response = client.get("/predictions/upcoming")
    assert response.status_code == 503
    assert "no champion" in response.json()["detail"]


def test_predictions_upcoming_returns_real_predictions(session_factory, monkeypatch):
    with session_factory() as s:
        _seed_champion(s)
        _seed_upcoming_match(s, home_name="Brazil", away_name="Norway")
        s.commit()

    client = _client(session_factory, monkeypatch)
    response = client.get("/predictions/upcoming")
    assert response.status_code == 200

    body = response.json()
    assert body["count"] == 1
    assert body["predictions"][0]["home_team_name"] == "Brazil"
    
def test_monitoring_performance_returns_summary(session_factory, monkeypatch):
    client = _client(session_factory, monkeypatch)
    response = client.get("/monitoring/performance")
    assert response.status_code == 200
    body = response.json()
    assert "n_predictions_logged" in body
    assert "calibration" in body
    
def test_dashboard_root_serves_html(session_factory, monkeypatch):
    client = _client(session_factory, monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "MatchCast" in response.text