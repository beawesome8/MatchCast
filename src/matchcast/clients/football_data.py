"""football-data.org API client.

The free tier allows 10 calls/minute. We self-throttle below that
(default 6/min, set in config) rather than reacting to HTTP 429s:
polite clients keep their free tier; greedy ones get their token
banned mid-tournament.

The throttle takes `clock` and `sleep` as injectable parameters so
tests can verify the waiting logic without actually waiting.
"""

import time

import httpx

from matchcast.config import settings


class FootballDataClient:
    BASE_URL = "https://api.football-data.org/v4"

    def __init__(
        self,
        token: str | None = None,
        calls_per_minute: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        cpm = calls_per_minute or settings.api_calls_per_minute
        self._min_interval = 60.0 / cpm
        self._last_request_at: float | None = None  # None = no request made yet
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers={"X-Auth-Token": token if token is not None else settings.football_data_token},
            timeout=20.0,
            transport=transport,
        )

    def _throttle(self, clock=time.monotonic, sleep=time.sleep) -> None:
        if self._last_request_at is not None:
            wait = self._last_request_at + self._min_interval - clock()
            if wait > 0:
                sleep(wait)
        self._last_request_at = clock()

    def get_competition_matches(self, competition: str = "WC") -> list[dict]:
        """All matches for a competition (past and scheduled)."""
        self._throttle()
        resp = self._client.get(f"/competitions/{competition}/matches")
        resp.raise_for_status()
        return resp.json()["matches"]

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()