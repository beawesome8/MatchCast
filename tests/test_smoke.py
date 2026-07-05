"""Phase 0 smoke tests: the package imports and config loads."""

from matchcast import __version__
from matchcast.config import Settings


def test_package_imports():
    assert __version__


def test_settings_have_safe_defaults(monkeypatch):
    # Explicitly clear these so the test is correct regardless of what
    # the OS, shell, VS Code, or CI happens to have set in the real
    # environment — GitHub Actions sets DATABASE_URL as a real env var
    # too, so "no ambient env vars" was never a safe assumption.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("FOOTBALL_DATA_TOKEN", raising=False)

    s = Settings(_env_file=None)
    assert s.database_url.startswith("sqlite")
    assert 0 < s.api_calls_per_minute <= 10