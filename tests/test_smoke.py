"""Phase 0 smoke tests: the package imports and config loads.

These look trivial, but they make CI meaningful from the first
commit — every later phase adds real tests to an already-green
pipeline instead of bolting CI on at the end.
"""

from matchcast import __version__
from matchcast.config import Settings


def test_package_imports():
    assert __version__


def test_settings_have_safe_defaults():
    s = Settings(_env_file=None)
    assert s.database_url.startswith("sqlite")
    assert 0 < s.api_calls_per_minute <= 10
