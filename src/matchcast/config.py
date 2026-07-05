"""Central configuration.

All environment-dependent values live here, loaded from environment
variables (or a local .env file). Code imports `settings` — nothing
else in the codebase reads os.environ directly.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    football_data_token: str = Field(default="", repr=False)
    database_url: str = Field(default="sqlite:///./local.db", repr=False)

    # football-data.org free tier: 10 calls/minute. We throttle below
    # that to leave headroom.
    api_calls_per_minute: int = 6

    # Promotion gate: challenger may be at most this much worse
    # (Brier score, lower is better) before it is rejected.
    promotion_brier_tolerance: float = 0.0


settings = Settings()