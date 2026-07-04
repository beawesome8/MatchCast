"""Database wiring."""

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from matchcast.config import settings
from matchcast.models import Base


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(settings.database_url, future=True)


def get_session_factory(engine=None):
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False, future=True)


def init_db(engine=None) -> None:
    """Create all tables that don't exist yet. Idempotent."""
    Base.metadata.create_all(engine or get_engine())