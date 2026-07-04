"""Ingestion pipeline: fetch -> normalize -> validate -> upsert.

Two properties matter here more than any individual line:

1.  **Idempotent.** Running this job twice produces the same database
    state as running it once (matches are keyed on source_match_id and
    updated in place). Scheduled jobs get retried, crons overlap,
    humans re-run things — a loader that duplicates rows on re-run is
    a time bomb.

2.  **Fail-closed.** If the batch fails validation, *nothing* is
    loaded; the raw batch is written to the quarantine table with the
    failure reasons. Bad data never reaches the tables the model
    trains on.

Run manually with:  python -m matchcast.ingest
"""

import json
from datetime import datetime

import pandas as pd
import pandera as pa

from matchcast.clients.football_data import FootballDataClient
from matchcast.db import get_session_factory, init_db
from matchcast.models import IngestionQuarantine, Match, Team
from matchcast.validation import MATCH_BATCH_SCHEMA

SOURCE = "football-data.org"


def normalize_match(raw: dict) -> dict:
    """Flatten one API match object into our canonical row shape."""
    score = raw.get("score") or {}
    full_time = score.get("fullTime") or {}
    return {
        "source_match_id": raw["id"],
        "tournament_id": "WC2026",
        "stage": raw.get("stage"),
        "status": raw["status"],
        "home_team_source_id": raw["homeTeam"]["id"],
        "away_team_source_id": raw["awayTeam"]["id"],
        "home_team_name": raw["homeTeam"]["name"],
        "away_team_name": raw["awayTeam"]["name"],
        "home_team_tla": raw["homeTeam"].get("tla"),
        "away_team_tla": raw["awayTeam"].get("tla"),
        "home_goals": full_time.get("home"),
        "away_goals": full_time.get("away"),
        "winner": score.get("winner"),
        "kickoff_utc": datetime.fromisoformat(raw["utcDate"].replace("Z", "+00:00")),
    }


def _upsert_teams(session, df: pd.DataFrame) -> dict[int, int]:
    """Insert/update teams; return mapping source_team_id -> our team id."""
    seen: dict[int, tuple[str, str | None]] = {}
    for side in ("home", "away"):
        for _, row in df.iterrows():
            sid = int(row[f"{side}_team_source_id"])
            tla = row[f"{side}_team_tla"]
            seen[sid] = (row[f"{side}_team_name"], None if pd.isna(tla) else tla)

    mapping: dict[int, int] = {}
    for sid, (name, tla) in seen.items():
        team = session.query(Team).filter_by(source_team_id=sid).one_or_none()
        if team is None:
            team = Team(source_team_id=sid, name=name, tla=tla)
            session.add(team)
            session.flush()  # assigns team.id
        else:
            team.name, team.tla = name, tla
        mapping[sid] = team.id
    return mapping


def _upsert_matches(session, df: pd.DataFrame, team_ids: dict[int, int]) -> int:
    count = 0
    for _, row in df.iterrows():
        source_id = int(row["source_match_id"])
        match = session.query(Match).filter_by(source_match_id=source_id).one_or_none()
        if match is None:
            match = Match(source_match_id=source_id)
            session.add(match)
        match.tournament_id = row["tournament_id"]
        match.stage = None if pd.isna(row["stage"]) else row["stage"]
        match.status = row["status"]
        match.kickoff_utc = pd.Timestamp(row["kickoff_utc"]).to_pydatetime()
        match.home_team_id = team_ids[int(row["home_team_source_id"])]
        match.away_team_id = team_ids[int(row["away_team_source_id"])]
        match.home_goals = None if pd.isna(row["home_goals"]) else int(row["home_goals"])
        match.away_goals = None if pd.isna(row["away_goals"]) else int(row["away_goals"])
        match.winner = None if pd.isna(row["winner"]) else row["winner"]
        count += 1
    return count


def run_ingest(session_factory, client: FootballDataClient) -> dict:
    raw_matches = client.get_competition_matches("WC")
    if not raw_matches:
        return {"fetched": 0, "loaded": 0, "quarantined": False}

    rows = [normalize_match(m) for m in raw_matches]
    df = pd.DataFrame(rows)

    with session_factory() as session:
        try:
            df = MATCH_BATCH_SCHEMA.validate(df, lazy=True)
        except pa.errors.SchemaErrors as exc:
            session.add(
                IngestionQuarantine(
                    source=SOURCE,
                    reason=exc.failure_cases.to_string()[:5000],
                    payload=json.dumps(rows, default=str)[:200_000],
                )
            )
            session.commit()
            return {"fetched": len(rows), "loaded": 0, "quarantined": True}

        team_ids = _upsert_teams(session, df)
        loaded = _upsert_matches(session, df, team_ids)
        session.commit()

    return {"fetched": len(rows), "loaded": loaded, "quarantined": False}


if __name__ == "__main__":
    init_db()
    with FootballDataClient() as api:
        print(run_ingest(get_session_factory(), api))