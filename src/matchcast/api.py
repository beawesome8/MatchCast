"""FastAPI serving layer — a thin HTTP wrapper around serving.py.
All prediction logic lives there so it's testable without spinning
up a real HTTP server."""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException

from matchcast.db import get_session_factory, init_db
from matchcast.monitoring import get_performance_summary
from matchcast.registry import get_current_champion
from matchcast.serving import get_upcoming_predictions

init_db()
app = FastAPI(title="MatchCast", version="0.1.0")


@app.get("/health")
def health():
    with get_session_factory()() as session:
        try:
            champion = get_current_champion(session)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"database unreachable: {exc}") from exc

    return {
        "status": "ok",
        "champion_model_version_id": champion.id if champion else None,
        "champion_holdout_brier": champion.holdout_brier if champion else None,
    }


@app.get("/predictions/upcoming")
def predictions_upcoming():
    with get_session_factory()() as session:
        try:
            predictions = get_upcoming_predictions(session)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"count": len(predictions), "predictions": [vars(p) for p in predictions]}


@app.get("/monitoring/performance")
def monitoring_performance():
    with get_session_factory()() as session:
        summary = get_performance_summary(session)
    return asdict(summary)