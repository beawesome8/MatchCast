"""The retraining pipeline: ingest -> build features -> train challenger
-> compare to champion -> promote or reject.

This is the file that runs on a schedule (see .github/workflows/retrain.yml).
It orchestrates Phases 1-2 into one decision, but contains no new modeling
or validation logic itself — everything it calls was already independently
tested. This file's own tests focus entirely on the PROMOTION DECISION,
using fakes for ingestion/training so they run in milliseconds and never
touch the network.

Promotion rule:
  - No champion exists yet: promote if the challenger beats the random
    guess baseline. A first model that can't beat a coin flip shouldn't
    become champion just because it's first.
  - Champion exists: promote only if challenger's holdout Brier is better
    than the champion's by at least `promotion_brier_tolerance`. Ties and
    small improvements within tolerance are NOT promoted — this prevents
    the champion flapping between near-identical models on noisy data.
  - Anything else: reject, with a specific, readable reason recorded.
"""

from dataclasses import dataclass

from matchcast.clients.football_data import FootballDataClient
from matchcast.config import settings
from matchcast.features import build_feature_table
from matchcast.ingest import run_ingest
from matchcast.registry import get_current_champion, promote_to_champion, register_model
from matchcast.train import TrainingResult, train_model


@dataclass
class PipelineResult:
    ingest_summary: dict
    n_feature_rows: int
    challenger_result: TrainingResult
    decision: str  # "promoted" or "rejected"
    reason: str
    model_version_id: int


def decide_promotion(
    challenger: TrainingResult,
    champion: TrainingResult | None,
    tolerance: float = 0.0,
) -> tuple[bool, str]:
    """Pure decision logic, deliberately separated from any I/O so it
    can be tested exhaustively without a database or a trained model."""
    if not challenger.beats_random_baseline:
        return False, (
            f"challenger holdout Brier {challenger.holdout_brier:.4f} does not "
            "beat the random-guess baseline; rejected regardless of champion"
        )

    if champion is None:
        return True, "no existing champion; challenger beats random baseline"

    improvement = champion.holdout_brier - challenger.holdout_brier
    if improvement >= tolerance:
        return True, (
            f"challenger holdout Brier {challenger.holdout_brier:.4f} improves on "
            f"champion's {champion.holdout_brier:.4f} by {improvement:.4f} "
            f"(tolerance {tolerance:.4f})"
        )

    return False, (
        f"challenger holdout Brier {challenger.holdout_brier:.4f} does not improve "
        f"on champion's {champion.holdout_brier:.4f} by the required tolerance "
        f"{tolerance:.4f} (actual improvement: {improvement:.4f})"
    )


def run_pipeline(session_factory, client: FootballDataClient) -> PipelineResult:
    with session_factory() as session:
        ingest_summary = run_ingest(session_factory, client)

        rows = build_feature_table(session)
        _, challenger = train_model(rows)

        current_champion_entry = get_current_champion(session)
        champion_result = (
            TrainingResult(
                model_path=current_champion_entry.model_path,
                data_hash=current_champion_entry.data_hash,
                n_train=current_champion_entry.n_train,
                n_holdout=current_champion_entry.n_holdout,
                train_brier=current_champion_entry.train_brier,
                holdout_brier=current_champion_entry.holdout_brier,
                holdout_log_loss=current_champion_entry.holdout_log_loss,
                beats_random_baseline=current_champion_entry.beats_random_baseline,
                trained_at="",  # not needed for comparison
            )
            if current_champion_entry is not None
            else None
        )

        should_promote, reason = decide_promotion(
            challenger, champion_result, tolerance=settings.promotion_brier_tolerance
        )

        if should_promote:
            entry = register_model(
                session, challenger, status="rejected", rejection_reason="pending promotion"
            )
            promote_to_champion(session, entry)
            entry.rejection_reason = None
            decision = "promoted"
        else:
            entry = register_model(session, challenger, status="rejected", rejection_reason=reason)
            decision = "rejected"

        session.commit()

        return PipelineResult(
            ingest_summary=ingest_summary,
            n_feature_rows=len(rows),
            challenger_result=challenger,
            decision=decision,
            reason=reason,
            model_version_id=entry.id,
        )


if __name__ == "__main__":
    import json
    from dataclasses import asdict

    from matchcast.db import get_session_factory, init_db

    init_db()
    with FootballDataClient() as api:
        result = run_pipeline(get_session_factory(), api)
    print(json.dumps(asdict(result), indent=2, default=str))