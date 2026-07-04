# MatchCast — Design

**Goal:** demonstrate production ML operations — automated retraining, safe model promotion, versioning, validation, and monitoring — using the live FIFA World Cup 2026 as the data source. The model is intentionally simple; the operational machinery is the product.

**Constraint:** the tournament ends July 19, 2026. Every component is scoped to ship while live matches remain, because the promotion gate and prediction log only accumulate value from real, pre-registered predictions.

## Data sources

| Source | Role | Cost |
|---|---|---|
| football-data.org (free tier) | Live fixtures, scores, status. 10 calls/min limit — client must throttle. | $0 |
| openfootball worldcup.json | Historical bootstrap (past World Cups + WC26 backfill). Updated ~daily, never used live. | $0 |
| eloratings.net / Kaggle internationals | Pre-tournament team strength (Elo) for feature bootstrap. | $0 |

## Database schema (Postgres)

- `teams` — canonical team registry (id, name, fifa_code)
- `matches` — all matches, historical + WC26 (teams, score, stage, status, kickoff_utc)
- `features` — computed per-match feature vectors, versioned by feature config hash
- `model_versions` — the registry: version, artifact path, training data hash, feature config, metrics, status (champion / rejected / retired)
- `predictions_log` — every prediction, written **before kickoff**: match id, model version, class probabilities, created_at; outcome + Brier contribution filled in after full time
- `ingestion_quarantine` — batches that failed validation, with reason

## Model

- **Task:** 3-class win/draw/loss from the home-listed team's perspective.
- **Features (v1):** Elo differential, rolling form (last 10 points-per-game), rolling goal difference, tournament stage, knockout flag.
- **Knockout handling:** knockout matches cannot end in a draw at the outcome level. The model is trained on 90-minute results (draws exist); for knockout predictions, the draw probability is redistributed to the two teams proportional to their win probabilities, and this is documented in the API response.
- **Trainer:** XGBoost multiclass. Deterministic seeds. Training data snapshot is hashed and recorded in the registry.

## Retraining & promotion (the core of the project)

A scheduled GitHub Actions workflow (cron, after each matchday) runs:

1. **Ingest** new results → validate → load.
2. **Train challenger** on the expanded dataset.
3. **Evaluate** challenger vs. current champion: Brier score and log-loss on (a) a fixed holdout of pre-2026 matches and (b) all WC26 matches predicted so far (from `predictions_log`, re-scored under the challenger counterfactually for comparison only).
4. **Gate:** promote only if challenger Brier ≤ champion Brier + tolerance on both sets. Otherwise record the rejection with metrics.
5. **Log** the decision to the registry either way.

Rationale for incremental retraining over online learning: with ~100 tournament matches total, per-sample weight updates add complexity without benefit; full retrains on an expanding dataset are cheap, reproducible, and auditable.

## Serving

FastAPI. Key endpoints:

- `GET /predictions/upcoming` — probabilities for scheduled matches, with model version
- `GET /simulations/latest` — Monte Carlo tournament-winner probabilities
- `GET /live/{match_id}` — Bayesian in-match win probability (score + time based Poisson update on the pre-match prior)
- `GET /monitoring/performance` — Brier over time per model version, calibration bins, data freshness

## Monitoring

- Brier score and hit rate per model version, computed only on genuinely pre-registered predictions
- Calibration table (predicted probability bins vs. observed frequencies)
- Pipeline health: last successful ingest, last retrain, API error counts

## Out of scope (deliberately)

- Player-level predictions (goal scorer, MOTM) — requires paid event-level data; low signal for an MLOps portfolio
- Hosted MLflow / heavyweight registry — the minimal registry table is appropriate at this scale; the README notes what would replace it at production scale
- Betting-adjacent use — free-tier data latency makes this unsuitable, and it is not the goal
- Real pre-tournament Elo bootstrap (eloratings.net snapshot) — deferred under
  tournament timeline pressure; teams currently seed at neutral 1500 and Elo
  differentials sharpen only as WC2026 matches accumulate. Documented limitation
  for METHODOLOGY.md: holdout Brier only modestly beats the 0.667 random-guess
  baseline early in this dataset's life for exactly this reason.

## Post-tournament plan

Schema is tournament-agnostic (`tournament_id` on matches). After July 19: publish a retrospective (retrain count, gate rejections, final Brier vs. a naive Elo baseline and a coin-flip baseline), then point the pipeline at a league season where incremental retraining has a longer runway.
