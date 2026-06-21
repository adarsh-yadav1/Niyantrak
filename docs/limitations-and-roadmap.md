# Limitations & Roadmap

This page is the single source of truth for what Niyantrak does **not** yet do. It exists so that every other doc can link here instead of repeating (and risking inconsistent restatements of) the same list.

## Current Limitations

### No live traffic feed

Forecasts are historical-pattern-based (lag, rolling, corridor/cluster profiles) plus whatever live event details the user enters in the form — not live road-speed or live congestion sensor data. The system does not currently ingest a real-time traffic API.

### No automatic alerting

Deployment orders are generated and displayed on the dashboard for an operator to read and act on. They are not automatically sent to officers via SMS, push notification, or any messaging channel.

### Feedback is not yet retraining

Post-event feedback (actual duration, actual officers/barricades used, actual incident count, notes) is stored in `data/post_event_feedback.csv` for audit and analysis. Scaffolding toward scheduled retraining already exists (`src/features/feedback_training.py`, `scripts/retrain_30_days.py` — see [Feedback & Retraining](feedback-and-retraining.md)), but it is not yet wired into `train_all.py` or triggered automatically by new feedback submissions.

### Diversion is corridor-graph based, with live scoring layered on top

The diversion engine routes over a predefined graph of known corridors and their alternates, with each candidate additionally scored live using historical load, alert probability, volatility, and forecasted incidents (see [Operational Outputs](operational-outputs.md#state-aware-diversion-ranking)). It is not live, turn-by-turn, dynamically-routed road-network routing (e.g. it does not query a live routing API or react to real-time road closures it wasn't told about).

### Weather and crowd size are manual inputs

The user selects weather (Clear / Light Rain / Heavy Rain / Fog) and crowd size (Small / Medium / Large / Mega) from the form. There is no live weather API integration and no live crowd-sensing (e.g. from CCTV, mobile density, or ticketing data) yet.

### Two forecasting tiers, manually maintained in parallel

The primary spatial-cluster model and the corridor-hour fallback model are trained by separate scripts (`train_spatial_timeseries_model.py`, `train_timeseries_model.py`) and must both be re-run together via `train_all.py` to stay in sync. There is currently no automated check that flags the two tiers drifting apart if one is retrained without the other.

### Duplicated multiplier logic

Crowd and weather multiplier values are currently defined in two places (`dashboard/services/ml_engine.py` for the dashboard, `src/inference/predict_traffic_risk.py` for the CLI predictor) with identical values today. If either is edited independently in the future without updating the other, the dashboard and CLI predictor could silently diverge — see [Event Impact Scoring](event-impact-scoring.md#crowd-size-adjustment).

### Orphaned resource-recommendation code

`src/recommendation/resource_recommender.py` defines a `recommend_resources()` function that is not called anywhere in the live system — the active implementation lives in `dashboard/services/ml_engine.py`. The unused file should either be removed or clearly marked as inactive to avoid confusion for anyone reading the codebase — see [Architecture](architecture.md#project-structure).

## Future Improvements

- Integrate live traffic speed and weather feeds, replacing the manual weather input and supplementing the historical-pattern forecast with real-time signal
- Send automatic deployment alerts to officers (SMS, push notification, or an internal messaging integration) instead of requiring manual reading of the dashboard
- Use an OpenStreetMap road graph for dynamic diversions, replacing or augmenting the current predefined corridor graph with real road-network routing
- Wire `feedback_training.py`'s output into `retrain_30_days.py` (or directly into `train_all.py` on a schedule) to close the loop from stored feedback to actual scheduled retraining
- Replace the EIS severity proxy (see [Event Impact Scoring](event-impact-scoring.md#severity-proxy)) with real officer-labelled outcomes once enough feedback data has accumulated
- Add model drift monitoring and an MLflow model registry, particularly to track the two forecasting tiers independently
- Add a real-time congestion heatmap and CCTV/GPS/sensor integration for live crowd and traffic density signals
- Consolidate the duplicated crowd/weather multiplier logic into a single shared module used by both the dashboard and the CLI predictor
- Remove or clearly mark the unused `resource_recommender.py` implementation

## How to Read This List

Every item above is phrased as "not yet," not "can't be done" — these are scope boundaries chosen for a hackathon-timeline prototype, not architectural dead ends. Several (the two-tier forecasting split, the feedback storage, the diversion graph) already have the scaffolding in place for their corresponding future improvement; what's missing in those cases is the final wiring step, not the underlying design.

## Related Docs

- [Feedback & Retraining](feedback-and-retraining.md) — full detail on the retraining gap specifically
- [Judge / Reviewer Notes](judge-notes.md) — how to talk about these limitations honestly in a review setting
