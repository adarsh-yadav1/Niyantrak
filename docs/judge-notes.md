# Judge / Reviewer Notes

This page collects the honest, evidence-based explanations meant for anyone evaluating this project (hackathon judges, interviewers, or collaborators reviewing the design). The goal is to be precise about what's validated, what's assumed, and what's explicitly out of scope — rather than overselling any single component.

## One-Paragraph System Summary

This system predicts event-driven traffic impact using a coordinate-first pipeline. The user selects an exact event location on the map, and the backend infers the nearest real corridor, spatial cluster, hotspot distance, and historical traffic profile.

The forecasting layer runs two tiers of a zero-inflated CatBoost hurdle model — a primary, point-aware spatial-cluster model, and a corridor-hour fallback used when location-resolution confidence is lower — because traffic incidents are rare: more than 90% of corridor-hour rows are zero. In each tier, a classifier first estimates whether an incident is likely, and a regressor then estimates the incident count when risk exists.

The system combines this ML forecast with live event severity, road closure, crowd size, weather, and historical cause risk to calculate a calibrated Event Impact Score (EIS). The dashboard shows final risk, affected map radius, officers required, barricades required, diversion routes with a live state-aware ranking, prediction interval, similar historical events, and a deployment order.

## Design Choices That Were Validated, Not Assumed

| Design Choice | How It Was Validated |
|---|---|
| KMeans spatial cluster fallback | Tested via an ablation study comparing normal corridor-hour profiles vs. forced cluster fallback — see [Location Intelligence](location-intelligence.md#cluster-fallback-ablation-study). Result: cluster fallback is *weaker* than corridor history when corridor matching is reliable, so it's used only as a fallback, not a default. |
| EIS weight formula | Micro-calibrated against historical event outcomes using a practical severity proxy (duration, same-hour incident volume, road closure, cause severity) — see [Event Impact Scoring](event-impact-scoring.md#eis-weight-micro-calibration). The lowest-MAE candidate formula was selected, not chosen by intuition. |
| Zero-inflated hurdle architecture | Chosen specifically because of the >90% zero-incident base rate in both the corridor-hour and spatial-cluster-hour datasets — see [Data & Features](data-and-features.md#zero-heavy-target-distribution) and [Forecasting Model](forecasting-model.md). |
| Two-tier forecasting (spatial primary, corridor-hour fallback) | The system selects whichever tier has higher location-resolution confidence for the given coordinates, rather than relying on a single model regardless of how well the location was resolved — see [Forecasting Model](forecasting-model.md#two-forecasting-tiers). |

## On the R² Metric

R² is reported honestly at around 0.23, and that is **not** treated as a failure to hide. This is a sparse, rare-event forecasting problem — exact incident-count regression is inherently hard when most corridor-hours have zero incidents. The alert classifier (ROC-AUC ≈ 0.85, PR-AUC ≈ 0.42, recall ≈ 0.71) is the layer that actually drives operational decisions, because the practical question is *"will something happen here,"* not *"exactly how many incidents will there be."* Full detail in [Forecasting Model](forecasting-model.md#why-r²-around-023-is-acceptable).

## On the Feedback Loop

The feedback collection module stores post-event outcomes for audit and future retraining. It does **not** currently retrain the model automatically, and feedback is not yet wired into the live training pipeline. Scaffolding toward scheduled (not automatic) retraining already exists — `src/features/feedback_training.py` converts stored feedback into training-ready rows, and `scripts/retrain_30_days.py` can re-run training on a rolling window — but neither is triggered automatically yet. Use "feedback collection system with scheduled-retraining scaffolding in place" — avoid "automatic learning loop" or "self-learning model," which would overstate the current implementation. See [Feedback & Retraining](feedback-and-retraining.md).

## On the Diversion Engine

The diversion engine routes over a predefined corridor graph (nodes = corridors, edges = known alternates), with each candidate route additionally scored live using historical load, alert probability, volatility, and ML-forecasted incident count. It is a planning aid grounded in known corridor relationships plus live risk scoring, not turn-by-turn live road-network routing. See [Operational Outputs](operational-outputs.md#diversion-recommendation).

## On Historical Context Variables

The "Historical Context Variables" shown on the dashboard (lag, rolling, corridor average) are computed from the trained feature store, refreshed on each training run — not recalculated live from a streaming database per request. This is worth stating proactively rather than waiting for the question: it reflects a deliberate scope decision for a coordinate-aware feature store, with a live-refresh pipeline as a natural next step rather than a gap in the modeling approach itself.

## Full List of Known Limitations

See [Limitations & Roadmap](limitations-and-roadmap.md) for the complete, current list — kept separate so it's easy to update as features land.

## Related Docs

- [Architecture](architecture.md)
- [Forecasting Model](forecasting-model.md)
- [Location Intelligence](location-intelligence.md)
- [Event Impact Scoring](event-impact-scoring.md)
- [Limitations & Roadmap](limitations-and-roadmap.md)
