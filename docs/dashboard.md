# Dashboard

The dashboard is built with Django: the `traffic_web/` package holds the project-level settings, URLs, and WSGI/ASGI entry points, while the `dashboard/` app holds the actual page. Templates live under `dashboard/templates/dashboard/`, styling in `dashboard/static/dashboard/style.css`, and the core ML/business logic for the dashboard lives in `dashboard/services/ml_engine.py` and `dashboard/services/feedback_store.py`.

## What the Dashboard Shows

```text
Hero operational decision
Final risk level
EIS score
Impact map circles
Officers required
Barricades required
Deployment order
Model metrics in operational language
R² explanation panel
Location intelligence
State-aware diversion ranking
Pre-event vs post-event comparison
80% prediction interval
Cluster fallback ablation result
EIS calibration evidence
Similar historical events
Feedback collection form
```

## Impact Map Visualization

The dashboard uses Leaflet / OpenStreetMap to render:

- event marker
- primary impact circle
- secondary spillover circle
- risk color
- legend
- popup details

Map semantics:

| Element | Meaning |
|---|---|
| Marker | event location |
| Inner circle | primary affected zone |
| Outer circle | secondary spillover zone |

This visually answers: *what area will be affected?*

## Historical Similar Events

File: `src/inference/similar_events.py`

When a new event is submitted, the system finds similar historical events using:

```text
event cause match
inferred corridor match
hour similarity
geographic distance
```

The dashboard shows, for each similar event found:

```text
similar past event
corridor
time
distance
duration
road closure
similarity score
```

**Why this matters:** it grounds the ML output in actual historical examples, rather than presenting a risk score in isolation. Example:

```text
Last similar event:
vehicle breakdown near ORR East 1
duration: 72 minutes
road closure: False
distance: 340m
```

## Model Validation Panels

The dashboard surfaces model metrics and evidence directly, rather than hiding them behind a separate report:

- model metrics (MAE, RMSE, R², Alert Precision/Recall/F1, ROC-AUC, PR-AUC) with a plain-language explanation of why R² is modest (see [Forecasting Model](forecasting-model.md#why-r²-around-023-is-acceptable))
- cluster fallback ablation result (see [Location Intelligence](location-intelligence.md#cluster-fallback-ablation-study))
- EIS weight calibration evidence (see [Event Impact Scoring](event-impact-scoring.md#eis-weight-micro-calibration))

## Historical Context Variables

The dashboard's "Historical Context Variables" panel (lag, rolling, and corridor-average values) is computed from the trained coordinate-aware feature store, refreshed whenever `train_all.py` or `scripts/prepare_feature_store.py` is re-run — it is not recalculated live from a streaming database on every request. This is an intentional design choice for a corridor-history feature, not a limitation specific to this panel; see [Feedback & Retraining](feedback-and-retraining.md) for how this could evolve toward more frequent refresh cycles.

## Related Docs

- [Operational Outputs](operational-outputs.md) — the deployment order, resource plan, and diversion ranking shown on the dashboard
- [Feedback & Retraining](feedback-and-retraining.md) — the feedback form shown on the dashboard
- [Judge / Reviewer Notes](judge-notes.md) — talking points for walking a reviewer through the dashboard
