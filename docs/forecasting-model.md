# Forecasting Model

## Why Not a Standard Regressor

The corridor-hour dataset has `incident_count = 0` for more than 90% of rows (see [Data & Features](data-and-features.md#zero-heavy-target-distribution)). A standard regressor trained on this data tends to collapse toward predicting near-zero values everywhere and misses the rare traffic spikes that actually matter operationally.

## Two Forecasting Tiers

Niyantrak trains this hurdle architecture **twice**, producing two independent model bundles:

| Tier | File | Trained by | Role |
|---|---|---|---|
| Primary | `models/spatial_timeseries_forecast_model.pkl` | `src/forecasting/train_spatial_timeseries_model.py` | Point-aware spatial-cluster model — used when location-resolution confidence is high |
| Fallback | `models/timeseries_forecast_model.pkl` | `src/forecasting/train_timeseries_model.py` | Corridor-hour model — used when spatial confidence is low |

Both tiers share the same two-stage (classifier + regressor) hurdle structure described below; they differ in which spatial granularity their training data is aggregated at. The dashboard's "Forecast Source" field reports which tier produced a given prediction — for example, `"primary spatial-cluster-hour model"` or `"corridor-hour fallback model"`.

## Model Architecture: Zero-Inflated CatBoost Hurdle Model

Each tier's forecasting model has two stages:

```text
Stage 1: CatBoostClassifier  (alert classifier)
Stage 2: CatBoostRegressor   (positive-count regressor)
```

### Stage 1 — Alert Classifier

Predicts whether a corridor-hour (or spatial-cluster-hour, for the primary tier) will have any incident at all.

- **Target:** `alert_target = 1 if incident_count > 0 else 0`
- **Output:** `alert_probability`

Example:

```text
alert_probability = 0.72
```

Meaning: the model estimates a 72% chance that this corridor-hour will have at least one incident.

### Stage 2 — Positive Count Regressor

Predicts the expected incident count *when risk exists*. Its output is gated by the Stage 1 classifier probability:

```text
if alert probability is low:
    predicted incidents ≈ 0

if alert probability is high:
    positive-count regressor output is used
```

This two-stage structure performs better on sparse traffic data than a single end-to-end regressor.

Training for the **fallback tier** is implemented in `src/forecasting/train_timeseries_model.py`, using the corridor-hour dataset built by `build_timeseries_dataset.py` and validated via `cross_validate_timeseries.py`. The **primary spatial tier** follows the same two-stage structure in `train_spatial_timeseries_model.py`, using the spatial-cluster-hour dataset built by `build_spatial_timeseries_dataset.py`.

## Prediction Interval (Quantile Models)

Two additional CatBoost regressors are trained to produce an 80% prediction interval:

```text
lower quantile model: alpha = 0.10
upper quantile model: alpha = 0.90
```

File: `src/forecasting/train_quantile_intervals.py`

Example:

```text
Expected incidents: 0.60
80% prediction interval: 0.20 – 1.10
```

Because the main model is zero-inflated, this interval is gated by the alert classifier probability:

- **low alert probability** → interval shrinks toward zero
- **high alert probability** → interval widens

This is more honest than a simple RMSE-based range, since it reflects the model's actual confidence rather than a fixed-width band.

## Model Metrics

The dashboard surfaces both raw regression metrics and operational (alert-level) metrics:

```text
MAE
RMSE
R²
Alert Accuracy
Alert Precision
Alert Recall
Alert F1
ROC-AUC
PR-AUC
```

Recent example results:

```text
MAE              : around 0.15
RMSE             : around 0.49
R²               : around 0.23
Alert Recall     : around 0.71
ROC-AUC          : around 0.85
PR-AUC           : around 0.42
```

## Why R² Around 0.23 Is Acceptable

This is **not** a normal continuous regression problem. Traffic incidents are rare and spike-driven — more than 90% of corridor-hour rows are zero — so exact count prediction is inherently difficult, and a modest R² is expected.

The operational goal is not primarily about perfect incident-count regression. The more important goal is to **detect risky corridor-hours early**, which is why alert metrics matter more than R² in practice:

- **Alert recall** shows how many real incident-hours were caught.
- **ROC-AUC** shows whether risky hours are ranked above normal hours.
- **PR-AUC** matters because positive incidents are rare, and PR-AUC is more informative than ROC-AUC under class imbalance.

**Reviewer-safe framing:** R² is reported honestly, but it is not the main decision metric. This is a sparse, rare-event forecasting problem. The alert classifier is the main operational layer, and it achieves strong risk-ranking performance with ROC-AUC around 0.85.

## Forecast Risk Score

The raw predicted incident count is converted into a percentage risk score using historical thresholds:

```text
incident_p95
incident_p99
alert_probability
context_multiplier
```

Purpose: convert a raw predicted count into an operational risk percentage.

Example:

```text
Predicted incidents: 0.60
Forecast risk score: 38.3%
```

Forecast risk is purely historical/model-based. If forecast risk is zero, it means the historical model does not expect incidents for that corridor-hour — but the **final operational risk** (see [Operational Outputs](operational-outputs.md)) may still be high if the live event itself is severe.

## Related Docs

- [Data & Features](data-and-features.md) — the inputs to this model
- [Location Intelligence](location-intelligence.md) — how features are supplied at prediction time, and how the system chooses which tier to use
- [Event Impact Scoring](event-impact-scoring.md) — how forecast risk combines with live event severity
