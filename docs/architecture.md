# Architecture

## System Pipeline

```text
Raw Traffic Event Data
        ↓
Data Cleaning and Timestamp Normalization
        ↓
Corridor-Hour and Spatial-Cluster Time-Series Datasets
        ↓
Lag / Rolling / Calendar / Spatial Risk Feature Engineering
        ↓
Two-Tier Forecasting Layer
  (primary: spatial-cluster hurdle model
   fallback: corridor-hour hurdle model)
        ↓
Coordinate-Aware Feature Store
        ↓
Location Resolver
        ↓
KMeans Spatial Cluster Fallback
        ↓
Forecast Risk Score
        ↓
Event Impact Score
        ↓
Crowd + Weather Adjustment
        ↓
Calibrated EIS Score
        ↓
Final Operational Risk
        ↓
Map Impact Circles + Deployment Recommendation
        ↓
Feedback Collection + Validation Evidence
```

## Why Coordinate-First

Earlier versions of this system depended on the user manually selecting a corridor name. That approach was weak because real incidents can happen:

- between corridors
- near service roads
- near junctions
- near temporary event venues
- at unknown or new locations

The system was upgraded to work directly from coordinates instead:

- latitude
- longitude
- nearest corridor
- nearest hotspot
- nearest spatial cluster
- spatial density
- cluster-hour fallback profile

The user now clicks on the map or enters coordinates directly. The backend infers the corridor internally — there is no need to know corridor names in advance.

## Main User Flow

1. User opens the dashboard.
2. User clicks an event location on the map.
3. Latitude and longitude are auto-filled.
4. User enters event details (cause, vehicle type, road closure, crowd size, weather).
5. Backend validates the coordinates against Bengaluru bounds and any restricted zones.
6. Backend infers the nearest corridor and spatial cluster, with a confidence level.
7. The feature store supplies historical lag, rolling, and calendar-aware features for that location and hour.
8. The system selects the primary spatial-cluster model when location confidence is high, or the corridor-hour fallback model otherwise, and runs the hurdle prediction.
9. The event scoring layer calculates live impact from crowd, weather, and event cause.
10. The EIS layer combines forecast risk, live event impact, and historical cause risk.
11. The dashboard shows the final decision: risk, affected area, deployment, and diversion — including a live-ranked diversion candidate table.

## What Is ML vs. Statistical vs. Rule-Based

Niyantrak is intentionally a hybrid system. Traffic operations need explainable recommendations, not only black-box predictions.

**Machine Learning**
- `CatBoostClassifier` (alert/incident-likelihood classifier) — trained twice, once per forecasting tier
- `CatBoostRegressor` (positive incident-count regressor) — trained twice, once per forecasting tier
- CatBoost quantile regressors (prediction intervals)
- KMeans spatial clustering
- Time-series cross-validation
- Feature importance analysis

**Statistical Feature Store**
- Lag features
- Rolling features
- Corridor-hour profiles
- Cluster-hour profiles
- Risk percentiles
- Spatial density
- Hotspot points
- Calendar-aware event context

**Rule-Based Operational Intelligence**
- Event cause weights
- Vehicle type weights
- Road closure boost
- Crowd multiplier
- Weather multiplier
- EIS formula
- Resource allocation rules
- Affected radius rules
- Diversion graph + live pressure-score ranking
- Deployment order generation
- Feedback storage

## Project Structure

```text
NIYANTRAK/
config.py
train_all.py
requirements.txt

scripts/
    predict.py
    prepare_feature_store.py
    train_spatial_model.py
    retrain_30_days.py

src/
    preprocessing/
        load_data.py
        clean.py

    features/
        event_calender.py
        feedback_training.py

    forecasting/
        build_timeseries_dataset.py
        build_spatial_timeseries_dataset.py
        cross_validate_timeseries.py
        train_timeseries_model.py
        train_spatial_timeseries_model.py
        forecast_predictor.py
        spatial_forecast_predictor.py
        forecast_feature_importance.py
        train_quantile_intervals.py

    inference/
        feature_store.py
        location_resolver.py
        location_validity_guard.py
        police_station_resolver.py
        active_event_memory.py
        predict_traffic_risk.py
        similar_events.py

    scoring/
        event_impact.py
        risk_score.py

    recommendation/
        resource_recommender.py

    routing/
        diversion_engine.py

    evaluation/
        cluster_fallback_ablation.py
        eis_weight_calibration.py

traffic_web/
    settings.py
    urls.py
    wsgi.py
    asgi.py

dashboard/
    services/
        ml_engine.py
        feedback_store.py

    templates/
        dashboard/
            index.html

    static/
        dashboard/
            style.css

models/
    timeseries_forecast_model.pkl
    timeseries_forecast.pkl
    spatial_timeseries_forecast_model.pkl
    traffic_feature_store.pkl
    cluster_fallback_ablation.json
    eis_weight_calibration.json
```

> `traffic_web/` is the Django project package (settings, URLs, WSGI/ASGI entry points) that wraps the `dashboard` app shown above. `train_all.py`, `scripts/predict.py`, and `manage.py` are the canonical entry points — see [Setup & Usage](setup-and-usage.md).

> **Note on `src/recommendation/resource_recommender.py`:** this file exists in the codebase but is not currently called by the dashboard. The resource recommendation logic actually used at prediction time lives in `dashboard/services/ml_engine.py` (`recommend_resources`) — see [Operational Outputs](operational-outputs.md#resource-recommendation). The standalone file uses different logic (keyed by raw predicted volume rather than risk level) and should not be treated as the active implementation.

## Related Docs

- [Data & Features](data-and-features.md) — what feeds into the model
- [Forecasting Model](forecasting-model.md) — how the hurdle models work
- [Location Intelligence](location-intelligence.md) — how coordinates become corridors
- [Event Impact Scoring](event-impact-scoring.md) — how EIS is calculated
- [Operational Outputs](operational-outputs.md) — what the system recommends
