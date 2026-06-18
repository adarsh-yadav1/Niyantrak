# Coordinate-First Event-Driven Traffic Intelligence Engine

A machine learning based traffic intelligence system for forecasting event-driven congestion impact, estimating operational risk, recommending traffic deployment, and visualizing affected areas on a live map.

The system is designed for scenarios such as:

* Accidents
* Vehicle breakdowns
* Congestion
* VIP movements
* Public events
* Processions
* Protests
* Construction
* Water logging
* Road closures

The core idea is simple:

```text
User selects event location on map
        ↓
System extracts latitude and longitude
        ↓
Nearest corridor and spatial cluster are inferred automatically
        ↓
ML model forecasts traffic incident risk
        ↓
Event impact is quantified
        ↓
Dashboard shows risk, affected area, deployment, diversion, and evidence
```

---

## 1. Project Objective

Current traffic planning is often reactive and experience-driven. Event impact is usually not quantified in advance, and post-event feedback is rarely stored in a structured way.

This system addresses that by providing:

```text
1. Coordinate-first traffic risk prediction
2. Event impact score from 0 to 100
3. Forecasted incident volume
4. Final operational risk level
5. Officer and barricade recommendation
6. Map-based affected area visualization
7. Diversion recommendation
8. Similar historical event lookup
9. Model validation metrics on dashboard
10. Post-event feedback collection for future retraining
```

---

## 2. Final System Architecture

```text
Raw Traffic Event Data
        ↓
Data Cleaning and Timestamp Normalization
        ↓
Corridor-Hour Time-Series Dataset
        ↓
Lag / Rolling / Spatial Risk Feature Engineering
        ↓
Zero-Inflated CatBoost Hurdle Model
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

---

## 3. Why Coordinate-First?

Earlier versions depended heavily on the user selecting a corridor name manually.

That was weak because real incidents can happen:

```text
between corridors
near service roads
near junctions
near temporary event venues
at unknown or new locations
```

So the system was upgraded to use:

```text
latitude
longitude
nearest corridor
nearest hotspot
nearest spatial cluster
spatial density
cluster-hour fallback profile
```

Now the user does not need to know the corridor name. The user can click on the map or enter coordinates, and the backend infers the corridor internally.

---

## 4. Main User Flow

```text
1. User opens dashboard
2. User clicks event location on map
3. Latitude and longitude are auto-filled
4. User enters event details
5. Backend validates Bengaluru bounds
6. Backend infers nearest corridor and spatial cluster
7. Feature store supplies historical lag/rolling features
8. Hurdle model predicts incident risk
9. Event scoring layer calculates live impact
10. EIS layer combines forecast + event + cause risk
11. Dashboard shows final decision
```

---

## 5. Dataset Structure

The raw dataset contains traffic event records.

Important columns:

```text
id
event_type
latitude
longitude
endlatitude
endlongitude
address
end_address
event_cause
requires_road_closure
start_datetime
end_datetime
status
corridor
priority
veh_type
veh_no
junction
zone
police_station
```

Each row represents one traffic-related event.

Example:

```text
accident on ORR East 1 at 09:12
vehicle breakdown near CBD at 18:30
public event near Mysore Road at 17:00
```

---

## 6. Time-Series Dataset Generation

Raw events are converted into a corridor-hour time-series dataset.

Raw format:

```text
one row = one event
```

Training format:

```text
one row = one corridor-hour
```

Example:

```text
Raw events:
ORR East 1 | 2024-04-01 09:12 | accident
ORR East 1 | 2024-04-01 09:47 | congestion

After aggregation:
ORR East 1 | 2024-04-01 09:00 | incident_count = 2
```

Target column:

```text
incident_count
```

This means:

```text
number of traffic incidents in a corridor during a specific hour
```

---

## 7. Zero-Heavy Target Distribution

The dataset is highly imbalanced.

Most corridor-hour rows have no incidents:

```text
incident_count = 0 for more than 90% of rows
```

This is why a normal regression model is not enough.

A standard regressor may learn to predict near-zero values most of the time and miss rare traffic spikes.

To handle this, the system uses a zero-inflated hurdle model.

---

## 8. ML Model Architecture

The forecasting model is a:

```text
Zero-Inflated CatBoost Hurdle Model
```

It has two stages:

```text
Stage 1: CatBoostClassifier
Stage 2: CatBoostRegressor
```

### Stage 1 — Alert Classifier

The classifier predicts whether a corridor-hour will have any incident.

Target:

```text
alert_target = 1 if incident_count > 0 else 0
```

Output:

```text
alert_probability
```

Example:

```text
alert_probability = 0.72
```

Meaning:

```text
The model estimates a 72% chance that this corridor-hour will have at least one incident.
```

### Stage 2 — Positive Count Regressor

The regressor predicts expected incident count when risk exists.

The final predicted incident count is controlled by the classifier probability.

Conceptually:

```text
if alert probability is low:
    predicted incidents ≈ 0

if alert probability is high:
    positive-count regressor output is used
```

This structure works better for sparse traffic data than a single regressor.

---

## 9. Time-Series Features

The model uses time-based features:

```text
hour
weekday
month
hour_sin
hour_cos
```

### Why use `hour_sin` and `hour_cos`?

Hour is cyclical.

```text
23:00 and 00:00 are close in real life
but numerically 23 and 0 look far apart
```

Cyclical encoding solves this.

---

## 10. Lag Features

The model uses lag features:

```text
lag_1
lag_2
lag_3
lag_24
lag_48
lag_72
lag_168
```

Meaning:

```text
lag_1    = incident count 1 hour ago
lag_24   = same hour yesterday
lag_168  = same hour last week
```

Why useful:

Traffic has memory. If a corridor recently had incidents, congestion may continue into later hours.

---

## 11. Rolling Features

Rolling features:

```text
rolling_6
rolling_12
rolling_24
rolling_168
```

Meaning:

```text
rolling_6    = average incident count over last 6 hours
rolling_24   = average incident count over last 24 hours
rolling_168  = average incident count over last week
```

These features capture recent corridor pressure.

---

## 12. Spatial and Historical Risk Features

The system also uses:

```text
corridor_avg
corridor_volatility
zone_risk
junction_risk
cause_risk
closure_risk
cluster_risk
```

Meaning:

| Feature               | Meaning                                     |
| --------------------- | ------------------------------------------- |
| `corridor_avg`        | Average incident count for that corridor    |
| `corridor_volatility` | How unstable or spike-prone the corridor is |
| `zone_risk`           | Historical event density in that zone       |
| `junction_risk`       | Historical risk around a junction           |
| `cause_risk`          | Historical frequency/risk of event cause    |
| `closure_risk`        | Historical risk linked to road closure      |
| `cluster_risk`        | Spatial cluster event density               |

These features make the model more location-aware.

---

## 13. Coordinate-Aware Feature Store

File:

```text
src/inference/feature_store.py
```

The feature store stores historical values required at prediction time.

At prediction time, the user provides:

```text
latitude
longitude
event cause
vehicle type
road closure
crowd size
weather
```

But the model needs:

```text
lag features
rolling features
corridor average
corridor volatility
zone risk
junction risk
cause risk
closure risk
cluster risk
```

The feature store supplies these values.

Stored objects include:

```text
corridor_hour_profiles
corridor_profiles
global_profile
corridor_location_profiles
corridor_location_points
hotspot_points
spatial_cluster_model
spatial_cluster_centers
spatial_cluster_hour_profiles
spatial_cluster_profiles
risk thresholds
```

---

## 14. Location Resolver

File:

```text
src/inference/location_resolver.py
```

The location resolver takes:

```text
latitude
longitude
feature store
```

And returns:

```python
{
    "corridor": "ORR East 1",
    "matched_by": "nearest real corridor historical point",
    "distance_m": 420.5,
    "confidence": "HIGH",
    "spatial_cluster_id": 7,
    "nearest_hotspot_distance_m": 180.4,
    "spatial_density_at_point": 0.72
}
```

It performs:

```text
coordinate validation
Bengaluru boundary validation
nearest real corridor matching
nearest corridor centroid matching
nearest spatial cluster lookup
nearest hotspot distance calculation
spatial density calculation
```

---

## 15. Bengaluru Bounding Box Validation

The system validates that coordinates are inside Bengaluru coverage.

Example invalid input:

```text
Latitude: 28.6139
Longitude: 77.2090
```

This is Delhi, not Bengaluru.

The system stops prediction and returns:

```text
Selected location is outside Bengaluru coverage area.
Please choose a location within Bengaluru.
```

This prevents fake corridor matching for out-of-city coordinates.

---

## 16. KMeans Spatial Cluster Fallback

The system includes a KMeans spatial fallback mechanism.

Why needed:

If a coordinate cannot be matched confidently to a known corridor, the model should not fall back to all-zero lag features.

Bad old behavior:

```text
unknown corridor
        ↓
lag_1 = 0
rolling_24 = 0
corridor_avg = 0
        ↓
weak prediction
```

New behavior:

```text
unknown or weak location
        ↓
find nearest KMeans spatial cluster
        ↓
use cluster-hour historical profile
        ↓
prediction still has meaningful history
```

Fallback order:

```text
1. exact inferred corridor-hour profile
2. nearest inferred corridor-hour profile
3. inferred corridor-level profile
4. spatial cluster-hour profile
5. nearest spatial cluster-hour profile
6. spatial cluster-level profile
7. global fallback profile
```

---

## 17. Cluster Fallback Ablation Study

File:

```text
src/evaluation/cluster_fallback_ablation.py
```

Purpose:

```text
Test whether cluster fallback is useful instead of just claiming it.
```

The ablation compares:

```text
Method A: normal corridor-hour profiles
Method B: forced spatial cluster fallback profiles
```

Recent result:

```text
Rows tested       : 5000
Normal MAE        : 0.1537
Cluster MAE       : 0.2384
MAE Delta         : -0.0848
Improvement       : -55.18%
```

Conclusion:

```text
Cluster fallback is weaker than corridor-hour history when corridor matching is reliable.
Therefore, cluster fallback is not used as a replacement.
It is used only when corridor matching is weak or unavailable.
```

This makes the fallback claim evidence-based.

---

## 18. Forecast Risk Score

The forecast model outputs predicted incident count.

This is converted into a percentage risk score using historical thresholds:

```text
incident_p95
incident_p99
alert_probability
context_multiplier
```

Purpose:

```text
Convert raw predicted count into operational risk percentage.
```

Example:

```text
Predicted incidents: 0.60
Forecast risk score: 38.3%
```

Forecast risk is purely historical/model-based.

If forecast risk is zero, it means:

```text
The historical model does not expect incidents for that corridor-hour.
```

But the final operational risk may still be high if the live event is severe.

---

## 19. Event Impact Score

File:

```text
src/scoring/event_impact.py
```

The event impact score uses live event details:

```text
event_cause
vehicle_type
road_closure
rush_hour
```

Examples:

```text
accident + heavy vehicle + closure       → high impact
vehicle breakdown + LCV + no closure     → moderate impact
public event + large crowd + rush hour   → high impact
```

This captures the immediate operational severity of the event.

---

## 20. Crowd Size Adjustment

User input:

```text
Small  < 500
Medium 500 - 5,000
Large  5,000 - 50,000
Mega   > 50,000
```

Multipliers:

```text
small  → 1.00
medium → 1.08
large  → 1.18
mega   → 1.30
```

Why useful:

Crowd size is a major factor for:

```text
public events
rallies
processions
festivals
sports events
protests
```

The same event with a mega crowd should produce higher impact than the same event with a small crowd.

---

## 21. Weather Adjustment

User input:

```text
Clear
Light Rain
Heavy Rain
Fog
```

Multipliers:

```text
clear       → 1.00
light_rain  → 1.10
heavy_rain  → 1.25
fog         → 1.20
```

Weather affects:

```text
event impact
duration estimate
operational pressure
```

Heavy rain and fog increase traffic instability.

---

## 22. Composite Event Impact Score — EIS

The final Event Impact Score is a 0–100 score.

It combines:

```text
forecast risk
live event impact
historical cause risk
```

Formula:

```text
EIS =
    forecast_weight × forecast_score
  + event_weight × adjusted_event_score
  + cause_weight × cause_risk_score
```

The EIS level is:

```text
0–25     LOW
25–50    MODERATE
50–75    HIGH
75–100   CRITICAL
```

Why EIS matters:

The problem statement asks to quantify event impact in advance.

Instead of only saying:

```text
HIGH risk
```

The system says:

```text
EIS Score: 72.4%
EIS Level: HIGH
```

This gives judges and operators a comparable numeric score.

---

## 23. EIS Weight Micro-Calibration

File:

```text
src/evaluation/eis_weight_calibration.py
```

Purpose:

```text
Avoid choosing EIS weights blindly.
```

The system tests multiple EIS weight combinations against historical event outcomes using a practical severity proxy.

Severity proxy uses:

```text
actual duration
same corridor-hour incident count
road closure status
event cause severity prior
```

Target definition:

```text
Actual severity proxy =
45% duration score
+ 25% same corridor-hour incident volume score
+ 20% road closure score
+ 10% event cause severity prior
```

Candidate formulas are tested, and the lowest-MAE formula is selected.

Outputs:

```text
models/eis_weight_calibration.json
EIS_WEIGHT_CALIBRATION.md
```

Judge-safe explanation:

```text
The EIS weights were not chosen blindly. Multiple candidate formulas were evaluated against historical event outcomes using a proxy severity target. The lowest-MAE formula is used in the dashboard. This proxy can later be replaced with officer-labelled feedback data.
```

---

## 24. Final Operational Risk

The final risk combines:

```text
forecast risk
calibrated EIS score
```

This is important because sometimes the historical forecast is low but the live event is severe.

Example:

```text
Forecast risk: LOW
Event: accident + heavy vehicle + road closure
EIS: HIGH
Final operational risk: HIGH
```

This prevents under-response to serious live events.

---

## 25. Pre-Event vs Post-Event Comparison

The dashboard shows:

```text
normal condition
with event
delta
```

Example:

```text
Incident volume
0.15 → 0.60
+0.46

Risk score
38.3% → 42.5%
+4.2 percentage points
```

Note:

The incident volume delta is shown as an absolute difference, not a percentage, because small baselines can create misleading percentage jumps.

---

## 26. Prediction Interval

The system now uses CatBoost quantile models for prediction intervals.

File:

```text
src/forecasting/train_quantile_intervals.py
```

Two additional CatBoost regressors are trained:

```text
lower quantile model: alpha = 0.10
upper quantile model: alpha = 0.90
```

Together they produce:

```text
80% prediction interval
```

Example:

```text
Expected incidents: 0.60
80% prediction interval: 0.20 – 1.10
```

Because the main model is zero-inflated, the interval is gated by the alert classifier probability.

Meaning:

```text
low alert probability  → interval shrinks toward zero
high alert probability → interval widens
```

This is more honest than using a simple RMSE-based range.

---

## 27. Model Metrics

The dashboard shows both raw and operational metrics.

Important metrics:

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

Example recent results:

```text
MAE              : around 0.15
RMSE             : around 0.49
R²               : around 0.23
Alert Recall     : around 0.71
ROC-AUC          : around 0.85
PR-AUC           : around 0.42
```

---

## 28. Why R² Around 0.23 Is Acceptable

This is not normal continuous regression.

Traffic incidents are rare and spike-driven:

```text
more than 90% of corridor-hour rows are zero
```

Exact count prediction is naturally difficult.

The operational goal is not only perfect incident count regression. The more important goal is:

```text
detect risky corridor-hours early
```

That is why alert metrics matter more:

```text
Alert recall shows how many real incident-hours were caught.
ROC-AUC shows whether risky hours are ranked above normal hours.
PR-AUC is important because positive incidents are rare.
```

Judge-safe explanation:

```text
R² is reported honestly, but it is not the main decision metric. This is a sparse rare-event forecasting problem. The alert classifier is the main operational layer, and it achieves strong risk-ranking performance with ROC-AUC around 0.85.
```

---

## 29. Historical Similar Events

File:

```text
src/inference/similar_events.py
```

When a new event is submitted, the system finds similar historical events using:

```text
event cause match
inferred corridor match
hour similarity
geographic distance
```

Dashboard shows:

```text
similar past event
corridor
time
distance
duration
road closure
similarity score
```

Why useful:

This grounds ML output in actual historical examples.

Example:

```text
Last similar event:
vehicle breakdown near ORR East 1
duration: 72 minutes
road closure: False
distance: 340m
```

---

## 30. Diversion Recommendation

File:

```text
src/routing/diversion_engine.py
```

The diversion engine uses a graph of corridors.

Each corridor is a node.

Edges represent possible route alternatives.

Output:

```text
primary_detour
secondary_detour
support_corridors
diversion_action
```

Example:

```text
Affected corridor : ORR East 1
Primary detour    : ORR East 2
Secondary detour  : Old Airport Road
Support corridors : Varthur Road, CBD 2, Mysore Road
```

---

## 31. Resource Recommendation

The system recommends:

```text
officers
barricades
```

Base logic:

```text
LOW      → 2 officers, 0 barricades
MODERATE → 4 officers, 1 barricade
HIGH     → 6 officers, 2 barricades
CRITICAL → 8 officers, 4 barricades
```

Additional resources are added for:

```text
road closure
high predicted incidents
critical final risk
```

---

## 32. Impact Map Visualization

The dashboard uses Leaflet/OpenStreetMap.

It shows:

```text
event marker
primary impact circle
secondary spillover circle
risk color
legend
popup details
```

Map meaning:

```text
inner circle = primary affected zone
outer circle = secondary spillover zone
marker = event location
```

This visually answers:

```text
What area will be affected?
```

---

## 33. Affected Radius Calculation

The system estimates:

```text
affected_radius_m
secondary_radius_m
```

Based on:

```text
final risk level
predicted incidents
road closure
event cause
```

High-spread causes increase the radius:

```text
accident
congestion
vip_movement
protest
procession
public_event
water_logging
```

---

## 34. Deployment Order

The dashboard generates a formatted deployment order:

```text
TRAFFIC DEPLOYMENT ORDER
----------------------------------------
Risk Level       : HIGH
Risk Score       : 67.20%
Event Cause      : accident
Location         : 12.920000, 77.620000
Inferred Corridor: ORR East 1
Duration Estimate: 95 minutes

RESOURCE PLAN
----------------------------------------
Officers Required: 7
Barricades Needed: 4

DIVERSION PLAN
----------------------------------------
Primary Detour   : ORR East 2
Secondary Detour : Old Airport Road
Support Corridors: Varthur Road, CBD 2

ACTION
----------------------------------------
Deploy officers, prepare barricades, and keep diversion support ready.
```

This makes the output operational, not just analytical.

---

## 35. Post-Event Feedback Collection

The system includes a feedback collection module.

It stores:

```text
actual duration
actual officers deployed
actual barricades used
actual road closure
actual incident count
officer notes
```

Output file:

```text
data/post_event_feedback.csv
```

Important honesty note:

```text
The current prototype stores feedback for audit, analysis, and future retraining.
It does not automatically retrain the ML model after each feedback entry.
```

Correct wording:

```text
feedback collection system with retraining capability planned
```

Avoid saying:

```text
automatic learning loop
self-learning model
continuous retraining
```

---

## 36. Final Training Pipeline

The final training pipeline is run using:

```bash
python train_all.py
```

It performs:

```text
1. Load data
2. Build time-series dataset
3. Run cross-validation
4. Train zero-inflated CatBoost hurdle model
5. Save feature importance
6. Build coordinate-aware feature store
7. Run cluster fallback ablation
8. Train quantile interval models
9. Run EIS weight calibration
```

---

## 37. Generated Files

After training, the following files are generated:

```text
models/timeseries_forecast_model.pkl
models/timeseries_forecast.pkl
models/traffic_feature_store.pkl
models/cluster_fallback_ablation.json
models/eis_weight_calibration.json
EIS_WEIGHT_CALIBRATION.md
forecast_feature_importance.png
```

---

## 38. Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run full training pipeline:

```bash
python train_all.py
```

Run Django dashboard:

```bash
python manage.py runserver
```

Run terminal predictor:

```bash
python predict.py
```

---

## 39. Project Structure

```text
config.py
train_all.py
predict.py
prepare_feature_store.py

src/
    preprocessing/
        load_data.py

    forecasting/
        build_timeseries_dataset.py
        cross_validate_timeseries.py
        train_timeseries_model.py
        forecast_predictor.py
        forecast_feature_importance.py
        train_quantile_intervals.py

    inference/
        feature_store.py
        location_resolver.py
        predict_traffic_risk.py
        similar_events.py

    scoring/
        event_impact.py
        risk_score.py

    routing/
        diversion_engine.py

    evaluation/
        cluster_fallback_ablation.py
        eis_weight_calibration.py

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
    traffic_feature_store.pkl
    cluster_fallback_ablation.json
    eis_weight_calibration.json
```

---

## 40. Dashboard Output

The dashboard shows:

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
Pre-event vs post-event comparison
80% prediction interval
Cluster fallback ablation result
EIS calibration evidence
Similar historical events
Feedback collection form
```

---

## 41. What Is ML and What Is Rule-Based?

### Machine Learning

```text
CatBoostClassifier
CatBoostRegressor
CatBoost quantile regressors
KMeans spatial clustering
Time-series cross-validation
Feature importance
```

### Statistical Feature Store

```text
lag features
rolling features
corridor-hour profiles
cluster-hour profiles
risk percentiles
spatial density
hotspot points
```

### Rule-Based Operational Intelligence

```text
event cause weights
vehicle type weights
road closure boost
crowd multiplier
weather multiplier
EIS formula
resource allocation rules
affected radius rules
diversion graph
deployment order generation
feedback storage
```

This hybrid approach is intentional.

Traffic operations need explainable recommendations, not only black-box predictions.

---

## 42. Strengths of the Final System

```text
coordinate-first prediction
no manual corridor dependency
Bengaluru boundary validation
nearest real corridor resolver
KMeans spatial fallback
ablation-tested fallback mechanism
zero-inflated hurdle model
CatBoost quantile prediction interval
calibrated EIS weights
dashboard model metrics
operational metric translation
R² context explanation
map impact visualization
deployment recommendation
diversion planning
similar historical events
feedback collection for future retraining
```

---

## 43. Known Limitations

```text
real-time traffic API is not yet integrated
SMS/WhatsApp/email alert sending is not yet implemented
feedback is stored but not yet used for automatic retraining
similar event lookup is heuristic
diversion is corridor-level, not live road-network routing
weather is manually selected, not API-driven
crowd size is manually entered
```

---

## 44. Future Improvements

```text
integrate live traffic speed feeds
integrate live weather API
send automatic alerts to officers
use OpenStreetMap road graph for dynamic diversions
use officer feedback for scheduled retraining
add MLflow model registry
add model drift monitoring
add real-time congestion heatmap
add CCTV/GPS/sensor integration
```

---

## 45. One-Line Summary

```text
Map coordinates → location intelligence → hurdle forecast model → calibrated EIS → final operational risk → impact map → deployment and diversion recommendation
```

---

## 46. Judge Explanation

```text
This system predicts event-driven traffic impact using a coordinate-first pipeline. The user selects an exact event location on the map, and the backend infers the nearest real corridor, spatial cluster, hotspot distance, and historical traffic profile.

The forecast model is a zero-inflated CatBoost hurdle model because traffic incidents are rare and more than 90% of corridor-hour rows are zero. First, a classifier estimates whether an incident is likely, and then a regressor estimates the incident count when risk exists.

The system combines the ML forecast with live event severity, road closure, crowd size, weather, and historical cause risk to calculate a calibrated Event Impact Score. The dashboard shows final risk, affected map radius, officers required, barricades required, diversion routes, prediction interval, similar historical events, and a deployment order.

We also validate key design choices. Cluster fallback was tested with an ablation study, and EIS weights were micro-calibrated against historical event outcomes. Post-event feedback is collected for audit and future retraining, but the prototype does not claim automatic retraining yet.
```
