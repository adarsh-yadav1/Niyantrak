# ML System Documentation — Event-Driven Traffic Intelligence Engine

## 1. System Purpose

This machine learning system is designed to support traffic operations during planned and unplanned events.

The system answers this operational question:

```text
Given a corridor, timestamp, and live event details,
how risky is the traffic situation and what operational response is needed?
```

The output is not only a prediction. It is a decision-support result.

The ML system produces:

```text
Predicted incident volume
Forecast risk score
Forecast risk level
Live event impact score
Live event impact level
Final operational risk score
Final operational risk level
Officer requirement
Barricade requirement
Diversion recommendation
Affected duration estimate
Historical context used
```

The system is built for event-driven congestion cases such as:

```text
accident
vehicle breakdown
congestion
VIP movement
public event
protest
procession
construction
water logging
road closure
heavy vehicle involvement
```

---

## 2. High-Level ML Architecture

The final ML system is a hybrid architecture.

It combines:

```text
Historical machine learning forecast
+
Live event impact scoring
+
Operational decision rules
```

Full ML pipeline:

```text
Raw Traffic Event Dataset
        ↓
Data Loading and Validation
        ↓
Time-Series Dataset Construction
        ↓
Feature Engineering
        ↓
Zero-Inflated Hurdle Forecast Model
        ↓
Model Evaluation
        ↓
Feature Importance
        ↓
Feature Store Generation
        ↓
Live Prediction Input
        ↓
Forecast Risk Calculation
        ↓
Event Impact Calculation
        ↓
Final Operational Risk Calculation
        ↓
Resource Recommendation
        ↓
Diversion Recommendation
```

The core idea is:

```text
Historical data tells what usually happens.
Live event information tells what is happening now.
Final operational risk combines both.
```

---

## 3. Why We Did Not Use Only One Regression Model

The dataset is sparse.

The target distribution showed:

```text
incident_count
0.0     74072
1.0      4133
2.0       817
3.0       245
4.0       104
5.0        35
6.0        17
7.0        10
8.0         6
13.0        5
12.0        4
10.0        3
22.0        2
14.0        2
9.0         2
```

Zero ratio:

```text
0.9321
```

This means around:

```text
93.21% of corridor-hour rows have zero incidents.
```

This is not a normal regression problem.

A simple model that predicts exact incident count will struggle because:

```text
Most rows are zero.
Positive rows are rare.
Large spikes are very rare.
R² becomes misleading.
A small number of missed spikes can heavily damage R².
```

So we changed the architecture from:

```text
Single CatBoostRegressor
```

to:

```text
Zero-Inflated Hurdle Forecast Model
```

---

## 4. Final ML Model Type

The final forecasting model is a **zero-inflated hurdle model**.

It has two stages:

```text
Stage 1: CatBoostClassifier
Predict whether any incident is likely.

Stage 2: CatBoostRegressor
Predict expected incident count only for positive incident situations.
```

Final count prediction:

```text
expected_count = probability_strength × positive_count_prediction
```

Where:

```text
probability_strength = calibrated strength above alert threshold
```

This avoids predicting small incident counts for many zero rows.

---

## 5. Why Zero-Inflated Hurdle Model Is Suitable

Traffic incident forecasting has many zero values.

Example:

```text
Most corridor-hours have no event.
Some corridor-hours have 1 event.
Rare corridor-hours have multiple incidents.
```

A zero-inflated model handles this better because it separates two questions:

```text
Question 1:
Will anything happen?

Question 2:
If something happens, how much?
```

This is better than forcing one regressor to learn both zero-detection and count estimation.

---

## 6. Active ML File Architecture

The active ML system uses these files:

```text
config.py

train_all.py
prepare_feature_store.py
predict.py

src/preprocessing/load_data.py

src/forecasting/build_timeseries_dataset.py
src/forecasting/cross_validate_timeseries.py
src/forecasting/train_timeseries_model.py
src/forecasting/forecast_predictor.py
src/forecasting/forecast_feature_importance.py

src/inference/feature_store.py
src/inference/predict_traffic_risk.py

src/scoring/event_impact.py
src/scoring/risk_score.py

src/routing/diversion_engine.py
```

These files form the current final ML pipeline.

---

## 7. Optional Old Severity Classifier Files

These files exist but are not part of the current final prediction flow:

```text
src/preprocessing/create_target.py
src/preprocessing/feature_engineering.py
src/preprocessing/advanced_features.py

src/training/train_catboost.py
src/training/cross_validation.py
src/training/feature_importance.py
src/training/shap_analysis.py
```

These were originally used for a severity classification model.

That older model predicted:

```text
LOW
MODERATE
HIGH
CRITICAL
```

from event features.

The current final system instead uses:

```text
forecast model
+
event impact scoring
+
final operational risk scoring
```

So the old severity classifier is optional and not required for the final dashboard or terminal predictor.

---

# Part A — Data Layer

## 8. Dataset Input

The system expects a traffic event dataset.

Important columns:

```text
start_datetime
corridor
latitude
longitude
zone
junction
event_cause
requires_road_closure
veh_type
```

Useful optional columns:

```text
event_type
priority
police_station
end_datetime
endlatitude
endlongitude
```

Each row in the raw dataset represents one event or traffic incident.

Example:

```text
event_cause = vehicle_breakdown
corridor = ORR East 1
start_datetime = 2024-03-01 09:15
latitude = 12.9200
longitude = 77.6200
veh_type = bmtc_bus
requires_road_closure = False
```

---

## 9. `config.py`

Purpose:

```text
Stores common project paths.
```

Main variables:

```python
DATA_PATH = "data/traffic_events.csv"

FORECAST_MODEL_PATH = "models/timeseries_forecast.pkl"

FORECAST_MODEL_COMPAT_PATH = "models/timeseries_forecast_model.pkl"

FEATURE_STORE_PATH = "models/traffic_feature_store.pkl"

FORECAST_FEATURE_IMPORTANCE_PATH = "forecast_feature_importance.png"
```

Why this matters:

```text
All major scripts use the same path configuration.
It prevents path mismatch between training and inference.
```

---

## 10. `src/preprocessing/load_data.py`

Purpose:

```text
Load dataset safely.
Validate file path.
Print dataset shape.
Print dataset columns.
Warn about missing important columns.
```

Supported formats:

```text
CSV
Excel
Parquet
```

This is the first checkpoint in the ML system.

It verifies that the data exists and has the required schema before downstream processing begins.

---

## 11. Datetime Safety

The project encountered a possible timezone bug:

```text
Cannot compare tz-naive and tz-aware timestamps.
```

This happens when some timestamps have timezone information and some do not.

Example:

```text
2024-03-01 09:00:00
2024-03-01 09:00:00+00:00
```

Fix used in the active forecasting pipeline:

```python
pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)
```

This converts all timestamps to UTC and strips timezone information.

Result:

```text
All datetime values become comparable and sortable.
```

This is important because the forecasting model uses chronological train/test splits.

---

# Part B — Time-Series Dataset Builder

## 12. `src/forecasting/build_timeseries_dataset.py`

This is one of the most important ML files.

Purpose:

```text
Convert raw event-level records into corridor-hour time-series data.
```

Raw data format:

```text
one row = one event
```

Model training format:

```text
one row = one corridor-hour
```

Example transformation:

```text
Raw:
ORR East 1, 2024-03-01 09:15, accident
ORR East 1, 2024-03-01 09:40, congestion

After aggregation:
ORR East 1, 2024-03-01 09:00, incident_count = 2
```

---

## 13. Time Bucket Creation

The file creates:

```text
time_bucket
```

using hourly flooring:

```python
df["time_bucket"] = df["start_datetime"].dt.floor("h")
```

Example:

```text
2024-03-01 09:15 → 2024-03-01 09:00
2024-03-01 09:59 → 2024-03-01 09:00
```

---

## 14. Corridor-Hour Aggregation

The system groups by:

```text
time_bucket
corridor
```

and counts events:

```text
incident_count
```

Example:

```text
corridor       time_bucket           incident_count
ORR East 1     2024-03-01 09:00      2
Mysore Road    2024-03-01 09:00      1
CBD 1          2024-03-01 09:00      0
```

---

## 15. Full Corridor-Hour Grid

A major improvement was creating a full grid:

```text
all corridors × all hourly timestamps
```

Why?

If only event hours are kept, the model never learns normal quiet periods.

Bad version:

```text
Only rows where incidents occurred.
```

Correct version:

```text
Every corridor-hour exists.
If no incident happened, incident_count = 0.
```

This created realistic sparse data.

That is why zero ratio became:

```text
93.21%
```

This is expected and correct.

---

## 16. Time Features

For each corridor-hour row, we create:

```text
hour
weekday
month
```

Examples:

```text
hour = 9
weekday = 4
month = 4
```

These allow the model to learn traffic patterns by time.

---

## 17. Cyclic Hour Encoding

The model also uses:

```text
hour_sin
hour_cos
```

Formula:

```text
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
```

Why?

Time is cyclic.

Numerically:

```text
23 and 0 look far apart.
```

But in reality:

```text
23:00 and 00:00 are close.
```

Sine/cosine encoding solves this.

---

## 18. Lag Features

The time-series dataset creates corridor-specific lag features:

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
lag_1   = incidents in previous hour on same corridor
lag_24  = same hour yesterday on same corridor
lag_168 = same hour last week on same corridor
```

Important:

```text
Lags are grouped by corridor.
```

So:

```text
ORR East 1 lag_1 does not use Mysore Road's previous value.
```

This prevents wrong temporal relationships.

---

## 19. Rolling Features

Rolling averages:

```text
rolling_6
rolling_12
rolling_24
rolling_168
```

Meaning:

```text
rolling_6   = average incidents in previous 6 hours
rolling_24  = average incidents in previous 24 hours
rolling_168 = average incidents in previous 7 days
```

Important anti-leakage rule:

```text
shift first, then rolling
```

Reason:

```text
The current row's incident_count must not be included in its own features.
```

Correct logic:

```text
previous values only → rolling mean
```

This prevents target leakage.

---

## 20. Corridor Statistics

The dataset includes:

```text
corridor_avg
corridor_volatility
```

`corridor_avg` means:

```text
average historical incident count for that corridor
```

`corridor_volatility` means:

```text
how unstable or spiky the corridor is
```

A corridor with high volatility may have sudden spikes.

---

## 21. Spatial and Risk Features

The dataset builder also creates:

```text
zone_risk
junction_risk
cause_risk
closure_risk
cluster_risk
```

### zone_risk

Counts historical event activity in a zone.

### junction_risk

Counts historical event activity near junctions.

Junctions are important because they are bottlenecks.

### cause_risk

Captures frequency or risk associated with event causes.

### closure_risk

Captures historical relationship between road closure and incidents.

### cluster_risk

Uses spatial clustering information to capture hotspot density.

These are added so the forecast model does not rely only on time.

---

# Part C — Forecast Features

## 22. Final Forecast Feature Set

The final forecasting model uses:

```text
corridor

hour
weekday
month

hour_sin
hour_cos

lag_1
lag_2
lag_3
lag_24
lag_48
lag_72
lag_168

rolling_6
rolling_12
rolling_24
rolling_168

corridor_avg
corridor_volatility

zone_risk
junction_risk
cause_risk
closure_risk
cluster_risk
```

Target:

```text
incident_count
```

---

## 23. Feature Meaning Summary

| Feature                | Meaning                            |
| ---------------------- | ---------------------------------- |
| `corridor`             | Traffic corridor name              |
| `hour`                 | Hour of day                        |
| `weekday`              | Day of week                        |
| `month`                | Month number                       |
| `hour_sin`, `hour_cos` | Cyclic time encoding               |
| `lag_1`                | Previous hour incident count       |
| `lag_24`               | Same hour previous day             |
| `lag_168`              | Same hour previous week            |
| `rolling_6`            | Previous 6-hour average            |
| `rolling_24`           | Previous 24-hour average           |
| `rolling_168`          | Previous 7-day average             |
| `corridor_avg`         | Average incident level of corridor |
| `corridor_volatility`  | Spike tendency of corridor         |
| `zone_risk`            | Historical zone activity           |
| `junction_risk`        | Historical junction activity       |
| `cause_risk`           | Historical event-cause activity    |
| `closure_risk`         | Road closure-related activity      |
| `cluster_risk`         | Spatial hotspot activity           |

---

# Part D — Model Training

## 24. `src/forecasting/train_timeseries_model.py`

This file trains the final ML forecast model.

It performs:

```text
dataset validation
chronological train/test split
threshold calibration
hurdle model training
holdout evaluation
production model training
model saving
```

---

## 25. Chronological Split

The system does not use random split.

It sorts by:

```text
time_bucket
corridor
```

Then splits by time:

```text
oldest 80% time buckets → train
newest 20% time buckets → test
```

Why?

Traffic forecasting is time-dependent.

Random split would leak future patterns into the past.

Chronological split simulates real deployment:

```text
train on past → predict future
```

---

## 26. Model Stage 1 — CatBoostClassifier

Stage 1 predicts:

```text
incident_count > 0
```

So the target becomes:

```text
0 = no incident
1 = incident
```

Model:

```text
CatBoostClassifier
```

Main settings:

```text
iterations = 700
depth = 6
learning_rate = 0.03
loss_function = Logloss
eval_metric = AUC
auto_class_weights = Balanced
```

Why balanced class weights?

Because positives are rare:

```text
Only around 6.79% rows are positive.
```

Without balancing, the model might learn to predict zero too often.

---

## 27. Model Stage 2 — CatBoostRegressor

Stage 2 trains only on positive rows:

```text
incident_count > 0
```

Target is transformed using:

```text
log1p(incident_count)
```

Why log transform?

Incident counts are skewed.

Example:

```text
1 incident is common.
10+ incidents are rare.
```

Log transform reduces spike dominance.

Prediction is converted back using:

```text
expm1(prediction)
```

---

## 28. Hurdle Model Bundle

The model is saved as a bundle:

```text
classifier
regressor
alert_threshold
positive_count_mean
features
cat_features
model_type
```

The bundle type is:

```text
zero_inflated_hurdle_v1
```

Because this is not a single model, we added:

```text
HurdleModelBundle
```

This behaves like a dictionary but supports:

```python
model.predict(X)
```

This solves compatibility with test code and old inference expectations.

---

## 29. `src/forecasting/forecast_predictor.py`

This file handles prediction from the hurdle model.

Main function:

```python
predict_forecast_count(model_bundle, X)
```

It returns:

```text
expected_count
raw_expected_count
alert_probability
alert_prediction
positive_count_prediction
model_type
```

It supports:

```text
new hurdle model bundle
old single regressor fallback
```

---

## 30. Threshold-Gated Count Prediction

Old formula:

```text
expected_count = alert_probability × positive_count_prediction
```

Problem:

```text
This overpredicted many zero rows.
```

Because with 93% zeros, even small probabilities create false counts.

New formula:

```text
probability_strength =
    (alert_probability - alert_threshold)
    /
    (1 - alert_threshold)

expected_count =
    probability_strength × positive_count_prediction
```

Then clipped to:

```text
0 to 1 strength range
```

So:

```text
alert_probability below threshold → expected_count close to 0
alert_probability above threshold → count starts increasing
```

This improved count prediction.

---

## 31. Alert Threshold Calibration

The system tunes alert threshold using:

```text
F2 score
```

Why F2?

F2 gives more importance to recall.

Traffic operations prefer:

```text
catch more real incidents
```

even if some extra alerts happen.

Reason:

```text
Missing a real accident is worse than giving an early warning.
```

Latest calibrated threshold:

```text
0.55
```

---

# Part E — Model Evaluation

## 32. `src/forecasting/cross_validate_timeseries.py`

This file performs time-series cross-validation.

It splits data by time buckets, not random rows.

For every fold:

```text
train on earlier time range
test on later time range
```

Metrics:

```text
MAE
RMSE
R²
Precision
Recall
Alert F1
ROC-AUC
PR-AUC
```

---

## 33. Latest Cross-Validation Result

Latest mean cross-validation:

```text
Mean MAE       : 0.1007
Mean RMSE      : 0.5118
Mean R²        : 0.1523
Mean Precision : 0.3764
Mean Recall    : 0.3987
Mean Alert F1  : 0.3865
Mean ROC-AUC   : 0.8458
Mean PR-AUC    : 0.3795
```

Interpretation:

```text
MAE is low.
ROC-AUC is strong.
PR-AUC is strong for a rare-positive problem.
R² is modest because incident spikes are rare and hard to predict.
```

---

## 34. Latest Holdout Result After F2 Tuning

Latest holdout:

```text
MAE              : 0.1548
RMSE             : 0.4890
R²               : 0.2340

Precision        : 0.2242
Recall           : 0.7124
F1               : 0.3410
ROC-AUC          : 0.8541
PR-AUC           : 0.4217
```

---

## 35. Metric Interpretation

### MAE = 0.1548

Average absolute error:

```text
around 0.15 incidents per corridor-hour
```

This is good for sparse count forecasting.

### R² = 0.2340

R² is positive but not high.

Reason:

```text
rare spikes are difficult to predict from incident history alone
```

This is expected.

### ROC-AUC = 0.8541

This is strong.

It means:

```text
the model ranks risky corridor-hours higher than quiet corridor-hours well
```

### PR-AUC = 0.4217

This is strong for a positive rate of around 6.79%.

Random baseline PR-AUC would be approximately:

```text
0.0679
```

So:

```text
0.4217 is much better than random
```

### Recall = 0.7124

This means:

```text
the system catches around 71% of actual incident hours
```

This is important for operations.

### Precision = 0.2242

This means:

```text
not every alert becomes an actual incident
```

But this is acceptable because traffic operations prefer early warning over missed incidents.

---

## 36. Why R² Is Not the Main Metric

For sparse event forecasting, R² can be misleading.

Reasons:

```text
Most values are zero.
Few rows are positive.
Spikes are rare.
Missing a few spikes heavily hurts R².
```

Better operational metrics:

```text
MAE
Recall
ROC-AUC
PR-AUC
Alert F2
Top-k alert precision
```

In this project, the model is judged mainly by:

```text
low MAE
strong ROC-AUC
strong PR-AUC
high recall
useful operational output
```

---

# Part F — Feature Importance

## 37. `src/forecasting/forecast_feature_importance.py`

This file explains which features matter.

For the hurdle model, it combines:

```text
classifier feature importance
regressor feature importance
```

Combined importance:

```text
0.55 × classifier_importance
+
0.45 × regressor_importance
```

This is because the classifier is slightly more important in sparse data.

The system saves:

```text
forecast_feature_importance.png
```

---

## 38. Example Feature Importance

Recent important features included:

```text
hour_cos
hour_sin
hour
rolling_168
lag_1
corridor_avg
weekday
rolling_24
corridor_volatility
month
rolling_6
closure_risk
junction_risk
rolling_12
lag_24
zone_risk
cause_risk
lag_2
lag_168
cluster_risk
```

Interpretation:

```text
time patterns are important
weekly patterns are important
recent incident history is important
corridor behavior matters
spatial-risk features contribute
```

---

# Part G — Feature Store

## 39. `src/inference/feature_store.py`

The feature store is essential for live inference.

Problem:

The forecast model requires features like:

```text
lag_1
lag_24
rolling_6
corridor_avg
junction_risk
cause_risk
cluster_risk
```

But at prediction time, the user only gives:

```text
corridor
timestamp
event details
```

So the feature store supplies historical values.

---

## 40. Feature Store Contents

The feature store saves:

```text
features
profile_features
corridor_hour_profiles
corridor_profiles
global_profile
incident_p50
incident_p75
incident_p90
incident_p95
incident_p99
corridors
```

---

## 41. Corridor-Hour Profile

Key format:

```text
corridor__hour
```

Example:

```text
ORR East 1__9
Mysore Road__20
CBD 1__14
```

Each stores median historical values for the profile features.

Example stored values:

```text
lag_1
lag_2
lag_3
lag_24
rolling_6
rolling_24
rolling_168
corridor_avg
corridor_volatility
zone_risk
junction_risk
cause_risk
closure_risk
cluster_risk
```

---

## 42. Fallback Strategy

If exact corridor-hour profile exists:

```text
use exact corridor-hour history
```

If not:

```text
use nearest hour profile for same corridor
```

If corridor exists but no hour profile:

```text
use corridor-level fallback
```

If corridor is unknown:

```text
use global fallback
```

Fallback hierarchy:

```text
exact corridor-hour
nearest corridor-hour
corridor-level profile
global profile
```

This prevents prediction failure when a user enters a corridor/time combination not seen exactly in training.

---

# Part H — Terminal Inference

## 43. `src/inference/predict_traffic_risk.py`

This file runs terminal-based prediction.

It asks for:

```text
corridor
hour
weekday
month
event_cause
veh_type
road closure flag
```

Then it performs:

```text
load model
load feature store
resolve corridor
build input row
predict incident count
calculate forecast risk
calculate event impact
calculate final risk
recommend officers
recommend barricades
recommend diversion
print report
```

---

## 44. Model Prediction During Inference

The prediction uses:

```python
predict_single_forecast(model, X)
```

This returns:

```text
predicted_incidents
forecast_details
```

`forecast_details` may include:

```text
alert_probability
alert_prediction
positive_count_prediction
raw_expected_count
```

---

# Part I — Forecast Risk Scoring

## 45. `src/scoring/risk_score.py`

This file contains:

```text
get_risk_level()
calculate_forecast_risk_score()
calculate_final_operational_risk()
```

---

## 46. Forecast Risk Score

Forecast risk converts predicted count into a percentage.

Important inputs:

```text
predicted_incidents
incident_p95
incident_p99
alert_probability
context_multiplier
```

The improved risk scaling avoids making forecast risk 100% too easily.

This is important because sparse data can make percentile thresholds small.

---

## 47. Non-Corridor Handling

`Non-corridor` is a broad fallback bucket.

It aggregates many unrelated event locations.

So it can have inflated historical risk.

To avoid over-aggressive forecast risk, we apply a context multiplier.

Example:

```text
Non-corridor multiplier = 0.65
```

This reduces risk inflation for broad fallback categories.

Real corridors are better for demo and operational prediction.

---

# Part J — Event Impact Scoring

## 48. `src/scoring/event_impact.py`

This file calculates live event impact.

Inputs:

```text
event_cause
veh_type
road_closure
rush_hour
```

The scoring is rule-based and explainable.

---

## 49. Event Cause Weights

Example cause weights:

```text
vehicle_breakdown → 45
accident          → 85
construction      → 75
water_logging     → 78
tree_fall         → 72
congestion        → 75
public_event      → 82
procession        → 88
vip_movement      → 92
protest           → 92
others            → 40
```

---

## 50. Vehicle Weights

Example vehicle weights:

```text
heavy_vehicle → 30
truck         → 30
bmtc_bus      → 30
ksrtc_bus     → 28
private_bus   → 28
lcv           → 18
private_car   → 10
taxi          → 8
auto          → 6
```

---

## 51. Event Impact Formula

Simplified:

```text
impact_score =
    0.58 × cause_score
  + 0.20 × vehicle_score
  + 0.12 × closure_score
  + rush_score
  + public_transport_boost
```

Then clamped:

```text
0 to 100
```

Impact level:

```text
0-25    LOW
25-50   MODERATE
50-75   HIGH
75-100  CRITICAL
```

---

## 52. Why Event Impact Layer Is Needed

The forecast model only knows historical incident patterns.

But live events can be severe even in historically quiet corridors.

Example:

```text
Forecast Risk: LOW
Event: accident + heavy_vehicle + road closure
```

Final output should not remain LOW.

The event impact layer solves this.

---

# Part K — Final Operational Risk

## 53. Final Risk Logic

Final risk combines:

```text
forecast_risk_score
event_impact_score
```

Formula:

```text
weighted_score =
    0.45 × forecast_risk_score
  + 0.55 × event_impact_score
```

Then event-floor logic:

```text
event_floor_score = 0.85 × event_impact_score
```

Final score:

```text
max(weighted_score, event_floor_score)
```

Escalation rules:

```text
if event_impact_score >= 85:
    final_score at least 75

if event_impact_score >= 70:
    final_score at least 55
```

This ensures high-impact live events are not suppressed by low historical forecast.

---

## 54. Risk Levels

Risk level mapping:

```text
0-25    LOW
25-50   MODERATE
50-75   HIGH
75-100  CRITICAL
```

---

## 55. Example Final Risk Behavior

Example 1:

```text
Forecast Risk: LOW
Event Impact: HIGH
Final Risk: HIGH
```

Example 2:

```text
Forecast Risk: CRITICAL
Event Impact: MODERATE
Final Risk: HIGH
```

Example 3:

```text
Forecast Risk: LOW
Event Impact: LOW
Final Risk: LOW
```

This gives balanced operational behavior.

---

# Part L — Resource Recommendation

## 56. Officer and Barricade Recommendation

The system recommends deployment based on:

```text
final_risk_level
predicted_incidents
road_closure
```

Base allocation:

```text
LOW      → 2 officers, 0 barricades
MODERATE → 4 officers, 1 barricade
HIGH     → 6 officers, 2 barricades
CRITICAL → 8 officers, 4 barricades
```

Extra rules:

```text
predicted_incidents >= 3 → add officer
predicted_incidents >= 5 → add barricade
predicted_incidents >= 8 → add more resources
road_closure = true → add officer and barricades
```

---

## 57. Example Resource Output

For:

```text
accident + heavy_vehicle + road closure
```

Possible output:

```text
Officers Needed   : 7
Barricades Needed : 4
```

For:

```text
vehicle_breakdown + lcv + no closure
```

Possible output:

```text
Officers Needed   : 4
Barricades Needed : 1
```

---

# Part M — Diversion Recommendation

## 58. `src/routing/diversion_engine.py`

This file recommends diversion corridors.

It uses:

```text
NetworkX graph
```

Each corridor is a node.

Edges represent possible corridor connections.

Example graph edges:

```text
ORR East 1 → ORR East 2
ORR East 1 → Old Airport Road
ORR East 1 → Varthur Road

Mysore Road → CBD 2
Mysore Road → Magadi Road
Mysore Road → Bannerghata Road
```

---

## 59. Diversion Output

The engine returns:

```text
status
message
primary_detour
secondary_detour
support_corridors
```

Example:

```text
Primary Detour    : ORR East 2
Secondary Detour  : Old Airport Road
Support Corridors : Varthur Road, CBD 2, Mysore Road, CBD 1, Hosur Road
```

---

## 60. Non-Corridor Diversion Rule

`Non-corridor` should not be a primary diversion route unless the affected corridor itself is `Non-corridor`.

This prevents bad outputs like:

```text
Primary Detour: Non-corridor
```

The ranking function penalizes `Non-corridor`.

---

# Part N — Training Orchestration

## 61. `train_all.py`

This is the main training script.

It runs:

```text
load_data
build_timeseries_dataset
cross_validate_timeseries
train_timeseries_model
forecast_feature_importance
```

Command:

```bash
python train_all.py
```

Generated artifacts:

```text
models/timeseries_forecast_model.pkl
models/timeseries_forecast.pkl
forecast_feature_importance.png
```

---

## 62. `prepare_feature_store.py`

This builds the feature store.

Command:

```bash
python prepare_feature_store.py
```

Generated artifact:

```text
models/traffic_feature_store.pkl
```

This must be run after training and after dataset changes.

---

## 63. `predict.py`

This runs terminal inference.

Command:

```bash
python predict.py
```

It calls:

```text
src/inference/predict_traffic_risk.py
```

---

# Part O — Model Artifacts

## 64. Final ML Artifacts

Final required artifacts:

```text
models/timeseries_forecast_model.pkl
models/timeseries_forecast.pkl
models/traffic_feature_store.pkl
```

Optional artifacts:

```text
forecast_feature_importance.png
feature_importance.png
shap_summary.png
models/catboost_severity.pkl
```

---

## 65. Artifact Meaning

| Artifact                          | Meaning                                     |
| --------------------------------- | ------------------------------------------- |
| `timeseries_forecast_model.pkl`   | Main zero-inflated hurdle forecasting model |
| `timeseries_forecast.pkl`         | Compatibility copy of forecast model        |
| `traffic_feature_store.pkl`       | Historical feature profiles for inference   |
| `forecast_feature_importance.png` | Forecast model explanation plot             |
| `catboost_severity.pkl`           | Optional old severity classifier            |

---

# Part P — Health Check

## 66. `health_check.py`

A health-check file was added to validate the ML project.

It checks:

```text
required files
optional files
package imports
config path
dataset loading
datetime safety
model artifacts
feature store loading
model loading
model.predict compatibility
sample prediction
Django setup
common bug patterns
```

Command:

```bash
python health_check.py
```

This helps detect:

```text
missing files
missing models
bad dataset path
timezone bug
hurdle model wrapper missing
feature store mismatch
import errors
```

---

# Part Q — Known Bugs and Validation

## 67. Bug 1 — Mixed Timezone Datetimes

Issue:

```text
tz-naive and tz-aware timestamps cannot be compared.
```

Affects active project:

```text
Yes, if start_datetime contains mixed timezone formats.
```

Correct fix location:

```text
src/forecasting/build_timeseries_dataset.py
```

Fix:

```python
pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)
```

---

## 68. Bug 2 — Hurdle Model Bundle Does Not Support `.predict()`

Issue:

```text
train_timeseries_model() returns a dict-like model bundle.
Some tests expect model.predict(X).
```

Affects active project:

```text
Yes.
```

Correct fix:

```text
HurdleModelBundle class in forecast_predictor.py
```

The bundle now supports:

```python
model.predict(X)
model.predict_details(X)
model.predict_proba(X)
```

---

## 69. Bug 3 — Missing Spatial Features in Old Classifier Pipeline

Issue:

```text
old advanced_features.py may not create all features used by train_catboost.py
```

Affects active project:

```text
No.
```

Reason:

```text
train_catboost.py is optional and not used in the current final flow.
```

So it does not block final ML system.

---

# Part R — Example Outputs

## 70. High Scenario

Input:

```text
Corridor: ORR East 1
Hour: 9
Weekday: 4
Month: 4
Event Cause: accident
Vehicle Type: heavy_vehicle
Road Closure Required: yes
```

Output observed:

```text
Forecast Risk Level   : LOW
Event Impact Level    : HIGH
Final Risk Level      : HIGH

Officers Needed       : 7
Barricades Needed     : 4

Primary Detour        : ORR East 2
Secondary Detour      : Old Airport Road
```

Interpretation:

```text
Historical forecast is low,
but live event is severe,
so final risk becomes HIGH.
```

This is correct hybrid behavior.

---

## 71. Non-Corridor Scenario

Input:

```text
Corridor: Non-corridor
Hour: 3
Event Cause: vehicle_breakdown
Vehicle Type: lcv
Road Closure Required: no
```

Observed:

```text
Forecast Risk Level : CRITICAL
Event Impact Level  : MODERATE
Final Risk Level    : HIGH
```

Reason:

```text
Non-corridor is a broad fallback bucket.
It aggregates many unrelated locations.
Its historical risk can be inflated.
```

Recommendation:

```text
Use actual corridors for clean demo:
ORR East 1
CBD 1
Mysore Road
Old Airport Road
Hosur Road
```

---

# Part S — Strengths of the ML System

## 72. Strengths

The ML system includes:

```text
proper time-series split
full corridor-hour grid
zero rows included
corridor-specific lag features
rolling features without leakage
zero-inflated hurdle model
F2-based threshold tuning
strong recall
good ROC-AUC
good PR-AUC
feature store for inference
event-aware final risk scoring
resource recommendation
diversion recommendation
health-check validation
```

The system is not just a prediction script.

It is an end-to-end ML decision engine.

---

# Part T — Limitations

## 73. Current Limitations

The model uses incident-history data only.

It does not yet use:

```text
live traffic speed
vehicle flow
road capacity
weather
rainfall
holiday calendar
event crowd size
CCTV feed
GPS probe data
real-time signal status
lane-level road graph
```

Because of that:

```text
exact spike prediction is limited
R² remains modest
historical forecast can be low even for severe live events
```

The event impact layer compensates for some of this.

---

# Part U — Future ML Improvements

## 74. Recommended Future Enhancements

Possible upgrades:

```text
weather features
holiday/event calendar features
road-capacity features
real-time speed and volume data
GPS-based congestion signal
OpenStreetMap road graph
lane-level diversion modeling
calibrated probability models
top-k alert precision metric
post-event feedback learning
online retraining
MLflow model registry
model drift monitoring
```

A very important future model:

```text
High-risk alert classifier
```

Target:

```text
Will this corridor-hour become operationally risky?
```

Metrics:

```text
recall
precision
F2
PR-AUC
top-k precision
```

This may be more useful than exact incident count.

---

# Part V — Final Judge Explanation

## 75. Technical Judge Summary

The ML system converts raw traffic event logs into corridor-hour time-series data. It includes zero-incident hours so the model learns both normal and abnormal conditions.

Because 93% of the target values are zero, we use a zero-inflated hurdle model instead of plain regression. The first stage predicts whether any incident is likely, and the second stage estimates count only for positive incident conditions.

The alert threshold is tuned with F2 score because recall is more important in traffic operations. Missing a real accident is more costly than raising an early warning.

The predicted incident count is converted into forecast risk, then combined with a live event impact score based on event cause, vehicle type, road closure, rush hour, and priority. This produces a final operational risk level.

Finally, the system converts risk into officer deployment, barricade requirement, affected area estimate, and diversion route recommendation.

---

# 76. One-Line ML Architecture

```text
Raw traffic events → corridor-hour time-series → hurdle forecast model → feature store → live event impact scoring → final operational risk → resource and diversion recommendation
```

---

# 77. Final ML System Status

Current model status:

```text
Sparse-data-aware forecasting: complete
Feature store: complete
Terminal inference: complete
Risk scoring: complete
Event impact scoring: complete
Resource recommendation: complete
Diversion recommendation: complete
Health check: complete
```

Latest useful model quality:

```text
MAE      : around 0.15
R²       : around 0.23
Recall   : around 0.71
ROC-AUC  : around 0.85
PR-AUC   : around 0.42
```

This is suitable for a prototype traffic command-center decision-support system.
