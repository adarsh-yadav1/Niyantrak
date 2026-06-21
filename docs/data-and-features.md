# Data & Features

## Raw Dataset Structure

The raw dataset (`data.csv`) contains traffic event records. Important columns:

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

Each row represents one traffic-related event, for example:

```text
accident on ORR East 1 at 09:12
vehicle breakdown near CBD at 18:30
public event near Mysore Road at 17:00
```

## Time-Series Dataset Generation

Raw events are converted into time-series datasets before training, at two different spatial granularities:

| | Format |
|---|---|
| Raw | one row = one event |
| Corridor-hour training (fallback model) | one row = one corridor-hour |
| Spatial-cluster-hour training (primary model) | one row = one spatial-cluster-hour |

Example aggregation:

```text
Raw events:
ORR East 1 | 2024-04-01 09:12 | accident
ORR East 1 | 2024-04-01 09:47 | congestion

After aggregation:
ORR East 1 | 2024-04-01 09:00 | incident_count = 2
```

The target column is `incident_count`: the number of traffic incidents in a corridor (or spatial cluster) during a specific hour.

Corridor-hour aggregation is handled by `src/forecasting/build_timeseries_dataset.py`. The parallel spatial-cluster-hour aggregation, used to train the primary forecasting tier, is handled by `src/forecasting/build_spatial_timeseries_dataset.py` — see [Forecasting Model](forecasting-model.md#two-forecasting-tiers).

## Zero-Heavy Target Distribution

Both time-series datasets are highly imbalanced — `incident_count = 0` for more than 90% of rows. A standard regressor tends to learn to predict near-zero values most of the time and miss rare traffic spikes. This is the core reason the forecasting model uses a zero-inflated hurdle architecture rather than a single regressor (see [Forecasting Model](forecasting-model.md)).

## Time-Based Features

```text
hour
weekday
month
hour_sin
hour_cos
```

**Why `hour_sin` and `hour_cos`?** Hour is cyclical — 23:00 and 00:00 are close in real life, but numerically 23 and 0 look far apart. Cyclical (sine/cosine) encoding solves this discontinuity.

## Calendar-Aware Event Features

```text
is_event_day
calendar_event_type
calendar_event_intensity
```

| Feature | Meaning |
|---|---|
| `is_event_day` | Whether the prediction timestamp falls on a known or user-declared event day |
| `calendar_event_type` | The category of the calendar event (e.g. public_event, protest, festival), when known |
| `calendar_event_intensity` | A relative intensity signal for the calendar event |

File: `src/features/event_calender.py`

These let the model condition its forecast on whether the target hour coincides with a known or declared event, separately from the live event details the user enters at prediction time.

## Lag Features

```text
lag_1
lag_2
lag_3
lag_24
lag_48
lag_72
lag_168
```

| Feature | Meaning |
|---|---|
| `lag_1` | incident count 1 hour ago |
| `lag_24` | same hour yesterday |
| `lag_168` | same hour last week |

Traffic has memory: if a corridor recently had incidents, congestion can continue into later hours. Lag features let the model capture that.

## Short-Window Incident-Recency Features

```text
any_incident_last_3h
incidents_last_24h
above_corridor_avg
```

| Feature | Meaning |
|---|---|
| `any_incident_last_3h` | Whether any incident occurred on this corridor in the last 3 hours |
| `incidents_last_24h` | Count of incidents on this corridor in the last 24 hours |
| `above_corridor_avg` | Whether current incident pressure is above this corridor's historical average |

These sit alongside the lag features above and give the model a faster-reacting recency signal, distinct from the longer-window lag and rolling features.

## Rolling Features

```text
rolling_6
rolling_12
rolling_24
rolling_168
```

| Feature | Meaning |
|---|---|
| `rolling_6` | average incident count over the last 6 hours |
| `rolling_24` | average incident count over the last 24 hours |
| `rolling_168` | average incident count over the last week |

These capture recent corridor pressure rather than a single point in time.

## Spatial and Historical Risk Features

```text
corridor_avg
corridor_volatility
zone_risk
junction_risk
cause_risk
closure_risk
cluster_risk
```

| Feature | Meaning |
|---|---|
| `corridor_avg` | Average incident count for that corridor |
| `corridor_volatility` | How unstable or spike-prone the corridor is |
| `zone_risk` | Historical event density in that zone |
| `junction_risk` | Historical risk around a junction |
| `cause_risk` | Historical frequency/risk of the event cause |
| `closure_risk` | Historical risk linked to road closure |
| `cluster_risk` | Spatial cluster event density |

These features make the model location-aware, not just time-aware.

## Where These Features Are Stored

All of the above (lag, recency, rolling, corridor, zone, junction, cause, closure, cluster, and calendar-aware values) are precomputed and stored in the **coordinate-aware feature store**, so they can be looked up at prediction time without recomputing from raw history. See [Location Intelligence](location-intelligence.md) for how this store is structured and queried.

## Related Docs

- [Forecasting Model](forecasting-model.md) — how these features feed the hurdle models
- [Location Intelligence](location-intelligence.md) — the feature store and location resolver
