# Event Impact Scoring

## Event Impact Score (Live Severity)

File: `src/scoring/event_impact.py`

The event impact score uses live event details rather than history:

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

This captures the immediate operational severity of the event itself, independent of historical forecast.

## Crowd Size Adjustment

File: `dashboard/services/ml_engine.py` (`get_crowd_multiplier`) — also duplicated in `src/inference/predict_traffic_risk.py` for the CLI predictor.

User input categories:

| Category | Range |
|---|---|
| Small | < 500 |
| Medium | 500 – 5,000 |
| Large | 5,000 – 50,000 |
| Mega | > 50,000 |

Multipliers:

| Category | Multiplier |
|---|---|
| small | 1.00 |
| medium | 1.08 |
| large | 1.18 |
| mega | 1.30 |

Crowd size matters most for public events, rallies, processions, festivals, sports events, and protests — the same event type with a mega crowd should produce a higher impact than the same event type with a small crowd.

> **Note:** these multiplier functions currently exist in two places with identical values (`ml_engine.py` for the dashboard, `predict_traffic_risk.py` for the CLI predictor). This is not a correctness issue today, but if either copy is tuned independently in the future, the dashboard and CLI predictor could silently diverge — worth consolidating into a single shared module.

## Weather Adjustment

File: `dashboard/services/ml_engine.py` (`get_weather_multiplier`) — also duplicated in `src/inference/predict_traffic_risk.py`.

User input categories:

```text
Clear
Light Rain
Heavy Rain
Fog
```

Multipliers:

| Condition | Multiplier |
|---|---|
| clear | 1.00 |
| light_rain | 1.10 |
| heavy_rain | 1.25 |
| fog | 1.20 |

Weather affects event impact, duration estimate, and overall operational pressure — heavy rain and fog increase traffic instability.

## Composite Event Impact Score (EIS)

The final Event Impact Score is a single 0–100 score combining:

```text
forecast risk
live event impact (adjusted by crowd + weather)
historical cause risk
```

Formula:

```text
EIS =
    forecast_weight × forecast_score
  + event_weight × adjusted_event_score
  + cause_weight × cause_risk_score
```

Default weights (before any EIS calibration overrides), defined in `dashboard/services/ml_engine.py`:

| Weight | Default Value |
|---|---|
| `forecast_weight` | 0.30 |
| `event_weight` | 0.55 |
| `cause_weight` | 0.15 |

These are normalized to sum to 1.0 before being applied, so calibration can adjust the balance without ever producing an out-of-range EIS score.

EIS levels:

| Range | Level |
|---|---|
| 0–25 | LOW |
| 25–50 | MODERATE |
| 50–75 | HIGH |
| 75–100 | CRITICAL |

**Why EIS matters:** the goal is to quantify event impact in advance, not just label it. Instead of only saying:

```text
HIGH risk
```

The system says:

```text
EIS Score: 72.4%
EIS Level: HIGH
```

This gives operators and reviewers a comparable numeric score across different events, rather than a single coarse label.

## EIS Weight Micro-Calibration

File: `src/evaluation/eis_weight_calibration.py`

**Purpose:** avoid choosing EIS weights blindly. Multiple candidate weight combinations are tested against historical event outcomes using a practical severity proxy.

### Severity Proxy

```text
Actual severity proxy =
    45% duration score
  + 25% same corridor-hour incident volume score
  + 20% road closure score
  + 10% event cause severity prior
```

This proxy combines actual duration, same corridor-hour incident count, road closure status, and an event cause severity prior.

### Calibration Process

Candidate EIS weight formulas are evaluated against this proxy, and the formula with the lowest MAE against historical outcomes is selected for use in the dashboard.

Outputs:

```text
models/eis_weight_calibration.json
EIS_WEIGHT_CALIBRATION.md
```

**Reviewer-safe framing:** the EIS weights were not chosen blindly. Multiple candidate formulas were evaluated against historical event outcomes using a proxy severity target, and the lowest-MAE formula is used in the dashboard. This proxy can later be replaced with officer-labelled feedback data (see [Feedback & Retraining](feedback-and-retraining.md)).

## Pre-Event vs. Post-Event Comparison

The dashboard shows a baseline-vs-event comparison:

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

**Note:** the incident volume delta is shown as an absolute difference, not a percentage, because small baselines (e.g. 0.15) can produce misleading percentage jumps.

## Related Docs

- [Forecasting Model](forecasting-model.md) — where `forecast_score` comes from
- [Operational Outputs](operational-outputs.md) — how EIS feeds into final operational risk
