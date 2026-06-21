# Feedback & Retraining

## Post-Event Feedback Collection

The system includes a feedback collection module (`dashboard/services/feedback_store.py`) that stores, after an event has concluded:

```text
actual duration
actual officers deployed
actual barricades used
actual road closure
actual incident count
officer notes
```

Output file: `data/post_event_feedback.csv`

## Current Scope

**Important honesty note:** the current prototype stores feedback for audit, analysis, and future retraining. It does **not** automatically retrain the ML model after each feedback entry — feedback is not yet wired into the live training pipeline.

### Correct wording to use when describing this feature

```text
feedback collection system with scheduled-retraining scaffolding in place
```

### Wording to avoid

```text
automatic learning loop
self-learning model
continuous retraining
```

This distinction matters for accurately describing the system to reviewers, collaborators, or in any documentation — overstating this capability is the easiest way to lose credibility on an otherwise well-evidenced system.

## What Already Exists Toward Retraining

Beyond simply storing feedback, two pieces of scaffolding for **scheduled** (not automatic) retraining already exist in the codebase:

- **`src/features/feedback_training.py`** — transforms the raw `data/post_event_feedback.csv` records into training-ready rows (`time_bucket`, `corridor`, `actual_incident_count`, `feedback_rows`), bridging the gap between free-form officer feedback and a format the forecasting pipeline could consume.
- **`scripts/retrain_30_days.py`** — a standalone script that re-runs the relevant training commands on a rolling window and writes a result to `models/retrain_log.json`.

Neither of these is currently invoked automatically by `train_all.py` or triggered by a new feedback submission — they exist as manually-run tools, not an active pipeline. This is still meaningfully short of "the model retrains itself," but it is further along than "nothing exists yet."

## How This Data Could Be Used Later

The collected feedback is structured so that it *could* support:

- scheduled (not automatic) retraining of the forecasting model, using the scaffolding above
- replacing the current EIS severity proxy (see [Event Impact Scoring](event-impact-scoring.md#severity-proxy)) with real officer-labelled outcomes
- auditing how accurate past risk/EIS predictions were against what actually happened

Wiring `feedback_training.py`'s output into `retrain_30_days.py` (or into `train_all.py` directly, run on a schedule) is the remaining step toward making this a real, if still manually-triggered, retraining loop — see [Limitations & Roadmap](limitations-and-roadmap.md) for the current state and planned direction.

## Related Docs

- [Event Impact Scoring](event-impact-scoring.md) — the severity proxy this feedback could eventually replace
- [Limitations & Roadmap](limitations-and-roadmap.md) — full list of what's not yet implemented
