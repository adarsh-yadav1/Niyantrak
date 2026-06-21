# Setup & Usage

This expands on the README's Setup section with more detail on what each step actually generates, plus the CLI predictor as an alternative to the dashboard.

## Prerequisites

- Python 3.10+
- pip

## Step 1 — Clone the repository

```bash
git clone https://github.com/adarsh-yadav1/Niyantrak.git
cd Niyantrak
```

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

**Windows users:** `requirements.txt` includes Jupyter-related packages (`jupyter_server_terminals`, `terminado`) that depend on `pywinpty`. On Windows, `pywinpty` sometimes fails to resolve automatically through the bulk `requirements.txt` install. If you hit an error related to `pywinpty` during or after this step, run:

```bash
pip install pywinpty==3.0.3
```

Then re-run `pip install -r requirements.txt` to pick up anything that failed to install before `pywinpty` was resolved. This step is **not needed on macOS or Linux** — `pywinpty` is a Windows-only package and pip will simply skip it on other platforms.

## Step 3 — Run the training pipeline

```bash
python train_all.py
```

This single command runs the full pipeline end to end. What it actually does, in order:

1. Loads and cleans `data.csv` (`src/preprocessing/load_data.py`, `clean.py`)
2. Builds the corridor-hour time-series dataset (`src/forecasting/build_timeseries_dataset.py`)
3. Builds the spatial-cluster-hour time-series dataset (`src/forecasting/build_spatial_timeseries_dataset.py`)
4. Trains the corridor-hour hurdle model — classifier + regressor (`train_timeseries_model.py`) → `models/timeseries_forecast_model.pkl`, `models/timeseries_forecast.pkl`
5. Trains the spatial-cluster-hour hurdle model — the primary forecasting tier (`train_spatial_timeseries_model.py`) → `models/spatial_timeseries_forecast_model.pkl`
6. Builds the coordinate-aware feature store, including spatial clustering (`src/inference/feature_store.py`) → `models/traffic_feature_store.pkl`
7. Runs the cluster fallback ablation study (`src/evaluation/cluster_fallback_ablation.py`) → `models/cluster_fallback_ablation.json`
8. Trains the quantile interval models for the 80% prediction interval (`train_quantile_intervals.py`)
9. Runs EIS weight calibration (`src/evaluation/eis_weight_calibration.py`) → `models/eis_weight_calibration.json`, `EIS_WEIGHT_CALIBRATION.md`
10. Generates a feature importance plot → `forecast_feature_importance.png`

See [Generated Model Artifacts](../README.md#generated-model-artifacts) in the README for the full file list this produces.

## Step 4 — Run the dashboard

```bash
python manage.py runserver
```

Open the local server URL printed in the terminal (typically `http://127.0.0.1:8000/`) to use the map-based dashboard.

## Alternative — Run the terminal predictor

```bash
python scripts/predict.py
```

Useful for a quick CLI-based prediction without starting the web dashboard — for example, to sanity-check a single coordinate and event combination during development.

## Standalone Feature Store Build

If you only need to rebuild the feature store (for example, after editing `data.csv` without retraining the forecasting models), you can run it independently:

```bash
python scripts/prepare_feature_store.py
```

## Standalone Spatial Model Training

To retrain only the primary spatial-cluster model without re-running the entire pipeline:

```bash
python scripts/train_spatial_model.py
```

## Scheduled Retraining Helper

A rolling-window retraining script exists for periodically refreshing the models without re-running the full pipeline manually each time:

```bash
python scripts/retrain_30_days.py
```

This is not currently triggered automatically — see [Feedback & Retraining](feedback-and-retraining.md) for its current scope and what it would take to make this a fully automated loop.

## Related Docs

- [Architecture](architecture.md) — what each pipeline stage does conceptually
- [Limitations & Roadmap](limitations-and-roadmap.md) — what's intentionally out of scope for now
