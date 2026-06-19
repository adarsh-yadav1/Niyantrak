import os
import traceback



import csv

# The data including the header and your specific rows
data = [
    ['police_station', 'latitude', 'longitude'],
    ['Cubbon Park Traffic Police Station', 12.9766, 77.5993],
    ['Ashok Nagar Traffic Police Station', 12.9662, 77.6068],
    ['Indiranagar Traffic Police Station', 12.9784, 77.6408],
    ['Madiwala Traffic Police Station', 12.9212, 77.6175],
    ['Whitefield Traffic Police Station', 12.9698, 77.7500]
]

# 'w' mode creates the file (or overwrites it if it already exists)
with open('data/police_station.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print("File 'police_station.csv' has been created successfully.")


import csv
import os

# Define the file path
file_path = os.path.join('data', 'restricted_zones.csv')

# The data you provided
zones_data = [
    {"zone_name": "Ulsoor Lake", "zone_type": "lake", "latitude": 12.9822, "longitude": 77.6192, "radius_m": 650},
    {"zone_name": "Sankey Tank", "zone_type": "lake", "latitude": 13.0067, "longitude": 77.5736, "radius_m": 550},
    {"zone_name": "Hebbal Lake", "zone_type": "lake", "latitude": 13.0469, "longitude": 77.5865, "radius_m": 750},
    {"zone_name": "Bellandur Lake", "zone_type": "lake", "latitude": 12.9352, "longitude": 77.6755, "radius_m": 1800},
    {"zone_name": "Varthur Lake", "zone_type": "lake", "latitude": 12.9413, "longitude": 77.7412, "radius_m": 1500},
    {"zone_name": "Madiwala Lake", "zone_type": "lake", "latitude": 12.9121, "longitude": 77.6179, "radius_m": 900},
    {"zone_name": "Agara Lake", "zone_type": "lake", "latitude": 12.9216, "longitude": 77.6406, "radius_m": 650},
    {"zone_name": "Lalbagh Botanical Garden", "zone_type": "park", "latitude": 12.9507, "longitude": 77.5848, "radius_m": 900},
    {"zone_name": "Cubbon Park", "zone_type": "park", "latitude": 12.9763, "longitude": 77.5929, "radius_m": 850}
]

# Write the data to the CSV
try:
    with open(file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["zone_name", "zone_type", "latitude", "longitude", "radius_m"])
        writer.writeheader()
        writer.writerows(zones_data)
    print(f"Success! Data has been written to {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")



from config import (
    DATA_PATH,
    FEATURE_STORE_PATH,
)

from src.preprocessing.load_data import load_data

from src.forecasting.build_timeseries_dataset import (
    build_timeseries_dataset,
)

from src.forecasting.cross_validate_timeseries import (
    cross_validate_timeseries,
)

from src.forecasting.train_timeseries_model import (
    train_timeseries_model,
)

from src.forecasting.forecast_feature_importance import (
    forecast_feature_importance,
)

from src.inference.feature_store import (
    build_feature_store,
)

from src.forecasting.train_spatial_timeseries_model import (
    train_spatial_timeseries_model,
)

from src.evaluation.cluster_fallback_ablation import (
    run_cluster_fallback_ablation,
)

from src.forecasting.train_quantile_intervals import (
    train_quantile_intervals,
)

from src.evaluation.eis_weight_calibration import (
    run_eis_weight_calibration,
)


# ============================================================
# FINAL FEATURE LIST
# Corridor-hour fallback model features
# Must match:
# - build_timeseries_dataset.py
# - train_timeseries_model.py
# - dashboard/services/ml_engine.py
# - feature_store.py
# ============================================================

FEATURES = [
    "corridor",

    "hour",
    "weekday",
    "month",

    "hour_sin",
    "hour_cos",

    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",

    # New compressed lag-signal features
    "any_incident_last_3h",
    "incidents_last_24h",
    "above_corridor_avg",

    "rolling_6",
    "rolling_12",
    "rolling_24",
    "rolling_168",

    "corridor_avg",
    "corridor_volatility",

    "zone_risk",
    "junction_risk",
    "cause_risk",
    "closure_risk",
    "cluster_risk",
]


def print_banner(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_step(step_no, title):
    print("\n" + "-" * 90)
    print(f"STEP {step_no}: {title}")
    print("-" * 90)


def run_required_step(step_no, title, func, *args, **kwargs):
    print_step(step_no, title)

    try:
        result = func(*args, **kwargs)
        print(f"\n[SUCCESS] {title}")
        return result

    except Exception as e:
        print(f"\n[FAILED] {title}")
        print(str(e))
        traceback.print_exc()
        raise


def run_optional_step(step_no, title, func, *args, **kwargs):
    """
    Optional steps should not break the whole training pipeline.

    Used for:
    - validation/evidence files
    - calibration reports
    - interval add-ons

    If these fail, the main model still exists.
    """

    print_step(step_no, title)

    try:
        result = func(*args, **kwargs)
        print(f"\n[SUCCESS] {title}")
        return result

    except Exception as e:
        print(f"\n[WARNING] Optional step failed: {title}")
        print(str(e))
        traceback.print_exc()
        return None


def main():
    print_banner(
        "FULL TRAFFIC INTELLIGENCE ML TRAINING PIPELINE"
    )

    os.makedirs(
        "models",
        exist_ok=True,
    )

    # ========================================================
    # STEP 1: LOAD RAW DATA
    # ========================================================

    df = run_required_step(
        1,
        "Loading Raw Traffic Dataset",
        load_data,
        DATA_PATH,
    )

    # ========================================================
    # STEP 2: BUILD CORRIDOR-HOUR TIME-SERIES DATASET
    # ========================================================

    ts_df = run_required_step(
        2,
        "Building Corridor-Hour Time-Series Dataset",
        build_timeseries_dataset,
        df,
    )

    print("\nFinal Corridor-Hour Forecast Dataset Shape:")
    print(ts_df.shape)

    # ========================================================
    # STEP 3: TIME-SERIES CROSS VALIDATION
    # ========================================================

    run_required_step(
        3,
        "Running Time-Series Cross Validation",
        cross_validate_timeseries,
        ts_df,
    )

    # ========================================================
    # STEP 4: TRAIN CORRIDOR-HOUR HURDLE MODEL
    # This is the fallback model.
    # ========================================================

    corridor_model = run_required_step(
        4,
        "Training Corridor-Hour Zero-Inflated CatBoost Hurdle Model",
        train_timeseries_model,
        ts_df,
    )

    # ========================================================
    # STEP 5: FORECAST FEATURE IMPORTANCE
    # ========================================================

    run_optional_step(
        5,
        "Saving Forecast Feature Importance",
        forecast_feature_importance,
        corridor_model,
        FEATURES,
    )

    # ========================================================
    # STEP 6: BUILD COORDINATE-AWARE FEATURE STORE
    # Needed by:
    # - coordinate resolver
    # - spatial cluster model
    # - inference
    # - ablation
    # - EIS calibration
    # ========================================================

    run_required_step(
        6,
        "Building Coordinate-Aware Feature Store",
        build_feature_store,
        data_path=DATA_PATH,
        output_path=FEATURE_STORE_PATH,
    )

    # ========================================================
    # STEP 7: TRAIN PRIMARY SPATIAL-CLUSTER MODEL
    # This fixes the earlier corridor-only limitation.
    #
    # Old:
    #   corridor × hour
    #
    # New primary:
    #   spatial_cluster_id × hour
    #   + latitude / longitude
    #   + hotspot distance
    #   + corridor distance
    #   + spatial density
    #   + lag-derived features
    # ========================================================

    run_required_step(
        7,
        "Training Primary Spatial-Cluster-Hour Forecast Model",
        train_spatial_timeseries_model,
        data_path=DATA_PATH,
        feature_store_path=FEATURE_STORE_PATH,
        output_path="models/spatial_timeseries_forecast_model.pkl",
    )

    # ========================================================
    # STEP 8: CLUSTER FALLBACK ABLATION STUDY
    # Proves whether spatial cluster fallback should replace
    # or only support corridor-hour history.
    # ========================================================

    run_optional_step(
        8,
        "Running Cluster Fallback Ablation Study",
        run_cluster_fallback_ablation,
        data_path=DATA_PATH,
        output_path="models/cluster_fallback_ablation.json",
        max_rows=5000,
    )

    # ========================================================
    # STEP 9: TRAIN CATBOOST QUANTILE INTERVAL MODELS
    # Adds 80% prediction interval support.
    # Attaches quantile models to the forecast bundle.
    # ========================================================

    run_optional_step(
        9,
        "Training CatBoost Quantile Interval Models",
        train_quantile_intervals,
        data_path=DATA_PATH,
    )

    # ========================================================
    # STEP 10: EIS WEIGHT MICRO-CALIBRATION
    # Selects EIS weights using historical severity proxy.
    # ========================================================

    run_optional_step(
        10,
        "Running EIS Weight Micro-Calibration",
        run_eis_weight_calibration,
        data_path=DATA_PATH,
        output_path="models/eis_weight_calibration.json",
        sample_size=20,
    )

    # ========================================================
    # DONE
    # ========================================================

    print_banner(
        "FULL ML TRAINING PIPELINE COMPLETE"
    )

    print("\nGenerated / Updated Files:")
    print("- models/timeseries_forecast_model.pkl")
    print("- models/timeseries_forecast.pkl")
    print("- models/spatial_timeseries_forecast_model.pkl")
    print("- models/traffic_feature_store.pkl")
    print("- models/cluster_fallback_ablation.json")
    print("- models/eis_weight_calibration.json")
    print("- EIS_WEIGHT_CALIBRATION.md")
    print("- forecast_feature_importance.png")

    print("\nModel Roles:")
    print("- spatial_timeseries_forecast_model.pkl  -> primary point-aware spatial-cluster model")
    print("- timeseries_forecast_model.pkl          -> corridor-hour fallback model")
    print("- traffic_feature_store.pkl              -> coordinate resolver + historical profile store")
    print("- cluster_fallback_ablation.json         -> evidence for fallback behavior")
    print("- eis_weight_calibration.json            -> calibrated EIS weights")
    print("- quantile models inside forecast bundle -> 80% prediction interval")

    print("\nNext command:")
    print("python manage.py runserver")


if __name__ == "__main__":
    main()