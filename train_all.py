import os

from config import DATA_PATH
from config import FEATURE_STORE_PATH

from src.preprocessing.load_data import load_data

from src.forecasting.build_timeseries_dataset import (
    build_timeseries_dataset
)

from src.forecasting.cross_validate_timeseries import (
    cross_validate_timeseries
)

from src.forecasting.train_timeseries_model import (
    train_timeseries_model
)

from src.forecasting.forecast_feature_importance import (
    forecast_feature_importance
)

from src.inference.feature_store import (
    build_feature_store
)

from src.evaluation.cluster_fallback_ablation import (
    run_cluster_fallback_ablation
)

from src.forecasting.train_quantile_intervals import (
    train_quantile_intervals
)

from src.evaluation.eis_weight_calibration import (
    run_eis_weight_calibration
)


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


def print_step(step_no, title):
    print("\n" + "=" * 80)
    print(f"STEP {step_no}: {title}")
    print("=" * 80)


def main():

    print("\n" + "=" * 80)
    print("TRAINING FULL TRAFFIC INTELLIGENCE PIPELINE")
    print("=" * 80)

    os.makedirs(
        "models",
        exist_ok=True
    )

    # =====================================================
    # STEP 1: LOAD DATA
    # =====================================================

    print_step(
        1,
        "Loading Data"
    )

    df = load_data(
        DATA_PATH
    )

    # =====================================================
    # STEP 2: BUILD TIME-SERIES DATASET
    # =====================================================

    print_step(
        2,
        "Building Time-Series Forecast Dataset"
    )

    ts_df = build_timeseries_dataset(
        df
    )

    print("\nFinal Forecast Dataset Shape:")
    print(
        ts_df.shape
    )

    # =====================================================
    # STEP 3: CROSS VALIDATION
    # =====================================================

    print_step(
        3,
        "Running Time-Series Cross Validation"
    )

    cross_validate_timeseries(
        ts_df
    )

    # =====================================================
    # STEP 4: TRAIN FINAL HURDLE FORECAST MODEL
    # =====================================================

    print_step(
        4,
        "Training Final Zero-Inflated Forecast Model"
    )

    model = train_timeseries_model(
        ts_df
    )

    # =====================================================
    # STEP 5: FEATURE IMPORTANCE
    # =====================================================

    print_step(
        5,
        "Saving Forecast Feature Importance"
    )

    forecast_feature_importance(
        model,
        FEATURES
    )

    # =====================================================
    # STEP 6: BUILD COORDINATE-AWARE FEATURE STORE
    # =====================================================

    print_step(
        6,
        "Building Coordinate-Aware Feature Store"
    )

    build_feature_store(
        data_path=DATA_PATH,
        output_path=FEATURE_STORE_PATH
    )

    # =====================================================
    # STEP 7: CLUSTER FALLBACK ABLATION STUDY
    # =====================================================

    print_step(
        7,
        "Running Cluster Fallback Ablation Study"
    )

    run_cluster_fallback_ablation(
        data_path=DATA_PATH,
        output_path="models/cluster_fallback_ablation.json"
    )

    # =====================================================
    # STEP 8: TRAIN CATBOOST QUANTILE INTERVAL MODELS
    # =====================================================

    print_step(
        8,
        "Training CatBoost Quantile Interval Models"
    )

    train_quantile_intervals(
        data_path=DATA_PATH
    )

    # =====================================================
    # STEP 9: EIS WEIGHT MICRO-CALIBRATION
    # =====================================================

    print_step(
        9,
        "Running EIS Weight Micro-Calibration"
    )

    run_eis_weight_calibration(
        data_path=DATA_PATH,
        output_path="models/eis_weight_calibration.json",
        sample_size=20
    )

    # =====================================================
    # COMPLETE
    # =====================================================

    print("\n" + "=" * 80)
    print("FULL TRAINING PIPELINE COMPLETE")
    print("=" * 80)

    print("\nGenerated / Updated Files:")
    print("- models/timeseries_forecast_model.pkl")
    print("- models/timeseries_forecast.pkl")
    print("- models/traffic_feature_store.pkl")
    print("- models/cluster_fallback_ablation.json")
    print("- models/eis_weight_calibration.json")
    print("- EIS_WEIGHT_CALIBRATION.md")
    print("- forecast_feature_importance.png")

    print("\nNext command:")
    print("python manage.py runserver")


if __name__ == "__main__":
    main()