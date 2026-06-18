import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from config import DATA_PATH

from src.preprocessing.load_data import load_data
from src.forecasting.build_timeseries_dataset import build_timeseries_dataset
from src.forecasting.forecast_predictor import predict_single_forecast

from src.inference.location_resolver import (
    resolve_corridor_from_coordinates,
    get_profile_with_spatial_fallback,
    find_nearest_cluster_hour_profile,
    clean_profile,
    make_key,
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


MODEL_PATHS = [
    "models/timeseries_forecast_model.pkl",
    "models/timeseries_forecast.pkl",
]

FEATURE_STORE_PATH = "models/traffic_feature_store.pkl"

OUTPUT_PATH = "models/cluster_fallback_ablation.json"


def safe_float(value, fallback=0.0):
    try:
        if value is None:
            return fallback

        if pd.isna(value):
            return fallback

        return float(value)

    except Exception:
        return fallback


def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path)

    raise FileNotFoundError(
        "Forecast model not found. Run python train_all.py first."
    )


def load_feature_store():
    if not os.path.exists(FEATURE_STORE_PATH):
        raise FileNotFoundError(
            "Feature store not found. Run python prepare_feature_store.py first."
        )

    return joblib.load(FEATURE_STORE_PATH)


def build_input_row(
    corridor,
    hour,
    weekday,
    month,
    profile
):
    hour = int(hour)
    weekday = int(weekday)
    month = int(month)

    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    row = {}

    for feature in FEATURES:
        if feature == "corridor":
            row[feature] = corridor

        elif feature == "hour":
            row[feature] = hour

        elif feature == "weekday":
            row[feature] = weekday

        elif feature == "month":
            row[feature] = month

        elif feature == "hour_sin":
            row[feature] = hour_sin

        elif feature == "hour_cos":
            row[feature] = hour_cos

        else:
            row[feature] = safe_float(
                profile.get(feature),
                0.0
            )

    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


def get_corridor_location(
    corridor,
    store
):
    profiles = store.get(
        "corridor_location_profiles",
        {}
    )

    exact = profiles.get(
        str(corridor)
    )

    if exact is not None:
        return (
            safe_float(exact.get("latitude")),
            safe_float(exact.get("longitude"))
        )

    corridor_clean = (
        str(corridor)
        .strip()
        .lower()
    )

    for key, value in profiles.items():
        if (
            str(key)
            .strip()
            .lower()
            ==
            corridor_clean
        ):
            return (
                safe_float(value.get("latitude")),
                safe_float(value.get("longitude"))
            )

    return None, None


def get_forced_cluster_profile(
    store,
    cluster_id,
    hour
):
    if cluster_id is None:
        return None, "no cluster id"

    cluster_hour_profiles = store.get(
        "spatial_cluster_hour_profiles",
        {}
    )

    cluster_profiles = store.get(
        "spatial_cluster_profiles",
        {}
    )

    key = make_key(
        cluster_id,
        hour
    )

    if key in cluster_hour_profiles:
        return (
            clean_profile(cluster_hour_profiles[key]),
            "exact spatial cluster-hour profile"
        )

    nearest_profile, nearest_hour = find_nearest_cluster_hour_profile(
        store,
        cluster_id,
        hour
    )

    if nearest_profile is not None:
        return (
            clean_profile(nearest_profile),
            f"nearest spatial cluster-hour profile, hour {nearest_hour}"
        )

    cluster_profile = cluster_profiles.get(
        str(cluster_id)
    )

    if cluster_profile is not None:
        return (
            clean_profile(cluster_profile),
            "spatial cluster-level profile"
        )

    return None, "cluster profile unavailable"


def predict_count(
    model,
    X
):
    pred, _ = predict_single_forecast(
        model,
        X
    )

    return max(
        safe_float(pred),
        0.0
    )


def prepare_holdout_rows(
    ts_df,
    holdout_fraction=0.20,
    max_rows=5000
):
    ts_df = ts_df.copy()

    ts_df = ts_df.sort_values(
        [
            "time_bucket",
            "corridor"
        ]
    )

    unique_times = (
        ts_df["time_bucket"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    split_idx = int(
        len(unique_times) * (1 - holdout_fraction)
    )

    holdout_times = set(
        unique_times[split_idx:]
    )

    holdout_df = ts_df[
        ts_df["time_bucket"].isin(
            holdout_times
        )
    ].copy()

    if len(holdout_df) > max_rows:
        holdout_df = holdout_df.sample(
            n=max_rows,
            random_state=42
        )

    return holdout_df


def run_cluster_fallback_ablation(
    data_path=DATA_PATH,
    output_path=OUTPUT_PATH,
    max_rows=5000
):
    print("\n" + "=" * 70)
    print("CLUSTER FALLBACK ABLATION STUDY")
    print("=" * 70)

    model = load_model()
    store = load_feature_store()

    df = load_data(
        data_path
    )

    ts_df = build_timeseries_dataset(
        df
    )

    holdout_df = prepare_holdout_rows(
        ts_df,
        max_rows=max_rows
    )

    print("\nHoldout rows selected:")
    print(len(holdout_df))

    y_true = []
    normal_preds = []
    cluster_preds = []

    skipped = 0
    examples = []

    source_counts = {
        "normal": {},
        "cluster": {},
    }

    for _, row in holdout_df.iterrows():
        corridor = str(row["corridor"])
        hour = int(row["hour"])
        weekday = int(row["weekday"])
        month = int(row["month"])
        target = safe_float(
            row["incident_count"]
        )

        latitude, longitude = get_corridor_location(
            corridor,
            store
        )

        if latitude is None or longitude is None:
            skipped += 1
            continue

        location_match = resolve_corridor_from_coordinates(
            latitude=latitude,
            longitude=longitude,
            store=store
        )

        if location_match.get("outside_bengaluru"):
            skipped += 1
            continue

        cluster_id = location_match.get(
            "spatial_cluster_id"
        )

        normal_profile, normal_source, _ = get_profile_with_spatial_fallback(
            store=store,
            corridor=corridor,
            hour=hour,
            location_match=location_match
        )

        cluster_profile, cluster_source = get_forced_cluster_profile(
            store=store,
            cluster_id=cluster_id,
            hour=hour
        )

        if cluster_profile is None:
            skipped += 1
            continue

        X_normal = build_input_row(
            corridor=corridor,
            hour=hour,
            weekday=weekday,
            month=month,
            profile=normal_profile
        )

        X_cluster = build_input_row(
            corridor=corridor,
            hour=hour,
            weekday=weekday,
            month=month,
            profile=cluster_profile
        )

        normal_pred = predict_count(
            model,
            X_normal
        )

        cluster_pred = predict_count(
            model,
            X_cluster
        )

        y_true.append(target)
        normal_preds.append(normal_pred)
        cluster_preds.append(cluster_pred)

        source_counts["normal"][normal_source] = (
            source_counts["normal"].get(normal_source, 0)
            +
            1
        )

        source_counts["cluster"][cluster_source] = (
            source_counts["cluster"].get(cluster_source, 0)
            +
            1
        )

        if len(examples) < 5:
            examples.append({
                "corridor": corridor,
                "hour": hour,
                "actual": round(target, 3),
                "normal_prediction": round(normal_pred, 3),
                "cluster_prediction": round(cluster_pred, 3),
                "normal_source": normal_source,
                "cluster_source": cluster_source,
                "spatial_cluster_id": cluster_id,
            })

    if not y_true:
        raise RuntimeError(
            "No valid rows available for cluster fallback ablation."
        )

    y_true = np.array(y_true)
    normal_preds = np.array(normal_preds)
    cluster_preds = np.array(cluster_preds)

    normal_mae = mean_absolute_error(
        y_true,
        normal_preds
    )

    cluster_mae = mean_absolute_error(
        y_true,
        cluster_preds
    )

    normal_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            normal_preds
        )
    )

    cluster_rmse = np.sqrt(
        mean_squared_error(
            y_true,
            cluster_preds
        )
    )

    try:
        normal_r2 = r2_score(
            y_true,
            normal_preds
        )

    except Exception:
        normal_r2 = None

    try:
        cluster_r2 = r2_score(
            y_true,
            cluster_preds
        )

    except Exception:
        cluster_r2 = None

    mae_delta = normal_mae - cluster_mae

    if normal_mae > 0:
        mae_improvement_pct = (
            mae_delta
            /
            normal_mae
        ) * 100

    else:
        mae_improvement_pct = 0.0

    if mae_improvement_pct > 1:
        conclusion = (
            "Cluster fallback reduced MAE on the holdout comparison, "
            "supporting its use for weak or unknown locations."
        )

    elif mae_improvement_pct < -1:
        conclusion = (
            "Cluster fallback was weaker than corridor-hour history on this holdout comparison. "
            "It should be used only when corridor matching is weak or unavailable."
        )

    else:
        conclusion = (
            "Cluster fallback performed similarly to corridor-hour history. "
            "It remains useful as a safety fallback for unknown locations."
        )

    result = {
        "title": "Cluster Fallback Ablation Study",
        "generated_at": pd.Timestamp.now().isoformat(),

        "rows_tested": int(len(y_true)),
        "rows_skipped": int(skipped),

        "normal_profile": {
            "label": "Corridor-hour profile",
            "mae": float(normal_mae),
            "rmse": float(normal_rmse),
            "r2": None if normal_r2 is None else float(normal_r2),
        },

        "cluster_fallback": {
            "label": "Forced spatial cluster fallback",
            "mae": float(cluster_mae),
            "rmse": float(cluster_rmse),
            "r2": None if cluster_r2 is None else float(cluster_r2),
        },

        "comparison": {
            "mae_delta": float(mae_delta),
            "mae_improvement_pct": float(mae_improvement_pct),
            "better_method": (
                "cluster_fallback"
                if cluster_mae < normal_mae
                else "corridor_hour_profile"
            ),
        },

        "source_counts": source_counts,

        "examples": examples,

        "conclusion": conclusion,

        "note": (
            "This ablation compares normal corridor-hour historical profiles against forced spatial-cluster fallback profiles on a chronological holdout sample. "
            "It is used to validate whether the fallback mechanism is reasonable for weak or unknown locations."
        ),
    }

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            result,
            file,
            indent=4
        )

    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)

    print(f"Rows tested       : {len(y_true)}")
    print(f"Rows skipped      : {skipped}")
    print(f"Normal MAE        : {normal_mae:.4f}")
    print(f"Cluster MAE       : {cluster_mae:.4f}")
    print(f"MAE Delta         : {mae_delta:.4f}")
    print(f"Improvement       : {mae_improvement_pct:.2f}%")
    print(f"Normal RMSE       : {normal_rmse:.4f}")
    print(f"Cluster RMSE      : {cluster_rmse:.4f}")
    print("\nConclusion:")
    print(conclusion)

    print("\nSaved:")
    print(output_path)

    return result


if __name__ == "__main__":
    run_cluster_fallback_ablation()