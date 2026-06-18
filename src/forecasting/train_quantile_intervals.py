import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor

from config import DATA_PATH

from src.preprocessing.load_data import load_data
from src.forecasting.build_timeseries_dataset import build_timeseries_dataset


MODEL_PATHS = [
    "models/timeseries_forecast_model.pkl",
    "models/timeseries_forecast.pkl",
]


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


def load_existing_model_bundle():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path), path

    raise FileNotFoundError(
        "Forecast model not found. Run python train_all.py first."
    )


def prepare_features(ts_df, features):
    X = ts_df[features].copy()

    if "corridor" in X.columns:
        X["corridor"] = (
            X["corridor"]
            .fillna("UNKNOWN")
            .astype(str)
        )

    for col in X.columns:
        if col == "corridor":
            continue

        X[col] = pd.to_numeric(
            X[col],
            errors="coerce"
        )

        X[col] = X[col].fillna(
            X[col].median()
        )

    return X


def chronological_split(X, y, train_ratio=0.80):
    split_idx = int(
        len(X) * train_ratio
    )

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def train_quantile_model(
    X_train,
    y_train,
    alpha,
    cat_features
):
    model = CatBoostRegressor(
        iterations=700,
        depth=6,
        learning_rate=0.04,
        loss_function=f"Quantile:alpha={alpha}",
        random_seed=42,
        verbose=100
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features
    )

    return model


def evaluate_interval(
    lower_pred,
    upper_pred,
    y_true
):
    lower_pred = np.array(lower_pred)
    upper_pred = np.array(upper_pred)
    y_true = np.array(y_true)

    lower_pred = np.minimum(
        lower_pred,
        upper_pred
    )

    upper_pred = np.maximum(
        lower_pred,
        upper_pred
    )

    coverage = np.mean(
        (y_true >= lower_pred)
        &
        (y_true <= upper_pred)
    )

    avg_width = np.mean(
        upper_pred - lower_pred
    )

    return float(coverage), float(avg_width)


def train_quantile_intervals(
    data_path=DATA_PATH
):
    print("\n" + "=" * 70)
    print("TRAINING CATBOOST QUANTILE INTERVAL MODELS")
    print("=" * 70)

    bundle, model_path = load_existing_model_bundle()

    print("\nLoaded forecast bundle:")
    print(model_path)

    features = bundle.get(
        "features",
        FEATURES
    ) if isinstance(bundle, dict) else FEATURES

    df = load_data(
        data_path
    )

    ts_df = build_timeseries_dataset(
        df
    )

    ts_df = ts_df.sort_values(
        "time_bucket"
    )

    X = prepare_features(
        ts_df,
        features
    )

    y = ts_df["incident_count"].astype(float)

    X_train, X_test, y_train, y_test = chronological_split(
        X,
        y
    )

    # Positive-only interval models are better for this zero-inflated system.
    # The classifier decides whether incident risk exists.
    # Quantile regressors estimate lower/upper range when incident risk exists.
    positive_mask = y_train > 0

    X_train_pos = X_train.loc[
        positive_mask
    ]

    y_train_pos = y_train.loc[
        positive_mask
    ]

    if len(X_train_pos) < 50:
        print("\nWARNING:")
        print("Very few positive rows found. Training quantile models on all rows.")
        X_train_pos = X_train
        y_train_pos = y_train

    cat_features = []

    if "corridor" in X_train_pos.columns:
        cat_features = [
            "corridor"
        ]

    print("\nPositive rows used for quantile training:")
    print(len(X_train_pos))

    print("\nTraining lower quantile model alpha=0.10")
    lower_model = train_quantile_model(
        X_train=X_train_pos,
        y_train=y_train_pos,
        alpha=0.10,
        cat_features=cat_features
    )

    print("\nTraining upper quantile model alpha=0.90")
    upper_model = train_quantile_model(
        X_train=X_train_pos,
        y_train=y_train_pos,
        alpha=0.90,
        cat_features=cat_features
    )

    lower_raw = lower_model.predict(
        X_test
    )

    upper_raw = upper_model.predict(
        X_test
    )

    lower_raw = np.maximum(
        lower_raw,
        0
    )

    upper_raw = np.maximum(
        upper_raw,
        0
    )

    lower_raw = np.minimum(
        lower_raw,
        upper_raw
    )

    upper_raw = np.maximum(
        lower_raw,
        upper_raw
    )

    coverage, avg_width = evaluate_interval(
        lower_pred=lower_raw,
        upper_pred=upper_raw,
        y_true=y_test
    )

    print("\n" + "=" * 70)
    print("QUANTILE INTERVAL HOLDOUT CHECK")
    print("=" * 70)

    print(f"Coverage       : {coverage:.4f}")
    print(f"Average width  : {avg_width:.4f}")

    if not isinstance(bundle, dict):
        raise TypeError(
            "Expected model bundle dictionary. Current model is not compatible with quantile interval attachment."
        )

    bundle["quantile_lower_model"] = lower_model
    bundle["quantile_upper_model"] = upper_model
    bundle["quantile_alpha_lower"] = 0.10
    bundle["quantile_alpha_upper"] = 0.90
    bundle["quantile_interval_label"] = "80% CatBoost quantile interval"

    bundle["quantile_interval_metrics"] = {
        "coverage": coverage,
        "average_width": avg_width,
        "trained_on_positive_rows": int(len(X_train_pos)),
        "holdout_rows": int(len(X_test)),
        "note": (
            "Lower and upper CatBoost quantile regressors are trained on positive incident rows. "
            "At inference time, the interval is gated by the hurdle classifier alert probability."
        )
    }

    for path in MODEL_PATHS:
        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            bundle,
            path
        )

        print("\nSaved updated model bundle:")
        print(path)

    print("\nQuantile interval models attached successfully.")

    return bundle


if __name__ == "__main__":
    train_quantile_intervals()