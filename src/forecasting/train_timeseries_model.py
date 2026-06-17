import os
import joblib
import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from src.forecasting.forecast_predictor import (
    predict_forecast_count,
    HurdleModelBundle
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
    "cluster_risk"
]


CAT_FEATURES = [
    "corridor"
]


TARGET = "incident_count"




def validate_dataset(ts_df):

    missing = [
        col
        for col in FEATURES + [TARGET, "time_bucket"]
        if col not in ts_df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns for forecasting:\n"
            + str(missing)
        )


def split_by_time(
    df,
    train_ratio=0.8
):

    unique_times = sorted(
        df["time_bucket"].unique()
    )

    split_idx = int(
        len(unique_times)
        *
        train_ratio
    )

    train_times = unique_times[:split_idx]
    test_times = unique_times[split_idx:]

    train_df = df[
        df["time_bucket"].isin(train_times)
    ].copy()

    test_df = df[
        df["time_bucket"].isin(test_times)
    ].copy()

    return train_df, test_df

def tune_alert_threshold(
    y_true_alert,
    alert_proba
):

    best_threshold = 0.50
    best_score = -1

    thresholds = np.arange(
        0.20,
        0.91,
        0.05
    )

    beta = 2.0

    for threshold in thresholds:

        preds = (
            alert_proba >= threshold
        ).astype(int)

        precision = precision_score(
            y_true_alert,
            preds,
            zero_division=0
        )

        recall = recall_score(
            y_true_alert,
            preds,
            zero_division=0
        )

        if precision == 0 and recall == 0:
            score = 0

        else:
            score = (
                (1 + beta ** 2)
                *
                precision
                *
                recall
            ) / (
                (beta ** 2 * precision)
                +
                recall
                +
                1e-9
            )

        if score > best_score:

            best_score = score
            best_threshold = threshold

    return float(best_threshold), float(best_score)


def fit_hurdle_bundle(
    X_train,
    y_train,
    alert_threshold=0.35
):

    y_train = y_train.astype(float)

    y_alert = (
        y_train > 0
    ).astype(int)

    classifier = CatBoostClassifier(
        iterations=700,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=False
    )

    classifier.fit(
        X_train,
        y_alert,
        cat_features=CAT_FEATURES
    )

    positive_mask = (
        y_train > 0
    )

    positive_count_mean = 1.0

    regressor = None

    if positive_mask.sum() >= 20:

        X_pos = X_train.loc[
            positive_mask
        ]

        y_pos = y_train.loc[
            positive_mask
        ]

        positive_count_mean = float(
            y_pos.mean()
        )

        y_pos_log = np.log1p(
            y_pos
        )

        regressor = CatBoostRegressor(
            iterations=700,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=42,
            verbose=False
        )

        regressor.fit(
            X_pos,
            y_pos_log,
            cat_features=CAT_FEATURES
        )

    else:

        if positive_mask.sum() > 0:

            positive_count_mean = float(
                y_train.loc[positive_mask].mean()
            )

    bundle = HurdleModelBundle({
        "model_type": "zero_inflated_hurdle_v1",

        "classifier": classifier,
        "regressor": regressor,

        "features": FEATURES,
        "cat_features": CAT_FEATURES,

        "alert_threshold": float(alert_threshold),
        "positive_count_mean": float(positive_count_mean),
    })

    return bundle


def build_calibrated_threshold(
    train_df
):

    if len(train_df) < 100:

        return 0.35

    calibration_train_df, calibration_df = split_by_time(
        train_df,
        train_ratio=0.80
    )

    if len(calibration_train_df) == 0 or len(calibration_df) == 0:

        return 0.35

    X_cal_train = calibration_train_df[FEATURES]
    y_cal_train = calibration_train_df[TARGET]

    X_cal = calibration_df[FEATURES]
    y_cal = calibration_df[TARGET]

    temp_bundle = fit_hurdle_bundle(
        X_train=X_cal_train,
        y_train=y_cal_train,
        alert_threshold=0.35
    )

    pred = predict_forecast_count(
        temp_bundle,
        X_cal
    )

    y_cal_alert = (
        y_cal > 0
    ).astype(int)

    threshold, best_f1 = tune_alert_threshold(
        y_true_alert=y_cal_alert,
        alert_proba=pred["alert_probability"]
    )

    print("\nCalibrated Alert Threshold:")
    print(f"Threshold : {threshold:.2f}")
    print(f"Calib F2  : {best_f1:.4f}")

    return threshold


def evaluate_hurdle_model(
    bundle,
    X_test,
    y_test
):

    pred = predict_forecast_count(
        bundle,
        X_test
    )

    expected_count = pred[
        "expected_count"
    ]

    alert_proba = pred[
        "alert_probability"
    ]

    alert_pred = pred[
        "alert_prediction"
    ]

    y_test = y_test.astype(float)

    y_alert = (
        y_test > 0
    ).astype(int)

    mae = mean_absolute_error(
        y_test,
        expected_count
    )

    rmse = mean_squared_error(
        y_test,
        expected_count
    ) ** 0.5

    r2 = r2_score(
        y_test,
        expected_count
    )

    precision = precision_score(
        y_alert,
        alert_pred,
        zero_division=0
    )

    recall = recall_score(
        y_alert,
        alert_pred,
        zero_division=0
    )

    alert_f1 = f1_score(
        y_alert,
        alert_pred,
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_alert,
            alert_proba
        )

    except Exception:

        roc_auc = 0.0

    try:

        pr_auc = average_precision_score(
            y_alert,
            alert_proba
        )

    except Exception:

        pr_auc = 0.0

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),

        "alert_precision": float(precision),
        "alert_recall": float(recall),
        "alert_f1": float(alert_f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
    }

    return metrics


def train_timeseries_model(
    ts_df
):

    print("\n" + "=" * 60)
    print("TRAINING ZERO-INFLATED TRAFFIC FORECAST MODEL")
    print("=" * 60)

    validate_dataset(
        ts_df
    )

    df = (
        ts_df
        .sort_values(
            [
                "time_bucket",
                "corridor"
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    train_df, test_df = split_by_time(
        df,
        train_ratio=0.8
    )

    print("\nTrain rows:", len(train_df))
    print("Test rows :", len(test_df))

    print(
        "Train time range:",
        train_df["time_bucket"].min(),
        "to",
        train_df["time_bucket"].max()
    )

    print(
        "Test time range :",
        test_df["time_bucket"].min(),
        "to",
        test_df["time_bucket"].max()
    )

    print("\nTarget Distribution:")
    print(
        df[TARGET]
        .value_counts()
        .head(15)
    )

    print("\nZero Ratio:")
    print(
        round(
            float((df[TARGET] == 0).mean()),
            4
        )
    )

    alert_threshold = build_calibrated_threshold(
        train_df
    )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("\nTraining holdout evaluation model...")

    eval_bundle = fit_hurdle_bundle(
        X_train=X_train,
        y_train=y_train,
        alert_threshold=alert_threshold
    )

    metrics = evaluate_hurdle_model(
        bundle=eval_bundle,
        X_test=X_test,
        y_test=y_test
    )

    print("\n" + "=" * 60)
    print("FORECAST HOLDOUT RESULTS")
    print("=" * 60)

    print(f"MAE              : {metrics['mae']:.4f}")
    print(f"RMSE             : {metrics['rmse']:.4f}")
    print(f"R²               : {metrics['r2']:.4f}")

    print("\n" + "-" * 60)
    print("ALERT CLASSIFICATION RESULTS")
    print("-" * 60)

    print(f"Precision        : {metrics['alert_precision']:.4f}")
    print(f"Recall           : {metrics['alert_recall']:.4f}")
    print(f"F1               : {metrics['alert_f1']:.4f}")
    print(f"ROC-AUC          : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC           : {metrics['pr_auc']:.4f}")

    print("\nTraining production hurdle model on full dataset...")

    X_full = df[FEATURES]
    y_full = df[TARGET]

    final_bundle = fit_hurdle_bundle(
        X_train=X_full,
        y_train=y_full,
        alert_threshold=alert_threshold
    )

    final_bundle["holdout_metrics"] = metrics

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        final_bundle,
        "models/timeseries_forecast_model.pkl"
    )

    joblib.dump(
        final_bundle,
        "models/timeseries_forecast.pkl"
    )

    print("\nModel Saved:")
    print("models/timeseries_forecast_model.pkl")
    print("models/timeseries_forecast.pkl")

    return final_bundle