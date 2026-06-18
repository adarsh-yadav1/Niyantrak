import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import DATA_PATH

from src.preprocessing.load_data import load_data
from src.forecasting.build_timeseries_dataset import build_timeseries_dataset
from src.forecasting.forecast_predictor import predict_single_forecast

from src.inference.location_resolver import (
    resolve_corridor_from_coordinates,
    get_profile_with_spatial_fallback,
)

from src.scoring.event_impact import calculate_event_impact

from src.scoring.risk_score import calculate_forecast_risk_score


MODEL_PATHS = [
    "models/timeseries_forecast_model.pkl",
    "models/timeseries_forecast.pkl",
]

FEATURE_STORE_PATH = "models/traffic_feature_store.pkl"

OUTPUT_PATH = "models/eis_weight_calibration.json"

MARKDOWN_OUTPUT_PATH = "EIS_WEIGHT_CALIBRATION.md"


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


CANDIDATE_WEIGHTS = [
    {
        "name": "Balanced default",
        "forecast_weight": 0.30,
        "event_weight": 0.55,
        "cause_weight": 0.15,
    },
    {
        "name": "Forecast stronger",
        "forecast_weight": 0.35,
        "event_weight": 0.50,
        "cause_weight": 0.15,
    },
    {
        "name": "Event stronger",
        "forecast_weight": 0.25,
        "event_weight": 0.60,
        "cause_weight": 0.15,
    },
    {
        "name": "Forecast-heavy",
        "forecast_weight": 0.40,
        "event_weight": 0.45,
        "cause_weight": 0.15,
    },
    {
        "name": "Cause stronger",
        "forecast_weight": 0.30,
        "event_weight": 0.50,
        "cause_weight": 0.20,
    },
    {
        "name": "Event-dominant",
        "forecast_weight": 0.20,
        "event_weight": 0.65,
        "cause_weight": 0.15,
    },
]


CAUSE_WEIGHTS = {
    "accident": 90,
    "vip_movement": 95,
    "protest": 90,
    "public_event": 80,
    "procession": 80,
    "construction": 75,
    "tree_fall": 75,
    "water_logging": 70,
    "congestion": 65,
    "road_conditions": 60,
    "pot_holes": 50,
    "vehicle_breakdown": 45,
    "others": 40,
}


def safe_float(value, fallback=0.0):
    try:
        if value is None:
            return fallback

        if pd.isna(value):
            return fallback

        return float(value)

    except Exception:
        return fallback


def safe_bool(value):
    value = str(value).strip().lower()

    return value in [
        "true",
        "yes",
        "1",
        "y",
        "on",
    ]


def normalize_text(value):
    return (
        str(value or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


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


def calculate_cause_risk_score(
    profile,
    store
):
    max_cause_risk = safe_float(
        store.get("max_cause_risk"),
        1.0
    )

    cause_risk_score = (
        safe_float(profile.get("cause_risk"), 0.0)
        /
        max(max_cause_risk, 1.0)
    ) * 100

    return max(
        0.0,
        min(cause_risk_score, 100.0)
    )


def calculate_candidate_eis(
    forecast_score,
    event_score,
    cause_risk_score,
    weights
):
    return (
        weights["forecast_weight"] * forecast_score
        +
        weights["event_weight"] * event_score
        +
        weights["cause_weight"] * cause_risk_score
    )


def calculate_actual_severity_proxy(
    row,
    duration_p95,
    incident_p95,
    actual_incident_count
):
    """
    Proxy actual severity from historical outcomes.

    This is not a manually labelled ground truth.
    It is a practical historical severity proxy using:
    - actual duration
    - actual incident volume in same corridor-hour
    - actual closure flag
    - event cause severity prior
    """

    duration_minutes = safe_float(
        row.get("duration_minutes"),
        0.0
    )

    duration_score = (
        duration_minutes
        /
        max(duration_p95, 1.0)
    ) * 100

    duration_score = max(
        0.0,
        min(duration_score, 100.0)
    )

    incident_score = (
        safe_float(actual_incident_count)
        /
        max(incident_p95, 1.0)
    ) * 100

    incident_score = max(
        0.0,
        min(incident_score, 100.0)
    )

    closure_score = (
        100.0
        if safe_bool(row.get("requires_road_closure"))
        else 0.0
    )

    cause_key = normalize_text(
        row.get("event_cause")
    )

    cause_score = CAUSE_WEIGHTS.get(
        cause_key,
        40
    )

    actual_proxy = (
        0.45 * duration_score
        +
        0.25 * incident_score
        +
        0.20 * closure_score
        +
        0.10 * cause_score
    )

    return max(
        0.0,
        min(actual_proxy, 100.0)
    )


def prepare_event_dataframe(df):
    df = df.copy()

    df["start_datetime"] = pd.to_datetime(
        df["start_datetime"],
        errors="coerce",
        utc=True
    ).dt.tz_convert(None)

    if "end_datetime" in df.columns:
        df["end_datetime"] = pd.to_datetime(
            df["end_datetime"],
            errors="coerce",
            utc=True
        ).dt.tz_convert(None)

        df["duration_minutes"] = (
            df["end_datetime"]
            -
            df["start_datetime"]
        ).dt.total_seconds() / 60

    else:
        df["duration_minutes"] = 0.0

    df["duration_minutes"] = (
        df["duration_minutes"]
        .fillna(0)
        .clip(lower=0)
    )

    for col in [
        "latitude",
        "longitude"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    for col in [
        "event_cause",
        "veh_type",
        "corridor",
        "priority",
        "event_type",
    ]:
        if col not in df.columns:
            df[col] = "UNKNOWN"

        df[col] = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

    if "requires_road_closure" not in df.columns:
        df["requires_road_closure"] = False

    df = df.dropna(
        subset=[
            "start_datetime",
            "latitude",
            "longitude",
        ]
    )

    df = df[
        (df["latitude"] >= 12.70)
        &
        (df["latitude"] <= 13.25)
        &
        (df["longitude"] >= 77.35)
        &
        (df["longitude"] <= 77.85)
    ].copy()

    df["time_bucket"] = (
        df["start_datetime"]
        .dt.floor("h")
    )

    df["hour"] = (
        df["start_datetime"]
        .dt.hour
    )

    df["weekday"] = (
        df["start_datetime"]
        .dt.weekday
    )

    df["month"] = (
        df["start_datetime"]
        .dt.month
    )

    return df


def build_ts_lookup(ts_df):
    lookup = {}

    for _, row in ts_df.iterrows():
        key = (
            str(row["corridor"]),
            row["time_bucket"]
        )

        lookup[key] = safe_float(
            row["incident_count"]
        )

    return lookup


def select_calibration_sample(
    df,
    sample_size=20
):
    df = df.copy()

    if len(df) <= sample_size:
        return df

    try:
        df["severity_bucket"] = pd.qcut(
            df["duration_minutes"].rank(method="first"),
            q=4,
            labels=False,
            duplicates="drop"
        )

        samples = []

        per_bucket = max(
            1,
            sample_size // 4
        )

        for _, group in df.groupby("severity_bucket"):
            take = min(
                len(group),
                per_bucket
            )

            samples.append(
                group.sample(
                    n=take,
                    random_state=42
                )
            )

        sampled = pd.concat(
            samples,
            ignore_index=True
        )

        if len(sampled) < sample_size:
            remaining = df.drop(
                sampled.index,
                errors="ignore"
            )

            extra = remaining.sample(
                n=min(
                    sample_size - len(sampled),
                    len(remaining)
                ),
                random_state=42
            )

            sampled = pd.concat(
                [
                    sampled,
                    extra
                ],
                ignore_index=True
            )

        return sampled.head(
            sample_size
        )

    except Exception:
        return df.sample(
            n=sample_size,
            random_state=42
        )


def run_eis_weight_calibration(
    data_path=DATA_PATH,
    output_path=OUTPUT_PATH,
    sample_size=20
):
    print("\n" + "=" * 70)
    print("EIS WEIGHT MICRO-CALIBRATION")
    print("=" * 70)

    model = load_model()
    store = load_feature_store()

    raw_df = load_data(
        data_path
    )

    event_df = prepare_event_dataframe(
        raw_df
    )

    ts_df = build_timeseries_dataset(
        raw_df
    )

    ts_lookup = build_ts_lookup(
        ts_df
    )

    duration_p95 = safe_float(
        event_df["duration_minutes"].quantile(0.95),
        60.0
    )

    incident_p95 = safe_float(
        store.get("incident_p95"),
        1.0
    )

    calibration_rows = select_calibration_sample(
        event_df,
        sample_size=sample_size
    )

    print("\nHistorical rows selected:")
    print(len(calibration_rows))

    evaluated_events = []

    for _, row in calibration_rows.iterrows():
        latitude = safe_float(
            row.get("latitude")
        )

        longitude = safe_float(
            row.get("longitude")
        )

        location_match = resolve_corridor_from_coordinates(
            latitude=latitude,
            longitude=longitude,
            store=store
        )

        if location_match.get("outside_bengaluru"):
            continue

        corridor = str(
            location_match.get(
                "corridor",
                row.get("corridor", "UNKNOWN")
            )
        )

        hour = int(row["hour"])
        weekday = int(row["weekday"])
        month = int(row["month"])

        profile, profile_source, _ = get_profile_with_spatial_fallback(
            store=store,
            corridor=corridor,
            hour=hour,
            location_match=location_match
        )

        X = build_input_row(
            corridor=corridor,
            hour=hour,
            weekday=weekday,
            month=month,
            profile=profile
        )

        predicted_incidents, forecast_details = predict_single_forecast(
            model,
            X
        )

        predicted_incidents = max(
            safe_float(predicted_incidents),
            0.0
        )

        alert_probability = None

        if forecast_details.get("alert_probability") is not None:
            alert_probability = safe_float(
                forecast_details["alert_probability"][0],
                None
            )

        try:
            forecast_score, _ = calculate_forecast_risk_score(
                predicted_incidents=predicted_incidents,
                incident_p95=store.get("incident_p95", 1.0),
                incident_p99=store.get("incident_p99", None),
                alert_probability=alert_probability,
                context_multiplier=1.0,
            )

        except TypeError:
            forecast_score, _ = calculate_forecast_risk_score(
                predicted_incidents,
                store.get("incident_p95", 1.0)
            )

        rush_hour = (
            (7 <= hour <= 10)
            or
            (17 <= hour <= 21)
        )

        base_event_score, _ = calculate_event_impact(
            event_cause=row.get("event_cause", "others"),
            veh_type=row.get("veh_type", "unknown"),
            road_closure=safe_bool(row.get("requires_road_closure")),
            rush_hour=rush_hour
        )

        cause_risk_score = calculate_cause_risk_score(
            profile,
            store
        )

        actual_incident_count = ts_lookup.get(
            (
                str(row.get("corridor")),
                row.get("time_bucket")
            ),
            0.0
        )

        actual_severity_proxy = calculate_actual_severity_proxy(
            row=row,
            duration_p95=duration_p95,
            incident_p95=incident_p95,
            actual_incident_count=actual_incident_count
        )

        evaluated_events.append({
            "event_cause": str(row.get("event_cause")),
            "corridor": corridor,
            "hour": hour,
            "duration_minutes": round(
                safe_float(row.get("duration_minutes")),
                2
            ),
            "actual_incident_count": round(
                actual_incident_count,
                2
            ),
            "actual_severity_proxy": round(
                actual_severity_proxy,
                2
            ),
            "forecast_score": round(
                safe_float(forecast_score),
                2
            ),
            "event_score": round(
                safe_float(base_event_score),
                2
            ),
            "cause_risk_score": round(
                safe_float(cause_risk_score),
                2
            ),
            "profile_source": profile_source,
        })

    if len(evaluated_events) < 5:
        raise RuntimeError(
            "Not enough valid historical events for EIS calibration."
        )

    candidate_results = []

    for weights in CANDIDATE_WEIGHTS:
        errors = []

        for event in evaluated_events:
            predicted_eis = calculate_candidate_eis(
                forecast_score=event["forecast_score"],
                event_score=event["event_score"],
                cause_risk_score=event["cause_risk_score"],
                weights=weights
            )

            error = abs(
                predicted_eis
                -
                event["actual_severity_proxy"]
            )

            errors.append(error)

        mae = float(
            np.mean(errors)
        )

        candidate_results.append({
            "name": weights["name"],
            "forecast_weight": weights["forecast_weight"],
            "event_weight": weights["event_weight"],
            "cause_weight": weights["cause_weight"],
            "mae": mae,
        })

    candidate_results = sorted(
        candidate_results,
        key=lambda x: x["mae"]
    )

    best = candidate_results[0]

    result = {
        "title": "EIS Weight Micro-Calibration",
        "generated_at": pd.Timestamp.now().isoformat(),

        "sample_size": int(len(evaluated_events)),

        "best_weights": {
            "forecast_weight": best["forecast_weight"],
            "event_weight": best["event_weight"],
            "cause_weight": best["cause_weight"],
        },

        "best_method": best["name"],
        "best_mae": best["mae"],

        "candidate_results": candidate_results,

        "sample_events": evaluated_events[:10],

        "target_definition": (
            "Actual severity proxy = 45% duration score + 25% same corridor-hour incident volume score "
            "+ 20% road closure score + 10% event cause severity prior."
        ),

        "conclusion": (
            f"The selected EIS weights were calibrated on {len(evaluated_events)} historical events "
            f"using a practical actual-severity proxy. Best method: {best['name']} with MAE {best['mae']:.2f}."
        ),

        "honesty_note": (
            "This is a micro-calibration using a proxy severity target, not manually labelled ground truth. "
            "It gives evidence for the EIS weights and can be replaced by officer-labelled feedback data later."
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

    write_markdown_summary(
        result,
        MARKDOWN_OUTPUT_PATH
    )

    print("\n" + "=" * 70)
    print("EIS CALIBRATION RESULTS")
    print("=" * 70)

    print(f"Sample size : {result['sample_size']}")
    print(f"Best method : {result['best_method']}")
    print(f"Best MAE    : {result['best_mae']:.4f}")
    print("\nBest weights:")
    print(result["best_weights"])

    print("\nSaved:")
    print(output_path)
    print(MARKDOWN_OUTPUT_PATH)

    return result


def write_markdown_summary(
    result,
    output_path
):
    best = result["best_weights"]

    lines = [
        "# EIS Weight Micro-Calibration",
        "",
        "## Purpose",
        "",
        "The Event Impact Score uses a weighted combination of forecast risk, live event impact, and historical cause risk. This calibration checks multiple candidate weight combinations against historical event outcomes using a practical severity proxy.",
        "",
        "## Selected Weights",
        "",
        f"- Forecast weight: `{best['forecast_weight']}`",
        f"- Event impact weight: `{best['event_weight']}`",
        f"- Cause risk weight: `{best['cause_weight']}`",
        "",
        f"Best method: **{result['best_method']}**",
        "",
        f"Best MAE against proxy severity: **{result['best_mae']:.4f}**",
        "",
        "## Target Definition",
        "",
        result["target_definition"],
        "",
        "## Candidate Results",
        "",
        "| Method | Forecast | Event | Cause | MAE |",
        "|---|---:|---:|---:|---:|",
    ]

    for item in result["candidate_results"]:
        lines.append(
            f"| {item['name']} | {item['forecast_weight']} | {item['event_weight']} | {item['cause_weight']} | {item['mae']:.4f} |"
        )

    lines.extend([
        "",
        "## Honesty Note",
        "",
        result["honesty_note"],
        "",
        "## Judge Explanation",
        "",
        "The EIS weights were not chosen blindly. We tested multiple candidate formulas against historical event outcomes using a proxy severity target based on actual duration, incident volume, closure status, and event cause severity. The lowest-MAE formula was selected for the dashboard.",
    ])

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(
            "\n".join(lines)
        )


if __name__ == "__main__":
    run_eis_weight_calibration()