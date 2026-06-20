import numpy as np
import pandas as pd
from src.features.event_calender import (
    add_event_calendar_features_to_corridor_timeseries,
)
from src.features.feedback_training import (
    apply_feedback_to_timeseries_counts,
)

FEATURES = [
    "corridor",

    "hour",
    "weekday",
    "month",
    "is_event_day",
    "calendar_event_type",
    "calendar_event_intensity",

    "hour_sin",
    "hour_cos",

    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",

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


PROFILE_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_72",
    "lag_168",

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


def parse_datetime_robust(series):
    """
    Robust datetime parser.

    First pass:
        normal pd.to_datetime(..., utc=True)

    Second pass:
        format='mixed' for rows that failed.

    Returns timezone-naive UTC timestamps.
    """

    raw_series = series.copy()

    first_pass = pd.to_datetime(
        raw_series,
        errors="coerce",
        utc=True
    )

    failed_mask = (
        first_pass.isna()
        &
        raw_series.notna()
    )

    if failed_mask.any():
        try:
            second_pass = pd.to_datetime(
                raw_series.loc[failed_mask],
                format="mixed",
                errors="coerce",
                utc=True
            )

        except Exception:
            second_pass = pd.to_datetime(
                raw_series.loc[failed_mask],
                errors="coerce",
                utc=True
            )

        first_pass.loc[failed_mask] = second_pass

    return first_pass.dt.tz_convert(None)


def clean_text(value):
    return (
        str(value or "UNKNOWN")
        .strip()
    )


def safe_numeric(series, fallback=0.0):
    return (
        pd.to_numeric(
            series,
            errors="coerce"
        )
        .fillna(fallback)
    )


def ensure_column(df, col, default="UNKNOWN"):
    if col not in df.columns:
        df[col] = default

    return df


def add_lag_and_rolling_features(ts_df):
    ts_df = ts_df.sort_values(
        [
            "corridor",
            "time_bucket"
        ]
    ).copy()

    grouped = ts_df.groupby(
        "corridor"
    )["incident_count"]

    # ------------------------------------------------
    # Lag features
    # ------------------------------------------------

    for lag in [
        1,
        2,
        3,
        24,
        48,
        72,
        168
    ]:
        ts_df[f"lag_{lag}"] = (
            grouped
            .shift(lag)
        )

    # ------------------------------------------------
    # Rolling features
    # shifted to avoid current-hour leakage
    # ------------------------------------------------

    shifted = grouped.shift(1)

    for window in [
        6,
        12,
        24,
        168
    ]:
        ts_df[f"rolling_{window}"] = (
            shifted
            .groupby(ts_df["corridor"])
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # ------------------------------------------------
    # Corridor average / volatility
    # ------------------------------------------------

    corridor_avg = (
        ts_df
        .groupby("corridor")["incident_count"]
        .mean()
    )

    corridor_std = (
        ts_df
        .groupby("corridor")["incident_count"]
        .std()
    )

    ts_df["corridor_avg"] = (
        ts_df["corridor"]
        .map(corridor_avg)
        .fillna(0)
    )

    ts_df["corridor_volatility"] = (
        ts_df["corridor"]
        .map(corridor_std)
        .fillna(0)
    )

    # ------------------------------------------------
    # Safe fill base features
    # ------------------------------------------------

    base_features = [
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

    for col in base_features:
        if col not in ts_df.columns:
            ts_df[col] = 0.0

        ts_df[col] = pd.to_numeric(
            ts_df[col],
            errors="coerce"
        ).fillna(0.0)

    # ------------------------------------------------
    # New compressed lag-signal features
    # ------------------------------------------------

    ts_df["any_incident_last_3h"] = (
        (
            (ts_df["lag_1"] > 0)
            |
            (ts_df["lag_2"] > 0)
            |
            (ts_df["lag_3"] > 0)
        )
        .astype(int)
    )

    ts_df["incidents_last_24h"] = (
        ts_df["rolling_24"]
        *
        24
    )

    ts_df["incidents_last_24h"] = (
        ts_df["incidents_last_24h"]
        .replace(
            [
                float("inf"),
                -float("inf")
            ],
            0
        )
        .fillna(0)
    )

    ts_df["above_corridor_avg"] = (
        ts_df["rolling_6"]
        >
        ts_df["corridor_avg"]
    ).astype(int)

    # ------------------------------------------------
    # Final safe fill
    # ------------------------------------------------

    for col in PROFILE_FEATURES:
        if col not in ts_df.columns:
            ts_df[col] = 0.0

        ts_df[col] = pd.to_numeric(
            ts_df[col],
            errors="coerce"
        ).fillna(0.0)

    return ts_df


def build_static_corridor_features(df):
    df = df.copy()

    zone_counts = (
        df["zone"]
        .value_counts()
    )

    junction_counts = (
        df["junction"]
        .value_counts()
    )

    cause_counts = (
        df["event_cause"]
        .value_counts()
    )

    closure_counts = (
        df["requires_road_closure"]
        .value_counts()
    )

    cluster_counts = (
        df["corridor"]
        .value_counts()
    )

    df["zone_risk_raw"] = (
        df["zone"]
        .map(zone_counts)
        .fillna(0)
    )

    df["junction_risk_raw"] = (
        df["junction"]
        .map(junction_counts)
        .fillna(0)
    )

    df["cause_risk_raw"] = (
        df["event_cause"]
        .map(cause_counts)
        .fillna(0)
    )

    df["closure_risk_raw"] = (
        df["requires_road_closure"]
        .map(closure_counts)
        .fillna(0)
    )

    df["cluster_risk_raw"] = (
        df["corridor"]
        .map(cluster_counts)
        .fillna(0)
    )

    static_features = (
        df
        .groupby("corridor")
        .agg(
            zone_risk=("zone_risk_raw", "mean"),
            junction_risk=("junction_risk_raw", "mean"),
            cause_risk=("cause_risk_raw", "mean"),
            closure_risk=("closure_risk_raw", "mean"),
            cluster_risk=("cluster_risk_raw", "mean"),
        )
        .reset_index()
    )

    return static_features


def build_timeseries_dataset(df):
    print("\n" + "=" * 70)
    print("BUILDING CORRIDOR-HOUR TIME-SERIES DATASET")
    print("=" * 70)

    df = df.copy()

    # ==================================================
    # Required columns
    # ==================================================

    required_defaults = {
        "corridor": "Non-corridor",
        "zone": "UNKNOWN",
        "junction": "UNKNOWN",
        "event_cause": "others",
        "requires_road_closure": "False",
    }

    for col, default in required_defaults.items():
        ensure_column(
            df,
            col,
            default
        )

    # ==================================================
    # Robust datetime parsing
    # ==================================================

    df["start_datetime_raw"] = df["start_datetime"].copy()

    initial_parse = pd.to_datetime(
        df["start_datetime_raw"],
        errors="coerce",
        utc=True
    )

    initial_failed = int(
        initial_parse.isna().sum()
    )

    df["start_datetime"] = parse_datetime_robust(
        df["start_datetime_raw"]
    )

    final_failed = int(
        df["start_datetime"].isna().sum()
    )

    recovered = (
        initial_failed
        -
        final_failed
    )

    print("\nDatetime Parse Recovery")
    print("-" * 50)
    print(f"Initially failed : {initial_failed}")
    print(f"Recovered        : {recovered}")
    print(f"Still failed     : {final_failed}")

    df = df.dropna(
        subset=[
            "start_datetime"
        ]
    ).copy()

    # ==================================================
    # Clean categorical columns
    # ==================================================

    for col in [
        "corridor",
        "zone",
        "junction",
        "event_cause",
        "requires_road_closure",
    ]:
        df[col] = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

    df["corridor"] = (
        df["corridor"]
        .replace(
            {
                "": "Non-corridor",
                "nan": "Non-corridor",
                "None": "Non-corridor",
            }
        )
    )

    # ==================================================
    # Build hourly buckets
    # ==================================================

    df["time_bucket"] = (
        df["start_datetime"]
        .dt.floor("h")
    )

    min_time = df["time_bucket"].min()
    max_time = df["time_bucket"].max()

    corridors = sorted(
        df["corridor"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not corridors:
        corridors = [
            "Non-corridor"
        ]

    all_hours = pd.date_range(
        start=min_time,
        end=max_time,
        freq="h"
    )

    full_index = pd.MultiIndex.from_product(
        [
            all_hours,
            corridors
        ],
        names=[
            "time_bucket",
            "corridor"
        ]
    )

    # ==================================================
    # Aggregate target
    # ==================================================

    event_counts = (
        df
        .groupby(
            [
                "time_bucket",
                "corridor"
            ]
        )
        .size()
        .reindex(
            full_index,
            fill_value=0
        )
        .reset_index(name="incident_count")
    )

    ts_df = event_counts.copy()

    ts_df = apply_feedback_to_timeseries_counts(
        ts_df
    )

    # ==================================================
    # Add static corridor risk features
    # ==================================================

    static_features = build_static_corridor_features(
        df
    )

    ts_df = ts_df.merge(
        static_features,
        on="corridor",
        how="left"
    )

    risk_cols = [
        "zone_risk",
        "junction_risk",
        "cause_risk",
        "closure_risk",
        "cluster_risk",
    ]

    for col in risk_cols:
        ts_df[col] = (
            pd.to_numeric(
                ts_df[col],
                errors="coerce"
            )
            .fillna(0)
        )

    # ==================================================
    # Time features
    # ==================================================

    ts_df["hour"] = (
        ts_df["time_bucket"]
        .dt.hour
        .astype(int)
    )

    ts_df["weekday"] = (
        ts_df["time_bucket"]
        .dt.weekday
        .astype(int)
    )

    ts_df["month"] = (
        ts_df["time_bucket"]
        .dt.month
        .astype(int)
    )

    ts_df["hour_sin"] = np.sin(
        2 * np.pi * ts_df["hour"] / 24
    )

    ts_df["hour_cos"] = np.cos(
        2 * np.pi * ts_df["hour"] / 24
    )

    # ==================================================
    # Lag, rolling, and derived lag features
    # ==================================================

    ts_df = add_lag_and_rolling_features(
        ts_df
    )

    # ==================================================
    # Final column order
    # ==================================================

    for col in FEATURES:
        if col not in ts_df.columns:
            if col == "calendar_event_type":
                ts_df[col] = "none"
            else:
                ts_df[col] = 0

    ts_df = ts_df[
        [
            "time_bucket",
            "incident_count"
        ]
        +
        FEATURES
    ]

    print("\nFinal Time-Series Dataset Shape:")
    print(
        ts_df.shape
    )

    print("\nTarget Distribution:")
    print(
        ts_df["incident_count"]
        .value_counts()
        .head(15)
    )

    print("\nZero Ratio:")
    print(
        round(
            (ts_df["incident_count"] == 0).mean(),
            4
        )
    )

    print("\nFeatures:")
    print(
        FEATURES
    )

    return ts_df