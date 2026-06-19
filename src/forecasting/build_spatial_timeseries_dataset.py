import joblib
import numpy as np
import pandas as pd

from src.inference.location_resolver import (
    haversine_distance_meters,
    is_bad_corridor_name,
)


SPATIAL_FEATURES = [
    "spatial_cluster_id",

    "latitude",
    "longitude",

    "dominant_corridor",
    "nearest_corridor_distance_m",
    "nearest_hotspot_distance_m",
    "spatial_density_at_point",

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


def safe_float(value, fallback=0.0):
    try:
        if value is None:
            return fallback

        if pd.isna(value):
            return fallback

        return float(value)

    except Exception:
        return fallback


def normalize_datetime_utc_naive(series):
    raw_series = series.copy()

    parsed = pd.to_datetime(
        raw_series,
        errors="coerce",
        utc=True
    )

    failed_mask = (
        parsed.isna()
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

        parsed.loc[failed_mask] = second_pass

    return parsed.dt.tz_convert(None)


def ensure_column(df, col, default="UNKNOWN"):
    if col not in df.columns:
        df[col] = default

    return df


def prepare_raw_location_events(df):
    df = df.copy()

    ensure_column(
        df,
        "corridor",
        "Non-corridor"
    )

    ensure_column(
        df,
        "zone",
        "UNKNOWN"
    )

    ensure_column(
        df,
        "junction",
        "UNKNOWN"
    )

    ensure_column(
        df,
        "event_cause",
        "others"
    )

    ensure_column(
        df,
        "requires_road_closure",
        "False"
    )

    df["start_datetime_raw"] = df["start_datetime"].copy()

    initial_parse = pd.to_datetime(
        df["start_datetime_raw"],
        errors="coerce",
        utc=True
    )

    initial_failed = int(
        initial_parse.isna().sum()
    )

    df["start_datetime"] = normalize_datetime_utc_naive(
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

    print("\nSpatial Datetime Parse Recovery")
    print("-" * 50)
    print(f"Initially failed : {initial_failed}")
    print(f"Recovered        : {recovered}")
    print(f"Still failed     : {final_failed}")

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

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

    df = df.dropna(
        subset=[
            "start_datetime",
            "latitude",
            "longitude",
        ]
    ).copy()

    # Bengaluru bounds
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

    return df


def assign_spatial_clusters(df, store):
    df = df.copy()

    model = store.get(
        "spatial_cluster_model"
    )

    if model is None:
        raise ValueError(
            "Feature store does not contain spatial_cluster_model. "
            "Run prepare_feature_store.py before training spatial model."
        )

    input_df = pd.DataFrame({
        "latitude": df["latitude"].astype(float),
        "longitude": df["longitude"].astype(float),
    })

    df["spatial_cluster_id"] = (
        model
        .predict(input_df)
        .astype(int)
    )

    return df


def get_cluster_centers(store, events=None):
    centers = store.get(
        "spatial_cluster_centers",
        {}
    )

    output = {}

    for cluster_id, center in centers.items():
        output[int(cluster_id)] = {
            "latitude": safe_float(
                center.get("latitude")
            ),
            "longitude": safe_float(
                center.get("longitude")
            ),
        }

    if output:
        return output

    # fallback if centers are not stored
    if events is not None and len(events) > 0:
        grouped = (
            events
            .groupby("spatial_cluster_id")
            .agg(
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
            )
            .reset_index()
        )

        for _, row in grouped.iterrows():
            output[int(row["spatial_cluster_id"])] = {
                "latitude": safe_float(row["latitude"]),
                "longitude": safe_float(row["longitude"]),
            }

    return output


def get_dominant_corridor_map(events):
    dominant = {}

    for cluster_id, group in events.groupby("spatial_cluster_id"):
        corridors = (
            group["corridor"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

        corridors = corridors[
            ~corridors.map(is_bad_corridor_name)
        ]

        if len(corridors) == 0:
            dominant[int(cluster_id)] = "Non-corridor"

        else:
            dominant[int(cluster_id)] = (
                corridors
                .mode()
                .iloc[0]
            )

    return dominant


def get_nearest_corridor_distance_map(
    store,
    centers,
    dominant_corridor_map
):
    corridor_profiles = store.get(
        "corridor_location_profiles",
        {}
    )

    output = {}

    for cluster_id, center in centers.items():
        dominant_corridor = dominant_corridor_map.get(
            cluster_id,
            "Non-corridor"
        )

        corridor_profile = corridor_profiles.get(
            dominant_corridor
        )

        if corridor_profile is None:
            output[cluster_id] = 9999.0
            continue

        output[cluster_id] = haversine_distance_meters(
            center["latitude"],
            center["longitude"],
            corridor_profile["latitude"],
            corridor_profile["longitude"]
        )

    return output


def get_nearest_hotspot_distance_map(
    store,
    centers
):
    hotspots = store.get(
        "hotspot_points",
        []
    )

    output = {}

    for cluster_id, center in centers.items():
        best_distance = None

        for point in hotspots:
            try:
                distance = haversine_distance_meters(
                    center["latitude"],
                    center["longitude"],
                    point["latitude"],
                    point["longitude"]
                )

                if best_distance is None or distance < best_distance:
                    best_distance = distance

            except Exception:
                continue

        output[cluster_id] = (
            9999.0
            if best_distance is None
            else best_distance
        )

    return output


def build_static_cluster_features(events, store):
    centers = get_cluster_centers(
        store,
        events
    )

    dominant_corridor_map = get_dominant_corridor_map(
        events
    )

    nearest_corridor_distance_map = get_nearest_corridor_distance_map(
        store,
        centers,
        dominant_corridor_map
    )

    nearest_hotspot_distance_map = get_nearest_hotspot_distance_map(
        store,
        centers
    )

    cluster_event_count = (
        events
        .groupby("spatial_cluster_id")
        .size()
    )

    max_cluster_count = max(
        cluster_event_count.max(),
        1
    )

    zone_counts = (
        events["zone"]
        .value_counts()
    )

    junction_counts = (
        events["junction"]
        .value_counts()
    )

    cause_counts = (
        events["event_cause"]
        .value_counts()
    )

    closure_counts = (
        events["requires_road_closure"]
        .value_counts()
    )

    events["zone_risk_raw"] = (
        events["zone"]
        .map(zone_counts)
        .fillna(0)
    )

    events["junction_risk_raw"] = (
        events["junction"]
        .map(junction_counts)
        .fillna(0)
    )

    events["cause_risk_raw"] = (
        events["event_cause"]
        .map(cause_counts)
        .fillna(0)
    )

    events["closure_risk_raw"] = (
        events["requires_road_closure"]
        .map(closure_counts)
        .fillna(0)
    )

    static_rows = []

    for cluster_id, center in centers.items():
        group = events[
            events["spatial_cluster_id"] == cluster_id
        ]

        if len(group) == 0:
            zone_risk = 0.0
            junction_risk = 0.0
            cause_risk = 0.0
            closure_risk = 0.0
            count = 0

        else:
            zone_risk = group["zone_risk_raw"].mean()
            junction_risk = group["junction_risk_raw"].mean()
            cause_risk = group["cause_risk_raw"].mean()
            closure_risk = group["closure_risk_raw"].mean()
            count = int(len(group))

        static_rows.append({
            "spatial_cluster_id": int(cluster_id),

            "latitude": center["latitude"],
            "longitude": center["longitude"],

            "dominant_corridor": dominant_corridor_map.get(
                cluster_id,
                "Non-corridor"
            ),

            "nearest_corridor_distance_m": nearest_corridor_distance_map.get(
                cluster_id,
                9999.0
            ),

            "nearest_hotspot_distance_m": nearest_hotspot_distance_map.get(
                cluster_id,
                9999.0
            ),

            "spatial_density_at_point": count / max_cluster_count,

            "zone_risk": safe_float(zone_risk),
            "junction_risk": safe_float(junction_risk),
            "cause_risk": safe_float(cause_risk),
            "closure_risk": safe_float(closure_risk),
            "cluster_risk": safe_float(count),
        })

    return pd.DataFrame(
        static_rows
    )


def add_lag_and_rolling_features(spatial_ts):
    spatial_ts = spatial_ts.sort_values(
        [
            "spatial_cluster_id",
            "time_bucket"
        ]
    ).copy()

    grouped = spatial_ts.groupby(
        "spatial_cluster_id"
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
        spatial_ts[f"lag_{lag}"] = (
            grouped
            .shift(lag)
        )

    # ------------------------------------------------
    # Rolling features
    # shifted to avoid leakage
    # ------------------------------------------------

    shifted_incidents = grouped.shift(1)

    for window in [
        6,
        12,
        24,
        168
    ]:
        spatial_ts[f"rolling_{window}"] = (
            shifted_incidents
            .groupby(spatial_ts["spatial_cluster_id"])
            .rolling(window)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # ------------------------------------------------
    # Cluster average / volatility
    # ------------------------------------------------

    cluster_avg = (
        spatial_ts
        .groupby("spatial_cluster_id")["incident_count"]
        .mean()
    )

    cluster_std = (
        spatial_ts
        .groupby("spatial_cluster_id")["incident_count"]
        .std()
    )

    spatial_ts["corridor_avg"] = (
        spatial_ts["spatial_cluster_id"]
        .map(cluster_avg)
        .fillna(0)
    )

    spatial_ts["corridor_volatility"] = (
        spatial_ts["spatial_cluster_id"]
        .map(cluster_std)
        .fillna(0)
    )

    # ------------------------------------------------
    # Fill base features one by one
    # ------------------------------------------------

    base_numeric_features = [
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

    for col in base_numeric_features:
        if col not in spatial_ts.columns:
            spatial_ts[col] = 0.0

        spatial_ts[col] = pd.to_numeric(
            spatial_ts[col],
            errors="coerce"
        ).fillna(0.0)

    # ------------------------------------------------
    # Compressed lag-signal features
    # ------------------------------------------------

    spatial_ts["any_incident_last_3h"] = (
        (
            (spatial_ts["lag_1"] > 0)
            |
            (spatial_ts["lag_2"] > 0)
            |
            (spatial_ts["lag_3"] > 0)
        )
        .astype(int)
    )

    spatial_ts["incidents_last_24h"] = (
        spatial_ts["rolling_24"]
        *
        24
    )

    spatial_ts["incidents_last_24h"] = (
        spatial_ts["incidents_last_24h"]
        .replace(
            [
                float("inf"),
                -float("inf")
            ],
            0
        )
        .fillna(0)
    )

    spatial_ts["above_corridor_avg"] = (
        spatial_ts["rolling_6"]
        >
        spatial_ts["corridor_avg"]
    ).astype(int)

    # ------------------------------------------------
    # Final safe fill
    # ------------------------------------------------

    for col in PROFILE_FEATURES:
        if col not in spatial_ts.columns:
            spatial_ts[col] = 0.0

        spatial_ts[col] = pd.to_numeric(
            spatial_ts[col],
            errors="coerce"
        ).fillna(0.0)

    return spatial_ts


def build_spatial_timeseries_dataset(df, store):
    print("\n" + "=" * 70)
    print("BUILDING SPATIAL-CLUSTER TIME-SERIES DATASET")
    print("=" * 70)

    events = prepare_raw_location_events(
        df
    )

    if len(events) == 0:
        raise ValueError(
            "No valid Bengaluru location events available for spatial model."
        )

    events = assign_spatial_clusters(
        events,
        store
    )

    static_features = build_static_cluster_features(
        events,
        store
    )

    centers = get_cluster_centers(
        store,
        events
    )

    all_clusters = sorted(
        list(centers.keys())
    )

    min_time = events["time_bucket"].min()
    max_time = events["time_bucket"].max()

    all_hours = pd.date_range(
        start=min_time,
        end=max_time,
        freq="h"
    )

    full_index = pd.MultiIndex.from_product(
        [
            all_hours,
            all_clusters
        ],
        names=[
            "time_bucket",
            "spatial_cluster_id"
        ]
    )

    spatial_ts = (
        events
        .groupby(
            [
                "time_bucket",
                "spatial_cluster_id"
            ]
        )
        .size()
        .reindex(
            full_index,
            fill_value=0
        )
        .reset_index(name="incident_count")
    )

    spatial_ts = spatial_ts.merge(
        static_features,
        on="spatial_cluster_id",
        how="left"
    )

    spatial_ts["hour"] = (
        spatial_ts["time_bucket"]
        .dt.hour
        .astype(int)
    )

    spatial_ts["weekday"] = (
        spatial_ts["time_bucket"]
        .dt.weekday
        .astype(int)
    )

    spatial_ts["month"] = (
        spatial_ts["time_bucket"]
        .dt.month
        .astype(int)
    )

    spatial_ts["hour_sin"] = np.sin(
        2 * np.pi * spatial_ts["hour"] / 24
    )

    spatial_ts["hour_cos"] = np.cos(
        2 * np.pi * spatial_ts["hour"] / 24
    )

    spatial_ts = add_lag_and_rolling_features(
        spatial_ts
    )

    for col in SPATIAL_FEATURES:
        if col not in spatial_ts.columns:
            spatial_ts[col] = 0

    spatial_ts = spatial_ts[
        [
            "time_bucket",
            "incident_count"
        ]
        +
        SPATIAL_FEATURES
    ]

    print("\nSpatial Dataset Shape:")
    print(
        spatial_ts.shape
    )

    print("\nSpatial Target Distribution:")
    print(
        spatial_ts["incident_count"]
        .value_counts()
        .head(15)
    )

    print("\nSpatial Zero Ratio:")
    print(
        round(
            (spatial_ts["incident_count"] == 0).mean(),
            4
        )
    )

    return spatial_ts