import os
import joblib
import pandas as pd

from sklearn.cluster import KMeans

from src.preprocessing.load_data import load_data
from src.forecasting.build_timeseries_dataset import build_timeseries_dataset


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


PROFILE_FEATURES = [
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


def make_key(
    name,
    hour
):
    return f"{str(name)}__{int(hour)}"


def safe_float(
    value
):
    try:
        if pd.isna(value):
            return 0.0

        return float(value)

    except Exception:
        return 0.0


def normalize_datetime_utc_naive(
    series
):
    return (
        pd.to_datetime(
            series,
            errors="coerce",
            utc=True
        )
        .dt.tz_convert(None)
    )


def clean_common_columns(
    df
):
    df = df.copy()

    if "start_datetime" in df.columns:
        df["start_datetime"] = normalize_datetime_utc_naive(
            df["start_datetime"]
        )

    for col in [
        "corridor",
        "zone",
        "junction",
        "event_cause",
        "veh_type"
    ]:
        if col not in df.columns:
            df[col] = "UNKNOWN"

        df[col] = (
            df[col]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

    if "requires_road_closure" not in df.columns:
        df["requires_road_closure"] = False

    df["requires_road_closure"] = (
        df["requires_road_closure"]
        .fillna(False)
        .astype(bool)
    )

    df["latitude"] = pd.to_numeric(
        df.get("latitude"),
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df.get("longitude"),
        errors="coerce"
    )

    return df


def add_basic_risk_columns(
    df
):
    df = df.copy()

    zone_risk = (
        df.groupby("zone")
        .size()
    )

    junction_risk = (
        df.groupby("junction")
        .size()
    )

    cause_risk = (
        df.groupby("event_cause")
        .size()
    )

    closure_risk = (
        df.groupby("requires_road_closure")
        .size()
    )

    df["zone_risk"] = (
        df["zone"]
        .map(zone_risk)
        .fillna(0)
    )

    df["junction_risk"] = (
        df["junction"]
        .map(junction_risk)
        .fillna(0)
    )

    df["cause_risk"] = (
        df["event_cause"]
        .map(cause_risk)
        .fillna(0)
    )

    df["closure_risk"] = (
        df["requires_road_closure"]
        .map(closure_risk)
        .fillna(0)
    )

    return df


def build_location_and_cluster_store(
    df,
    n_clusters=30
):
    location_df = clean_common_columns(
        df
    )

    location_df = location_df.dropna(
        subset=[
            "latitude",
            "longitude",
            "start_datetime"
        ]
    )

    location_df = location_df[
        (location_df["latitude"] >= -90)
        &
        (location_df["latitude"] <= 90)
        &
        (location_df["longitude"] >= -180)
        &
        (location_df["longitude"] <= 180)
    ].copy()

    if len(location_df) == 0:
        return {
            "corridor_location_profiles": {},
            "corridor_location_points": [],
            "hotspot_points": [],
            "spatial_cluster_model": None,
            "spatial_cluster_centers": {},
            "spatial_cluster_hour_profiles": {},
            "spatial_cluster_profiles": {},
            "max_cause_risk": 1.0,
        }

    location_df = add_basic_risk_columns(
        location_df
    )

    coords = location_df[
        [
            "latitude",
            "longitude"
        ]
    ]

    usable_clusters = min(
        n_clusters,
        max(2, len(location_df))
    )

    kmeans = KMeans(
        n_clusters=usable_clusters,
        random_state=42,
        n_init=10
    )

    location_df["spatial_cluster_id"] = kmeans.fit_predict(
        coords
    )

    cluster_risk = (
        location_df.groupby("spatial_cluster_id")
        .size()
    )

    location_df["cluster_risk"] = (
        location_df["spatial_cluster_id"]
        .map(cluster_risk)
        .fillna(0)
    )

    # =====================================================
    # CORRIDOR LOCATION PROFILES
    # =====================================================

    corridor_location_profiles = {}

    grouped_corridor = (
        location_df
        .groupby("corridor")
        .agg(
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            event_count=("corridor", "size")
        )
        .reset_index()
    )

    for _, row in grouped_corridor.iterrows():
        corridor_location_profiles[
            str(row["corridor"])
        ] = {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "event_count": int(row["event_count"]),
        }

    # Sample event points for nearest-point resolver.
    corridor_location_points = []

    for corridor, group in location_df.groupby("corridor"):
        sample_size = min(
            len(group),
            300
        )

        sample = group.sample(
            n=sample_size,
            random_state=42
        )

        for _, row in sample.iterrows():
            corridor_location_points.append({
                "corridor": str(corridor),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "spatial_cluster_id": int(row["spatial_cluster_id"]),
            })

    # =====================================================
    # HOTSPOT POINTS
    # =====================================================

    location_df["lat_round"] = (
        location_df["latitude"]
        .round(3)
    )

    location_df["lon_round"] = (
        location_df["longitude"]
        .round(3)
    )

    hotspot_df = (
        location_df
        .groupby(
            [
                "lat_round",
                "lon_round"
            ]
        )
        .size()
        .reset_index(name="event_count")
        .sort_values(
            "event_count",
            ascending=False
        )
        .head(500)
    )

    hotspot_points = []

    for _, row in hotspot_df.iterrows():
        hotspot_points.append({
            "latitude": float(row["lat_round"]),
            "longitude": float(row["lon_round"]),
            "event_count": int(row["event_count"]),
        })

    # =====================================================
    # CLUSTER CENTERS
    # =====================================================

    spatial_cluster_centers = {}

    for cluster_id, center in enumerate(kmeans.cluster_centers_):
        spatial_cluster_centers[
            str(cluster_id)
        ] = {
            "latitude": float(center[0]),
            "longitude": float(center[1]),
        }

    # =====================================================
    # CLUSTER-HOUR TIME SERIES PROFILES
    # =====================================================

    location_df["time_bucket"] = (
        location_df["start_datetime"]
        .dt.floor("h")
    )

    min_time = location_df["time_bucket"].min()
    max_time = location_df["time_bucket"].max()

    all_hours = pd.date_range(
        start=min_time,
        end=max_time,
        freq="h"
    )

    all_clusters = sorted(
        location_df["spatial_cluster_id"]
        .dropna()
        .unique()
        .tolist()
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

    cluster_ts = (
        location_df
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

    static_features = (
        location_df
        .groupby("spatial_cluster_id")
        .agg(
            zone_risk=("zone_risk", "mean"),
            junction_risk=("junction_risk", "mean"),
            cause_risk=("cause_risk", "mean"),
            closure_risk=("closure_risk", "mean"),
            cluster_risk=("cluster_risk", "mean")
        )
        .reset_index()
    )

    cluster_ts = cluster_ts.merge(
        static_features,
        on="spatial_cluster_id",
        how="left"
    )

    cluster_ts = cluster_ts.sort_values(
        [
            "spatial_cluster_id",
            "time_bucket"
        ]
    )

    cluster_ts["hour"] = (
        cluster_ts["time_bucket"]
        .dt.hour
    )

    grouped = cluster_ts.groupby(
        "spatial_cluster_id"
    )["incident_count"]

    for lag in [
        1,
        2,
        3,
        24,
        48,
        72,
        168
    ]:
        cluster_ts[f"lag_{lag}"] = (
            grouped
            .shift(lag)
        )

    shifted = grouped.shift(1)

    cluster_ts["rolling_6"] = (
        shifted
        .groupby(cluster_ts["spatial_cluster_id"])
        .rolling(6)
        .mean()
        .reset_index(level=0, drop=True)
    )

    cluster_ts["rolling_12"] = (
        shifted
        .groupby(cluster_ts["spatial_cluster_id"])
        .rolling(12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    cluster_ts["rolling_24"] = (
        shifted
        .groupby(cluster_ts["spatial_cluster_id"])
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    cluster_ts["rolling_168"] = (
        shifted
        .groupby(cluster_ts["spatial_cluster_id"])
        .rolling(168)
        .mean()
        .reset_index(level=0, drop=True)
    )

    cluster_avg = (
        cluster_ts
        .groupby("spatial_cluster_id")["incident_count"]
        .mean()
    )

    cluster_std = (
        cluster_ts
        .groupby("spatial_cluster_id")["incident_count"]
        .std()
    )

    cluster_ts["corridor_avg"] = (
        cluster_ts["spatial_cluster_id"]
        .map(cluster_avg)
    )

    cluster_ts["corridor_volatility"] = (
        cluster_ts["spatial_cluster_id"]
        .map(cluster_std)
        .fillna(0)
    )

    cluster_ts[PROFILE_FEATURES] = (
        cluster_ts[PROFILE_FEATURES]
        .fillna(0)
    )

    spatial_cluster_hour_profiles = {}

    grouped_ch = (
        cluster_ts
        .groupby(
            [
                "spatial_cluster_id",
                "hour"
            ]
        )[PROFILE_FEATURES]
        .median()
        .reset_index()
    )

    for _, row in grouped_ch.iterrows():
        key = make_key(
            int(row["spatial_cluster_id"]),
            int(row["hour"])
        )

        profile = {}

        for feature in PROFILE_FEATURES:
            profile[feature] = safe_float(
                row[feature]
            )

        spatial_cluster_hour_profiles[key] = profile

    spatial_cluster_profiles = {}

    grouped_c = (
        cluster_ts
        .groupby("spatial_cluster_id")[PROFILE_FEATURES]
        .median()
        .reset_index()
    )

    for _, row in grouped_c.iterrows():
        cluster_id = str(
            int(row["spatial_cluster_id"])
        )

        profile = {}

        for feature in PROFILE_FEATURES:
            profile[feature] = safe_float(
                row[feature]
            )

        spatial_cluster_profiles[cluster_id] = profile

    max_cause_risk = safe_float(
        location_df["cause_risk"].max()
    )

    if max_cause_risk <= 0:
        max_cause_risk = 1.0

    return {
        "corridor_location_profiles": corridor_location_profiles,
        "corridor_location_points": corridor_location_points,
        "hotspot_points": hotspot_points,
        "spatial_cluster_model": kmeans,
        "spatial_cluster_centers": spatial_cluster_centers,
        "spatial_cluster_hour_profiles": spatial_cluster_hour_profiles,
        "spatial_cluster_profiles": spatial_cluster_profiles,
        "max_cause_risk": max_cause_risk,
    }


def build_feature_store(
    data_path="data/traffic_events.csv",
    output_path="models/traffic_feature_store.pkl"
):
    print("\n" + "=" * 60)
    print("BUILDING COORDINATE-FIRST TRAFFIC FEATURE STORE")
    print("=" * 60)

    df = load_data(
        data_path
    )

    ts_df = build_timeseries_dataset(
        df
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    # =====================================================
    # CORRIDOR + HOUR PROFILE
    # =====================================================

    corridor_hour_profiles = {}

    grouped_ch = (
        ts_df
        .groupby(
            [
                "corridor",
                "hour"
            ]
        )[PROFILE_FEATURES]
        .median()
        .reset_index()
    )

    for _, row in grouped_ch.iterrows():
        key = make_key(
            row["corridor"],
            row["hour"]
        )

        profile = {}

        for feature in PROFILE_FEATURES:
            profile[feature] = safe_float(
                row[feature]
            )

        corridor_hour_profiles[key] = profile

    # =====================================================
    # CORRIDOR FALLBACK PROFILE
    # =====================================================

    corridor_profiles = {}

    grouped_c = (
        ts_df
        .groupby("corridor")[PROFILE_FEATURES]
        .median()
        .reset_index()
    )

    for _, row in grouped_c.iterrows():
        corridor = str(
            row["corridor"]
        )

        profile = {}

        for feature in PROFILE_FEATURES:
            profile[feature] = safe_float(
                row[feature]
            )

        corridor_profiles[corridor] = profile

    # =====================================================
    # GLOBAL PROFILE
    # =====================================================

    global_profile = {}

    for feature in PROFILE_FEATURES:
        global_profile[feature] = safe_float(
            ts_df[feature].median()
        )

    # =====================================================
    # INCIDENT THRESHOLDS
    # =====================================================

    incident_p50 = safe_float(
        ts_df["incident_count"].quantile(0.50)
    )

    incident_p75 = safe_float(
        ts_df["incident_count"].quantile(0.75)
    )

    incident_p90 = safe_float(
        ts_df["incident_count"].quantile(0.90)
    )

    incident_p95 = safe_float(
        ts_df["incident_count"].quantile(0.95)
    )

    incident_p99 = safe_float(
        ts_df["incident_count"].quantile(0.99)
    )

    if incident_p95 <= 0:
        incident_p95 = max(
            safe_float(ts_df["incident_count"].max()),
            1.0
        )

    if incident_p99 <= 0:
        incident_p99 = incident_p95

    # =====================================================
    # LOCATION + SPATIAL CLUSTER STORE
    # =====================================================

    location_store = build_location_and_cluster_store(
        df
    )

    store = {
        "features": FEATURES,
        "profile_features": PROFILE_FEATURES,

        "corridor_hour_profiles": corridor_hour_profiles,
        "corridor_profiles": corridor_profiles,
        "global_profile": global_profile,

        "incident_p50": incident_p50,
        "incident_p75": incident_p75,
        "incident_p90": incident_p90,
        "incident_p95": incident_p95,
        "incident_p99": incident_p99,

        "corridors": sorted(
            list(corridor_profiles.keys())
        ),
    }

    store.update(
        location_store
    )

    joblib.dump(
        store,
        output_path
    )

    print("\nFeature store saved:")
    print(output_path)

    print("\nTotal corridor-hour profiles:")
    print(len(corridor_hour_profiles))

    print("\nTotal corridor profiles:")
    print(len(corridor_profiles))

    print("\nLocation profiles:")
    print(len(store.get("corridor_location_profiles", {})))

    print("\nLocation resolver points:")
    print(len(store.get("corridor_location_points", [])))

    print("\nHotspot points:")
    print(len(store.get("hotspot_points", [])))

    print("\nSpatial cluster-hour profiles:")
    print(len(store.get("spatial_cluster_hour_profiles", {})))

    print("\nSpatial cluster profiles:")
    print(len(store.get("spatial_cluster_profiles", {})))

    print("\nRisk thresholds:")
    print(f"P50: {incident_p50:.2f}")
    print(f"P75: {incident_p75:.2f}")
    print(f"P90: {incident_p90:.2f}")
    print(f"P95: {incident_p95:.2f}")
    print(f"P99: {incident_p99:.2f}")

    print("\nSample corridors:")
    for corridor in store["corridors"][:20]:
        print("-", corridor)

    return store