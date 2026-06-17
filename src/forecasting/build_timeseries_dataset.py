import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def normalize_datetime_utc_naive(series):

    return (
        pd.to_datetime(
            series,
            errors="coerce",
            utc=True
        )
        .dt.tz_convert(None)
    )



def build_timeseries_dataset(df):

    print("\nBuilding Advanced Time Series Dataset...")

    # ====================================================
    # DATETIME CLEANING
    # ====================================================

    df = df.copy()

    df["start_datetime"] = normalize_datetime_utc_naive(
        df["start_datetime"]
    )

    df = df.dropna(
        subset=["start_datetime"]
    )

    # ====================================================
    # BASIC CLEANING
    # ====================================================

    df["corridor"] = (
        df["corridor"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    df["zone"] = (
        df["zone"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    df["junction"] = (
        df["junction"]
        .fillna("UNKNOWN")
        .astype(str)
    )

    df["event_cause"] = (
        df["event_cause"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["requires_road_closure"] = (
        df["requires_road_closure"]
        .fillna(False)
        .astype(bool)
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["latitude"] = df["latitude"].fillna(
        df["latitude"].median()
    )

    df["longitude"] = df["longitude"].fillna(
        df["longitude"].median()
    )

    # ====================================================
    # TIME BUCKET
    # ====================================================

    df["time_bucket"] = (
        df["start_datetime"]
        .dt.floor("h")
    )

    # ====================================================
    # SPATIAL CLUSTERING
    # ====================================================

    coords = df[
        ["latitude", "longitude"]
    ].values

    clustering = DBSCAN(
        eps=0.005,
        min_samples=10
    )

    df["cluster_id"] = clustering.fit_predict(
        coords
    )

    # ====================================================
    # RAW RISK MAPS
    # ====================================================

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

    cluster_risk = (
        df.groupby("cluster_id")
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

    df["cluster_risk"] = (
        df["cluster_id"]
        .map(cluster_risk)
        .fillna(0)
    )

    # ====================================================
    # INCIDENT COUNT PER CORRIDOR-HOUR
    # ====================================================

    hourly = (
        df.groupby(
            [
                "corridor",
                "time_bucket"
            ]
        )
        .size()
        .reset_index(name="incident_count")
    )

    # ====================================================
    # COMPLETE CORRIDOR-HOUR GRID
    # This ensures missing hours become 0 incidents.
    # Very important for correct lag/rolling features.
    # ====================================================

    min_time = hourly["time_bucket"].min()
    max_time = hourly["time_bucket"].max()

    all_hours = pd.date_range(
        start=min_time,
        end=max_time,
        freq="h"
    )

    corridors = sorted(
        hourly["corridor"].unique()
    )

    grid_parts = []

    for corridor in corridors:

        temp = pd.DataFrame({
            "time_bucket": all_hours
        })

        temp["corridor"] = corridor

        grid_parts.append(temp)

    full_grid = pd.concat(
        grid_parts,
        ignore_index=True
    )

    ts_df = full_grid.merge(
        hourly,
        on=[
            "corridor",
            "time_bucket"
        ],
        how="left"
    )

    ts_df["incident_count"] = (
        ts_df["incident_count"]
        .fillna(0)
        .astype(float)
    )

    # ====================================================
    # CORRIDOR-LEVEL SPATIAL FEATURES
    # ====================================================

    spatial_features = (
        df.groupby("corridor")
        .agg(
            zone_risk=("zone_risk", "mean"),
            junction_risk=("junction_risk", "mean"),
            cause_risk=("cause_risk", "mean"),
            closure_risk=("closure_risk", "mean"),
            cluster_risk=("cluster_risk", "mean")
        )
        .reset_index()
    )

    ts_df = ts_df.merge(
        spatial_features,
        on="corridor",
        how="left"
    )

    spatial_cols = [
        "zone_risk",
        "junction_risk",
        "cause_risk",
        "closure_risk",
        "cluster_risk"
    ]

    for col in spatial_cols:

        ts_df[col] = (
            ts_df[col]
            .fillna(ts_df[col].median())
        )

    # ====================================================
    # TIME FEATURES
    # ====================================================

    ts_df["hour"] = (
        ts_df["time_bucket"]
        .dt.hour
    )

    ts_df["weekday"] = (
        ts_df["time_bucket"]
        .dt.weekday
    )

    ts_df["month"] = (
        ts_df["time_bucket"]
        .dt.month
    )

    ts_df["hour_sin"] = np.sin(
        2 * np.pi * ts_df["hour"] / 24
    )

    ts_df["hour_cos"] = np.cos(
        2 * np.pi * ts_df["hour"] / 24
    )

    # ====================================================
    # SORT BEFORE LAGS
    # ====================================================

    ts_df = ts_df.sort_values(
        [
            "corridor",
            "time_bucket"
        ]
    ).reset_index(drop=True)

    # ====================================================
    # CORRIDOR-SPECIFIC LAGS
    # IMPORTANT: Never use global shift here.
    # ====================================================

    group = ts_df.groupby("corridor")["incident_count"]

    ts_df["lag_1"] = group.shift(1)
    ts_df["lag_2"] = group.shift(2)
    ts_df["lag_3"] = group.shift(3)

    ts_df["lag_24"] = group.shift(24)
    ts_df["lag_48"] = group.shift(48)
    ts_df["lag_72"] = group.shift(72)
    ts_df["lag_168"] = group.shift(168)

    # ====================================================
    # CORRIDOR-SPECIFIC ROLLING WINDOWS
    # Shift first to avoid using current target in rolling.
    # ====================================================

    shifted = (
        ts_df.groupby("corridor")["incident_count"]
        .shift(1)
    )

    ts_df["rolling_6"] = (
        shifted
        .groupby(ts_df["corridor"])
        .rolling(6)
        .mean()
        .reset_index(level=0, drop=True)
    )

    ts_df["rolling_12"] = (
        shifted
        .groupby(ts_df["corridor"])
        .rolling(12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    ts_df["rolling_24"] = (
        shifted
        .groupby(ts_df["corridor"])
        .rolling(24)
        .mean()
        .reset_index(level=0, drop=True)
    )

    ts_df["rolling_168"] = (
        shifted
        .groupby(ts_df["corridor"])
        .rolling(168)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ====================================================
    # CORRIDOR STATS
    # Computed on full grid, so zero-incident hours matter.
    # ====================================================

    corridor_avg = (
        ts_df.groupby("corridor")["incident_count"]
        .mean()
    )

    corridor_std = (
        ts_df.groupby("corridor")["incident_count"]
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

    # ====================================================
    # DROP NA FROM LAGS / ROLLINGS
    # ====================================================

    ts_df = ts_df.dropna().reset_index(drop=True)

    # ====================================================
    # DEBUG OUTPUT
    # ====================================================

    print("\nDataset Shape:")
    print(ts_df.shape)

    print("\nIncident Count Stats:")
    print(ts_df["incident_count"].describe())

    print("\nTop Incident Counts:")
    print(
        ts_df["incident_count"]
        .value_counts()
        .head(10)
    )

    print("\nTop Corridors By Rows:")
    print(
        ts_df.groupby("corridor")
        .size()
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nColumns:")
    print(ts_df.columns.tolist())

    return ts_df