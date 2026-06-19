import os
from pathlib import Path

import pandas as pd

from src.inference.location_resolver import (
    haversine_distance_meters,
)


BAD_STATION_NAMES = {
    "",
    "unknown",
    "nan",
    "none",
    "null",
    "not available",
    "not_applicable",
    "na",
}


DEFAULT_POLICE_STATION_PATH = (
    Path("data")
    / "police_stations.csv"
)


def clean_station_name(value):
    return (
        str(value or "")
        .strip()
    )


def is_bad_station_name(value):
    value = (
        str(value or "")
        .strip()
        .lower()
    )

    return value in BAD_STATION_NAMES


def safe_float(value, fallback=None):
    try:
        if value is None:
            return fallback

        if pd.isna(value):
            return fallback

        value = str(value).strip()

        if value == "":
            return fallback

        return float(value)

    except Exception:
        return fallback


def load_official_police_station_points(
    path=DEFAULT_POLICE_STATION_PATH
):
    if not os.path.exists(path):
        return []

    df = pd.read_csv(
        path
    )

    if "police_station" not in df.columns:
        if "station_name" in df.columns:
            df["police_station"] = df["station_name"]

        elif "name" in df.columns:
            df["police_station"] = df["name"]

        else:
            return []

    if "latitude" not in df.columns or "longitude" not in df.columns:
        return []

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df["police_station"] = (
        df["police_station"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    )

    points = []

    for _, row in df.iterrows():
        station = clean_station_name(
            row["police_station"]
        )

        if is_bad_station_name(station):
            continue

        points.append({
            "police_station": station,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "source": "official_station_coordinates",
        })

    return points


def build_police_station_store_from_dataset(df):
    """
    Fallback when official police station coordinates are unavailable.

    This does not represent the exact station building location.
    It estimates the operational centroid of historical incidents
    assigned to each police station.
    """

    if "police_station" not in df.columns:
        return {
            "police_station_points": [],
            "police_station_profiles": {},
            "police_station_source": "missing_police_station_column",
        }

    if "latitude" not in df.columns or "longitude" not in df.columns:
        return {
            "police_station_points": [],
            "police_station_profiles": {},
            "police_station_source": "missing_coordinates",
        }

    station_df = df.copy()

    station_df["police_station"] = (
        station_df["police_station"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
    )

    station_df["latitude"] = pd.to_numeric(
        station_df["latitude"],
        errors="coerce"
    )

    station_df["longitude"] = pd.to_numeric(
        station_df["longitude"],
        errors="coerce"
    )

    station_df = station_df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    )

    station_df = station_df[
        ~station_df["police_station"].map(
            is_bad_station_name
        )
    ].copy()

    # Bengaluru bounds
    station_df = station_df[
        (station_df["latitude"] >= 12.70)
        &
        (station_df["latitude"] <= 13.25)
        &
        (station_df["longitude"] >= 77.35)
        &
        (station_df["longitude"] <= 77.85)
    ].copy()

    if len(station_df) == 0:
        return {
            "police_station_points": [],
            "police_station_profiles": {},
            "police_station_source": "no_valid_station_rows",
        }

    grouped = (
        station_df
        .groupby("police_station")
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            event_count=("police_station", "size"),
        )
        .reset_index()
    )

    points = []
    profiles = {}

    for _, row in grouped.iterrows():
        station = clean_station_name(
            row["police_station"]
        )

        point = {
            "police_station": station,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "event_count": int(row["event_count"]),
            "source": "historical_event_centroid",
        }

        points.append(point)

        profiles[station] = point

    return {
        "police_station_points": points,
        "police_station_profiles": profiles,
        "police_station_source": "historical_event_centroid",
    }


def build_police_station_store(df):
    official_points = load_official_police_station_points()

    if official_points:
        profiles = {}

        for point in official_points:
            station = point["police_station"]
            profiles[station] = point

        return {
            "police_station_points": official_points,
            "police_station_profiles": profiles,
            "police_station_source": "official_station_coordinates",
        }

    return build_police_station_store_from_dataset(
        df
    )


def resolve_nearest_police_station(
    latitude,
    longitude,
    store,
    max_distance_m=10000
):
    points = store.get(
        "police_station_points",
        []
    )

    if not points:
        return {
            "police_station": "Unknown",
            "matched_by": "no police station store available",
            "distance_m": None,
            "confidence": "LOW",
            "source": "unavailable",
        }

    latitude = safe_float(
        latitude,
        None
    )

    longitude = safe_float(
        longitude,
        None
    )

    if latitude is None or longitude is None:
        return {
            "police_station": "Unknown",
            "matched_by": "invalid coordinates",
            "distance_m": None,
            "confidence": "LOW",
            "source": "unavailable",
        }

    best_point = None
    best_distance = None

    for point in points:
        try:
            distance = haversine_distance_meters(
                latitude,
                longitude,
                point["latitude"],
                point["longitude"]
            )

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_point = point

        except Exception:
            continue

    if best_point is None:
        return {
            "police_station": "Unknown",
            "matched_by": "no valid police station point",
            "distance_m": None,
            "confidence": "LOW",
            "source": "unavailable",
        }

    if best_distance <= 1500:
        confidence = "HIGH"

    elif best_distance <= 4000:
        confidence = "MEDIUM"

    elif best_distance <= max_distance_m:
        confidence = "LOW"

    else:
        confidence = "LOW"

    return {
        "police_station": best_point["police_station"],
        "matched_by": "nearest police station by coordinates",
        "distance_m": round(best_distance, 2),
        "confidence": confidence,
        "source": best_point.get("source", "unknown"),
        "event_count": best_point.get("event_count"),
    }