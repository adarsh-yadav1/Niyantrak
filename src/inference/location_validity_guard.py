from pathlib import Path

import pandas as pd

from src.inference.location_resolver import haversine_distance_meters


RESTRICTED_ZONE_PATH = Path("data/restricted_zones.csv")


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


def load_restricted_zones(path=RESTRICTED_ZONE_PATH):
    path = Path(path)

    if not path.exists():
        return []

    df = pd.read_csv(path)

    required = [
        "zone_name",
        "zone_type",
        "latitude",
        "longitude",
        "radius_m",
    ]

    for col in required:
        if col not in df.columns:
            return []

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["radius_m"] = pd.to_numeric(df["radius_m"], errors="coerce")

    df = df.dropna(
        subset=[
            "latitude",
            "longitude",
            "radius_m",
        ]
    )

    zones = []

    for _, row in df.iterrows():
        zones.append({
            "zone_name": str(row["zone_name"]).strip(),
            "zone_type": str(row["zone_type"]).strip().lower(),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "radius_m": float(row["radius_m"]),
        })

    return zones


def build_restricted_zone_store():
    return {
        "restricted_zones": load_restricted_zones()
    }


def check_restricted_zone(latitude, longitude, store):
    zones = store.get("restricted_zones", [])

    if not zones:
        return {
            "is_restricted": False,
            "reason": None,
            "zone_name": None,
            "zone_type": None,
            "distance_m": None,
        }

    latitude = safe_float(latitude)
    longitude = safe_float(longitude)

    if latitude is None or longitude is None:
        return {
            "is_restricted": True,
            "reason": "Invalid coordinates.",
            "zone_name": None,
            "zone_type": None,
            "distance_m": None,
        }

    nearest_zone = None
    nearest_distance = None

    for zone in zones:
        distance = haversine_distance_meters(
            latitude,
            longitude,
            zone["latitude"],
            zone["longitude"]
        )

        if nearest_distance is None or distance < nearest_distance:
            nearest_distance = distance
            nearest_zone = zone

        if distance <= zone["radius_m"]:
            return {
                "is_restricted": True,
                "reason": (
                    f"Selected point falls inside or very close to "
                    f"{zone['zone_name']} ({zone['zone_type']}). "
                    f"Please select a point on a nearby road instead."
                ),
                "zone_name": zone["zone_name"],
                "zone_type": zone["zone_type"],
                "distance_m": round(distance, 2),
            }

    return {
        "is_restricted": False,
        "reason": None,
        "zone_name": nearest_zone["zone_name"] if nearest_zone else None,
        "zone_type": nearest_zone["zone_type"] if nearest_zone else None,
        "distance_m": round(nearest_distance, 2) if nearest_distance is not None else None,
    }


def check_road_corridor_proximity(
    location_match,
    max_high_confidence_distance_m=1500,
    max_allowed_distance_m=3000
):
    if not location_match:
        return {
            "is_valid": False,
            "reason": "Could not resolve this location to a known road corridor.",
        }

    distance_m = safe_float(
        location_match.get("distance_m"),
        None
    )

    confidence = (
        str(location_match.get("confidence", "LOW"))
        .strip()
        .upper()
    )

    if distance_m is None:
        return {
            "is_valid": False,
            "reason": "Nearest corridor distance is unavailable.",
        }

    if distance_m > max_allowed_distance_m:
        return {
            "is_valid": False,
            "reason": (
                f"Selected point is {distance_m:.0f} m away from the nearest known corridor. "
                f"Please select a point closer to a road."
            ),
        }

    if confidence == "LOW" and distance_m > max_high_confidence_distance_m:
        return {
            "is_valid": False,
            "reason": (
                f"Location match confidence is LOW and the point is {distance_m:.0f} m "
                f"from the nearest corridor. Please choose a road-side point."
            ),
        }

    return {
        "is_valid": True,
        "reason": None,
    }


def validate_prediction_location(
    latitude,
    longitude,
    store,
    location_match
):
    restricted_check = check_restricted_zone(
        latitude=latitude,
        longitude=longitude,
        store=store
    )

    if restricted_check["is_restricted"]:
        return {
            "is_valid": False,
            "reason": restricted_check["reason"],
            "blocked_by": "restricted_zone",
            "restricted_zone": restricted_check,
        }

    corridor_check = check_road_corridor_proximity(
        location_match
    )

    if not corridor_check["is_valid"]:
        return {
            "is_valid": False,
            "reason": corridor_check["reason"],
            "blocked_by": "corridor_distance",
            "restricted_zone": restricted_check,
        }

    return {
        "is_valid": True,
        "reason": None,
        "blocked_by": None,
        "restricted_zone": restricted_check,
    }