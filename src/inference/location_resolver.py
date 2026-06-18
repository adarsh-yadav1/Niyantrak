import math
import pandas as pd


BENGALURU_BOUNDS = {
    "min_lat": 12.70,
    "max_lat": 13.25,
    "min_lon": 77.35,
    "max_lon": 77.85,
}


BAD_CORRIDOR_NAMES = {
    "",
    "unknown",
    "nan",
    "none",
    "null",
    "non-corridor",
    "non corridor",
    "non_corridor",
    "out_of_bengaluru",
    "outside_bengaluru",
    "auto inferred after prediction",
    "will be inferred after prediction",
}


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


def make_key(name, hour):
    return f"{str(name)}__{int(hour)}"


def normalize_corridor_name(value):
    return (
        str(value or "")
        .strip()
    )


def is_bad_corridor_name(value):
    value = (
        str(value or "")
        .strip()
        .lower()
    )

    return value in BAD_CORRIDOR_NAMES


def is_valid_coordinate(latitude, longitude):
    try:
        latitude = float(latitude)
        longitude = float(longitude)

        return (
            -90 <= latitude <= 90
            and
            -180 <= longitude <= 180
        )

    except Exception:
        return False


def is_inside_bengaluru(latitude, longitude):
    try:
        latitude = float(latitude)
        longitude = float(longitude)

        return (
            BENGALURU_BOUNDS["min_lat"] <= latitude <= BENGALURU_BOUNDS["max_lat"]
            and
            BENGALURU_BOUNDS["min_lon"] <= longitude <= BENGALURU_BOUNDS["max_lon"]
        )

    except Exception:
        return False


def haversine_distance_meters(lat1, lon1, lat2, lon2):
    radius = 6371000

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return radius * c


def find_nearest_hotspot(latitude, longitude, store):
    hotspots = store.get("hotspot_points", [])

    if not hotspots:
        return None, None

    best_hotspot = None
    best_distance = None

    for point in hotspots:
        try:
            distance = haversine_distance_meters(
                latitude,
                longitude,
                point["latitude"],
                point["longitude"]
            )

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_hotspot = point

        except Exception:
            continue

    return best_hotspot, best_distance


def estimate_spatial_density(latitude, longitude, store, radius_m=500):
    points = store.get("corridor_location_points", [])

    if not points:
        return 0.0

    count = 0

    for point in points:
        try:
            distance = haversine_distance_meters(
                latitude,
                longitude,
                point["latitude"],
                point["longitude"]
            )

            if distance <= radius_m:
                count += 1

        except Exception:
            continue

    density = count / 25.0

    return max(
        0.0,
        min(density, 1.0)
    )

def resolve_spatial_cluster(latitude, longitude, store):
    model = store.get("spatial_cluster_model")
    centers = store.get("spatial_cluster_centers", {})

    if model is None:
        return None, None

    try:
        input_df = pd.DataFrame(
            [
                {
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                }
            ]
        )

        cluster_id = int(
            model.predict(input_df)[0]
        )

        center = centers.get(str(cluster_id))

        if center is None:
            center = centers.get(cluster_id)

        distance = None

        if center is not None:
            distance = haversine_distance_meters(
                latitude,
                longitude,
                center["latitude"],
                center["longitude"]
            )

        return cluster_id, distance

    except Exception:
        return None, None

def nearest_point_match(
    latitude,
    longitude,
    points,
    allow_bad_corridor=False
):
    best_point = None
    best_distance = None

    for point in points:
        corridor = normalize_corridor_name(
            point.get("corridor")
        )

        if (
            not allow_bad_corridor
            and is_bad_corridor_name(corridor)
        ):
            continue

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

    return best_point, best_distance


def nearest_corridor_centroid_match(
    latitude,
    longitude,
    corridor_profiles,
    allow_bad_corridor=False
):
    best_corridor = None
    best_distance = None

    for corridor, profile in corridor_profiles.items():
        corridor = normalize_corridor_name(corridor)

        if (
            not allow_bad_corridor
            and is_bad_corridor_name(corridor)
        ):
            continue

        try:
            distance = haversine_distance_meters(
                latitude,
                longitude,
                profile["latitude"],
                profile["longitude"]
            )

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_corridor = corridor

        except Exception:
            continue

    return best_corridor, best_distance


def nearest_real_corridor_inside_cluster(
    latitude,
    longitude,
    spatial_cluster_id,
    store
):
    if spatial_cluster_id is None:
        return None, None

    points = store.get("corridor_location_points", [])

    cluster_points = []

    for point in points:
        try:
            if int(point.get("spatial_cluster_id")) == int(spatial_cluster_id):
                cluster_points.append(point)
        except Exception:
            continue

    if not cluster_points:
        return None, None

    best_point, best_distance = nearest_point_match(
        latitude=latitude,
        longitude=longitude,
        points=cluster_points,
        allow_bad_corridor=False
    )

    if best_point is None:
        return None, None

    return normalize_corridor_name(best_point["corridor"]), best_distance


def resolve_corridor_from_coordinates(
    latitude,
    longitude,
    store,
    max_point_distance_m=2500,
    max_centroid_distance_m=5000
):
    """
    Coordinate-first resolver.

    Important fix:
    - Do NOT let Non-corridor dominate nearest-point matching.
    - Prefer real corridor names first.
    - Use spatial cluster fallback for history, but still try to infer a real corridor for model compatibility.
    """

    if not is_valid_coordinate(latitude, longitude):
        return {
            "corridor": "Non-corridor",
            "matched_by": "invalid coordinates fallback",
            "distance_m": None,
            "confidence": "LOW",
            "spatial_cluster_id": None,
            "spatial_cluster_distance_m": None,
            "nearest_hotspot_distance_m": None,
            "spatial_density_at_point": 0.0,
            "outside_bengaluru": False,
            "used_bad_corridor": True,
        }

    latitude = float(latitude)
    longitude = float(longitude)

    if not is_inside_bengaluru(latitude, longitude):
        return {
            "corridor": "OUTSIDE_BENGALURU",
            "matched_by": "outside Bengaluru boundary",
            "distance_m": None,
            "confidence": "INVALID",
            "spatial_cluster_id": None,
            "spatial_cluster_distance_m": None,
            "nearest_hotspot_distance_m": None,
            "spatial_density_at_point": 0.0,
            "outside_bengaluru": True,
            "used_bad_corridor": True,
        }

    spatial_cluster_id, cluster_distance = resolve_spatial_cluster(
        latitude,
        longitude,
        store
    )

    _, hotspot_distance = find_nearest_hotspot(
        latitude,
        longitude,
        store
    )

    spatial_density = estimate_spatial_density(
        latitude,
        longitude,
        store
    )

    location_points = store.get("corridor_location_points", [])

    # =====================================================
    # 1. Prefer nearest real corridor historical point
    # =====================================================

    best_point, best_point_distance = nearest_point_match(
        latitude=latitude,
        longitude=longitude,
        points=location_points,
        allow_bad_corridor=False
    )

    if (
        best_point is not None
        and
        best_point_distance is not None
        and
        best_point_distance <= max_point_distance_m
    ):
        confidence = "HIGH"

        if best_point_distance > 1000:
            confidence = "MEDIUM"

        return {
            "corridor": normalize_corridor_name(best_point["corridor"]),
            "matched_by": "nearest real corridor historical point",
            "distance_m": round(best_point_distance, 2),
            "confidence": confidence,
            "spatial_cluster_id": spatial_cluster_id,
            "spatial_cluster_distance_m": (
                None
                if cluster_distance is None
                else round(cluster_distance, 2)
            ),
            "nearest_hotspot_distance_m": (
                None
                if hotspot_distance is None
                else round(hotspot_distance, 2)
            ),
            "spatial_density_at_point": round(spatial_density, 4),
            "outside_bengaluru": False,
            "used_bad_corridor": False,
        }

    # =====================================================
    # 2. Try nearest real corridor centroid
    # =====================================================

    corridor_profiles = store.get("corridor_location_profiles", {})

    best_corridor, best_centroid_distance = nearest_corridor_centroid_match(
        latitude=latitude,
        longitude=longitude,
        corridor_profiles=corridor_profiles,
        allow_bad_corridor=False
    )

    if (
        best_corridor is not None
        and
        best_centroid_distance is not None
        and
        best_centroid_distance <= max_centroid_distance_m
    ):
        confidence = "MEDIUM"

        if best_centroid_distance > 3000:
            confidence = "LOW"

        return {
            "corridor": best_corridor,
            "matched_by": "nearest real corridor centroid",
            "distance_m": round(best_centroid_distance, 2),
            "confidence": confidence,
            "spatial_cluster_id": spatial_cluster_id,
            "spatial_cluster_distance_m": (
                None
                if cluster_distance is None
                else round(cluster_distance, 2)
            ),
            "nearest_hotspot_distance_m": (
                None
                if hotspot_distance is None
                else round(hotspot_distance, 2)
            ),
            "spatial_density_at_point": round(spatial_density, 4),
            "outside_bengaluru": False,
            "used_bad_corridor": False,
        }

    # =====================================================
    # 3. Use cluster to infer nearest real corridor
    # =====================================================

    cluster_corridor, cluster_corridor_distance = nearest_real_corridor_inside_cluster(
        latitude=latitude,
        longitude=longitude,
        spatial_cluster_id=spatial_cluster_id,
        store=store
    )

    if cluster_corridor is not None:
        return {
            "corridor": cluster_corridor,
            "matched_by": "nearest real corridor inside spatial cluster",
            "distance_m": (
                None
                if cluster_corridor_distance is None
                else round(cluster_corridor_distance, 2)
            ),
            "confidence": "LOW",
            "spatial_cluster_id": spatial_cluster_id,
            "spatial_cluster_distance_m": (
                None
                if cluster_distance is None
                else round(cluster_distance, 2)
            ),
            "nearest_hotspot_distance_m": (
                None
                if hotspot_distance is None
                else round(hotspot_distance, 2)
            ),
            "spatial_density_at_point": round(spatial_density, 4),
            "outside_bengaluru": False,
            "used_bad_corridor": False,
        }

    # =====================================================
    # 4. Last resort only: allow bad corridor
    # =====================================================

    fallback_point, fallback_distance = nearest_point_match(
        latitude=latitude,
        longitude=longitude,
        points=location_points,
        allow_bad_corridor=True
    )

    fallback_corridor = "Non-corridor"

    if fallback_point is not None:
        fallback_corridor = normalize_corridor_name(
            fallback_point.get("corridor", "Non-corridor")
        )

    if is_bad_corridor_name(fallback_corridor):
        fallback_corridor = "Non-corridor"

    return {
        "corridor": fallback_corridor,
        "matched_by": "spatial cluster fallback; no reliable corridor label",
        "distance_m": (
            None
            if fallback_distance is None
            else round(fallback_distance, 2)
        ),
        "confidence": "LOW",
        "spatial_cluster_id": spatial_cluster_id,
        "spatial_cluster_distance_m": (
            None
            if cluster_distance is None
            else round(cluster_distance, 2)
        ),
        "nearest_hotspot_distance_m": (
            None
            if hotspot_distance is None
            else round(hotspot_distance, 2)
        ),
        "spatial_density_at_point": round(spatial_density, 4),
        "outside_bengaluru": False,
        "used_bad_corridor": True,
    }


def clean_profile(profile):
    output = {}

    if profile is None:
        profile = {}

    for feature in PROFILE_FEATURES:
        value = profile.get(feature, 0.0)

        try:
            output[feature] = float(value)
        except Exception:
            output[feature] = 0.0

    return output


def find_nearest_corridor_hour_profile(store, corridor, requested_hour):
    profiles = store.get("corridor_hour_profiles", {})

    available_hours = []

    for key in profiles.keys():
        try:
            c, h = key.rsplit("__", 1)

            if c == corridor:
                available_hours.append(int(h))

        except Exception:
            continue

    if not available_hours:
        return None, None

    nearest_hour = min(
        available_hours,
        key=lambda h: min(
            abs(h - requested_hour),
            24 - abs(h - requested_hour)
        )
    )

    nearest_key = make_key(
        corridor,
        nearest_hour
    )

    return profiles.get(nearest_key), nearest_hour


def find_nearest_cluster_hour_profile(store, cluster_id, requested_hour):
    if cluster_id is None:
        return None, None

    profiles = store.get("spatial_cluster_hour_profiles", {})

    available_hours = []

    for key in profiles.keys():
        try:
            c, h = key.rsplit("__", 1)

            if int(c) == int(cluster_id):
                available_hours.append(int(h))

        except Exception:
            continue

    if not available_hours:
        return None, None

    nearest_hour = min(
        available_hours,
        key=lambda h: min(
            abs(h - requested_hour),
            24 - abs(h - requested_hour)
        )
    )

    nearest_key = make_key(
        cluster_id,
        nearest_hour
    )

    return profiles.get(nearest_key), nearest_hour


def get_profile_with_spatial_fallback(
    store,
    corridor,
    hour,
    location_match
):
    cluster_id = None
    confidence = "LOW"
    used_bad_corridor = False

    if location_match:
        cluster_id = location_match.get("spatial_cluster_id")
        confidence = location_match.get("confidence", "LOW")
        used_bad_corridor = location_match.get("used_bad_corridor", False)

    weak_location = (
        confidence in ["LOW", "INVALID"]
        or used_bad_corridor
        or is_bad_corridor_name(corridor)
    )

    corridor_hour_profiles = store.get("corridor_hour_profiles", {})
    corridor_profiles = store.get("corridor_profiles", {})
    cluster_hour_profiles = store.get("spatial_cluster_hour_profiles", {})
    cluster_profiles = store.get("spatial_cluster_profiles", {})
    global_profile = store.get("global_profile", {})

    # =====================================================
    # If location is weak, use spatial cluster first.
    # This is the key fix for unknown/new locations.
    # =====================================================

    if weak_location and cluster_id is not None:
        cluster_key = make_key(
            cluster_id,
            hour
        )

        if cluster_key in cluster_hour_profiles:
            return (
                clean_profile(cluster_hour_profiles[cluster_key]),
                "spatial cluster-hour fallback history",
                hour
            )

        nearest_cluster_profile, nearest_cluster_hour = find_nearest_cluster_hour_profile(
            store,
            cluster_id,
            hour
        )

        if nearest_cluster_profile is not None:
            return (
                clean_profile(nearest_cluster_profile),
                f"nearest spatial cluster-hour history, hour {nearest_cluster_hour}",
                nearest_cluster_hour
            )

        cluster_profile = cluster_profiles.get(str(cluster_id))

        if cluster_profile is not None:
            return (
                clean_profile(cluster_profile),
                "spatial cluster fallback history",
                None
            )

    # =====================================================
    # Strong location: use inferred real corridor first.
    # =====================================================

    if not is_bad_corridor_name(corridor):
        corridor_key = make_key(
            corridor,
            hour
        )

        if corridor_key in corridor_hour_profiles:
            return (
                clean_profile(corridor_hour_profiles[corridor_key]),
                "exact inferred corridor-hour history",
                hour
            )

        nearest_profile, nearest_hour = find_nearest_corridor_hour_profile(
            store,
            corridor,
            hour
        )

        if nearest_profile is not None:
            return (
                clean_profile(nearest_profile),
                f"nearest inferred corridor-hour history, hour {nearest_hour}",
                nearest_hour
            )

        if corridor in corridor_profiles:
            return (
                clean_profile(corridor_profiles[corridor]),
                "inferred corridor-level fallback history",
                None
            )

    # =====================================================
    # If corridor failed, try cluster anyway.
    # =====================================================

    if cluster_id is not None:
        cluster_key = make_key(
            cluster_id,
            hour
        )

        if cluster_key in cluster_hour_profiles:
            return (
                clean_profile(cluster_hour_profiles[cluster_key]),
                "spatial cluster-hour fallback history",
                hour
            )

        nearest_cluster_profile, nearest_cluster_hour = find_nearest_cluster_hour_profile(
            store,
            cluster_id,
            hour
        )

        if nearest_cluster_profile is not None:
            return (
                clean_profile(nearest_cluster_profile),
                f"nearest spatial cluster-hour history, hour {nearest_cluster_hour}",
                nearest_cluster_hour
            )

        cluster_profile = cluster_profiles.get(str(cluster_id))

        if cluster_profile is not None:
            return (
                clean_profile(cluster_profile),
                "spatial cluster fallback history",
                None
            )

    return (
        clean_profile(global_profile),
        "global fallback history",
        None
    )