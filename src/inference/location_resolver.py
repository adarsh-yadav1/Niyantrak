import math


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


def is_valid_coordinate(
    latitude,
    longitude
):
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


def haversine_distance_meters(
    lat1,
    lon1,
    lat2,
    lon2
):
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


def find_nearest_hotspot(
    latitude,
    longitude,
    store
):
    hotspots = store.get(
        "hotspot_points",
        []
    )

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


def estimate_spatial_density(
    latitude,
    longitude,
    store,
    radius_m=500
):
    points = store.get(
        "corridor_location_points",
        []
    )

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

    density = max(
        0.0,
        min(density, 1.0)
    )

    return density


def resolve_spatial_cluster(
    latitude,
    longitude,
    store
):
    model = store.get(
        "spatial_cluster_model"
    )

    centers = store.get(
        "spatial_cluster_centers",
        {}
    )

    if model is None:
        return None, None

    try:
        cluster_id = int(
            model.predict(
                [[float(latitude), float(longitude)]]
            )[0]
        )

        center = centers.get(
            str(cluster_id)
        )

        if center is None:
            center = centers.get(
                cluster_id
            )

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


def resolve_corridor_from_coordinates(
    latitude,
    longitude,
    store,
    max_point_distance_m=2500
):
    if not is_valid_coordinate(
        latitude,
        longitude
    ):
        return {
            "corridor": "Non-corridor",
            "matched_by": "invalid coordinates fallback",
            "distance_m": None,
            "confidence": "LOW",
            "spatial_cluster_id": None,
            "spatial_cluster_distance_m": None,
            "nearest_hotspot_distance_m": None,
            "spatial_density_at_point": 0.0,
        }

    latitude = float(latitude)
    longitude = float(longitude)

    spatial_cluster_id, cluster_distance = resolve_spatial_cluster(
        latitude,
        longitude,
        store
    )

    hotspot, hotspot_distance = find_nearest_hotspot(
        latitude,
        longitude,
        store
    )

    spatial_density = estimate_spatial_density(
        latitude,
        longitude,
        store
    )

    location_points = store.get(
        "corridor_location_points",
        []
    )

    best_point = None
    best_point_distance = None

    for point in location_points:
        try:
            distance = haversine_distance_meters(
                latitude,
                longitude,
                point["latitude"],
                point["longitude"]
            )

            if (
                best_point_distance is None
                or
                distance < best_point_distance
            ):
                best_point_distance = distance
                best_point = point

        except Exception:
            continue

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
            "corridor": best_point["corridor"],
            "matched_by": "nearest historical event point",
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
        }

    corridor_profiles = store.get(
        "corridor_location_profiles",
        {}
    )

    best_corridor = None
    best_centroid_distance = None

    for corridor, profile in corridor_profiles.items():
        try:
            distance = haversine_distance_meters(
                latitude,
                longitude,
                profile["latitude"],
                profile["longitude"]
            )

            if (
                best_centroid_distance is None
                or
                distance < best_centroid_distance
            ):
                best_centroid_distance = distance
                best_corridor = corridor

        except Exception:
            continue

    if best_corridor is not None:
        confidence = "MEDIUM"

        if (
            best_centroid_distance is not None
            and
            best_centroid_distance > 4000
        ):
            confidence = "LOW"

        return {
            "corridor": best_corridor,
            "matched_by": "nearest corridor centroid",
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
        }

    return {
        "corridor": "Non-corridor",
        "matched_by": "global fallback",
        "distance_m": None,
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
    }


def clean_profile(
    profile
):
    output = {}

    if profile is None:
        profile = {}

    for feature in PROFILE_FEATURES:
        value = profile.get(
            feature,
            0.0
        )

        try:
            output[feature] = float(value)
        except Exception:
            output[feature] = 0.0

    return output


def find_nearest_corridor_hour_profile(
    store,
    corridor,
    requested_hour
):
    profiles = store.get(
        "corridor_hour_profiles",
        {}
    )

    available_hours = []

    for key in profiles.keys():
        try:
            c, h = key.rsplit(
                "__",
                1
            )

            if c == corridor:
                available_hours.append(
                    int(h)
                )

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


def find_nearest_cluster_hour_profile(
    store,
    cluster_id,
    requested_hour
):
    if cluster_id is None:
        return None, None

    profiles = store.get(
        "spatial_cluster_hour_profiles",
        {}
    )

    available_hours = []

    for key in profiles.keys():
        try:
            c, h = key.rsplit(
                "__",
                1
            )

            if int(c) == int(cluster_id):
                available_hours.append(
                    int(h)
                )

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

    if location_match:
        cluster_id = location_match.get(
            "spatial_cluster_id"
        )

    confidence = "LOW"

    if location_match:
        confidence = location_match.get(
            "confidence",
            "LOW"
        )

    weak_location = (
        confidence == "LOW"
        or
        str(corridor).strip().lower() in [
            "unknown",
            "non-corridor",
            "non corridor"
        ]
    )

    corridor_hour_profiles = store.get(
        "corridor_hour_profiles",
        {}
    )

    corridor_profiles = store.get(
        "corridor_profiles",
        {}
    )

    cluster_hour_profiles = store.get(
        "spatial_cluster_hour_profiles",
        {}
    )

    cluster_profiles = store.get(
        "spatial_cluster_profiles",
        {}
    )

    global_profile = store.get(
        "global_profile",
        {}
    )

    # For weak/unknown locations, prefer cluster history first.
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

        cluster_profile = cluster_profiles.get(
            str(cluster_id)
        )

        if cluster_profile is not None:
            return (
                clean_profile(cluster_profile),
                "spatial cluster fallback history",
                None
            )

    # Strong location: use inferred corridor first.
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

    # If corridor failed, try cluster.
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

        cluster_profile = cluster_profiles.get(
            str(cluster_id)
        )

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