import os
from pathlib import Path
from datetime import datetime
import json
import joblib
import numpy as np
import pandas as pd

from django.conf import settings

from src.forecasting.forecast_predictor import predict_single_forecast
from src.forecasting.spatial_forecast_predictor import (
    predict_single_spatial_forecast
)
from src.inference.active_event_memory import (
    get_active_event_pressure,
    save_active_event,
)

from src.inference.location_resolver import (
    resolve_corridor_from_coordinates,
    get_profile_with_spatial_fallback,
    is_bad_corridor_name,
)

from src.inference.police_station_resolver import (
    resolve_nearest_police_station,
    is_bad_station_name,
)

from src.inference.similar_events import find_similar_events

from src.scoring.event_impact import calculate_event_impact

from src.scoring.risk_score import (
    calculate_forecast_risk_score,
    calculate_final_operational_risk,
)

from src.routing.diversion_engine import (
    recommend_diversions,
    get_candidate_corridors,
)


BASE_DIR = Path(settings.BASE_DIR)


ABLATION_PATH = (
    BASE_DIR
    / "models"
    / "cluster_fallback_ablation.json"
)

EIS_CALIBRATION_PATH = (
    BASE_DIR
    / "models"
    / "eis_weight_calibration.json"
)

MODEL_PATHS = [
    BASE_DIR / "models" / "timeseries_forecast_model.pkl",
    BASE_DIR / "models" / "timeseries_forecast.pkl",
]

SPATIAL_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "spatial_timeseries_forecast_model.pkl"
)

FEATURE_STORE_PATH = (
    BASE_DIR
    / "models"
    / "traffic_feature_store.pkl"
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


_MODEL_CACHE = None
_SPATIAL_MODEL_CACHE = None
_STORE_CACHE = None


def build_calendar_event_context_from_payload(payload):
    event_type = (
        str(payload.get("event_type", "unplanned"))
        .strip()
        .lower()
    )

    event_cause = (
        str(payload.get("event_cause", "others"))
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    crowd_size = (
        str(payload.get("crowd_size", "unknown"))
        .strip()
        .lower()
    )

    explicit_calendar_type = (
        str(payload.get("calendar_event_type", ""))
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    invalid_crowd_values = {
        "",
        "unknown",
        "none",
        "nan",
        "null",
        "not_applicable",
        "na",
    }

    planned_causes = {
        "public_event": "public_event",
        "protest": "protest",
        "procession": "procession",
        "vip_movement": "vip",
        "festival": "festival",
        "sports": "sports",
        "political": "political",
        "election": "election",
    }

    if explicit_calendar_type and explicit_calendar_type not in invalid_crowd_values:
        calendar_event_type = explicit_calendar_type

    elif event_cause in planned_causes:
        calendar_event_type = planned_causes[event_cause]

    elif event_type == "planned":
        calendar_event_type = "public_event"

    else:
        calendar_event_type = "none"

    if calendar_event_type == "none":
        return {
            "is_event_day": 0,
            "calendar_event_type": "none",
            "calendar_event_intensity": 0.0,
            "crowd_size_used_for_calendar": "not_applicable",
        }

    if crowd_size in invalid_crowd_values:
        resolved_crowd_size = "medium"
    else:
        resolved_crowd_size = crowd_size

    intensity_map = {
        "none": 0,
        "sports": 75,
        "festival": 70,
        "public_event": 70,
        "protest": 85,
        "procession": 75,
        "political": 85,
        "vip": 90,
        "election": 80,
        "construction": 55,
        "other": 45,
    }

    crowd_multiplier = {
        "small": 0.60,
        "medium": 0.80,
        "large": 1.00,
        "mega": 1.20,
    }.get(
        resolved_crowd_size,
        0.80
    )

    calendar_event_intensity = (
        intensity_map.get(
            calendar_event_type,
            45
        )
        *
        crowd_multiplier
    )

    calendar_event_intensity = clamp(
        calendar_event_intensity,
        0,
        100
    )

    return {
        "is_event_day": 1,
        "calendar_event_type": calendar_event_type,
        "calendar_event_intensity": calendar_event_intensity,
        "crowd_size_used_for_calendar": resolved_crowd_size,
    }


def parse_bool(value):
    value = str(value).strip().lower()

    return value in [
        "yes",
        "true",
        "1",
        "y",
        "on",
    ]


def safe_float(value, fallback=0.0):
    try:
        if value is None:
            return fallback

        value = str(value).strip()

        if value == "":
            return fallback

        return float(value)

    except Exception:
        return fallback


def clamp(value, low=0.0, high=100.0):
    return max(
        low,
        min(float(value), high)
    )


def load_ablation_results():
    if not os.path.exists(ABLATION_PATH):
        return {
            "available": False,
            "message": "Cluster fallback ablation has not been run yet.",
        }

    try:
        with open(
            ABLATION_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        data["available"] = True

        return data

    except Exception as e:
        return {
            "available": False,
            "message": f"Could not load ablation results: {e}",
        }


def load_eis_calibration_results():
    default_weights = {
        "forecast_weight": 0.30,
        "event_weight": 0.55,
        "cause_weight": 0.15,
    }

    if not os.path.exists(EIS_CALIBRATION_PATH):
        return {
            "available": False,
            "message": "EIS calibration has not been run yet.",
            "best_weights": default_weights,
            "best_method": "Default expert-weighted formula",
            "best_mae": None,
        }

    try:
        with open(
            EIS_CALIBRATION_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            data = json.load(file)

        data["available"] = True

        if "best_weights" not in data:
            data["best_weights"] = default_weights

        return data

    except Exception as e:
        return {
            "available": False,
            "message": f"Could not load EIS calibration: {e}",
            "best_weights": default_weights,
            "best_method": "Default expert-weighted formula",
            "best_mae": None,
        }


def load_model():
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    for path in MODEL_PATHS:
        if os.path.exists(path):
            _MODEL_CACHE = joblib.load(path)
            return _MODEL_CACHE

    raise FileNotFoundError(
        "Forecast model not found. Run python train_all.py first."
    )


def load_spatial_model_optional():
    global _SPATIAL_MODEL_CACHE

    if _SPATIAL_MODEL_CACHE is not None:
        return _SPATIAL_MODEL_CACHE

    if not os.path.exists(SPATIAL_MODEL_PATH):
        return None

    _SPATIAL_MODEL_CACHE = joblib.load(
        SPATIAL_MODEL_PATH
    )

    return _SPATIAL_MODEL_CACHE


def load_feature_store():
    global _STORE_CACHE

    if _STORE_CACHE is not None:
        return _STORE_CACHE

    if not os.path.exists(FEATURE_STORE_PATH):
        raise FileNotFoundError(
            "Feature store not found. Run python prepare_feature_store.py first."
        )

    _STORE_CACHE = joblib.load(FEATURE_STORE_PATH)

    return _STORE_CACHE


def resolve_corridor_name(corridor, store):
    corridor = str(corridor).strip()

    if is_bad_corridor_name(corridor):
        return "Non-corridor"

    user_clean = corridor.lower()

    for c in store.get("corridors", []):
        if str(c).strip().lower() == user_clean:
            return c

    return corridor


def normalize_timestamp(timestamp_string):
    if not timestamp_string:
        return datetime.now()

    try:
        return datetime.fromisoformat(timestamp_string)

    except Exception:
        return datetime.now()


def build_input_row(
        corridor,
        hour,
        weekday,
        month,
        profile,
        calendar_context=None
):
    if calendar_context is None:
        calendar_context = {
            "is_event_day": 0,
            "calendar_event_type": "none",
            "calendar_event_intensity": 0.0,
        }

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

        elif feature == "is_event_day":
            row[feature] = int(
                calendar_context.get("is_event_day", 0)
            )

        elif feature == "calendar_event_type":
            row[feature] = str(
                calendar_context.get("calendar_event_type", "none")
            )

        elif feature == "calendar_event_intensity":
            row[feature] = safe_float(
                calendar_context.get("calendar_event_intensity"),
                0.0
            )

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


def build_spatial_input_row(
    latitude,
    longitude,
    corridor,
    hour,
    weekday,
    month,
    profile,
    location_match,
    calendar_context=None
):
    if calendar_context is None:
        calendar_context = {
            "is_event_day": 0,
            "calendar_event_type": "none",
            "calendar_event_intensity": 0.0,
        }

    cluster_id = location_match.get(
        "spatial_cluster_id"
    )

    if cluster_id is None:
        return None

    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    row = {
        "spatial_cluster_id": str(cluster_id),

        "latitude": latitude,
        "longitude": longitude,

        "dominant_corridor": corridor,

        "nearest_corridor_distance_m": safe_float(
            location_match.get("distance_m"),
            9999.0
        ),

        "nearest_hotspot_distance_m": safe_float(
            location_match.get("nearest_hotspot_distance_m"),
            9999.0
        ),

        "spatial_density_at_point": safe_float(
            location_match.get("spatial_density_at_point"),
            0.0
        ),

        "hour": hour,
        "weekday": weekday,
        "month": month,

        "is_event_day": int(calendar_context.get("is_event_day", 0)),
        "calendar_event_type": str(calendar_context.get("calendar_event_type", "none")),
        "calendar_event_intensity": safe_float(calendar_context.get("calendar_event_intensity"), 0.0),

        "hour_sin": hour_sin,
        "hour_cos": hour_cos,

        "lag_1": safe_float(profile.get("lag_1"), 0.0),
        "lag_2": safe_float(profile.get("lag_2"), 0.0),
        "lag_3": safe_float(profile.get("lag_3"), 0.0),
        "lag_24": safe_float(profile.get("lag_24"), 0.0),
        "lag_48": safe_float(profile.get("lag_48"), 0.0),
        "lag_72": safe_float(profile.get("lag_72"), 0.0),
        "lag_168": safe_float(profile.get("lag_168"), 0.0),

        "any_incident_last_3h": safe_float(profile.get("any_incident_last_3h"), 0.0),
        "incidents_last_24h": safe_float(profile.get("incidents_last_24h"), 0.0),
        "above_corridor_avg": safe_float(profile.get("above_corridor_avg"), 0.0),

        "rolling_6": safe_float(profile.get("rolling_6"), 0.0),
        "rolling_12": safe_float(profile.get("rolling_12"), 0.0),
        "rolling_24": safe_float(profile.get("rolling_24"), 0.0),
        "rolling_168": safe_float(profile.get("rolling_168"), 0.0),

        "corridor_avg": safe_float(profile.get("corridor_avg"), 0.0),
        "corridor_volatility": safe_float(profile.get("corridor_volatility"), 0.0),

        "zone_risk": safe_float(profile.get("zone_risk"), 0.0),
        "junction_risk": safe_float(profile.get("junction_risk"), 0.0),
        "cause_risk": safe_float(profile.get("cause_risk"), 0.0),
        "closure_risk": safe_float(profile.get("closure_risk"), 0.0),
        "cluster_risk": safe_float(profile.get("cluster_risk"), 0.0),
    }

    spatial_features = [
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

    return pd.DataFrame(
        [row],
        columns=spatial_features
    )

def estimate_corridor_state_for_diversion(
    corridor,
    hour,
    weekday,
    month,
    model,
    store
):
    try:
        location_match_stub = {
            "confidence": "MEDIUM",
            "used_bad_corridor": False,
            "spatial_cluster_id": None,
        }

        profile, profile_source, matched_hour = get_profile_with_spatial_fallback(
            store=store,
            corridor=corridor,
            hour=hour,
            location_match=location_match_stub
        )

        X_candidate = build_input_row(
            corridor=corridor,
            hour=hour,
            weekday=weekday,
            month=month,
            profile=profile
        )

        candidate_predicted_incidents, candidate_details = predict_single_forecast(
            model,
            X_candidate
        )

        candidate_predicted_incidents = max(
            safe_float(candidate_predicted_incidents),
            0.0
        )

        candidate_alert_probability = None

        if candidate_details.get("alert_probability") is not None:
            candidate_alert_probability = float(
                candidate_details["alert_probability"][0]
            )

        try:
            ml_forecast_score, ml_forecast_level = calculate_forecast_risk_score(
                predicted_incidents=candidate_predicted_incidents,
                incident_p95=store.get("incident_p95", 1.0),
                incident_p99=store.get("incident_p99", None),
                alert_probability=candidate_alert_probability,
                context_multiplier=1.0,
            )

        except TypeError:
            ml_forecast_score, ml_forecast_level = calculate_forecast_risk_score(
                candidate_predicted_incidents,
                store.get("incident_p95", 1.0)
            )

        route_pressure = calculate_route_pressure_score(
            profile=profile,
            store=store,
            ml_forecast_score=ml_forecast_score,
            predicted_incidents=candidate_predicted_incidents,
            alert_probability=candidate_alert_probability
        )

        return {
            "forecast_score": route_pressure["route_pressure_score"],
            "forecast_level": route_pressure["route_pressure_level"],

            "ml_forecast_score": ml_forecast_score,
            "ml_forecast_level": ml_forecast_level,
            "predicted_incidents": candidate_predicted_incidents,

            "historical_load": route_pressure["historical_load"],
            "load_score": route_pressure["load_score"],
            "alert_score": route_pressure["alert_score"],
            "volatility_score": route_pressure["volatility_score"],

            "alert_probability": candidate_alert_probability,
            "profile_source": profile_source,
            "matched_hour": matched_hour,
            "status": "OK",
        }

    except Exception as e:
        return {
            "forecast_score": 50.0,
            "forecast_level": "UNKNOWN",

            "ml_forecast_score": 0.0,
            "ml_forecast_level": "UNKNOWN",
            "predicted_incidents": 0.0,

            "historical_load": 0.0,
            "load_score": 0.0,
            "alert_score": 0.0,
            "volatility_score": 0.0,

            "alert_probability": None,
            "profile_source": "state check failed",
            "matched_hour": None,
            "status": f"FAILED: {e}",
        }

def build_diversion_corridor_state_lookup(
    affected_corridor,
    hour,
    weekday,
    month,
    model,
    store
):
    candidates = get_candidate_corridors(
        affected_corridor,
        max_depth=2
    )

    corridor_states = {}

    for candidate in candidates:
        candidate_corridor = candidate["corridor"]

        corridor_states[candidate_corridor] = estimate_corridor_state_for_diversion(
            corridor=candidate_corridor,
            hour=hour,
            weekday=weekday,
            month=month,
            model=model,
            store=store
        )

    return corridor_states


def get_risk_level_from_score(score):
    score = float(score)

    if score < 25:
        return "LOW"

    if score < 50:
        return "MODERATE"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


def is_crowd_relevant_event(event_type, event_cause):
    event_type = (
        str(event_type or "")
        .strip()
        .lower()
    )

    event_cause = (
        str(event_cause or "")
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    crowd_relevant_causes = {
        "public_event",
        "protest",
        "procession",
        "vip_movement",
        "festival",
        "sports",
        "political",
        "election",
    }

    return (
        event_type == "planned"
        or
        event_cause in crowd_relevant_causes
    )


def get_crowd_multiplier(
    crowd_size,
    event_type="unplanned",
    event_cause="others"
):
    """
    Crowd size should only affect planned / crowd-bearing events.

    For unplanned events, crowd is usually unknown and should not boost risk.
    """

    if not is_crowd_relevant_event(
        event_type,
        event_cause
    ):
        return 1.00, "not_applicable"

    crowd_size = (
        str(crowd_size or "unknown")
        .strip()
        .lower()
    )

    if crowd_size in [
        "",
        "unknown",
        "none",
        "nan",
        "null",
    ]:
        # Conservative default only for planned events.
        crowd_size = "medium"

    multipliers = {
        "small": 1.00,
        "medium": 1.08,
        "large": 1.18,
        "mega": 1.30,
    }

    return (
        multipliers.get(crowd_size, 1.08),
        crowd_size
    )

def get_weather_multiplier(weather):
    weather = (
        str(weather or "clear")
        .strip()
        .lower()
    )

    return {
        "clear": 1.00,
        "light_rain": 1.10,
        "heavy_rain": 1.25,
        "fog": 1.20,
    }.get(
        weather,
        1.00
    )


def get_priority_boost(priority):
    return {
        "low": 0,
        "medium": 4,
        "high": 8,
        "critical": 12,
    }.get(
        str(priority).strip().lower(),
        4
    )


def get_event_type_boost(event_type):
    return {
        "planned": 2,
        "unplanned": 6,
    }.get(
        str(event_type).strip().lower(),
        4
    )

def calculate_adjusted_event_score(
    base_event_score,
    priority,
    event_type,
    crowd_size,
    weather,
    event_cause="others"
):
    priority_boost = get_priority_boost(priority)
    event_type_boost = get_event_type_boost(event_type)

    score = (
        safe_float(base_event_score)
        +
        priority_boost
        +
        event_type_boost
    )

    crowd_multiplier, resolved_crowd_size = get_crowd_multiplier(
        crowd_size=crowd_size,
        event_type=event_type,
        event_cause=event_cause
    )

    weather_multiplier = get_weather_multiplier(
        weather
    )

    score = (
        score
        *
        crowd_multiplier
        *
        weather_multiplier
    )

    score = clamp(score, 0, 100)

    return (
        score,
        crowd_multiplier,
        weather_multiplier,
        resolved_crowd_size
    )

def calculate_eis_score(
    forecast_score,
    adjusted_event_score,
    profile,
    store,
    weights=None
):
    if weights is None:
        weights = {
            "forecast_weight": 0.30,
            "event_weight": 0.55,
            "cause_weight": 0.15,
        }

    max_cause_risk = safe_float(
        store.get("max_cause_risk"),
        1.0
    )

    cause_risk_score = (
        safe_float(profile.get("cause_risk"), 0.0)
        /
        max(max_cause_risk, 1.0)
    ) * 100

    cause_risk_score = clamp(
        cause_risk_score,
        0,
        100
    )

    forecast_weight = safe_float(
        weights.get("forecast_weight"),
        0.30
    )

    event_weight = safe_float(
        weights.get("event_weight"),
        0.55
    )

    cause_weight = safe_float(
        weights.get("cause_weight"),
        0.15
    )

    total_weight = max(
        forecast_weight + event_weight + cause_weight,
        1e-6
    )

    forecast_weight /= total_weight
    event_weight /= total_weight
    cause_weight /= total_weight

    eis_score = (
        forecast_weight * safe_float(forecast_score)
        +
        event_weight * safe_float(adjusted_event_score)
        +
        cause_weight * cause_risk_score
    )

    eis_score = clamp(
        eis_score,
        0,
        100
    )

    return eis_score, cause_risk_score


def calculate_operational_after_event_index(
    normal_baseline,
    predicted_incidents,
    eis_score,
    road_closure,
    crowd_size,
    weather,
    store
):
    normal_baseline = max(
        safe_float(normal_baseline),
        0.0
    )

    predicted_incidents = max(
        safe_float(predicted_incidents),
        0.0
    )

    incident_p95 = max(
        safe_float(store.get("incident_p95"), 1.0),
        1.0
    )

    event_pressure = (
        safe_float(eis_score)
        /
        100.0
    ) * incident_p95

    if road_closure:
        event_pressure += 0.35 * incident_p95

    if str(crowd_size).lower() in ["large", "mega"]:
        event_pressure += 0.20 * incident_p95

    if str(weather).lower() in ["heavy_rain", "fog"]:
        event_pressure += 0.15 * incident_p95

    after_event_index = max(
        predicted_incidents,
        normal_baseline + event_pressure
    )

    expected_delta = max(
        after_event_index - normal_baseline,
        0.0
    )

    if normal_baseline > 0:
        percentage_increase = (
            expected_delta
            /
            normal_baseline
        ) * 100
    else:
        percentage_increase = 100 if after_event_index > 0 else 0

    percentage_increase = clamp(
        percentage_increase,
        0,
        999
    )

    return after_event_index, expected_delta, percentage_increase


def recommend_resources(
    final_risk_level,
    predicted_incidents,
    road_closure
):
    if final_risk_level == "LOW":
        officers = 2
        barricades = 0

    elif final_risk_level == "MODERATE":
        officers = 4
        barricades = 1

    elif final_risk_level == "HIGH":
        officers = 6
        barricades = 2

    else:
        officers = 8
        barricades = 4

    if predicted_incidents >= 3:
        officers += 1

    if predicted_incidents >= 5:
        barricades += 1

    if predicted_incidents >= 8:
        officers += 2
        barricades += 1

    if road_closure:
        officers += 1
        barricades += 2

    return officers, barricades


def get_action_message(final_risk_level, road_closure):
    if final_risk_level == "LOW":
        return "Normal monitoring is sufficient."

    if final_risk_level == "MODERATE":
        return "Increase patrol visibility and monitor CCTV feeds."

    if final_risk_level == "HIGH":
        if road_closure:
            return "Deploy officers, prepare barricades, and keep diversion support ready."

        return "Deploy traffic officers and prepare diversion support."

    if road_closure:
        return "Immediate deployment required. Activate barricading and diversion plan."

    return "Immediate deployment required. Deploy officers and monitor corridor continuously."


def estimate_duration_minutes(
    predicted_incidents,
    final_risk_level,
    road_closure,
    weather="clear"
):
    base = 30 + safe_float(predicted_incidents) * 20

    if final_risk_level == "HIGH":
        base += 20

    elif final_risk_level == "CRITICAL":
        base += 40

    if road_closure:
        base += 30

    weather_multiplier = get_weather_multiplier(weather)

    base *= weather_multiplier

    return int(
        max(
            15,
            min(base, 240)
        )
    )


def estimate_affected_radius_meters(
    final_risk_level,
    predicted_incidents,
    road_closure,
    event_cause
):
    event_cause = (
        str(event_cause)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if final_risk_level == "LOW":
        base_radius = 250

    elif final_risk_level == "MODERATE":
        base_radius = 500

    elif final_risk_level == "HIGH":
        base_radius = 900

    else:
        base_radius = 1400

    if road_closure:
        base_radius += 500

    if predicted_incidents >= 3:
        base_radius += 300

    if predicted_incidents >= 6:
        base_radius += 500

    high_spread_causes = [
        "accident",
        "congestion",
        "vip_movement",
        "protest",
        "procession",
        "public_event",
        "water_logging",
    ]

    if event_cause in high_spread_causes:
        base_radius += 300

    return int(
        min(
            max(base_radius, 200),
            3000
        )
    )


def build_prediction_interval(
    predicted_incidents,
    model_metrics,
    alert_probability=None,
    model=None,
    X=None
):
    predicted_incidents = safe_float(
        predicted_incidents,
        0.0
    )

    if (
        isinstance(model, dict)
        and
        X is not None
        and
        model.get("quantile_lower_model") is not None
        and
        model.get("quantile_upper_model") is not None
    ):
        try:
            lower_model = model["quantile_lower_model"]
            upper_model = model["quantile_upper_model"]

            raw_lower = safe_float(
                lower_model.predict(X)[0],
                0.0
            )

            raw_upper = safe_float(
                upper_model.predict(X)[0],
                predicted_incidents
            )

            raw_lower = max(
                0.0,
                raw_lower
            )

            raw_upper = max(
                raw_lower,
                raw_upper
            )

            threshold = safe_float(
                model.get("alert_threshold"),
                0.50
            )

            if alert_probability is not None:
                alert_probability = safe_float(
                    alert_probability,
                    0.0
                )

                if alert_probability <= threshold:
                    strength = 0.0

                else:
                    strength = (
                        alert_probability
                        -
                        threshold
                    ) / max(
                        1.0 - threshold,
                        1e-6
                    )

                strength = clamp(
                    strength,
                    0.0,
                    1.0
                )

            else:
                strength = 1.0

            lower = raw_lower * strength
            upper = raw_upper * max(
                strength,
                0.25
            )

            lower = min(
                lower,
                predicted_incidents
            )

            upper = max(
                upper,
                predicted_incidents
            )

            return {
                "expected": float(predicted_incidents),
                "lower": float(lower),
                "upper": float(upper),
                "method": "CatBoost Quantile models with hurdle probability gating",
                "label": "80% prediction interval",
                "type": "catboost_quantile",
                "coverage": (
                    model.get("quantile_interval_metrics", {})
                    .get("coverage")
                ),
                "average_width": (
                    model.get("quantile_interval_metrics", {})
                    .get("average_width")
                ),
            }

        except Exception:
            pass

    rmse = safe_float(
        model_metrics.get("rmse"),
        0.50
    )

    if rmse <= 0:
        rmse = 0.50

    uncertainty_width = rmse

    if alert_probability is not None:
        probability_uncertainty = 1 - abs(
            float(alert_probability) - 0.5
        ) * 2

        uncertainty_width *= (
            1 + 0.25 * probability_uncertainty
        )

    lower = max(
        0.0,
        predicted_incidents - uncertainty_width
    )

    upper = max(
        predicted_incidents,
        predicted_incidents + uncertainty_width
    )

    return {
        "expected": float(predicted_incidents),
        "lower": float(lower),
        "upper": float(upper),
        "method": "Estimated uncertainty using holdout RMSE",
        "label": "Estimated uncertainty (±1 RMSE)",
        "type": "rmse_fallback",
        "coverage": None,
        "average_width": None,
    }


def build_deployment_order(
    latitude,
    longitude,
    corridor,
    event_cause,
    final_level,
    final_score,
    officers,
    barricades,
    diversion,
    duration,
    action
):
    support_corridors = diversion.get(
        "support_corridors",
        []
    )

    support_text = (
        ", ".join(support_corridors)
        if support_corridors
        else "None"
    )

    return (
        "TRAFFIC DEPLOYMENT ORDER\n"
        "----------------------------------------\n"
        f"Risk Level       : {final_level}\n"
        f"Risk Score       : {final_score:.2f}%\n"
        f"Event Cause      : {event_cause}\n"
        f"Location         : {latitude:.6f}, {longitude:.6f}\n"
        f"Inferred Corridor: {corridor}\n"
        f"Duration Estimate: {duration} minutes\n\n"
        "RESOURCE PLAN\n"
        "----------------------------------------\n"
        f"Officers Required: {officers}\n"
        f"Barricades Needed: {barricades}\n\n"
        "DIVERSION PLAN\n"
        "----------------------------------------\n"
        f"Primary Detour   : {diversion.get('primary_detour')}\n"
        f"Secondary Detour : {diversion.get('secondary_detour')}\n"
        f"Support Corridors: {support_text}\n\n"
        "ACTION\n"
        "----------------------------------------\n"
        f"{action}"
    )


def metric_available(value):
    try:
        if value is None:
            return False

        value = float(value)

        return not np.isnan(value)

    except Exception:
        return False


def percent_text(value):
    if not metric_available(value):
        return "Not available"

    return f"{float(value) * 100:.1f}%"


def number_text(value, digits=3):
    if not metric_available(value):
        return "Not available"

    return f"{float(value):.{digits}f}"


def build_operational_metrics(model_metrics):
    mae = model_metrics.get("mae")
    rmse = model_metrics.get("rmse")
    r2 = model_metrics.get("r2")

    alert_accuracy = model_metrics.get("alert_accuracy")
    alert_precision = model_metrics.get("alert_precision")
    alert_recall = model_metrics.get("alert_recall")
    alert_f1 = model_metrics.get("alert_f1")
    roc_auc = model_metrics.get("roc_auc")
    pr_auc = model_metrics.get("pr_auc")

    summary_cards = [
        {
            "title": "Incident Capture",
            "value": percent_text(alert_recall),
            "description": (
                "Caught this share of real incident-hours during historical validation."
            ),
        },
        {
            "title": "Risk Ranking Quality",
            "value": number_text(roc_auc, 3),
            "description": (
                "ROC-AUC shows how well the model ranks risky hours above normal hours."
            ),
        },
        {
            "title": "Average Count Error",
            "value": number_text(mae, 3),
            "description": (
                "Average incident-count error per corridor-hour."
            ),
        },
        {
            "title": "Alert Correctness",
            "value": percent_text(alert_accuracy),
            "description": (
                "Share of historical hours where the alert/no-alert decision was correct."
            ),
        },
    ]

    explanation_rows = [
        {
            "metric": "Recall",
            "raw_value": number_text(alert_recall, 4),
            "operational_meaning": (
                f"The model caught {percent_text(alert_recall)} of real incident-hours."
            ),
        },
        {
            "metric": "Precision",
            "raw_value": number_text(alert_precision, 4),
            "operational_meaning": (
                f"When the system raised an alert, {percent_text(alert_precision)} were true incident-hours."
            ),
        },
        {
            "metric": "ROC-AUC",
            "raw_value": number_text(roc_auc, 4),
            "operational_meaning": (
                "Measures whether risky hours are ranked above quiet hours. "
                "This is more important than R² for alerting."
            ),
        },
        {
            "metric": "PR-AUC",
            "raw_value": number_text(pr_auc, 4),
            "operational_meaning": (
                "Measures alert quality when incident-hours are rare."
            ),
        },
        {
            "metric": "MAE",
            "raw_value": number_text(mae, 4),
            "operational_meaning": (
                "Average count error in predicted incidents per corridor-hour."
            ),
        },
        {
            "metric": "R²",
            "raw_value": number_text(r2, 4),
            "operational_meaning": (
                "Regression fit for exact incident count. It is secondary because the dataset is zero-heavy and spike-driven."
            ),
        },
    ]

    headline = (
        f"Caught {percent_text(alert_recall)} of real incident-hours in historical validation. "
        f"Risk ranking quality was ROC-AUC {number_text(roc_auc, 3)}."
    )

    return {
        "headline": headline,
        "summary_cards": summary_cards,
        "explanation_rows": explanation_rows,
        "raw": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "alert_accuracy": alert_accuracy,
            "alert_precision": alert_precision,
            "alert_recall": alert_recall,
            "alert_f1": alert_f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        },
    }


def calculate_route_pressure_score(
    profile,
    store,
    ml_forecast_score,
    predicted_incidents,
    alert_probability
):
    incident_p95 = max(
        safe_float(store.get("incident_p95"), 1.0),
        1.0
    )

    lag_1 = safe_float(
        profile.get("lag_1"),
        0.0
    )

    lag_24 = safe_float(
        profile.get("lag_24"),
        0.0
    )

    rolling_6 = safe_float(
        profile.get("rolling_6"),
        0.0
    )

    rolling_24 = safe_float(
        profile.get("rolling_24"),
        0.0
    )

    rolling_168 = safe_float(
        profile.get("rolling_168"),
        0.0
    )

    corridor_avg = safe_float(
        profile.get("corridor_avg"),
        0.0
    )

    corridor_volatility = safe_float(
        profile.get("corridor_volatility"),
        0.0
    )

    historical_load = max(
        lag_1,
        lag_24 * 0.80,
        rolling_6,
        rolling_24 * 0.90,
        rolling_168 * 0.70,
        corridor_avg * 0.85
    )

    load_score = clamp(
        (historical_load / incident_p95) * 100,
        0,
        100
    )

    volatility_score = clamp(
        (corridor_volatility / incident_p95) * 45,
        0,
        45
    )

    predicted_score = clamp(
        (safe_float(predicted_incidents) / incident_p95) * 100,
        0,
        100
    )

    if alert_probability is None:
        alert_score = 0.0
    else:
        alert_score = clamp(
            safe_float(alert_probability) * 100,
            0,
            100
        )

    route_pressure_score = (
        0.45 * load_score
        +
        0.25 * alert_score
        +
        0.20 * volatility_score
        +
        0.10 * predicted_score
    )

    route_pressure_score = max(
        route_pressure_score,
        safe_float(ml_forecast_score)
    )

    route_pressure_score = clamp(
        route_pressure_score,
        0,
        100
    )

    route_pressure_level = get_risk_level_from_score(
        route_pressure_score
    )

    return {
        "route_pressure_score": route_pressure_score,
        "route_pressure_level": route_pressure_level,
        "historical_load": historical_load,
        "load_score": load_score,
        "alert_score": alert_score,
        "volatility_score": volatility_score,
        "predicted_score": predicted_score,
    }


def predict_event_impact(payload):
    model = load_model()
    store = load_feature_store()

    time_context = {
        "event_datetime": payload.get("event_datetime")
    }

    timestamp = normalize_timestamp(
        time_context.get("event_datetime")
    )

    hour = timestamp.hour
    weekday = timestamp.weekday()
    month = timestamp.month

    calendar_context = build_calendar_event_context_from_payload(
        payload
    )

    latitude = safe_float(
        payload.get("latitude"),
        12.9716
    )

    longitude = safe_float(
        payload.get("longitude"),
        77.5946
    )

    end_latitude = safe_float(
        payload.get("end_latitude"),
        None
    )

    end_longitude = safe_float(
        payload.get("end_longitude"),
        None
    )

    location_match = resolve_corridor_from_coordinates(
        latitude=latitude,
        longitude=longitude,
        store=store
    )

    police_station_match = resolve_nearest_police_station(
        latitude=latitude,
        longitude=longitude,
        store=store
    )

    if location_match.get("outside_bengaluru"):
        raise ValueError(
            "Selected location is outside Bengaluru coverage area. "
            "Please choose a location within Bengaluru."
        )

    corridor = resolve_corridor_name(
        location_match["corridor"],
        store
    )

    user_police_station = payload.get(
        "police_station",
        ""
    )

    if is_bad_station_name(user_police_station):
        police_station = police_station_match.get(
            "police_station",
            "Unknown"
        )
    else:
        police_station = str(
            user_police_station
        ).strip()

    event_type = payload.get(
        "event_type",
        "unplanned"
    )

    event_cause = payload.get(
        "event_cause",
        "others"
    )

    priority = payload.get(
        "priority",
        "Medium"
    )

    veh_type = payload.get(
        "veh_type",
        "unknown"
    )

    crowd_size = payload.get(
        "crowd_size",
        "small"
    )

    weather = payload.get(
        "weather",
        "clear"
    )

    road_closure = parse_bool(
        payload.get(
            "requires_road_closure",
            False
        )
    )

    profile, profile_source, matched_hour = get_profile_with_spatial_fallback(
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
        profile=profile,
        calendar_context=calendar_context
    )

    spatial_model = load_spatial_model_optional()

    spatial_X = None
    forecast_source = "corridor-hour fallback model"

    if spatial_model is not None:
        spatial_X = build_spatial_input_row(
            latitude=latitude,
            longitude=longitude,
            corridor=corridor,
            hour=hour,
            weekday=weekday,
            month=month,
            profile=profile,
            location_match=location_match,
            calendar_context=calendar_context
        )

    if spatial_model is not None and spatial_X is not None:
        try:
            predicted_incidents, forecast_details = predict_single_spatial_forecast(
                spatial_model,
                spatial_X
            )

            forecast_source = "primary spatial-cluster-hour model"

        except Exception:
            predicted_incidents, forecast_details = predict_single_forecast(
                model,
                X
            )

            forecast_source = "corridor-hour fallback model after spatial failure"

    else:
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
        alert_probability = float(
            forecast_details["alert_probability"][0]
        )

    context_multiplier = 1.0

    if is_bad_corridor_name(corridor):
        context_multiplier = 0.65

    try:
        forecast_score, forecast_level = calculate_forecast_risk_score(
            predicted_incidents=predicted_incidents,
            incident_p95=store.get("incident_p95", 1.0),
            incident_p99=store.get("incident_p99", None),
            alert_probability=alert_probability,
            context_multiplier=context_multiplier,
        )

    except TypeError:
        forecast_score, forecast_level = calculate_forecast_risk_score(
            predicted_incidents,
            store.get("incident_p95", 1.0)
        )

    rush_hour = (
        (7 <= hour <= 10)
        or
        (17 <= hour <= 21)
    )

    base_event_score, base_event_level = calculate_event_impact(
        event_cause=event_cause,
        veh_type=veh_type,
        road_closure=road_closure,
        rush_hour=rush_hour
    )

    adjusted_event_score, crowd_multiplier, weather_multiplier, resolved_crowd_size = calculate_adjusted_event_score(
        base_event_score=base_event_score,
        priority=priority,
        event_type=event_type,
        crowd_size=crowd_size,
        weather=weather,
        event_cause=event_cause
    )

    adjusted_event_level = get_risk_level_from_score(
        adjusted_event_score
    )

    eis_calibration = load_eis_calibration_results()

    eis_weights = eis_calibration.get(
        "best_weights",
        {
            "forecast_weight": 0.30,
            "event_weight": 0.55,
            "cause_weight": 0.15,
        }
    )

    eis_score, cause_risk_score = calculate_eis_score(
        forecast_score=forecast_score,
        adjusted_event_score=adjusted_event_score,
        profile=profile,
        store=store,
        weights=eis_weights
    )

    eis_level = get_risk_level_from_score(
        eis_score
    )

    final_score, final_level = calculate_final_operational_risk(
        forecast_risk_score=forecast_score,
        event_impact_score=eis_score
    )

    # ------------------------------------------------------------
    # ACTIVE EVENT MEMORY
    # ------------------------------------------------------------

    active_event_pressure = get_active_event_pressure(
        current_datetime=time_context.get("event_datetime"),
        latitude=latitude,
        longitude=longitude,
        corridor=corridor,
        spatial_cluster_id=location_match.get("spatial_cluster_id")
    )

    base_final_score = final_score

    final_score = clamp(
        final_score
        +
        active_event_pressure["pressure_score"] * 0.20,
        0,
        100
    )

    final_level = get_risk_level_from_score(
        final_score
    )

    # ------------------------------------------------------------
    # RESOURCE AND DIVERSION
    # ------------------------------------------------------------

    officers, barricades = recommend_resources(
        final_risk_level=final_level,
        predicted_incidents=predicted_incidents,
        road_closure=road_closure
    )

    diversion_corridor_states = build_diversion_corridor_state_lookup(
        affected_corridor=corridor,
        hour=hour,
        weekday=weekday,
        month=month,
        model=model,
        store=store
    )

    diversion = recommend_diversions(
        affected_corridor=corridor,
        final_risk_level=final_level,
        road_closure=road_closure,
        corridor_states=diversion_corridor_states
    )

    # Inject the actual state values back into the diversion object for frontend display
    if "candidate_evaluations" in diversion:
        for item in diversion["candidate_evaluations"]:
            c = item["corridor"]
            if c in diversion_corridor_states:
                state = diversion_corridor_states[c]
                item["forecast_score"] = state["forecast_score"]
                item["forecast_level"] = state["forecast_level"]
                item["historical_load"] = state["historical_load"]
                item["predicted_incidents"] = state["predicted_incidents"]

    duration = estimate_duration_minutes(
        predicted_incidents=predicted_incidents,
        final_risk_level=final_level,
        road_closure=road_closure,
        weather=weather
    )

    affected_radius = estimate_affected_radius_meters(
        final_risk_level=final_level,
        predicted_incidents=predicted_incidents,
        road_closure=road_closure,
        event_cause=event_cause
    )

    secondary_radius = int(
        affected_radius * 1.75
    )

    action = get_action_message(
        final_risk_level=final_level,
        road_closure=road_closure
    )

    closure_probability = clamp(
        (
            forecast_score * 0.35
            +
            adjusted_event_score * 0.65
        ) / 100,
        0,
        1
    )

    normal_baseline = safe_float(
        profile.get("rolling_24"),
        0.0
    )

    if normal_baseline <= 0:
        normal_baseline = safe_float(
            profile.get("rolling_6"),
            0.0
        )

    if normal_baseline <= 0:
        normal_baseline = safe_float(
            profile.get("corridor_avg"),
            0.0
        )

    expected_after_event, expected_delta, percentage_increase = calculate_operational_after_event_index(
        normal_baseline=normal_baseline,
        predicted_incidents=predicted_incidents,
        eis_score=eis_score,
        road_closure=road_closure,
        crowd_size=crowd_size,
        weather=weather,
        store=store
    )

    risk_delta_points = (
        final_score
        -
        forecast_score
    )

    risk_delta_percent = 0.0

    if forecast_score > 0:
        risk_delta_percent = (
            risk_delta_points
            /
            forecast_score
        ) * 100

    risk_delta_points = round(
        risk_delta_points,
        2
    )

    risk_delta_percent = max(
        0.0,
        min(
            risk_delta_percent,
            999.0
        )
    )

    holdout_metrics = {}

    if isinstance(model, dict):
        holdout_metrics = model.get(
            "holdout_metrics",
            {}
        )

    model_metrics = {
        "mae": holdout_metrics.get("mae"),
        "rmse": holdout_metrics.get("rmse"),
        "r2": holdout_metrics.get("r2"),

        "alert_accuracy": holdout_metrics.get("alert_accuracy"),
        "alert_precision": holdout_metrics.get("alert_precision"),
        "alert_recall": holdout_metrics.get("alert_recall"),
        "alert_f1": holdout_metrics.get("alert_f1"),
        "roc_auc": holdout_metrics.get("roc_auc"),
        "pr_auc": holdout_metrics.get("pr_auc"),
    }

    operational_metrics = build_operational_metrics(
        model_metrics
    )

    prediction_interval = build_prediction_interval(
        predicted_incidents=predicted_incidents,
        model_metrics=model_metrics,
        alert_probability=alert_probability,
        model=model,
        X=X
    )

    similar_events = find_similar_events(
        event_cause=event_cause,
        corridor=corridor,
        latitude=latitude,
        longitude=longitude,
        hour=hour,
        limit=3
    )

    deployment_order = build_deployment_order(
        latitude=latitude,
        longitude=longitude,
        corridor=corridor,
        event_cause=event_cause,
        final_level=final_level,
        final_score=final_score,
        officers=officers,
        barricades=barricades,
        diversion=diversion,
        duration=duration,
        action=action
    )

    ablation_results = load_ablation_results()

    result = {
        "input": {
            "event_type": event_type,
            "event_cause": event_cause,
            "priority": priority,

            "corridor": corridor,
            "veh_type": veh_type,
            "police_station": police_station,
            "police_station_match": police_station_match,

            "timestamp": timestamp,
            "hour": hour,
            "weekday": weekday,
            "month": month,

            "road_closure": road_closure,

            "latitude": latitude,
            "longitude": longitude,
            "end_latitude": end_latitude,
            "end_longitude": end_longitude,

            "crowd_size": crowd_size,
            "weather": weather,

            "location_match": location_match,
        },

        "forecast": {
            "predicted_incidents": predicted_incidents,
            "forecast_score": forecast_score,
            "forecast_level": forecast_level,
            "alert_probability": alert_probability,
            "forecast_source": forecast_source,
        },

        "event": {
            "base_event_score": base_event_score,
            "base_event_level": base_event_level,

            "event_score": adjusted_event_score,
            "event_level": adjusted_event_level,

            "rush_hour": rush_hour,
            "crowd_size_used": resolved_crowd_size,
            "crowd_multiplier": crowd_multiplier,
            "weather_multiplier": weather_multiplier,
        },

        "eis": {
            "score": eis_score,
            "level": eis_level,
            "cause_risk_score": cause_risk_score,
            "weights": eis_weights,
        },

        "active_event_memory": {
            "base_final_score": base_final_score,
            "pressure_score": active_event_pressure["pressure_score"],
            "pressure_level": active_event_pressure["pressure_level"],
            "recent_events_found": active_event_pressure["recent_events_found"],
            "contributors": active_event_pressure["contributors"],
            "explanation": active_event_pressure["explanation"],
        },

        "baseline": {
            "normal_baseline": normal_baseline,

            "predicted_after_event": expected_after_event,

            "ml_predicted_incidents": predicted_incidents,

            "expected_delta": expected_delta,
            "percentage_increase": percentage_increase,

            "normal_risk_score": forecast_score,
            "with_event_risk_score": final_score,
            "risk_delta_points": risk_delta_points,
            "risk_delta_percent": risk_delta_percent,

            "baseline_source": profile_source,
            "baseline_reference": "rolling_24 → rolling_6 → corridor_avg",
        },

        "final": {
            "final_score": final_score,
            "final_level": final_level,
            "duration_minutes": duration,
            "closure_probability": closure_probability,
            "affected_radius_m": affected_radius,
            "secondary_radius_m": secondary_radius,
        },

        "confidence": prediction_interval,

        "history": {
            "profile_source": profile_source,
            "matched_hour": matched_hour,

            "lag_1": X.loc[0, "lag_1"],
            "lag_2": X.loc[0, "lag_2"],
            "lag_3": X.loc[0, "lag_3"],
            "lag_24": X.loc[0, "lag_24"],

            "any_incident_last_3h": X.loc[0, "any_incident_last_3h"],
            "incidents_last_24h": X.loc[0, "incidents_last_24h"],
            "above_corridor_avg": X.loc[0, "above_corridor_avg"],

            "rolling_6": X.loc[0, "rolling_6"],
            "rolling_24": X.loc[0, "rolling_24"],
            "rolling_168": X.loc[0, "rolling_168"],

            "corridor_avg": X.loc[0, "corridor_avg"],
            "corridor_volatility": X.loc[0, "corridor_volatility"],
        },

        "resources": {
            "officers": officers,
            "barricades": barricades,
            "station": police_station or "Unknown",
            "shift_hours": 2 if final_level in ["LOW", "MODERATE"] else 4,
        },

        "diversion": diversion,

        "metrics": model_metrics,

        "operational_metrics": operational_metrics,

        "ablation": ablation_results,

        "eis_calibration": eis_calibration,

        "map": {
            "latitude": latitude,
            "longitude": longitude,
            "end_latitude": end_latitude,
            "end_longitude": end_longitude,
            "affected_radius_m": affected_radius,
            "secondary_radius_m": secondary_radius,
        },

        "similar_events": similar_events,

        "deployment_order": deployment_order,

        "action": action,
    }

    if not result.get("blocked"):
        saved_event = save_active_event(
            event_datetime=time_context.get("event_datetime"),
            latitude=latitude,
            longitude=longitude,
            corridor=corridor,
            spatial_cluster_id=location_match.get("spatial_cluster_id"),
            event_type=event_type,
            event_cause=event_cause,
            priority=priority,
            road_closure=road_closure,
            event_score=adjusted_event_score,
            final_score=final_score,
            source="dashboard_prediction"
        )

        result["active_event_memory"]["current_event_saved"] = True
        result["active_event_memory"]["current_event_id"] = saved_event["event_id"]

    return result