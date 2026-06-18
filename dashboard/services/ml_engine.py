import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from django.conf import settings

from src.forecasting.forecast_predictor import (
    predict_single_forecast
)

from src.inference.location_resolver import (
    resolve_corridor_from_coordinates,
    get_profile_with_spatial_fallback
)

from src.scoring.event_impact import calculate_event_impact

from src.scoring.risk_score import (
    calculate_forecast_risk_score,
    calculate_final_operational_risk
)

from src.routing.diversion_engine import recommend_diversions


BASE_DIR = Path(settings.BASE_DIR)

MODEL_PATHS = [
    BASE_DIR / "models" / "timeseries_forecast_model.pkl",
    BASE_DIR / "models" / "timeseries_forecast.pkl",
]

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


_MODEL_CACHE = None
_STORE_CACHE = None


def parse_bool(
    value
):
    value = str(value).strip().lower()

    return value in [
        "yes",
        "true",
        "1",
        "y",
        "on"
    ]


def safe_float(
    value,
    fallback=0.0
):
    try:
        if value is None:
            return fallback

        value = str(value).strip()

        if value == "":
            return fallback

        return float(value)

    except Exception:
        return fallback


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


def load_feature_store():
    global _STORE_CACHE

    if _STORE_CACHE is not None:
        return _STORE_CACHE

    if not os.path.exists(FEATURE_STORE_PATH):
        raise FileNotFoundError(
            "Feature store not found. Run python prepare_feature_store.py first."
        )

    _STORE_CACHE = joblib.load(
        FEATURE_STORE_PATH
    )

    return _STORE_CACHE


def resolve_corridor_name(
    corridor,
    store
):
    user_clean = (
        str(corridor)
        .strip()
        .lower()
    )

    for c in store.get("corridors", []):
        if (
            str(c)
            .strip()
            .lower()
            ==
            user_clean
        ):
            return c

    return str(corridor).strip()


def normalize_timestamp(
    timestamp_string
):
    if not timestamp_string:
        return datetime.now()

    try:
        return datetime.fromisoformat(
            timestamp_string
        )

    except Exception:
        return datetime.now()


def build_input_row(
    corridor,
    hour,
    weekday,
    month,
    profile
):
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
            row[feature] = profile.get(
                feature,
                0.0
            )

    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


def get_risk_level_from_score(
    score
):
    if score < 25:
        return "LOW"

    if score < 50:
        return "MODERATE"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


def apply_crowd_weather_adjustment(
    event_score,
    crowd_size,
    weather
):
    crowd = str(
        crowd_size or "small"
    ).strip().lower()

    weather = str(
        weather or "clear"
    ).strip().lower()

    crowd_multiplier = {
        "small": 1.00,
        "medium": 1.08,
        "large": 1.18,
        "mega": 1.30,
    }.get(
        crowd,
        1.00
    )

    weather_multiplier = {
        "clear": 1.00,
        "light_rain": 1.10,
        "heavy_rain": 1.25,
        "fog": 1.20,
    }.get(
        weather,
        1.00
    )

    adjusted = (
        event_score
        *
        crowd_multiplier
        *
        weather_multiplier
    )

    adjusted = max(
        0,
        min(adjusted, 100)
    )

    return adjusted, crowd_multiplier, weather_multiplier


def calculate_eis_score(
    forecast_score,
    event_score,
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

    cause_risk_score = max(
        0,
        min(cause_risk_score, 100)
    )

    eis_score = (
        0.35 * forecast_score
        +
        0.50 * event_score
        +
        0.15 * cause_risk_score
    )

    eis_score = max(
        0,
        min(eis_score, 100)
    )

    return eis_score, cause_risk_score


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


def get_action_message(
    final_risk_level,
    road_closure
):
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
    base = 30 + predicted_incidents * 20

    if final_risk_level == "HIGH":
        base += 20

    elif final_risk_level == "CRITICAL":
        base += 40

    if road_closure:
        base += 30

    weather = str(weather or "clear").lower()

    if weather == "light_rain":
        base *= 1.10

    elif weather == "heavy_rain":
        base *= 1.25

    elif weather == "fog":
        base *= 1.20

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
        "water_logging"
    ]

    if event_cause in high_spread_causes:
        base_radius += 300

    return int(
        min(
            max(base_radius, 200),
            3000
        )
    )


def build_shift_plan(
    timestamp,
    duration_minutes,
    officers
):
    start_hour = timestamp.hour

    pre_event = max(
        1,
        officers // 3
    )

    peak = officers

    dispersal = max(
        1,
        officers // 2
    )

    return [
        {
            "phase": "Pre-event control",
            "time": f"{start_hour:02d}:00 - {(start_hour + 1) % 24:02d}:00",
            "officers": pre_event,
        },
        {
            "phase": "Peak impact control",
            "time": f"{(start_hour + 1) % 24:02d}:00 - {(start_hour + 3) % 24:02d}:00",
            "officers": peak,
        },
        {
            "phase": "Dispersal monitoring",
            "time": f"{(start_hour + 3) % 24:02d}:00 - {(start_hour + 4) % 24:02d}:00",
            "officers": dispersal,
        },
    ]


def predict_event_impact(
    payload
):
    model = load_model()
    store = load_feature_store()

    timestamp = normalize_timestamp(
        payload.get("timestamp")
    )

    hour = timestamp.hour
    weekday = timestamp.weekday()
    month = timestamp.month

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

    corridor = resolve_corridor_name(
        location_match["corridor"],
        store
    )

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

    police_station = payload.get(
        "police_station",
        "Unknown"
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
        profile=profile
    )

    predicted_incidents, forecast_details = predict_single_forecast(
        model,
        X
    )

    predicted_incidents = max(
        predicted_incidents,
        0.0
    )

    alert_probability = None

    if (
        forecast_details.get("alert_probability") is not None
    ):
        alert_probability = float(
            forecast_details["alert_probability"][0]
        )

    context_multiplier = 1.0

    if corridor.strip().lower() in [
        "non-corridor",
        "non corridor",
        "unknown"
    ]:
        context_multiplier = 0.65

    try:
        forecast_score, forecast_level = calculate_forecast_risk_score(
            predicted_incidents=predicted_incidents,
            incident_p95=store.get(
                "incident_p95",
                1.0
            ),
            incident_p99=store.get(
                "incident_p99",
                None
            ),
            alert_probability=alert_probability,
            context_multiplier=context_multiplier
        )

    except TypeError:
        forecast_score, forecast_level = calculate_forecast_risk_score(
            predicted_incidents,
            store.get(
                "incident_p95",
                1.0
            )
        )

    rush_hour = (
        (7 <= hour <= 10)
        or
        (17 <= hour <= 21)
    )

    event_score, event_level = calculate_event_impact(
        event_cause=event_cause,
        veh_type=veh_type,
        road_closure=road_closure,
        rush_hour=rush_hour
    )

    priority_boost = {
        "low": 0,
        "medium": 4,
        "high": 8,
        "critical": 12,
    }.get(
        str(priority).strip().lower(),
        4
    )

    event_type_boost = {
        "planned": 2,
        "unplanned": 6,
    }.get(
        str(event_type).strip().lower(),
        4
    )

    event_score = min(
        event_score
        +
        priority_boost
        +
        event_type_boost,
        100
    )

    event_score, crowd_multiplier, weather_multiplier = apply_crowd_weather_adjustment(
        event_score=event_score,
        crowd_size=crowd_size,
        weather=weather
    )

    event_level = get_risk_level_from_score(
        event_score
    )

    eis_score, cause_risk_score = calculate_eis_score(
        forecast_score=forecast_score,
        event_score=event_score,
        profile=profile,
        store=store
    )

    eis_level = get_risk_level_from_score(
        eis_score
    )

    final_score, final_level = calculate_final_operational_risk(
        forecast_risk_score=forecast_score,
        event_impact_score=eis_score
    )

    officers, barricades = recommend_resources(
        final_risk_level=final_level,
        predicted_incidents=predicted_incidents,
        road_closure=road_closure
    )

    diversion = recommend_diversions(
        affected_corridor=corridor,
        final_risk_level=final_level,
        road_closure=road_closure
    )

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

    closure_probability = min(
        max(
            (
                forecast_score * 0.35
                +
                event_score * 0.65
            ) / 100,
            0
        ),
        1
    )

    normal_baseline = safe_float(
        profile.get("rolling_24"),
        0.0
    )

    if normal_baseline <= 0:
        normal_baseline = safe_float(
            profile.get("corridor_avg"),
            0.0
        )

    expected_delta = max(
        predicted_incidents - normal_baseline,
        0.0
    )

    shift_plan = build_shift_plan(
        timestamp=timestamp,
        duration_minutes=duration,
        officers=officers
    )

    deployment_order = (
        f"Traffic deployment order: {final_level} risk event at "
        f"lat {latitude:.6f}, lon {longitude:.6f}. "
        f"Inferred corridor: {corridor}. "
        f"Deploy {officers} officers and {barricades} barricades. "
        f"Primary detour: {diversion.get('primary_detour')}. "
        f"Secondary detour: {diversion.get('secondary_detour')}. "
        f"Action: {action}"
    )

    holdout_metrics = {}

    if isinstance(model, dict):
        holdout_metrics = model.get(
            "holdout_metrics",
            {}
        )

    return {
        "input": {
            "event_type": event_type,
            "event_cause": event_cause,
            "priority": priority,
            "corridor": corridor,
            "veh_type": veh_type,
            "police_station": police_station,
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
        },

        "event": {
            "event_score": event_score,
            "event_level": event_level,
            "rush_hour": rush_hour,
            "crowd_multiplier": crowd_multiplier,
            "weather_multiplier": weather_multiplier,
        },

        "eis": {
            "score": eis_score,
            "level": eis_level,
            "cause_risk_score": cause_risk_score,
        },

        "final": {
            "final_score": final_score,
            "final_level": final_level,
            "duration_minutes": duration,
            "closure_probability": closure_probability,
            "affected_radius_m": affected_radius,
            "secondary_radius_m": secondary_radius,
        },

        "baseline": {
            "normal_baseline": normal_baseline,
            "predicted_after_event": predicted_incidents,
            "expected_delta": expected_delta,
        },

        "history": {
            "profile_source": profile_source,
            "matched_hour": matched_hour,
            "lag_1": X.loc[0, "lag_1"],
            "lag_2": X.loc[0, "lag_2"],
            "lag_3": X.loc[0, "lag_3"],
            "lag_24": X.loc[0, "lag_24"],
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

        "shift_plan": shift_plan,

        "diversion": diversion,

        "metrics": holdout_metrics,

        "map": {
            "latitude": latitude,
            "longitude": longitude,
            "end_latitude": end_latitude,
            "end_longitude": end_longitude,
            "affected_radius_m": affected_radius,
            "secondary_radius_m": secondary_radius,
        },

        "deployment_order": deployment_order,

        "action": action,
    }