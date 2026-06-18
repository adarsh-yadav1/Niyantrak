import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from django.conf import settings

from src.forecasting.forecast_predictor import predict_single_forecast

from src.inference.location_resolver import (
    resolve_corridor_from_coordinates,
    get_profile_with_spatial_fallback,
    is_bad_corridor_name,
)

from src.inference.similar_events import find_similar_events

from src.scoring.event_impact import calculate_event_impact

from src.scoring.risk_score import (
    calculate_forecast_risk_score,
    calculate_final_operational_risk,
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
            row[feature] = safe_float(
                profile.get(feature),
                0.0
            )

    return pd.DataFrame(
        [row],
        columns=FEATURES
    )


def get_risk_level_from_score(score):
    score = float(score)

    if score < 25:
        return "LOW"

    if score < 50:
        return "MODERATE"

    if score < 75:
        return "HIGH"

    return "CRITICAL"


def get_crowd_multiplier(crowd_size):
    crowd_size = (
        str(crowd_size or "small")
        .strip()
        .lower()
    )

    return {
        "small": 1.00,
        "medium": 1.08,
        "large": 1.18,
        "mega": 1.30,
    }.get(
        crowd_size,
        1.00
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
    weather
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

    crowd_multiplier = get_crowd_multiplier(crowd_size)
    weather_multiplier = get_weather_multiplier(weather)

    score = (
        score
        *
        crowd_multiplier
        *
        weather_multiplier
    )

    score = clamp(score, 0, 100)

    return score, crowd_multiplier, weather_multiplier


def calculate_eis_score(
    forecast_score,
    adjusted_event_score,
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

    cause_risk_score = clamp(
        cause_risk_score,
        0,
        100
    )

    # More operationally stable:
    # event impact is dominant,
    # forecast adds historical signal,
    # cause risk adds historical context.
    eis_score = (
        0.30 * safe_float(forecast_score)
        +
        0.55 * safe_float(adjusted_event_score)
        +
        0.15 * cause_risk_score
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
    """
    Separate display index from pure ML count.

    The ML predicted_incidents stays honest.
    This value estimates operational pressure after the event.
    """

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
    alert_probability=None
):
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
        safe_float(predicted_incidents) - uncertainty_width
    )

    upper = max(
        safe_float(predicted_incidents),
        safe_float(predicted_incidents) + uncertainty_width
    )

    return {
        "expected": float(predicted_incidents),
        "lower": float(lower),
        "upper": float(upper),
        "method": "Estimated uncertainty using holdout RMSE",
        "label": "Estimated uncertainty (±1 RMSE)",
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


def predict_event_impact(payload):
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

    if location_match.get("outside_bengaluru"):
        raise ValueError(
            "Selected location is outside Bengaluru coverage area. "
            "Please choose a location within Bengaluru."
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

    adjusted_event_score, crowd_multiplier, weather_multiplier = calculate_adjusted_event_score(
        base_event_score=base_event_score,
        priority=priority,
        event_type=event_type,
        crowd_size=crowd_size,
        weather=weather
    )

    adjusted_event_level = get_risk_level_from_score(
        adjusted_event_score
    )

    eis_score, cause_risk_score = calculate_eis_score(
        forecast_score=forecast_score,
        adjusted_event_score=adjusted_event_score,
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

    prediction_interval = build_prediction_interval(
        predicted_incidents=predicted_incidents,
        model_metrics=model_metrics,
        alert_probability=alert_probability
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
            "base_event_score": base_event_score,
            "base_event_level": base_event_level,

            "event_score": adjusted_event_score,
            "event_level": adjusted_event_level,

            "rush_hour": rush_hour,
            "crowd_multiplier": crowd_multiplier,
            "weather_multiplier": weather_multiplier,
        },

        "eis": {
            "score": eis_score,
            "level": eis_level,
            "cause_risk_score": cause_risk_score,
        },

        "baseline": {
            "normal_baseline": normal_baseline,

            # This is operational pressure after event, not pure ML count.
            "predicted_after_event": expected_after_event,

            # Pure model count is kept separately for honesty.
            "ml_predicted_incidents": predicted_incidents,

            "expected_delta": expected_delta,
            "percentage_increase": percentage_increase,

            # Correct risk comparison.
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