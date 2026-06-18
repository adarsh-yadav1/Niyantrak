import os
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

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


MODEL_PATHS = [
    "models/timeseries_forecast_model.pkl",
    "models/timeseries_forecast.pkl",
]

FEATURE_STORE_PATH = "models/traffic_feature_store.pkl"


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


def parse_bool(
    value
):
    value = str(value).strip().lower()

    return value in [
        "yes",
        "y",
        "true",
        "1"
    ]


def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path)

    raise FileNotFoundError(
        "Forecast model not found. Run python train_all.py first."
    )


def load_feature_store():
    if not os.path.exists(FEATURE_STORE_PATH):
        raise FileNotFoundError(
            "Feature store not found. Run python prepare_feature_store.py first."
        )

    return joblib.load(
        FEATURE_STORE_PATH
    )


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


def get_risk_level(
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


def predict_traffic_risk():
    model = load_model()
    store = load_feature_store()

    print("\nCoordinate-first prediction mode")
    print("-" * 60)

    latitude = float(
        input("Latitude: ").strip()
    )

    longitude = float(
        input("Longitude: ").strip()
    )

    hour = int(
        input("Hour (0-23): ").strip()
    )

    weekday = int(
        input("Weekday (0=Mon, 6=Sun): ").strip()
    )

    month = int(
        input("Month (1-12): ").strip()
    )

    print("\nEvent Details")
    print("-" * 60)

    event_cause = input(
        "Event Cause: "
    ).strip()

    veh_type = input(
        "Vehicle Type: "
    ).strip()

    road_closure = parse_bool(
        input("Road Closure Required (yes/no): ")
    )

    crowd_size = input(
        "Crowd Size (small/medium/large/mega): "
    ).strip() or "small"

    weather = input(
        "Weather (clear/light_rain/heavy_rain/fog): "
    ).strip() or "clear"

    location_match = resolve_corridor_from_coordinates(
        latitude=latitude,
        longitude=longitude,
        store=store
    )

    corridor = resolve_corridor_name(
        location_match["corridor"],
        store
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

    if forecast_details.get("alert_probability") is not None:
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

    event_score, crowd_multiplier, weather_multiplier = apply_crowd_weather_adjustment(
        event_score=event_score,
        crowd_size=crowd_size,
        weather=weather
    )

    event_level = get_risk_level(
        event_score
    )

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

    eis_level = get_risk_level(
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

    action = get_action_message(
        final_risk_level=final_level,
        road_closure=road_closure
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

    print("\n")
    print("=" * 60)
    print("TRAFFIC INTELLIGENCE REPORT")
    print("=" * 60)

    print("\nInput Summary")
    print("-" * 60)

    print(f"Latitude              : {latitude}")
    print(f"Longitude             : {longitude}")
    print(f"Inferred Corridor     : {corridor}")
    print(f"Location Match        : {location_match['matched_by']}")
    print(f"Match Distance        : {location_match['distance_m']}")
    print(f"Match Confidence      : {location_match['confidence']}")
    print(f"Spatial Cluster       : {location_match['spatial_cluster_id']}")
    print(f"Nearest Hotspot Dist. : {location_match['nearest_hotspot_distance_m']}")
    print(f"Spatial Density       : {location_match['spatial_density_at_point']}")

    print(f"Hour                  : {hour}")
    print(f"Weekday               : {weekday}")
    print(f"Month                 : {month}")
    print(f"Event Cause           : {event_cause}")
    print(f"Vehicle Type          : {veh_type}")
    print(f"Road Closure Required : {road_closure}")
    print(f"Crowd Size            : {crowd_size}")
    print(f"Weather               : {weather}")

    print("\nForecast Layer")
    print("-" * 60)
    print(f"Predicted Incidents   : {predicted_incidents:.2f}")
    print(f"Forecast Risk Score   : {forecast_score:.2f}%")
    print(f"Forecast Risk Level   : {forecast_level}")

    if alert_probability is not None:
        print(f"Alert Probability     : {alert_probability:.2f}")

    print("\nEvent Impact Layer")
    print("-" * 60)
    print(f"Event Impact Score    : {event_score:.2f}%")
    print(f"Event Impact Level    : {event_level}")
    print(f"Crowd Multiplier      : {crowd_multiplier:.2f}")
    print(f"Weather Multiplier    : {weather_multiplier:.2f}")

    print("\nComposite Event Impact Score")
    print("-" * 60)
    print(f"EIS Score             : {eis_score:.2f}%")
    print(f"EIS Level             : {eis_level}")
    print(f"Cause Risk Component  : {cause_risk_score:.2f}%")

    print("\nPre-event vs Post-event")
    print("-" * 60)
    print(f"Normal Baseline       : {normal_baseline:.2f}")
    print(f"Predicted After Event : {predicted_incidents:.2f}")
    print(f"Expected Delta        : {expected_delta:.2f}")

    print("\nFinal Operational Decision")
    print("-" * 60)
    print(f"Final Risk Score      : {final_score:.2f}%")
    print(f"Final Risk Level      : {final_level}")

    print("\nHistorical Context Used")
    print("-" * 60)
    print(f"Feature Source        : {profile_source}")
    print(f"Matched Hour          : {matched_hour}")
    print(f"Lag 1 Hour            : {X.loc[0, 'lag_1']:.2f}")
    print(f"Lag 2 Hours           : {X.loc[0, 'lag_2']:.2f}")
    print(f"Lag 3 Hours           : {X.loc[0, 'lag_3']:.2f}")
    print(f"Lag 24 Hours          : {X.loc[0, 'lag_24']:.2f}")
    print(f"Rolling 6 Hours       : {X.loc[0, 'rolling_6']:.2f}")
    print(f"Rolling 24 Hours      : {X.loc[0, 'rolling_24']:.2f}")
    print(f"Rolling 168 Hours     : {X.loc[0, 'rolling_168']:.2f}")
    print(f"Corridor Avg          : {X.loc[0, 'corridor_avg']:.2f}")
    print(f"Corridor Volatility   : {X.loc[0, 'corridor_volatility']:.2f}")

    print("\nRecommended Deployment")
    print("-" * 60)
    print(f"Officers Needed       : {officers}")
    print(f"Barricades Needed     : {barricades}")

    print("\nRecommended Diversion")
    print("-" * 60)
    print(f"Primary Detour        : {diversion.get('primary_detour')}")
    print(f"Secondary Detour      : {diversion.get('secondary_detour')}")
    print(
        "Support Corridors     : "
        +
        ", ".join(diversion.get("support_corridors", []))
    )
    print(f"Diversion Action      : {diversion.get('message')}")

    print("\nRecommended Action")
    print("-" * 60)
    print(action)
    print("=" * 60)