from pathlib import Path
from datetime import datetime
from datetime import timedelta
import uuid

import numpy as np
import pandas as pd

from src.inference.location_resolver import (
    haversine_distance_meters,
)

try:
    from src.routing.diversion_engine import build_corridor_graph
except Exception:
    build_corridor_graph = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTIVE_EVENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "active_event_memory.csv"
)


ACTIVE_EVENT_COLUMNS = [
    "event_id",
    "recorded_at",
    "event_datetime",

    "latitude",
    "longitude",
    "corridor",
    "spatial_cluster_id",

    "event_type",
    "event_cause",
    "priority",
    "road_closure",

    "event_score",
    "final_score",
    "severity_score",

    "source",
]


CAUSE_SEVERITY = {
    "accident": 85,
    "vehicle_breakdown": 55,
    "congestion": 70,
    "public_event": 70,
    "protest": 85,
    "procession": 80,
    "vip_movement": 90,
    "festival": 75,
    "sports": 75,
    "construction": 60,
    "water_logging": 78,
    "tree_fall": 65,
    "road_conditions": 55,
    "pot_holes": 45,
    "others": 45,
}


PRIORITY_BOOST = {
    "low": 0,
    "medium": 8,
    "high": 15,
    "critical": 22,
}


def safe_float(value, fallback=0.0):
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


def safe_bool(value):
    value = (
        str(value or "")
        .strip()
        .lower()
    )

    return value in [
        "true",
        "yes",
        "1",
        "y",
        "required",
    ]


def normalize_text(value, fallback="unknown"):
    value = (
        str(value or fallback)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if value in [
        "",
        "nan",
        "none",
        "null",
    ]:
        return fallback

    return value


def parse_datetime_value(value):
    if value is None:
        return datetime.now()

    value = str(value).strip()

    if value == "":
        return datetime.now()

    parsed = pd.to_datetime(
        value,
        errors="coerce"
    )

    if pd.isna(parsed):
        return datetime.now()

    if getattr(parsed, "tzinfo", None) is not None:
        try:
            parsed = parsed.tz_convert(None)
        except Exception:
            parsed = parsed.tz_localize(None)

    return parsed.to_pydatetime()


def get_risk_level(score):
    score = safe_float(
        score,
        0.0
    )

    if score >= 75:
        return "CRITICAL"

    if score >= 50:
        return "HIGH"

    if score >= 25:
        return "MODERATE"

    return "LOW"


def calculate_event_severity(
    event_cause,
    priority,
    road_closure,
    event_score=None,
    final_score=None
):
    event_cause = normalize_text(
        event_cause,
        "others"
    )

    priority = normalize_text(
        priority,
        "medium"
    )

    base = CAUSE_SEVERITY.get(
        event_cause,
        45
    )

    base += PRIORITY_BOOST.get(
        priority,
        8
    )

    if safe_bool(road_closure):
        base += 18

    if event_score is not None:
        base = max(
            base,
            safe_float(event_score, 0.0)
        )

    if final_score is not None:
        base = max(
            base,
            safe_float(final_score, 0.0) * 0.80
        )

    return max(
        0.0,
        min(base, 100.0)
    )


def get_time_weight(age_hours):
    """
    User-proposed 3-phase recency logic.

    0-24 hours      : strong
    24-48 hours     : medium
    48-168 hours    : weak
    older than 7d   : ignored
    """

    if age_hours < 0:
        return 0.0

    if age_hours <= 24:
        return 1.00

    if age_hours <= 48:
        return 0.45

    if age_hours <= 168:
        return 0.15

    return 0.0


def get_distance_weight(distance_m):
    if distance_m <= 1000:
        return 1.00

    if distance_m <= 3000:
        return 0.60

    if distance_m <= 5000:
        return 0.30

    if distance_m <= 8000:
        return 0.12

    return 0.03


def get_corridor_relation_weight(
    source_corridor,
    target_corridor
):
    source_corridor = str(
        source_corridor or ""
    ).strip()

    target_corridor = str(
        target_corridor or ""
    ).strip()

    if not source_corridor or not target_corridor:
        return 0.30

    if source_corridor == target_corridor:
        return 1.00

    if build_corridor_graph is None:
        return 0.30

    try:
        graph = build_corridor_graph()

        if source_corridor not in graph.nodes or target_corridor not in graph.nodes:
            return 0.30

        shortest_hops = nx_shortest_path_length_safe(
            graph,
            source_corridor,
            target_corridor
        )

        if shortest_hops == 1:
            return 0.65

        if shortest_hops == 2:
            return 0.45

        return 0.25

    except Exception:
        return 0.30


def nx_shortest_path_length_safe(
    graph,
    source,
    target
):
    try:
        import networkx as nx

        return nx.shortest_path_length(
            graph,
            source=source,
            target=target
        )

    except Exception:
        return 99


def ensure_active_event_file():
    ACTIVE_EVENT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not ACTIVE_EVENT_PATH.exists():
        pd.DataFrame(
            columns=ACTIVE_EVENT_COLUMNS
        ).to_csv(
            ACTIVE_EVENT_PATH,
            index=False
        )


def load_active_events():
    ensure_active_event_file()

    try:
        df = pd.read_csv(
            ACTIVE_EVENT_PATH
        )

    except Exception:
        df = pd.DataFrame(
            columns=ACTIVE_EVENT_COLUMNS
        )

    for col in ACTIVE_EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[
        ACTIVE_EVENT_COLUMNS
    ].copy()


def prune_active_events(
    reference_datetime=None,
    max_age_hours=168
):
    if reference_datetime is None:
        reference_datetime = datetime.now()

    df = load_active_events()

    if len(df) == 0:
        return df

    parsed_times = pd.to_datetime(
        df["event_datetime"],
        errors="coerce"
    )

    min_allowed_time = (
        reference_datetime
        -
        timedelta(hours=max_age_hours)
    )

    keep_mask = (
        parsed_times.notna()
        &
        (parsed_times >= min_allowed_time)
    )

    df = df.loc[
        keep_mask
    ].copy()

    df.to_csv(
        ACTIVE_EVENT_PATH,
        index=False
    )

    return df


def save_active_event(
    event_datetime,
    latitude,
    longitude,
    corridor,
    spatial_cluster_id,
    event_type,
    event_cause,
    priority,
    road_closure,
    event_score,
    final_score,
    source="dashboard_prediction"
):
    """
    Save the current reported event after prediction.

    This allows future predictions to react to it immediately
    without model retraining.
    """

    current_dt = parse_datetime_value(
        event_datetime
    )

    df = prune_active_events(
        reference_datetime=current_dt
    )

    severity_score = calculate_event_severity(
        event_cause=event_cause,
        priority=priority,
        road_closure=road_closure,
        event_score=event_score,
        final_score=final_score
    )

    row = {
        "event_id": str(uuid.uuid4()),
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "event_datetime": current_dt.isoformat(timespec="seconds"),

        "latitude": safe_float(latitude),
        "longitude": safe_float(longitude),
        "corridor": str(corridor or "Unknown"),
        "spatial_cluster_id": (
            "" if spatial_cluster_id is None else str(spatial_cluster_id)
        ),

        "event_type": str(event_type or "unknown"),
        "event_cause": str(event_cause or "others"),
        "priority": str(priority or "Medium"),
        "road_closure": bool(safe_bool(road_closure)),

        "event_score": safe_float(event_score),
        "final_score": safe_float(final_score),
        "severity_score": severity_score,

        "source": source,
    }

    df = pd.concat(
        [
            df,
            pd.DataFrame([row])
        ],
        ignore_index=True
    )

    df.to_csv(
        ACTIVE_EVENT_PATH,
        index=False
    )

    return row


def get_active_event_pressure(
    current_datetime,
    latitude,
    longitude,
    corridor,
    spatial_cluster_id=None,
    max_age_hours=168
):
    """
    Calculates how much recent reported events should influence
    the current prediction.

    This does not change the ML model output. It produces an
    operational pressure score that can boost final risk.
    """

    current_dt = parse_datetime_value(
        current_datetime
    )

    df = prune_active_events(
        reference_datetime=current_dt,
        max_age_hours=max_age_hours
    )

    if len(df) == 0:
        return {
            "pressure_score": 0.0,
            "pressure_level": "LOW",
            "recent_events_found": 0,
            "contributors": [],
            "explanation": "No active recent events found.",
        }

    lat = safe_float(
        latitude,
        None
    )

    lon = safe_float(
        longitude,
        None
    )

    if lat is None or lon is None:
        return {
            "pressure_score": 0.0,
            "pressure_level": "LOW",
            "recent_events_found": 0,
            "contributors": [],
            "explanation": "Invalid current coordinates.",
        }

    contributors = []

    for _, row in df.iterrows():
        event_dt = parse_datetime_value(
            row.get("event_datetime")
        )

        age_hours = (
            current_dt
            -
            event_dt
        ).total_seconds() / 3600.0

        time_weight = get_time_weight(
            age_hours
        )

        if time_weight <= 0:
            continue

        event_lat = safe_float(
            row.get("latitude"),
            None
        )

        event_lon = safe_float(
            row.get("longitude"),
            None
        )

        if event_lat is None or event_lon is None:
            continue

        distance_m = haversine_distance_meters(
            lat,
            lon,
            event_lat,
            event_lon
        )

        distance_weight = get_distance_weight(
            distance_m
        )

        source_corridor = row.get(
            "corridor",
            ""
        )

        corridor_weight = get_corridor_relation_weight(
            source_corridor=source_corridor,
            target_corridor=corridor
        )

        source_cluster = str(
            row.get("spatial_cluster_id", "")
        ).strip()

        target_cluster = (
            "" if spatial_cluster_id is None else str(spatial_cluster_id)
        ).strip()

        cluster_weight = 0.0

        if source_cluster and target_cluster and source_cluster == target_cluster:
            cluster_weight = 0.85

        relation_weight = max(
            corridor_weight,
            cluster_weight
        )

        severity_score = safe_float(
            row.get("severity_score"),
            0.0
        )

        closure_boost = (
            1.15
            if safe_bool(row.get("road_closure"))
            else 1.00
        )

        raw_pressure = (
            severity_score
            *
            time_weight
            *
            distance_weight
            *
            relation_weight
            *
            closure_boost
        )

        if raw_pressure <= 1:
            continue

        contributors.append({
            "event_id": row.get("event_id"),
            "event_cause": row.get("event_cause"),
            "corridor": source_corridor,
            "age_hours": round(age_hours, 2),
            "distance_m": round(distance_m, 2),
            "severity_score": round(severity_score, 2),
            "time_weight": round(time_weight, 2),
            "distance_weight": round(distance_weight, 2),
            "relation_weight": round(relation_weight, 2),
            "pressure": round(raw_pressure, 2),
        })

    if not contributors:
        return {
            "pressure_score": 0.0,
            "pressure_level": "LOW",
            "recent_events_found": 0,
            "contributors": [],
            "explanation": "Recent events exist, but none are close or relevant enough to affect this prediction.",
        }

    contributors = sorted(
        contributors,
        key=lambda x: x["pressure"],
        reverse=True
    )

    top_pressures = [
        item["pressure"]
        for item in contributors[:5]
    ]

    primary_pressure = top_pressures[0]
    secondary_pressure = sum(
        top_pressures[1:]
    ) * 0.35

    pressure_score = min(
        100.0,
        primary_pressure + secondary_pressure
    )

    pressure_level = get_risk_level(
        pressure_score
    )

    return {
        "pressure_score": round(pressure_score, 2),
        "pressure_level": pressure_level,
        "recent_events_found": len(contributors),
        "contributors": contributors[:5],
        "explanation": (
            "Recent-event pressure uses time decay, distance decay, "
            "corridor relation, spatial cluster relation, and event severity."
        ),
    }