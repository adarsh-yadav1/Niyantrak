import networkx as nx


BAD_DIVERSION_CORRIDORS = {
    "",
    "unknown",
    "nan",
    "none",
    "null",
    "non-corridor",
    "non corridor",
    "non_corridor",
    "outside_bengaluru",
    "out_of_bengaluru",
}


RISK_RANK = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "CRITICAL": 3,
    "UNKNOWN": 2,
}


def normalize_name(value):
    return (
        str(value or "")
        .strip()
    )


def is_bad_corridor(value):
    return (
        str(value or "")
        .strip()
        .lower()
        in BAD_DIVERSION_CORRIDORS
    )


def build_corridor_graph():

    graph = nx.Graph()

    edges = [

        # ORR / East side
        ("ORR East 1", "ORR East 2"),
        ("ORR East 1", "Old Airport Road"),
        ("ORR East 1", "Varthur Road"),
        ("ORR East 2", "Old Madras Road"),
        ("ORR East 2", "Hosur Road"),

        # ORR / North side
        ("ORR North 1", "ORR North 2"),
        ("ORR North 1", "Bellary Road 1"),
        ("ORR North 2", "Hennur Main Road"),
        ("ORR North 2", "IRR(Thanisandra road)"),

        # Bellary / Airport side
        ("Bellary Road 1", "Bellary Road 2"),
        ("Bellary Road 2", "Airport New South Road"),
        ("Bellary Road 1", "CBD 1"),
        ("Bellary Road 2", "ORR North 1"),

        # West side
        ("Tumkur Road", "West of Chord Road"),
        ("Tumkur Road", "ORR West 1"),
        ("West of Chord Road", "Magadi Road"),
        ("ORR West 1", "Mysore Road"),

        # South / Mysore side
        ("Mysore Road", "Magadi Road"),
        ("Mysore Road", "CBD 2"),
        ("Mysore Road", "Bannerghata Road"),

        # CBD
        ("CBD 1", "CBD 2"),
        ("CBD 1", "Old Airport Road"),
        ("CBD 2", "Hosur Road"),
        ("CBD 2", "Bannerghata Road"),

        # South / Hosur side
        ("Hosur Road", "Bannerghata Road"),
        ("Hosur Road", "ORR East 2"),
        ("Bannerghata Road", "ORR West 1"),

        # Old Madras / East
        ("Old Madras Road", "Varthur Road"),
        ("Old Madras Road", "ORR East 2"),
        ("Varthur Road", "Old Airport Road"),

        # fallback connectivity
        ("Non-corridor", "CBD 1"),
        ("Non-corridor", "CBD 2"),
        ("Non-corridor", "ORR East 1"),
        ("Non-corridor", "Mysore Road"),
    ]

    for u, v in edges:
        graph.add_edge(
            u,
            v,
            weight=1
        )

    return graph


def get_candidate_corridors(
    affected_corridor,
    max_depth=2
):
    """
    Returns direct and two-hop diversion candidates.

    Non-corridor is excluded from real diversion recommendations unless
    no graph match exists.
    """

    graph = build_corridor_graph()

    affected_corridor = normalize_name(
        affected_corridor
    )

    if affected_corridor not in graph.nodes:
        return []

    lengths = nx.single_source_shortest_path_length(
        graph,
        affected_corridor,
        cutoff=max_depth
    )

    candidates = []

    for corridor, hops in lengths.items():
        if corridor == affected_corridor:
            continue

        if is_bad_corridor(corridor):
            continue

        route_type = (
            "direct"
            if hops == 1
            else "secondary"
        )

        candidates.append({
            "corridor": corridor,
            "distance_hops": int(hops),
            "route_type": route_type,
        })

    candidates = sorted(
        candidates,
        key=lambda x: (
            x["distance_hops"],
            x["corridor"]
        )
    )

    return candidates


def get_state_value(
    corridor_states,
    corridor,
    key,
    default=None
):
    if not corridor_states:
        return default

    state = corridor_states.get(
        corridor,
        {}
    )

    return state.get(
        key,
        default
    )


def safe_float(value, fallback=0.0):
    try:
        if value is None:
            return fallback

        return float(value)

    except Exception:
        return fallback


def normalize_risk_level(value):
    value = (
        str(value or "UNKNOWN")
        .strip()
        .upper()
    )

    if value not in RISK_RANK:
        return "UNKNOWN"

    return value


def rank_diversion_candidates(
    candidates,
    corridor_states=None
):
    """
    Ranks candidates by predicted current risk.

    Lower score is better.

    Ranking logic:
    - LOW risk routes first
    - MODERATE next
    - HIGH/CRITICAL avoided unless no better route exists
    - shorter graph distance preferred
    - lower predicted incident count preferred
    """

    ranked = []

    state_aware = bool(
        corridor_states
    )

    for candidate in candidates:
        corridor = candidate["corridor"]

        risk_level = normalize_risk_level(
            get_state_value(
                corridor_states,
                corridor,
                "forecast_level",
                "UNKNOWN"
            )
        )

        forecast_score = safe_float(
            get_state_value(
                corridor_states,
                corridor,
                "forecast_score",
                50.0 if state_aware else 0.0
            )
        )

        predicted_incidents = safe_float(
            get_state_value(
                corridor_states,
                corridor,
                "predicted_incidents",
                0.0
            )
        )

        historical_load = safe_float(
            get_state_value(
                corridor_states,
                corridor,
                "historical_load",
                0.0
            )
        )

        ml_forecast_score = safe_float(
            get_state_value(
                corridor_states,
                corridor,
                "ml_forecast_score",
                forecast_score
            )
        )

        risk_rank = RISK_RANK.get(
            risk_level,
            2
        )

        distance_hops = int(
            candidate.get(
                "distance_hops",
                2
            )
        )

        ranking_score = (
            risk_rank * 100
            +
            forecast_score
            +
            predicted_incidents * 15
            +
            distance_hops * 8
        )

        is_safe = risk_level in [
            "LOW",
            "MODERATE"
        ]

        ranked.append({
            "corridor": corridor,
            "distance_hops": distance_hops,
            "route_type": candidate.get(
                "route_type",
                "secondary"
            ),
            "forecast_level": risk_level,
            "forecast_score": round(
                forecast_score,
                2
            ),
            "ml_forecast_score": round(
                ml_forecast_score,
                2
            ),
            "predicted_incidents": round(
                predicted_incidents,
                3
            ),
            "historical_load": round(
                historical_load,
                3
            ),
            "ranking_score": round(
                ranking_score,
                2
            ),
            "is_safe": is_safe,
            "profile_source": get_state_value(
                corridor_states,
                corridor,
                "profile_source",
                "not evaluated"
            ),
        })

    ranked = sorted(
        ranked,
        key=lambda x: (
            x["ranking_score"],
            x["distance_hops"],
            x["corridor"]
        )
    )

    return ranked


def recommend_diversions(
    affected_corridor,
    final_risk_level,
    road_closure=False,
    corridor_states=None
):

    graph = build_corridor_graph()

    affected_corridor = normalize_name(
        affected_corridor
    )

    if affected_corridor not in graph.nodes:

        return {
            "status": "NO_GRAPH_MATCH",
            "state_aware": False,
            "message": (
                "No corridor graph match found. Use manual local diversion planning."
            ),
            "primary_detour": "Manual diversion required",
            "secondary_detour": "Nearest available service road",
            "support_corridors": [],
            "candidate_evaluations": [],
        }

    candidates = get_candidate_corridors(
        affected_corridor,
        max_depth=2
    )

    if not candidates:

        return {
            "status": "NO_ALTERNATE",
            "state_aware": bool(corridor_states),
            "message": "No alternate corridor found.",
            "primary_detour": "Manual diversion required",
            "secondary_detour": "Nearest available service road",
            "support_corridors": [],
            "candidate_evaluations": [],
        }

    ranked = rank_diversion_candidates(
        candidates,
        corridor_states=corridor_states
    )

    safe_ranked = [
        item
        for item in ranked
        if item["is_safe"]
    ]

    if safe_ranked:
        selected = safe_ranked

    else:
        selected = ranked

    primary = selected[0]

    if len(selected) > 1:
        secondary = selected[1]

    elif len(ranked) > 1:
        secondary = ranked[1]

    else:
        secondary = {
            "corridor": "Local service road",
            "forecast_level": "UNKNOWN",
            "forecast_score": 0,
            "predicted_incidents": 0,
        }

    support_corridors = [
        item["corridor"]
        for item in ranked
        if item["corridor"] not in [
            primary["corridor"],
            secondary["corridor"]
        ]
    ][:5]

    state_aware = bool(
        corridor_states
    )

    if state_aware and not safe_ranked:
        action = (
            "All candidate diversions show elevated predicted risk. "
            "Use the least-risk route with officer control and continuous monitoring."
        )

    elif final_risk_level in [
        "HIGH",
        "CRITICAL"
    ] or road_closure:
        action = (
            "Activate state-aware diversion support and place officers at approach junctions."
        )

    else:
        action = (
            "Keep lowest-risk diversion route on standby; no full diversion needed yet."
        )

    return {
        "status": "OK",
        "state_aware": state_aware,
        "message": action,

        "primary_detour": primary["corridor"],
        "secondary_detour": secondary["corridor"],
        "support_corridors": support_corridors,

        "primary_detour_risk": {
            "level": primary.get("forecast_level"),
            "score": primary.get("forecast_score"),
            "predicted_incidents": primary.get("predicted_incidents"),
        },

        "secondary_detour_risk": {
            "level": secondary.get("forecast_level"),
            "score": secondary.get("forecast_score"),
            "predicted_incidents": secondary.get("predicted_incidents"),
        },

        "candidate_evaluations": ranked[:8],
    }