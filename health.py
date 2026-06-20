import os
import sys
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime


# ============================================================
# PROJECT ROOT SETUP
# ============================================================

def find_project_root():
    """
    Finds the real project root safely whether this file is placed in:
    - gridv1/scripts/project_health_check.py
    - gridv1/project_health_check.py
    - or run from another folder
    """

    start_paths = [
        Path(__file__).resolve().parent,
        Path.cwd().resolve(),
    ]

    for start in start_paths:
        for path in [start] + list(start.parents):
            has_core_files = (
                (path / "config.py").exists()
                and (path / "manage.py").exists()
                and (path / "src").is_dir()
                and (path / "dashboard").is_dir()
            )

            if has_core_files:
                return path

    return Path.cwd().resolve()


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORT_PATH = PROJECT_ROOT / "health_report.json"

print(f"\nDetected project root: {PROJECT_ROOT}\n")


# ============================================================
# HEALTH CHECK UTILS
# ============================================================

RESULTS = []


def log(status, name, message="", details=None):
    row = {
        "status": status,
        "name": name,
        "message": message,
        "details": details,
    }

    RESULTS.append(row)

    icon = {
        "PASS": "[OK]",
        "WARN": "[WARN]",
        "FAIL": "[FAIL]",
        "SKIP": "[SKIP]",
    }.get(status, "[INFO]")

    print(f"{icon} {name}")

    if message:
        print(f"     {message}")


def check_file(path, required=True):
    full_path = PROJECT_ROOT / path

    if full_path.exists():
        log(
            "PASS",
            f"File exists: {full_path.relative_to(PROJECT_ROOT)}",
        )
        return True

    if required:
        log(
            "FAIL",
            f"Missing file: {full_path.relative_to(PROJECT_ROOT)}",
        )
    else:
        log(
            "WARN",
            f"Optional file missing: {full_path.relative_to(PROJECT_ROOT)}",
        )

    return False


def check_dir(path, required=True):
    full_path = PROJECT_ROOT / path

    if full_path.exists() and full_path.is_dir():
        log(
            "PASS",
            f"Directory exists: {full_path.relative_to(PROJECT_ROOT)}",
        )
        return True

    if required:
        log(
            "FAIL",
            f"Missing directory: {full_path.relative_to(PROJECT_ROOT)}",
        )
    else:
        log(
            "WARN",
            f"Optional directory missing: {full_path.relative_to(PROJECT_ROOT)}",
        )

    return False


def check_import(module_name, required=True):
    try:
        __import__(module_name)

        log(
            "PASS",
            f"Import: {module_name}",
        )

        return True

    except Exception as e:
        status = "FAIL" if required else "WARN"

        log(
            status,
            f"Import failed: {module_name}",
            str(e),
        )

        return False


def read_text(path):
    full_path = PROJECT_ROOT / path

    if not full_path.exists():
        return ""

    return full_path.read_text(
        encoding="utf-8",
        errors="ignore",
    )


def check_text_contains(path, patterns, required=True):
    text = read_text(path)

    if not text:
        log(
            "FAIL" if required else "WARN",
            f"Cannot read: {path}",
        )
        return False

    ok = True

    for pattern in patterns:
        if pattern not in text:
            ok = False

            log(
                "FAIL" if required else "WARN",
                f"Missing text in {path}",
                pattern,
            )

    if ok:
        log(
            "PASS",
            f"Template/static content check: {path}",
        )

    return ok


def check_csv_schema(path, required_cols, required=True):
    try:
        import pandas as pd

        full_path = PROJECT_ROOT / path

        if not full_path.exists():
            log(
                "FAIL" if required else "WARN",
                f"CSV missing: {path}",
            )
            return False

        df = pd.read_csv(
            full_path,
        )

        missing = [
            col
            for col in required_cols
            if col not in df.columns
        ]

        if missing:
            log(
                "FAIL" if required else "WARN",
                f"CSV schema issue: {path}",
                f"Missing columns: {missing}",
            )
            return False

        log(
            "PASS",
            f"CSV schema valid: {path}",
            f"Rows: {len(df)}",
        )

        return True

    except Exception as e:
        log(
            "FAIL" if required else "WARN",
            f"CSV check crashed: {path}",
            str(e),
        )
        return False


# ============================================================
# BASIC STRUCTURE CHECKS
# ============================================================

def check_project_structure():
    print("\n" + "=" * 70)
    print("1. PROJECT STRUCTURE CHECK")
    print("=" * 70)

    required_files = [
        "config.py",
        "manage.py",
        "train_all.py",
        "requirements.txt",
        "README.md",
        ".gitignore",
    ]

    required_dirs = [
        "src",
        "src/forecasting",
        "src/inference",
        "src/features",
        "src/scoring",
        "src/routing",
        "src/recommendation",
        "src/evaluation",
        "dashboard",
        "dashboard/templates/dashboard",
        "dashboard/static/dashboard",
        "traffic_web",
        "data",
    ]

    for path in required_files:
        check_file(path)

    for path in required_dirs:
        check_dir(path)


# ============================================================
# PYTHON IMPORT CHECKS
# ============================================================

def check_imports():
    print("\n" + "=" * 70)
    print("2. PYTHON IMPORT CHECK")
    print("=" * 70)

    modules = [
        "config",

        "src.features.event_calendar",

        "src.preprocessing.load_data",

        "src.forecasting.build_timeseries_dataset",
        "src.forecasting.build_spatial_timeseries_dataset",
        "src.forecasting.train_timeseries_model",
        "src.forecasting.train_spatial_timeseries_model",
        "src.forecasting.train_quantile_intervals",
        "src.forecasting.forecast_predictor",
        "src.forecasting.spatial_forecast_predictor",
        "src.forecasting.cross_validate_timeseries",
        "src.forecasting.forecast_feature_importance",

        "src.inference.feature_store",
        "src.inference.location_resolver",
        "src.inference.location_validity_guard",
        "src.inference.police_station_resolver",
        "src.inference.active_event_memory",
        "src.inference.similar_events",

        "src.scoring.event_impact",
        "src.scoring.risk_score",

        "src.routing.diversion_engine",
        "src.recommendation.resource_recommender",
        "src.evaluation.cluster_fallback_ablation",
        "src.evaluation.eis_weight_calibration",
    ]

    for module in modules:
        check_import(module)


# ============================================================
# DATA FILE CHECKS
# ============================================================

def check_data_files():
    print("\n" + "=" * 70)
    print("3. DATA FILE CHECK")
    print("=" * 70)

    check_csv_schema(
        "data/events_calendar.csv",
        [
            "event_name",
            "event_type",
            "start_datetime",
            "end_datetime",
            "latitude",
            "longitude",
            "corridor",
            "impact_radius_m",
            "crowd_size",
        ],
        required=False,
    )

    check_csv_schema(
        "data/police_stations.csv",
        [
            "police_station",
            "latitude",
            "longitude",
        ],
        required=False,
    )

    check_csv_schema(
        "data/restricted_zones.csv",
        [
            "zone_name",
            "zone_type",
            "latitude",
            "longitude",
            "radius_m",
        ],
        required=False,
    )


# ============================================================
# FRONTEND CHECKS
# ============================================================

def check_frontend_files():
    print("\n" + "=" * 70)
    print("4. FRONTEND TEMPLATE / CSS CHECK")
    print("=" * 70)

    check_file(
        "dashboard/templates/dashboard/index.html",
    )

    check_file(
        "dashboard/static/dashboard/style.css",
    )

    check_text_contains(
        "dashboard/templates/dashboard/index.html",
        [
            "{% csrf_token %}",
            "name=\"event_type\"",
            "name=\"event_cause\"",
            "name=\"priority\"",
            "name=\"latitude\"",
            "name=\"longitude\"",
            "name=\"event_datetime\"",
            "id=\"map\"",
            "heroMap",
            "active_event_memory",
            "location_match",
            "police_station_match",
            "result.history",
            "post_event_feedback",
            "copyDeploymentOrder",
        ],
    )

    check_text_contains(
        "dashboard/static/dashboard/style.css",
        [
            ".manual-time-grid",
            ".selected-time-preview",
            ".memory-list",
            ".danger-card",
            ".history-note",
        ],
        required=False,
    )


# ============================================================
# DJANGO CHECK
# ============================================================

def check_django_setup():
    print("\n" + "=" * 70)
    print("5. DJANGO SETUP CHECK")
    print("=" * 70)

    try:
        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE",
            "traffic_web.settings",
        )

        import django

        django.setup()

        log(
            "PASS",
            "Django setup successful",
        )

        from django.urls import reverse

        try:
            reverse("dashboard")

            log(
                "PASS",
                "Django URL exists: dashboard",
            )

        except Exception:
            log(
                "WARN",
                "Django URL may be missing: dashboard",
            )

        try:
            reverse("post_event_feedback")

            log(
                "PASS",
                "Django URL exists: post_event_feedback",
            )

        except Exception:
            log(
                "WARN",
                "Django URL may be missing: post_event_feedback",
            )

    except Exception as e:
        log(
            "FAIL",
            "Django setup failed",
            str(e),
        )


# ============================================================
# MODEL ARTIFACT CHECKS
# ============================================================

def check_model_artifacts():
    print("\n" + "=" * 70)
    print("6. MODEL ARTIFACT CHECK")
    print("=" * 70)

    model_files = [
        "models/timeseries_forecast_model.pkl",
        "models/timeseries_forecast.pkl",
        "models/spatial_timeseries_forecast_model.pkl",
        "models/traffic_feature_store.pkl",
    ]

    for path in model_files:
        check_file(
            path,
            required=False,
        )

    try:
        import joblib

        feature_store_path = (
            PROJECT_ROOT
            / "models"
            / "traffic_feature_store.pkl"
        )

        if not feature_store_path.exists():
            log(
                "WARN",
                "Feature store not found",
                "Run python train_all.py or python scripts/prepare_feature_store.py",
            )
            return

        store = joblib.load(
            feature_store_path,
        )

        if not isinstance(store, dict):
            log(
                "FAIL",
                "Feature store type invalid",
                str(type(store)),
            )
            return

        expected_keys = [
            "restricted_zones",
            "police_station_points",
            "spatial_cluster_model",
            "spatial_cluster_centers",
            "corridor_location_profiles",
        ]

        for key in expected_keys:
            if key in store:
                log(
                    "PASS",
                    f"Feature store key exists: {key}",
                )
            else:
                log(
                    "WARN",
                    f"Feature store key missing: {key}",
                )

    except Exception as e:
        log(
            "FAIL",
            "Feature store load crashed",
            str(e),
        )


# ============================================================
# LOGIC UNIT CHECKS
# ============================================================

def check_active_event_memory_logic():
    print("\n" + "=" * 70)
    print("7. ACTIVE EVENT MEMORY LOGIC CHECK")
    print("=" * 70)

    try:
        from src.inference.active_event_memory import (
            get_time_weight,
            get_distance_weight,
            calculate_event_severity,
        )

        tests = [
            (
                get_time_weight(2),
                1.00,
                "time weight 0-24h",
            ),
            (
                get_time_weight(36),
                0.45,
                "time weight 24-48h",
            ),
            (
                get_time_weight(100),
                0.15,
                "time weight 48-168h",
            ),
            (
                get_time_weight(200),
                0.00,
                "time weight older than 7d",
            ),
        ]

        for actual, expected, name in tests:
            if abs(actual - expected) < 1e-9:
                log(
                    "PASS",
                    name,
                )
            else:
                log(
                    "FAIL",
                    name,
                    f"Expected {expected}, got {actual}",
                )

        if get_distance_weight(500) >= get_distance_weight(4000):
            log(
                "PASS",
                "distance decay is decreasing",
            )
        else:
            log(
                "FAIL",
                "distance decay logic invalid",
            )

        severity = calculate_event_severity(
            event_cause="accident",
            priority="Critical",
            road_closure=True,
        )

        if severity >= 80:
            log(
                "PASS",
                "event severity scoring works",
            )
        else:
            log(
                "WARN",
                "event severity scoring seems low",
                str(severity),
            )

    except Exception as e:
        log(
            "FAIL",
            "Active event memory logic crashed",
            str(e),
        )


def check_location_guard_logic():
    print("\n" + "=" * 70)
    print("8. LOCATION VALIDITY GUARD CHECK")
    print("=" * 70)

    try:
        from src.inference.location_validity_guard import (
            build_restricted_zone_store,
            check_restricted_zone,
            check_road_corridor_proximity,
        )

        store = build_restricted_zone_store()

        if store.get("restricted_zones"):
            log(
                "PASS",
                "Restricted zones loaded",
                f"Count: {len(store['restricted_zones'])}",
            )
        else:
            log(
                "WARN",
                "No restricted zones loaded",
                "Create data/restricted_zones.csv",
            )

        ulsoor_lake_check = check_restricted_zone(
            12.9822,
            77.6192,
            store,
        )

        if store.get("restricted_zones"):
            if ulsoor_lake_check.get("is_restricted"):
                log(
                    "PASS",
                    "Lake/restricted-zone blocking works",
                )
            else:
                log(
                    "WARN",
                    "Lake test did not block",
                    "Check restricted_zones.csv radius/coordinates",
                )

        fake_location_match = {
            "distance_m": 5000,
            "confidence": "LOW",
        }

        road_check = check_road_corridor_proximity(
            fake_location_match,
        )

        if not road_check["is_valid"]:
            log(
                "PASS",
                "Far-from-road blocking works",
            )
        else:
            log(
                "FAIL",
                "Far-from-road blocking failed",
            )

    except Exception as e:
        log(
            "FAIL",
            "Location guard logic crashed",
            str(e),
        )


def check_police_station_resolver_logic():
    print("\n" + "=" * 70)
    print("9. POLICE STATION RESOLVER CHECK")
    print("=" * 70)

    try:
        from src.inference.police_station_resolver import (
            load_official_police_station_points,
            resolve_nearest_police_station,
        )

        points = load_official_police_station_points()

        if points:
            log(
                "PASS",
                "Police station CSV loaded",
                f"Count: {len(points)}",
            )

            store = {
                "police_station_points": points,
            }

            result = resolve_nearest_police_station(
                12.9716,
                77.5946,
                store,
            )

            if result.get("police_station") != "Unknown":
                log(
                    "PASS",
                    "Nearest police station resolver works",
                    result.get("police_station"),
                )
            else:
                log(
                    "WARN",
                    "Police station resolver returned Unknown",
                )

        else:
            log(
                "WARN",
                "No official police station points loaded",
                "Create data/police_stations.csv",
            )

    except Exception as e:
        log(
            "FAIL",
            "Police station resolver crashed",
            str(e),
        )


def check_event_calendar_logic():
    print("\n" + "=" * 70)
    print("10. EVENT CALENDAR FEATURE CHECK")
    print("=" * 70)

    try:
        from src.features.event_calendar import (
            load_event_calendar,
            calculate_event_intensity,
        )

        calendar = load_event_calendar()

        if len(calendar) > 0:
            log(
                "PASS",
                "Event calendar loaded",
                f"Rows: {len(calendar)}",
            )
        else:
            log(
                "WARN",
                "Event calendar empty or missing",
                "Create data/events_calendar.csv",
            )

        intensity = calculate_event_intensity(
            event_type="protest",
            crowd_size="large",
        )

        if intensity > 50:
            log(
                "PASS",
                "Event calendar intensity scoring works",
            )
        else:
            log(
                "WARN",
                "Event intensity seems low",
                str(intensity),
            )

    except Exception as e:
        log(
            "FAIL",
            "Event calendar logic crashed",
            str(e),
        )


# ============================================================
# PREDICTION SMOKE TEST
# ============================================================

def backup_active_memory():
    path = (
        PROJECT_ROOT
        / "data"
        / "active_event_memory.csv"
    )

    backup = (
        PROJECT_ROOT
        / "data"
        / "active_event_memory.healthcheck.bak"
    )

    if path.exists():
        shutil.copy(
            path,
            backup,
        )

        return path, backup

    return path, None


def restore_active_memory(path, backup):
    try:
        if backup and backup.exists():
            shutil.copy(
                backup,
                path,
            )

            backup.unlink()

        elif path.exists():
            path.unlink()

    except Exception:
        pass


def check_prediction_smoke():
    print("\n" + "=" * 70)
    print("11. END-TO-END PREDICTION SMOKE TEST")
    print("=" * 70)

    required_artifacts = [
        PROJECT_ROOT / "models" / "traffic_feature_store.pkl",
        PROJECT_ROOT / "models" / "timeseries_forecast_model.pkl",
    ]

    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in required_artifacts
        if not path.exists()
    ]

    if missing:
        log(
            "SKIP",
            "Prediction smoke test skipped",
            f"Missing artifacts: {missing}. Run python train_all.py",
        )
        return

    memory_path, memory_backup = backup_active_memory()

    try:
        os.environ.setdefault(
            "DJANGO_SETTINGS_MODULE",
            "traffic_web.settings",
        )

        import django

        django.setup()

        from dashboard.services.ml_engine import predict_event_impact

        payload = {
            "event_type": "unplanned",
            "event_cause": "accident",
            "priority": "High",
            "latitude": "12.9716",
            "longitude": "77.5946",
            "end_latitude": "",
            "end_longitude": "",
            "corridor": "Non-corridor",
            "veh_type": "heavy_vehicle",
            "police_station": "",
            "event_datetime": datetime.now().strftime("%Y-%m-%dT%H:%M"),
            "requires_road_closure": "yes",
            "crowd_size": "unknown",
            "weather": "clear",
        }

        result = predict_event_impact(
            payload,
        )

        if not isinstance(result, dict):
            log(
                "FAIL",
                "Prediction output invalid",
                str(type(result)),
            )
            return

        if result.get("blocked"):
            log(
                "WARN",
                "Prediction returned blocked",
                result.get("message", "No message"),
            )
            return

        required_result_keys = [
            "final",
            "forecast",
            "event",
            "input",
            "resources",
            "diversion",
            "history",
        ]

        missing_keys = [
            key
            for key in required_result_keys
            if key not in result
        ]

        if missing_keys:
            log(
                "FAIL",
                "Prediction missing result keys",
                str(missing_keys),
            )
        else:
            log(
                "PASS",
                "Prediction smoke test successful",
            )

        final = result.get(
            "final",
            {},
        )

        log(
            "PASS",
            "Prediction final risk",
            f"{final.get('final_level')} / {final.get('final_score')}",
        )

    except Exception as e:
        log(
            "FAIL",
            "Prediction smoke test crashed",
            str(e),
        )

        print(
            traceback.format_exc(),
        )

    finally:
        restore_active_memory(
            memory_path,
            memory_backup,
        )


# ============================================================
# GITIGNORE CHECK
# ============================================================

def check_gitignore():
    print("\n" + "=" * 70)
    print("12. GITIGNORE CHECK")
    print("=" * 70)

    patterns = [
        ".venv/",
        "__pycache__/",
        "models/",
        "*.pkl",
        "*.joblib",
        "db.sqlite3",
        ".env",
        "data/post_event_feedback.csv",
        "data/active_event_memory.csv",
    ]

    text = read_text(
        ".gitignore",
    )

    if not text:
        log(
            "FAIL",
            ".gitignore missing or empty",
        )

        return

    for pattern in patterns:
        if pattern in text:
            log(
                "PASS",
                f".gitignore contains: {pattern}",
            )
        else:
            log(
                "WARN",
                f".gitignore missing: {pattern}",
            )


# ============================================================
# REPORT
# ============================================================

def write_report():
    counts = {
        "PASS": 0,
        "WARN": 0,
        "FAIL": 0,
        "SKIP": 0,
    }

    for row in RESULTS:
        counts[row["status"]] = counts.get(
            row["status"],
            0,
        ) + 1

    report = {
        "generated_at": datetime.now().isoformat(
            timespec="seconds",
        ),
        "project_root": str(PROJECT_ROOT),
        "summary": counts,
        "results": RESULTS,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FINAL HEALTH CHECK SUMMARY")
    print("=" * 70)

    print(f"PASS : {counts['PASS']}")
    print(f"WARN : {counts['WARN']}")
    print(f"FAIL : {counts['FAIL']}")
    print(f"SKIP : {counts['SKIP']}")
    print()
    print(f"Report saved: {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    if counts["FAIL"] > 0:
        print("\nStatus: FAILED — fix FAIL items first.")
        sys.exit(1)

    if counts["WARN"] > 0:
        print("\nStatus: PASSED WITH WARNINGS — acceptable for dev, review warnings.")

    else:
        print("\nStatus: CLEAN PASS")


def main():
    print("\n" + "=" * 70)
    print("GRIDLOCK IQ PROJECT HEALTH CHECK")
    print("=" * 70)

    check_project_structure()
    check_imports()
    check_data_files()
    check_frontend_files()
    check_django_setup()
    check_model_artifacts()
    check_active_event_memory_logic()
    check_location_guard_logic()
    check_police_station_resolver_logic()
    check_event_calendar_logic()
    check_prediction_smoke()
    check_gitignore()
    write_report()


if __name__ == "__main__":
    main()