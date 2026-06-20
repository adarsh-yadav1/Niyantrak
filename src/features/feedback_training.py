from pathlib import Path

import pandas as pd


DEFAULT_FEEDBACK_PATH = Path("data/post_event_feedback.csv")


TIME_COLUMNS = [
    "event_datetime",
    "start_datetime",
    "timestamp",
    "submitted_at",
    "created_at",
    "recorded_at",
    "feedback_timestamp",
]


REQUIRED_OUTPUT_COLUMNS = [
    "time_bucket",
    "corridor",
    "actual_incident_count",
    "feedback_rows",
]


def parse_datetime_robust(series):
    raw = series.copy()

    parsed = pd.to_datetime(
        raw,
        errors="coerce",
        utc=True
    )

    failed = parsed.isna() & raw.notna()

    if failed.any():
        second = pd.to_datetime(
            raw.loc[failed],
            format="mixed",
            errors="coerce",
            utc=True
        )

        parsed.loc[failed] = second

    return parsed.dt.tz_convert(None)


def find_time_column(df):
    for col in TIME_COLUMNS:
        if col in df.columns:
            return col

    return None


def load_post_event_feedback(
    feedback_path=DEFAULT_FEEDBACK_PATH
):
    path = Path(feedback_path)

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if len(df) == 0:
        return pd.DataFrame()

    time_col = find_time_column(df)

    if time_col is None:
        print(
            "\n[WARN] post_event_feedback.csv found, but no usable time column exists."
        )
        print(
            "Expected one of:",
            TIME_COLUMNS
        )
        return pd.DataFrame()

    if "corridor" not in df.columns:
        print(
            "\n[WARN] post_event_feedback.csv found, but corridor column missing."
        )
        return pd.DataFrame()

    if "actual_incident_count" not in df.columns:
        print(
            "\n[WARN] post_event_feedback.csv found, but actual_incident_count column missing."
        )
        return pd.DataFrame()

    df = df.copy()

    df["event_datetime_parsed"] = parse_datetime_robust(
        df[time_col]
    )

    df["corridor"] = (
        df["corridor"]
        .fillna("Non-corridor")
        .astype(str)
        .str.strip()
    )

    df["actual_incident_count"] = pd.to_numeric(
        df["actual_incident_count"],
        errors="coerce"
    ).fillna(0)

    df = df.dropna(
        subset=[
            "event_datetime_parsed",
            "corridor",
        ]
    ).copy()

    df["time_bucket"] = (
        df["event_datetime_parsed"]
        .dt.floor("h")
    )

    return df


def build_feedback_hourly_corrections(
    feedback_path=DEFAULT_FEEDBACK_PATH
):
    df = load_post_event_feedback(
        feedback_path
    )

    if df.empty:
        return pd.DataFrame(
            columns=REQUIRED_OUTPUT_COLUMNS
        )

    hourly = (
        df
        .groupby(
            [
                "time_bucket",
                "corridor"
            ],
            as_index=False
        )
        .agg(
            actual_incident_count=(
                "actual_incident_count",
                "max"
            ),
            feedback_rows=(
                "actual_incident_count",
                "size"
            )
        )
    )

    return hourly


def apply_feedback_to_timeseries_counts(
    ts_df,
    feedback_path=DEFAULT_FEEDBACK_PATH
):
    """
    Applies post-event feedback to the corridor-hour incident_count target.

    Rule:
    - If feedback exists for the same corridor-hour, use the larger value between:
        historical incident_count and actual_incident_count from feedback.

    Why max()?
    - It avoids double-counting when the raw event is already present.
    - It still allows officer feedback to correct undercounted event impact.
    """

    corrections = build_feedback_hourly_corrections(
        feedback_path
    )

    if corrections.empty:
        print("\nPost-event feedback training corrections: 0 rows")
        return ts_df

    ts_df = ts_df.copy()

    ts_df["time_bucket"] = pd.to_datetime(
        ts_df["time_bucket"],
        errors="coerce"
    )

    corrections["time_bucket"] = pd.to_datetime(
        corrections["time_bucket"],
        errors="coerce"
    )

    before_rows = len(ts_df)

    merged = ts_df.merge(
        corrections,
        on=[
            "time_bucket",
            "corridor"
        ],
        how="left"
    )

    has_feedback = (
        merged["actual_incident_count"]
        .notna()
    )

    corrected_count = int(
        has_feedback.sum()
    )

    merged.loc[
        has_feedback,
        "incident_count"
    ] = merged.loc[
        has_feedback,
        [
            "incident_count",
            "actual_incident_count"
        ]
    ].max(axis=1)

    merged = merged.drop(
        columns=[
            "actual_incident_count",
            "feedback_rows"
        ],
        errors="ignore"
    )

    print("\nPost-event Feedback Training Corrections")
    print("-" * 50)
    print(f"Feedback correction rows applied : {corrected_count}")
    print(f"Timeseries rows before           : {before_rows}")
    print(f"Timeseries rows after            : {len(merged)}")

    return merged