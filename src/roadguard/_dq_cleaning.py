"""Safe structural cleaning of raw RoadGuard datasets.

Exact duplicate observation rows are removed, conflicting duplicate keys are
rejected, and permitted weather missingness is forward-filled from the same
segment's strictly earlier non-missing values only. Outliers are preserved.
The final cleaned result runs the full cleaned validation. Latent segment
fields never cross the cleaned boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import pandas as pd

from roadguard._dq_rowchecks import validate_cleaned_dataset, validate_raw_dataset
from roadguard._dq_validation import (
    PUBLIC_SEGMENT_COLUMNS,
    WEATHER_COLUMNS,
    ValidationReport,
    rows_equal,
)
from roadguard.contracts import DatasetSpec
from roadguard.observations import OBSERVATION_COLUMNS

_OBS_INT_COLUMNS: Final[tuple[str, ...]] = (
    "traffic_volume",
    "road_age_days",
    "days_since_last_maintenance",
    "previous_repairs",
    "road_condition_score",
    "marking_condition_score",
    "guardrail_condition_score",
    "sign_condition_score",
    "accident_count_30d",
    "accident_count_365d",
)
_OBS_FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    "heavy_vehicle_ratio",
    "rainfall_mm",
    "temperature",
    "humidity",
)


@dataclass(frozen=True)
class CleaningResult:
    """Public cleaned frames plus the cleaned validation report."""

    segments: pd.DataFrame
    observations: pd.DataFrame
    targets: pd.DataFrame
    maintenance_events: pd.DataFrame
    report: ValidationReport


def clean_raw_dataset(
    segments: pd.DataFrame,
    observations: pd.DataFrame,
    targets: pd.DataFrame,
    maintenance_events: pd.DataFrame,
    spec: DatasetSpec,
) -> CleaningResult:
    """Structurally clean a raw dataset and return public frames.

    The raw input runs the complete raw validation (grid, observation-target
    alignment and target/event recomputation included); exact raw duplicates
    and permitted weather missingness remain warnings but never disable
    unrelated integrity checks. The cleaned result runs the complete cleaned
    validation.
    """
    public_segments = _project_public_segments(segments)
    raw_report = validate_raw_dataset(
        public_segments, observations, targets, maintenance_events, spec
    )
    if raw_report.error_count > 0:
        raise ValueError(
            "raw dataset validation failed with errors: "
            + ", ".join(sorted(raw_report.counts_by_code))
        )
    public_segments = _normalize_public_segments(public_segments)

    sorted_obs = observations.sort_values(["segment_id", "date"], kind="stable")
    by_key: dict[tuple[str, date], dict[str, Any]] = {}
    for _, row in sorted_obs.iterrows():
        key = (str(row["segment_id"]), pd.Timestamp(row["date"]).date())
        if key in by_key:
            if rows_equal(by_key[key], dict(row)):
                continue
            raise ValueError(f"conflicting observation rows share key {key}")
        by_key[key] = dict(row)

    cleaned = pd.DataFrame(list(by_key.values()), columns=list(OBSERVATION_COLUMNS))
    cleaned["date"] = pd.to_datetime(cleaned["date"])
    _forward_fill_weather(cleaned)
    for column in _OBS_INT_COLUMNS:
        cleaned[column] = cleaned[column].astype("int64")
    for column in _OBS_FLOAT_COLUMNS:
        cleaned[column] = cleaned[column].astype("float64")
    cleaned["segment_id"] = cleaned["segment_id"].astype(object)
    cleaned = cleaned.sort_values(["segment_id", "date"], kind="stable").reset_index(drop=True)

    cleaned_report = validate_cleaned_dataset(
        public_segments, cleaned, targets, maintenance_events, spec
    )
    if cleaned_report.error_count > 0:
        raise ValueError(
            "cleaned dataset validation failed with errors: "
            + ", ".join(sorted(cleaned_report.counts_by_code))
        )
    return CleaningResult(
        segments=public_segments,
        observations=cleaned,
        targets=targets.copy(),
        maintenance_events=maintenance_events.copy(),
        report=cleaned_report,
    )


def _project_public_segments(segments: pd.DataFrame) -> pd.DataFrame:
    """Project to the five public columns without coercing any values."""
    missing = [column for column in PUBLIC_SEGMENT_COLUMNS if column not in segments.columns]
    if missing:
        raise ValueError(f"segments frame is missing public columns: {missing}")
    for column in PUBLIC_SEGMENT_COLUMNS:
        if list(segments.columns).count(column) > 1:
            raise ValueError(f"segments frame has duplicate column label {column!r}")
    return segments[list(PUBLIC_SEGMENT_COLUMNS)].copy()


def _normalize_public_segments(segments: pd.DataFrame) -> pd.DataFrame:
    """Normalize construction_date to datetime64[ns] on a working copy."""
    normalized = segments.copy()
    try:
        converted = pd.to_datetime(normalized["construction_date"]).astype("datetime64[ns]")
    except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError) as exc:
        raise ValueError(f"construction_date outside datetime64[ns] range: {exc}") from exc
    normalized["construction_date"] = converted
    return normalized.sort_values("segment_id", kind="stable").reset_index(drop=True)


def _forward_fill_weather(observations: pd.DataFrame) -> None:
    for column in WEATHER_COLUMNS:
        last_by_segment: dict[str, float] = {}
        for _, row in observations.iterrows():
            segment_id = str(row["segment_id"])
            value = row[column]
            if pd.isna(value):
                if segment_id not in last_by_segment:
                    raise ValueError(
                        f"no earlier valid {column} value for segment {segment_id!r} "
                        f"at date {pd.Timestamp(row['date']).date()}"
                    )
                observations.loc[row.name, column] = last_by_segment[segment_id]
            else:
                last_by_segment[segment_id] = float(value)


__all__ = ["CleaningResult", "clean_raw_dataset"]
