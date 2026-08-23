"""Per-row and cross-field validation checks for RoadGuard datasets.

Called by the staged validation driver in ``_dq_validation``. Every check
appends deterministic ``ValidationIssue`` records and never lets raw pandas
exceptions escape.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime

import pandas as pd

from roadguard._dq_observation_checks import check_observation_rows
from roadguard._dq_validation import (
    PUBLIC_SEGMENT_COLUMNS,
    ValidationIssue,
    ValidationReport,
    extra_column_label,
)
from roadguard.contracts import DatasetSpec
from roadguard.events import EVENT_COLUMNS
from roadguard.observations import OBSERVATION_COLUMNS
from roadguard.segments import PROVINCES, ROAD_TYPES, SEGMENT_ID_PATTERN
from roadguard.targets import derive_observation_targets, maintenance_within_30_days


def validate_dataset(
    segments: pd.DataFrame,
    observations: pd.DataFrame,
    targets: pd.DataFrame,
    maintenance_events: pd.DataFrame,
    spec: DatasetSpec,
    cleaned: bool,
) -> ValidationReport:
    """Run staged validation and return a deterministic report."""
    issues: list[ValidationIssue] = []
    schema_ok = (
        check_schema(segments, "segments", PUBLIC_SEGMENT_COLUMNS, issues)
        and check_schema(observations, "observations", OBSERVATION_COLUMNS, issues)
        and check_schema(targets, "targets", TARGET_COLUMNS, issues)
        and check_schema(maintenance_events, "maintenance_events", EVENT_COLUMNS, issues)
    )
    if not schema_ok:
        return build_report(issues)
    check_table_dtypes(segments, observations, targets, maintenance_events, issues)
    check_construction_date_column(segments, issues)
    if any(issue.severity == "error" for issue in issues):
        return build_report(issues)

    segment_ids = collect_ids(segments, "segments", issues)
    observation_ids = collect_ids(observations, "observations", issues)
    target_ids = collect_ids(targets, "targets", issues)
    event_ids = collect_ids(maintenance_events, "maintenance_events", issues)

    check_ids(segments, "segments", issues)
    check_ids(observations, "observations", issues)
    check_ids(targets, "targets", issues)
    check_ids(maintenance_events, "maintenance_events", issues)
    check_dates(segments, "segments", "construction_date", issues)
    check_dates(observations, "observations", "date", issues)
    check_dates(targets, "targets", "date", issues)
    check_dates(maintenance_events, "maintenance_events", "maintenance_date", issues)
    if any(issue.severity == "error" for issue in issues):
        return build_report(issues)

    check_segment_rows(segments, issues, spec)
    check_observation_rows(observations, issues, segments, spec, cleaned)
    check_target_rows(targets, issues, observations, cleaned)
    check_event_rows(maintenance_events, segments, issues)
    check_foreign_keys(segment_ids, observation_ids, target_ids, event_ids, issues)
    check_target_event_consistency(observations, targets, maintenance_events, issues)
    return build_report(issues)


def _stable_scalar_description(value: object) -> str:
    """Deterministic label for arbitrary user-supplied scalars.

    Strings and numeric scalars render as themselves; every other value is
    described by its type name only, so unordered containers (sets,
    frozensets, dicts) never leak process-dependent representations.
    """
    if type(value) is str:
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return f"<{type(value).__name__}>"


def build_report(issues: list[ValidationIssue]) -> ValidationReport:
    ordered = sorted(
        issues,
        key=lambda issue: (
            issue.table,
            issue.code,
            _stable_scalar_description(issue.column),
            issue.row_key or "",
            issue.message,
        ),
    )
    return ValidationReport(tuple(ordered))


def check_schema(
    frame: pd.DataFrame, table: str, expected: tuple[str, ...], issues: list[ValidationIssue]
) -> bool:
    ok = True
    for column in expected:
        if not any(type(c) is str and c == column for c in frame.columns):
            issues.append(
                ValidationIssue(
                    "error",
                    "schema_missing_columns",
                    table,
                    column,
                    None,
                    f"{table} is missing required column {column!r}",
                )
            )
            ok = False
    for column in expected:
        if sum(1 for c in frame.columns if type(c) is str and c == column) > 1:
            issues.append(
                ValidationIssue(
                    "error",
                    "schema_duplicate_column",
                    table,
                    column,
                    None,
                    f"{table} has duplicate column label {column!r}",
                )
            )
            ok = False
    extras = [
        (position, column)
        for position, column in enumerate(frame.columns)
        if not (type(column) is str and column in expected)
    ]
    if extras:
        for position, column in extras:
            label = extra_column_label(position, column)
            issues.append(
                ValidationIssue(
                    "error",
                    "schema_extra_columns",
                    table,
                    label,
                    None,
                    f"{table} contains unexpected column {label!r}",
                )
            )
        ok = False
    return ok


def check_table_dtypes(
    segments: pd.DataFrame,
    observations: pd.DataFrame,
    targets: pd.DataFrame,
    maintenance_events: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    check_dtype(observations, "observations", "segment_id", "object", issues)
    check_dtype(observations, "observations", "date", "datetime64", issues)
    for column in _OBS_INT_COLUMNS:
        check_dtype(observations, "observations", column, "int64", issues)
    for column in _OBS_FLOAT_COLUMNS:
        check_dtype(observations, "observations", column, "float64", issues)
    check_dtype(segments, "segments", "segment_id", "object", issues)
    check_dtype(segments, "segments", "province", "object", issues)
    check_dtype(segments, "segments", "road_type", "object", issues)
    check_dtype(segments, "segments", "road_length_km", "float64", issues)
    check_dtype(targets, "targets", "segment_id", "object", issues)
    check_dtype(targets, "targets", "date", "datetime64", issues)
    for column in _TARGET_INT_COLUMNS:
        check_dtype(targets, "targets", column, "int64", issues)
    check_dtype(maintenance_events, "maintenance_events", "segment_id", "object", issues)
    check_dtype(maintenance_events, "maintenance_events", "maintenance_date", "datetime64", issues)


def check_dtype(
    frame: pd.DataFrame,
    table: str,
    column: str,
    expected: str,
    issues: list[ValidationIssue],
) -> None:
    dtype = str(frame[column].dtype)
    if expected == "object":
        ok = dtype == "object"
    elif expected == "datetime64":
        ok = dtype.startswith("datetime64")
    else:
        ok = dtype == expected
    if not ok:
        issues.append(
            ValidationIssue(
                "error",
                "invalid_dtype",
                table,
                column,
                None,
                f"{table} column {column} has dtype {dtype}, expected {expected}",
            )
        )


def check_construction_date_column(segments: pd.DataFrame, issues: list[ValidationIssue]) -> None:
    """Validate the construction_date boundary rule.

    Accepted forms: a datetime64 column, or an object column whose every
    value is a genuine ``date``/``datetime``/``Timestamp``. Strings and
    unsupported scalars produce contextual ``invalid_dtype`` issues.
    """
    dtype = str(segments["construction_date"].dtype)
    if dtype.startswith("datetime64"):
        return
    if dtype == "object":
        for index, value in enumerate(segments["construction_date"]):
            if not isinstance(value, (datetime, date)):
                issues.append(
                    ValidationIssue(
                        "error",
                        "invalid_dtype",
                        "segments",
                        "construction_date",
                        row_key_for(segments, index),
                        f"construction_date contains unsupported value of type "
                        f"{type(value).__name__}",
                    )
                )
        return
    issues.append(
        ValidationIssue(
            "error",
            "invalid_dtype",
            "segments",
            "construction_date",
            None,
            f"construction_date has incompatible dtype {dtype}; expected "
            "datetime64 or object of date values",
        )
    )


def collect_ids(frame: pd.DataFrame, table: str, issues: list[ValidationIssue]) -> set[str]:
    ids: set[str] = set()
    for value in frame["segment_id"]:
        if type(value) is str and re.fullmatch(SEGMENT_ID_PATTERN, value) is not None:
            ids.add(value)
    return ids


def _row_key_id_part(frame: pd.DataFrame, index: int, value: object) -> str:
    """Deterministic row-key id component.

    Valid string IDs may use their value; every other value is located by
    row position plus exact type, never by repr or equality.
    """
    if type(value) is str and re.fullmatch(SEGMENT_ID_PATTERN, value) is not None:
        return value
    return f"row[{index}]:{type(value).__name__}"


def check_ids(frame: pd.DataFrame, table: str, issues: list[ValidationIssue]) -> None:
    for index, value in enumerate(frame["segment_id"]):
        if type(value) is not str or re.fullmatch(SEGMENT_ID_PATTERN, value) is None:
            locator = _row_key_id_part(frame, index, value)
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_segment_id",
                    table,
                    "segment_id",
                    locator,
                    f"{table} contains malformed segment_id at {locator}",
                )
            )


def check_dates(
    frame: pd.DataFrame, table: str, column: str, issues: list[ValidationIssue]
) -> None:
    for index, value in enumerate(frame[column]):
        row_key = row_key_for(frame, index)
        try:
            timestamp = pd.Timestamp(value)
        except (ValueError, TypeError) as exc:
            issues.append(
                ValidationIssue("error", "invalid_date", table, column, row_key, str(exc))
            )
            continue
        if pd.isna(timestamp):
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_date",
                    table,
                    column,
                    row_key,
                    f"{table} column {column} contains a missing (NaT) date",
                )
            )
            continue
        if timestamp.tzinfo is not None:
            issues.append(
                ValidationIssue(
                    "error",
                    "timezone_aware_date",
                    table,
                    column,
                    row_key,
                    f"{table} column {column} must be timezone-naive",
                )
            )
            continue
        if (
            timestamp.hour
            or timestamp.minute
            or timestamp.second
            or timestamp.microsecond
            or timestamp.nanosecond
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "non_midnight_date",
                    table,
                    column,
                    row_key,
                    f"{table} column {column} contains non-midnight datetime {timestamp}",
                )
            )
            continue
        try:
            timestamp.as_unit("ns")
        except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError):
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_date",
                    table,
                    column,
                    row_key,
                    f"{table} column {column} contains {timestamp} outside datetime64[ns] range",
                )
            )


def row_key_for(frame: pd.DataFrame, index: int) -> str | None:
    try:
        segment_id = frame["segment_id"].iloc[index]
        if "date" in frame.columns:
            value = frame["date"].iloc[index]
        elif "maintenance_date" in frame.columns:
            value = frame["maintenance_date"].iloc[index]
        else:
            return _row_key_id_part(frame, index, segment_id)
        id_part = _row_key_id_part(frame, index, segment_id)
        try:
            return f"{id_part}|{pd.Timestamp(value).date()}"
        except (ValueError, TypeError):
            return f"{id_part}|<{type(value).__name__}>"
    except Exception:
        return None


def check_segment_rows(
    segments: pd.DataFrame, issues: list[ValidationIssue], spec: DatasetSpec
) -> None:
    if len(segments) != spec.dataset_segments:
        issues.append(
            ValidationIssue(
                "error",
                "segment_count",
                "segments",
                None,
                None,
                f"segments has {len(segments)} rows, expected {spec.dataset_segments}",
            )
        )
    for index, (_, row) in enumerate(segments.iterrows()):
        row_key = row_key_for(segments, index)
        province = row["province"]
        if type(province) is not str or province not in PROVINCES:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_category",
                    "segments",
                    "province",
                    row_key,
                    _category_message("province", province),
                )
            )
        road_type = row["road_type"]
        if type(road_type) is not str or road_type not in ROAD_TYPES:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_category",
                    "segments",
                    "road_type",
                    row_key,
                    _category_message("road_type", road_type),
                )
            )
        length = float(row["road_length_km"])
        if not math.isfinite(length) or length <= 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "road_length_range",
                    "segments",
                    "road_length_km",
                    row_key,
                    f"invalid road_length_km {length!r}",
                )
            )


def check_target_rows(
    targets: pd.DataFrame,
    issues: list[ValidationIssue],
    observations: pd.DataFrame,
    cleaned: bool,
) -> None:
    keys: dict[tuple[str, date], int] = {}
    for _, row in targets.iterrows():
        key = (str(row["segment_id"]), pd.Timestamp(row["date"]).date())
        keys[key] = keys.get(key, 0) + 1
    for key, count in keys.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_key",
                    "targets",
                    None,
                    f"{key[0]}|{key[1]}",
                    "duplicate target key",
                )
            )
    for _, row in targets.iterrows():
        key = (str(row["segment_id"]), pd.Timestamp(row["date"]).date())
        row_key = f"{key[0]}|{key[1]}"
        days = int(row["days_until_maintenance"])
        if days < 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "target_days_negative",
                    "targets",
                    "days_until_maintenance",
                    row_key,
                    "days_until_maintenance is negative",
                )
            )
        label = int(row["maintenance_within_30_days"])
        if label not in (0, 1):
            issues.append(
                ValidationIssue(
                    "error",
                    "target_label_invalid",
                    "targets",
                    "maintenance_within_30_days",
                    row_key,
                    "label is not 0 or 1",
                )
            )
        expected = int(maintenance_within_30_days(days))
        if label != expected:
            issues.append(
                ValidationIssue(
                    "error",
                    "target_label_mismatch",
                    "targets",
                    "maintenance_within_30_days",
                    row_key,
                    "label does not match the 30-day window contract",
                )
            )
    observation_keys = {
        (str(row["segment_id"]), pd.Timestamp(row["date"]).date())
        for _, row in observations.iterrows()
    }
    if set(keys) != observation_keys:
        issues.append(
            ValidationIssue(
                "error",
                "target_key_alignment",
                "targets",
                None,
                None,
                "target keys do not align exactly with observation keys",
            )
        )


def _category_message(column: str, value: object) -> str:
    """Build a deterministic category error message.

    Unsupported non-string values are described by their type name only, so
    unordered containers (sets, dicts) never leak process-dependent
    representations.
    """
    if type(value) is str:
        return f"unknown {column} {value!r}"
    return f"unknown {column}; unsupported value of type {type(value).__name__}"


def check_event_rows(
    maintenance_events: pd.DataFrame,
    segments: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    construction_by_id: dict[str, date] = {}
    for _, row in segments.iterrows():
        construction_by_id[str(row["segment_id"])] = pd.Timestamp(row["construction_date"]).date()
    keys: dict[tuple[str, date], int] = {}
    month_counts: dict[tuple[str, int, int], int] = {}
    for _, row in maintenance_events.iterrows():
        segment_id = str(row["segment_id"])
        event_date = pd.Timestamp(row["maintenance_date"]).date()
        key = (segment_id, event_date)
        keys[key] = keys.get(key, 0) + 1
        construction = construction_by_id.get(segment_id)
        if construction is not None and event_date < construction:
            issues.append(
                ValidationIssue(
                    "error",
                    "event_before_construction",
                    "maintenance_events",
                    "maintenance_date",
                    f"{segment_id}|{event_date}",
                    "maintenance event precedes the segment construction_date",
                )
            )
        month_key = (segment_id, event_date.year, event_date.month)
        month_counts[month_key] = month_counts.get(month_key, 0) + 1
    for key, count in keys.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_key",
                    "maintenance_events",
                    None,
                    f"{key[0]}|{key[1]}",
                    "duplicate maintenance event key",
                )
            )
    for month_key, count in month_counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    "error",
                    "event_month_conflict",
                    "maintenance_events",
                    "maintenance_date",
                    f"{month_key[0]}|{month_key[1]:04d}-{month_key[2]:02d}",
                    "more than one maintenance event in the same segment-month",
                )
            )


def check_foreign_keys(
    segment_ids: set[str],
    observation_ids: set[str],
    target_ids: set[str],
    event_ids: set[str],
    issues: list[ValidationIssue],
) -> None:
    for orphan in sorted(observation_ids - segment_ids):
        issues.append(
            ValidationIssue(
                "error",
                "foreign_key",
                "observations",
                "segment_id",
                orphan,
                "observation references unknown segment",
            )
        )
    for orphan in sorted(target_ids - segment_ids):
        issues.append(
            ValidationIssue(
                "error",
                "foreign_key",
                "targets",
                "segment_id",
                orphan,
                "target references unknown segment",
            )
        )
    for orphan in sorted(event_ids - segment_ids):
        issues.append(
            ValidationIssue(
                "error",
                "foreign_key",
                "maintenance_events",
                "segment_id",
                orphan,
                "maintenance event references unknown segment",
            )
        )


def check_target_event_consistency(
    observations: pd.DataFrame,
    targets: pd.DataFrame,
    maintenance_events: pd.DataFrame,
    issues: list[ValidationIssue],
) -> None:
    unique_observations = observations[["segment_id", "date"]].drop_duplicates()
    try:
        recomputed = derive_observation_targets(unique_observations, maintenance_events)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        issues.append(
            ValidationIssue(
                "error",
                "target_event_inconsistency",
                "targets",
                None,
                None,
                f"targets cannot be recomputed from events: {exc}",
            )
        )
        return
    expected = targets.sort_values(["segment_id", "date"]).reset_index(drop=True)
    recomputed = recomputed.sort_values(["segment_id", "date"]).reset_index(drop=True)
    if not expected.equals(recomputed):
        issues.append(
            ValidationIssue(
                "error",
                "target_event_inconsistency",
                "targets",
                None,
                None,
                "targets do not match targets recomputed from maintenance events",
            )
        )


_OBS_INT_COLUMNS: tuple[str, ...] = (
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
_OBS_FLOAT_COLUMNS: tuple[str, ...] = (
    "heavy_vehicle_ratio",
    "rainfall_mm",
    "temperature",
    "humidity",
)
_TARGET_INT_COLUMNS: tuple[str, ...] = (
    "days_until_maintenance",
    "maintenance_within_30_days",
)
TARGET_COLUMNS: tuple[str, ...] = (
    "segment_id",
    "date",
    "days_until_maintenance",
    "maintenance_within_30_days",
)


def validate_raw_dataset(
    segments: pd.DataFrame,
    observations: pd.DataFrame,
    targets: pd.DataFrame,
    maintenance_events: pd.DataFrame,
    spec: DatasetSpec,
) -> ValidationReport:
    """Validate raw data; permitted weather missingness and exact duplicates warn."""
    return validate_dataset(
        segments, observations, targets, maintenance_events, spec, cleaned=False
    )


def validate_cleaned_dataset(
    segments: pd.DataFrame,
    observations: pd.DataFrame,
    targets: pd.DataFrame,
    maintenance_events: pd.DataFrame,
    spec: DatasetSpec,
) -> ValidationReport:
    """Validate cleaned data; remaining missing values and duplicate keys are errors."""
    return validate_dataset(segments, observations, targets, maintenance_events, spec, cleaned=True)


__all__ = [
    "validate_cleaned_dataset",
    "validate_dataset",
    "validate_raw_dataset",
]
