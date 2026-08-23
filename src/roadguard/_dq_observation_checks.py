"""Observation-row validation checks (ranges, cross-fields, keys, cadence)."""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import pandas as pd

from roadguard._dq_validation import (
    RAINFALL_OUTLIER_MAX,
    TRAFFIC_VOLUME_OUTLIER_MAX,
    Severity,
    ValidationIssue,
    rows_equal,
)
from roadguard.contracts import DatasetSpec


def check_observation_rows(
    observations: pd.DataFrame,
    issues: list[ValidationIssue],
    segments: pd.DataFrame,
    spec: DatasetSpec,
    cleaned: bool,
) -> None:
    unique_dates = sorted(observations["date"].unique())
    if len(unique_dates) != spec.dataset_months_per_segment:
        issues.append(
            ValidationIssue(
                "error",
                "observation_date_count",
                "observations",
                "date",
                None,
                f"observations has {len(unique_dates)} unique dates, expected "
                f"{spec.dataset_months_per_segment}",
            )
        )
    for value in unique_dates:
        timestamp = pd.Timestamp(value)
        if timestamp.day != 1:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_cadence",
                    "observations",
                    "date",
                    str(timestamp.date()),
                    "observation dates must be first-of-month",
                )
            )
    for previous, current in zip(unique_dates, unique_dates[1:], strict=False):
        previous_month = previous.year * 12 + previous.month
        current_month = current.year * 12 + current.month
        if current_month != previous_month + 1:
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_cadence",
                    "observations",
                    "date",
                    None,
                    "observation dates must be consecutive months",
                )
            )

    construction_by_id: dict[str, date] = {}
    for _, row in segments.iterrows():
        construction_by_id[str(row["segment_id"])] = pd.Timestamp(row["construction_date"]).date()

    keys: dict[tuple[str, date], list[dict[str, Any]]] = {}
    for _, row in observations.iterrows():
        key = (str(row["segment_id"]), pd.Timestamp(row["date"]).date())
        keys.setdefault(key, []).append(dict(row))

    unique_keys = set(keys)
    if len(unique_keys) != spec.dataset_observations:
        issues.append(
            ValidationIssue(
                "error",
                "observation_key_count",
                "observations",
                None,
                None,
                f"observations has {len(unique_keys)} unique keys, expected "
                f"{spec.dataset_observations}",
            )
        )

    for key, rows in keys.items():
        if len(rows) > 1:
            if all(rows_equal(rows[0], row) for row in rows[1:]):
                severity: Severity = "error" if cleaned else "warning"
                for _ in rows[1:]:
                    issues.append(
                        ValidationIssue(
                            severity,
                            "duplicate_row" if severity == "warning" else "duplicate_key",
                            "observations",
                            None,
                            f"{key[0]}|{key[1]}",
                            f"exact duplicate observation row for key {key}",
                        )
                    )
            else:
                for _ in rows[1:]:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "conflicting_key",
                            "observations",
                            None,
                            f"{key[0]}|{key[1]}",
                            f"conflicting observation rows share key {key}",
                        )
                    )

    for _, row in observations.iterrows():
        segment_id = str(row["segment_id"])
        key = (segment_id, pd.Timestamp(row["date"]).date())
        row_key = f"{key[0]}|{key[1]}"
        construction = construction_by_id.get(segment_id)
        if construction is None:
            continue
        observation_date = key[1]
        if construction > observation_date:
            issues.append(
                ValidationIssue(
                    "error",
                    "construction_after_observation",
                    "observations",
                    "date",
                    row_key,
                    "construction_date is after the observation date",
                )
            )
        road_age = int(row["road_age_days"])
        expected_age = (observation_date - construction).days
        if road_age != expected_age:
            issues.append(
                ValidationIssue(
                    "error",
                    "road_age_mismatch",
                    "observations",
                    "road_age_days",
                    row_key,
                    f"road_age_days {road_age} != date - construction {expected_age}",
                )
            )
        dslm = int(row["days_since_last_maintenance"])
        if not (0 <= dslm <= road_age):
            issues.append(
                ValidationIssue(
                    "error",
                    "days_since_maintenance_range",
                    "observations",
                    "days_since_last_maintenance",
                    row_key,
                    f"days_since_last_maintenance {dslm} outside [0, {road_age}]",
                )
            )
        if int(row["previous_repairs"]) < 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "previous_repairs_negative",
                    "observations",
                    "previous_repairs",
                    row_key,
                    "previous_repairs is negative",
                )
            )
        traffic = int(row["traffic_volume"])
        if traffic < 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "traffic_volume_negative",
                    "observations",
                    "traffic_volume",
                    row_key,
                    "traffic_volume is negative",
                )
            )
        elif traffic > TRAFFIC_VOLUME_OUTLIER_MAX:
            issues.append(
                ValidationIssue(
                    "warning",
                    "operational_outlier",
                    "observations",
                    "traffic_volume",
                    row_key,
                    f"traffic_volume {traffic} exceeds operational threshold "
                    f"{TRAFFIC_VOLUME_OUTLIER_MAX}",
                )
            )
        heavy = float(row["heavy_vehicle_ratio"])
        if not math.isfinite(heavy):
            issues.append(
                ValidationIssue(
                    "error",
                    "non_finite",
                    "observations",
                    "heavy_vehicle_ratio",
                    row_key,
                    f"heavy_vehicle_ratio contains non-finite value {heavy!r}",
                )
            )
        elif not 0.0 <= heavy <= 1.0:
            issues.append(
                ValidationIssue(
                    "error",
                    "heavy_vehicle_ratio_range",
                    "observations",
                    "heavy_vehicle_ratio",
                    row_key,
                    f"heavy_vehicle_ratio {heavy} outside [0, 1]",
                )
            )
        rainfall = row["rainfall_mm"]
        if pd.isna(rainfall):
            issues.append(
                ValidationIssue(
                    "error" if cleaned else "warning",
                    "missing_value",
                    "observations",
                    "rainfall_mm",
                    row_key,
                    "missing rainfall_mm",
                )
            )
        else:
            rainfall = float(rainfall)
            if not math.isfinite(rainfall):
                issues.append(
                    ValidationIssue(
                        "error",
                        "non_finite",
                        "observations",
                        "rainfall_mm",
                        row_key,
                        f"rainfall_mm contains non-finite value {rainfall!r}",
                    )
                )
            elif rainfall < 0:
                issues.append(
                    ValidationIssue(
                        "error",
                        "rainfall_negative",
                        "observations",
                        "rainfall_mm",
                        row_key,
                        "rainfall_mm is negative",
                    )
                )
            elif rainfall > RAINFALL_OUTLIER_MAX:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "operational_outlier",
                        "observations",
                        "rainfall_mm",
                        row_key,
                        f"rainfall_mm {rainfall} exceeds operational threshold "
                        f"{RAINFALL_OUTLIER_MAX}",
                    )
                )
        for column, minimum, maximum in (
            ("temperature", -50.0, 60.0),
            ("humidity", 0.0, 100.0),
        ):
            value = row[column]
            if pd.isna(value):
                issues.append(
                    ValidationIssue(
                        "error" if cleaned else "warning",
                        "missing_value",
                        "observations",
                        column,
                        row_key,
                        f"missing {column}",
                    )
                )
            else:
                value = float(value)
                if not math.isfinite(value):
                    issues.append(
                        ValidationIssue(
                            "error",
                            "non_finite",
                            "observations",
                            column,
                            row_key,
                            f"{column} contains non-finite value {value!r}",
                        )
                    )
                elif not (minimum <= value <= maximum):
                    issues.append(
                        ValidationIssue(
                            "error",
                            f"{column}_range",
                            "observations",
                            column,
                            row_key,
                            f"{column} {value} outside [{minimum}, {maximum}]",
                        )
                    )
        for column in (
            "road_condition_score",
            "marking_condition_score",
            "guardrail_condition_score",
            "sign_condition_score",
        ):
            score = int(row[column])
            if not 1 <= score <= 100:
                issues.append(
                    ValidationIssue(
                        "error",
                        "condition_score_range",
                        "observations",
                        column,
                        row_key,
                        f"{column} {score} outside [1, 100]",
                    )
                )
        count_30d = int(row["accident_count_30d"])
        count_365d = int(row["accident_count_365d"])
        if count_30d < 0 or count_365d < 0:
            issues.append(
                ValidationIssue(
                    "error",
                    "accident_count_negative",
                    "observations",
                    "accident_count_30d",
                    row_key,
                    "accident count is negative",
                )
            )
        if count_30d > count_365d:
            issues.append(
                ValidationIssue(
                    "error",
                    "accident_count_inconsistent",
                    "observations",
                    "accident_count_30d",
                    row_key,
                    "accident_count_30d exceeds accident_count_365d",
                )
            )


__all__ = ["check_observation_rows"]
