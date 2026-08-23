"""Pure target-semantics helpers and event-derived supervised targets.

Observations are start-of-day snapshots. ``next_maintenance_date`` is the
first maintenance event on or after the observation date, so
``days_until_maintenance`` is never negative.

``derive_observation_targets`` derives supervised labels from the actual
Phase 2 maintenance-event keys: for every observation snapshot at ``t`` the
earliest event for the same segment with ``maintenance_date >= t`` is the
next maintenance, ``days_until_maintenance = next - t``, and
``maintenance_within_30_days`` is 1 when ``0 <= days <= 30``. Targets are
never generated independently, never randomized (no seed, no RNG), and are
returned in a frame physically separate from observation features. Missing
tail events are errors, never silently right-censored.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Final

import numpy as np
import pandas as pd

from roadguard.contracts import V1_MAINTENANCE_WINDOW_DAYS
from roadguard.segments import SEGMENT_ID_PATTERN

TARGET_COLUMNS: Final[tuple[str, ...]] = (
    "segment_id",
    "date",
    "days_until_maintenance",
    "maintenance_within_30_days",
)

_EPOCH: Final[date] = date(1970, 1, 1)

__all__ = [
    "TARGET_COLUMNS",
    "days_until_maintenance",
    "derive_observation_targets",
    "maintenance_within_30_days",
]


def days_until_maintenance(observation_date: date, next_maintenance_date: date) -> int:
    """Return calendar days from the observation snapshot to the next event.

    ``next_maintenance_date`` must be on or after ``observation_date``;
    a past event is a contract violation and raises ``ValueError``.
    """
    if next_maintenance_date < observation_date:
        raise ValueError("next_maintenance_date must be on or after observation_date")
    return (next_maintenance_date - observation_date).days


def maintenance_within_30_days(days: int) -> bool:
    """Return True when ``days`` falls inside the 30-day maintenance window.

    The window is inclusive on both ends: 0 and 30 are positive, 31 is not.
    """
    return 0 <= days <= V1_MAINTENANCE_WINDOW_DAYS


def derive_observation_targets(
    observations: pd.DataFrame,
    maintenance_events: pd.DataFrame,
) -> pd.DataFrame:
    """Derive supervised targets from actual maintenance-event keys.

    Returns exactly :data:`TARGET_COLUMNS`, one row per observation row,
    sorted by ``segment_id`` and ``date``. Only ``segment_id`` and
    ``maintenance_date`` from the event frame are used; extra event columns
    (cost, materials) are ignored. Inputs are validated at the boundary and
    never mutated. Any observation without a next maintenance event raises a
    contextual ``ValueError`` naming the segment and observation date.
    """
    _validate_required_columns(observations, ("segment_id", "date"), "observations")
    _validate_required_columns(
        maintenance_events,
        ("segment_id", "maintenance_date"),
        "maintenance_events",
    )
    _validate_segment_ids(observations, "observations")
    _validate_segment_ids(maintenance_events, "maintenance_events")
    _validate_date_column(observations, "date", "observations")
    _validate_date_column(maintenance_events, "maintenance_date", "maintenance_events")

    event_days_by_segment: dict[str, list[int]] = {}
    event_segments: set[str] = set()
    seen_event_keys: set[tuple[str, date]] = set()
    for _, row in maintenance_events.iterrows():
        segment_id = row["segment_id"]
        event_date = pd.Timestamp(row["maintenance_date"]).date()
        key = (segment_id, event_date)
        if key in seen_event_keys:
            raise ValueError(f"duplicate maintenance event key {key}")
        seen_event_keys.add(key)
        event_days_by_segment.setdefault(segment_id, []).append(_day_int(event_date))
        event_segments.add(segment_id)

    observation_rows: list[tuple[str, date, int]] = []
    seen_observation_keys: set[tuple[str, date]] = set()
    for _, row in observations.iterrows():
        segment_id = row["segment_id"]
        observation_date = pd.Timestamp(row["date"]).date()
        key = (segment_id, observation_date)
        if key in seen_observation_keys:
            raise ValueError(f"duplicate observation key {key}")
        seen_observation_keys.add(key)
        observation_rows.append((segment_id, observation_date, _day_int(observation_date)))

    observation_segments = {segment_id for segment_id, _, _ in observation_rows}
    unknown_events = event_segments - observation_segments
    if unknown_events:
        raise ValueError(
            "maintenance events reference segments absent from observations: "
            f"{sorted(unknown_events)}"
        )

    event_arrays = {
        segment_id: np.array(sorted(days), dtype=np.int64)
        for segment_id, days in event_days_by_segment.items()
    }

    records: list[dict[str, Any]] = []
    for segment_id, observation_date, t_day in observation_rows:
        event_days = event_arrays.get(segment_id)
        if event_days is None or len(event_days) == 0:
            raise ValueError(
                f"observation for segment {segment_id!r} at date "
                f"{observation_date} has no next maintenance event"
            )
        index = int(np.searchsorted(event_days, t_day, side="left"))
        if index == len(event_days):
            raise ValueError(
                f"observation for segment {segment_id!r} at date "
                f"{observation_date} has no next maintenance event"
            )
        next_date = _EPOCH + timedelta(days=int(event_days[index]))
        days = days_until_maintenance(observation_date, next_date)
        records.append(
            {
                "segment_id": segment_id,
                "date": observation_date,
                "days_until_maintenance": days,
                "maintenance_within_30_days": int(maintenance_within_30_days(days)),
            }
        )

    frame = pd.DataFrame(records, columns=list(TARGET_COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"])
    frame["segment_id"] = frame["segment_id"].astype(object)
    frame["days_until_maintenance"] = frame["days_until_maintenance"].astype("int64")
    frame["maintenance_within_30_days"] = frame["maintenance_within_30_days"].astype("int64")
    return frame.sort_values(["segment_id", "date"], kind="stable").reset_index(drop=True)


def _validate_required_columns(frame: pd.DataFrame, required: tuple[str, ...], label: str) -> None:
    for column in required:
        if column not in frame.columns:
            raise ValueError(f"{label} must contain columns: {list(required)}")
    for column in required:
        if list(frame.columns).count(column) > 1:
            raise ValueError(f"{label} has duplicate column label {column!r}")


def _validate_segment_ids(frame: pd.DataFrame, label: str) -> None:
    for value in frame["segment_id"]:
        if not isinstance(value, str) or re.fullmatch(SEGMENT_ID_PATTERN, value) is None:
            raise ValueError(f"{label} contains malformed segment_id {value!r}")


def _validate_date_column(frame: pd.DataFrame, column: str, label: str) -> None:
    series = frame[column]
    dtype = str(series.dtype)
    if dtype.startswith("datetime64"):
        if series.dt.tz is not None:
            raise ValueError(f"{label} column {column} must be timezone-naive")
        for value in series:
            if pd.isna(value):
                raise ValueError(f"{label} column {column} contains a missing (NaT) date")
            _validate_datetime_value(pd.Timestamp(value), label, column)
        return
    if series.dtype == object:
        for value in series:
            if not isinstance(value, (datetime, date)):
                raise ValueError(
                    f"{label} column {column} contains unsupported value of type "
                    f"{type(value).__name__}: {value!r}"
                )
            if pd.isna(value):
                raise ValueError(f"{label} column {column} contains a missing (NaT) date")
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is not None:
                raise ValueError(f"{label} column {column} must be timezone-naive")
            _validate_datetime_value(timestamp, label, column)
        return
    raise ValueError(
        f"{label} column {column} has incompatible dtype {dtype}; "
        "dates must be datetime64 or object values"
    )


def _validate_datetime_value(timestamp: pd.Timestamp, label: str, column: str) -> None:
    _require_midnight(timestamp, label, column)
    _require_ns_representable(timestamp, label, column)


def _require_ns_representable(timestamp: pd.Timestamp, label: str, column: str) -> None:
    try:
        timestamp.as_unit("ns")
    except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError) as exc:
        raise ValueError(
            f"{label} column {column} contains date {timestamp} outside datetime64[ns] range"
        ) from exc


def _require_midnight(timestamp: pd.Timestamp, label: str, column: str) -> None:
    if (
        timestamp.hour != 0
        or timestamp.minute != 0
        or timestamp.second != 0
        or timestamp.microsecond != 0
        or timestamp.nanosecond != 0
    ):
        raise ValueError(
            f"{label} column {column} contains non-midnight datetime {timestamp}; "
            "only start-of-day (midnight) values are allowed"
        )


def _day_int(value: date) -> int:
    return (value - _EPOCH).days
