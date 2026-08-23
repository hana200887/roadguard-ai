"""Deterministic raw-data corruption for observation features.

Builds a separate corrupted-raw representation: controlled missing values
(weather columns only, never the first observation of a segment),
domain-valid outliers (``traffic_volume``, ``rainfall_mm``) and exact
duplicate rows, with an exact manifest. Inputs are strictly validated before
any RNG use; nothing is silently cast.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from roadguard._dq_validation import extra_column_label
from roadguard.observations import OBSERVATION_COLUMNS
from roadguard.segments import SEGMENT_ID_PATTERN

WEATHER_COLUMNS: Final[tuple[str, ...]] = ("rainfall_mm", "temperature", "humidity")
OUTLIER_COLUMNS: Final[tuple[str, ...]] = ("traffic_volume", "rainfall_mm")

MISSING_RATE_DEFAULT: Final[float] = 0.02
OUTLIER_RATE_DEFAULT: Final[float] = 0.005
DUPLICATE_RATE_DEFAULT: Final[float] = 0.002
OUTLIER_MULTIPLIER_DEFAULT: Final[float] = 8.0
MAX_CORRUPTION_RATE: Final[float] = 0.25
CORRUPTION_RNG_NAMESPACE: Final[int] = 0x524735

_EPOCH: Final[date] = date(1970, 1, 1)
_PRE_EPOCH_BASE: Final[int] = 1 << 62
_INT64_MIN: Final[int] = np.iinfo(np.int64).min
_INT64_MAX: Final[int] = np.iinfo(np.int64).max
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

CorruptionKind = Literal["missing", "outlier", "duplicate"]


@dataclass(frozen=True)
class CorruptionEntry:
    """One corrupted cell or duplicated row (no full row dumps)."""

    kind: CorruptionKind
    segment_id: str
    date: date
    column: str | None = None


@dataclass(frozen=True)
class CorruptionManifest:
    """Deterministic record of every injected corruption."""

    entries: tuple[CorruptionEntry, ...]

    def counts_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.kind] = counts.get(entry.kind, 0) + 1
        return counts


def inject_observation_corruption(
    observations: pd.DataFrame,
    seed: int,
    *,
    missing_rate: float = MISSING_RATE_DEFAULT,
    outlier_rate: float = OUTLIER_RATE_DEFAULT,
    duplicate_rate: float = DUPLICATE_RATE_DEFAULT,
    outlier_multiplier: float = OUTLIER_MULTIPLIER_DEFAULT,
) -> tuple[pd.DataFrame, CorruptionManifest]:
    """Return a corrupted raw observation frame and its manifest.

    Missing values are injected only into ``WEATHER_COLUMNS`` (never into the
    first chronological observation of a segment), outliers only into
    ``OUTLIER_COLUMNS`` (skipped when the multiplication leaves the value
    unchanged), and duplicates are exact copies of the emitted rows. Keys,
    dates, road age, and target-related columns are never touched.
    """
    seed = _validate_corruption_seed(seed)
    missing_rate = _validate_rate(missing_rate, "missing_rate")
    outlier_rate = _validate_rate(outlier_rate, "outlier_rate")
    duplicate_rate = _validate_rate(duplicate_rate, "duplicate_rate")
    numerator, denominator, multiplier_float = _validate_outlier_multiplier(outlier_multiplier)
    _validate_corruption_input(observations)

    sorted_obs = observations.sort_values(["segment_id", "date"], kind="stable")
    rows: list[dict[str, Any]] = []
    entries: list[CorruptionEntry] = []
    seen_segments: set[str] = set()
    for _, row in sorted_obs.iterrows():
        segment_id = str(row["segment_id"])
        observation_date = pd.Timestamp(row["date"]).date()
        day_int = (observation_date - _EPOCH).days
        seed_component = _epoch_seed_component(day_int)
        key = int.from_bytes(segment_id.encode("ascii"), "big", signed=False)
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, key, CORRUPTION_RNG_NAMESPACE, seed_component])
        )
        corrupted = dict(row)
        is_first = segment_id not in seen_segments
        if not is_first:
            for column in WEATHER_COLUMNS:
                if rng.random() < missing_rate:
                    corrupted[column] = np.nan
                    entries.append(CorruptionEntry("missing", segment_id, observation_date, column))
        for column in OUTLIER_COLUMNS:
            if rng.random() < outlier_rate:
                base = corrupted[column]
                if pd.isna(base):
                    continue
                if column == "traffic_volume":
                    base_int = int(base)
                    if base_int > 0:
                        bound = (denominator * _INT64_MAX) // base_int
                        if numerator > bound:
                            raise ValueError(
                                f"outlier for segment {segment_id!r} at date "
                                f"{observation_date} column {column} overflows int64"
                            )
                    exact = Fraction(base_int * numerator, denominator)
                    new_value = round(exact)
                    if not (_INT64_MIN <= new_value <= _INT64_MAX):
                        raise ValueError(
                            f"outlier for segment {segment_id!r} at date "
                            f"{observation_date} column {column} overflows int64"
                        )
                else:
                    base = float(base)
                    derived = base * multiplier_float
                    if not math.isfinite(derived):
                        raise ValueError(
                            f"outlier for segment {segment_id!r} at date "
                            f"{observation_date} column {column} is not finite"
                        )
                    new_value = round(derived, 1)
                if new_value == base:
                    continue
                corrupted[column] = new_value
                entries.append(CorruptionEntry("outlier", segment_id, observation_date, column))
        if rng.random() < duplicate_rate:
            rows.append(dict(corrupted))
            entries.append(CorruptionEntry("duplicate", segment_id, observation_date, None))
        rows.append(corrupted)
        seen_segments.add(segment_id)

    frame = pd.DataFrame(rows, columns=list(OBSERVATION_COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"])
    for column in _OBS_INT_COLUMNS:
        frame[column] = frame[column].astype("int64")
    for column in _OBS_FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    frame["segment_id"] = frame["segment_id"].astype(object)
    frame = frame.sort_values(["segment_id", "date"], kind="stable").reset_index(drop=True)
    entries.sort(key=lambda e: (e.kind, e.segment_id, e.date, e.column or ""))
    return frame, CorruptionManifest(tuple(entries))


def _epoch_seed_component(day_int: int) -> int:
    """Encode a day offset into non-negative RNG entropy.

    Dates on or after 1970-01-01 use the plain day offset (stable historical
    streams); pre-epoch offsets map into a disjoint non-negative range so
    equal-distance dates before and after the epoch never share a stream.
    """
    if day_int >= 0:
        return day_int
    return _PRE_EPOCH_BASE + (-day_int)


def _validate_corruption_seed(seed: object) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, int):
        raise ValueError("seed must be a positive non-boolean integer")
    try:
        value = seed.__index__()
    except Exception as exc:
        raise ValueError("seed must be a positive non-boolean integer") from exc
    if type(value) is not int or value < 1:
        raise ValueError("seed must be a positive non-boolean integer")
    return value


def _validate_rate(rate: object, name: str) -> float:
    if isinstance(rate, (bool, np.bool_, str)):
        raise ValueError(f"{name} must be a finite number")
    if isinstance(rate, (int, np.integer)):
        try:
            value = rate.__index__()
        except Exception as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if type(value) is not int:
            raise ValueError(f"{name} must be a finite number")
        if value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
        if value > MAX_CORRUPTION_RATE:
            raise ValueError(f"{name} exceeds MAX_CORRUPTION_RATE ({MAX_CORRUPTION_RATE})")
        try:
            return float(value)
        except Exception as exc:
            raise ValueError(f"{name} must be a finite number") from exc
    if not isinstance(rate, (float, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    try:
        as_float = float(rate)
    except Exception as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(as_float) or as_float < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    if as_float > MAX_CORRUPTION_RATE:
        raise ValueError(f"{name} exceeds MAX_CORRUPTION_RATE ({MAX_CORRUPTION_RATE})")
    return as_float


def _validate_outlier_multiplier(multiplier: object) -> tuple[int, int, float]:
    """Validate the multiplier and return canonical traffic and rainfall forms.

    Returns the exact (numerator, denominator) pair for traffic arithmetic
    and one canonical Python float for rainfall, both derived from a single
    normalization so hostile subclasses can never mean different values in
    the two paths.
    """
    if isinstance(multiplier, (bool, np.bool_, str)):
        raise ValueError("outlier_multiplier must be a finite number")
    if isinstance(multiplier, (int, np.integer)):
        try:
            value = multiplier.__index__()
        except Exception as exc:
            raise ValueError("outlier_multiplier must be a finite number") from exc
        if type(value) is not int:
            raise ValueError("outlier_multiplier must be a finite number")
        if value <= 1:
            raise ValueError("outlier_multiplier must be finite and greater than 1")
        try:
            as_float = float(value)
        except Exception as exc:
            raise ValueError(
                "outlier_multiplier is too large to convert to float for rainfall corruption"
            ) from exc
        return value, 1, as_float
    if isinstance(multiplier, (float, np.floating)):
        try:
            as_float = float(multiplier)
        except Exception as exc:
            raise ValueError("outlier_multiplier must be a finite number") from exc
        if not math.isfinite(as_float):
            raise ValueError("outlier_multiplier must be a finite number")
        if as_float <= 1.0:
            raise ValueError("outlier_multiplier must be finite and greater than 1")
        numerator, denominator = as_float.as_integer_ratio()
        return numerator, denominator, as_float
    raise ValueError(
        f"outlier_multiplier must be a finite number, got type {type(multiplier).__name__}"
    )


def _validate_corruption_input(observations: pd.DataFrame) -> None:
    """Strictly validate the corruption input before any RNG use."""
    columns = list(observations.columns)
    missing = [
        column
        for column in OBSERVATION_COLUMNS
        if not any(type(c) is str and c == column for c in columns)
    ]
    if missing:
        raise ValueError(f"observations frame is missing columns: {missing}")
    duplicates = [
        column
        for column in OBSERVATION_COLUMNS
        if sum(1 for c in columns if type(c) is str and c == column) > 1
    ]
    if duplicates:
        raise ValueError(f"observations frame has duplicate column label {duplicates[0]!r}")
    extras = [
        (position, column)
        for position, column in enumerate(columns)
        if not (type(column) is str and column in OBSERVATION_COLUMNS)
    ]
    if extras:
        labels = ", ".join(extra_column_label(position, column) for position, column in extras)
        raise ValueError(f"observations frame contains extra columns: {labels}")
    if str(observations["segment_id"].dtype) != "object":
        raise ValueError("observations column segment_id must have object dtype")
    if not str(observations["date"].dtype).startswith("datetime64"):
        raise ValueError("observations date column must have datetime64 dtype")
    if observations["date"].dt.tz is not None:
        raise ValueError("observations date column must be timezone-naive")
    for column in _OBS_INT_COLUMNS:
        if str(observations[column].dtype) != "int64":
            raise ValueError(f"observations column {column} must have int64 dtype")
    for column in _OBS_FLOAT_COLUMNS:
        if str(observations[column].dtype) != "float64":
            raise ValueError(f"observations column {column} must have float64 dtype")

    for index, value in enumerate(observations["segment_id"]):
        if type(value) is not str or re.fullmatch(SEGMENT_ID_PATTERN, value) is None:
            if type(value) is str:
                raise ValueError(f"observations frame contains malformed segment_id {value!r}")
            raise ValueError(
                f"observations frame contains malformed segment_id at "
                f"row[{index}]:{type(value).__name__}"
            )
    for value in observations["date"]:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is not None:
            raise ValueError("observations frame contains an invalid date value")
        if (
            timestamp.hour
            or timestamp.minute
            or timestamp.second
            or timestamp.microsecond
            or timestamp.nanosecond
        ):
            raise ValueError(
                "observations frame contains a non-midnight date; "
                "corruption requires start-of-day values"
            )
        try:
            timestamp.as_unit("ns")
        except (pd.errors.OutOfBoundsDatetime, OverflowError, ValueError):
            raise ValueError(
                f"observations frame contains {timestamp} outside datetime64[ns] range"
            ) from None

    seen_keys: dict[tuple[str, date], dict[str, Any]] = {}
    for _, row in observations.iterrows():
        segment_id = str(row["segment_id"])
        observation_date = pd.Timestamp(row["date"]).date()
        key = (segment_id, observation_date)
        row_values = dict(row)
        if key in seen_keys:
            if all(
                seen_keys[key][column] == row_values[column]
                or (
                    isinstance(seen_keys[key][column], float)
                    and isinstance(row_values[column], float)
                    and math.isnan(seen_keys[key][column])
                    and math.isnan(row_values[column])
                )
                for column in OBSERVATION_COLUMNS
            ):
                raise ValueError(f"observations frame has duplicate key {key}")
            raise ValueError(f"observations frame has conflicting rows for key {key}")
        seen_keys[key] = row_values

        traffic = int(row["traffic_volume"])
        if traffic < 0:
            raise ValueError(f"traffic_volume is negative for key {key}")
        road_age = int(row["road_age_days"])
        if road_age < 0:
            raise ValueError(f"road_age_days is negative for key {key}")
        dslm = int(row["days_since_last_maintenance"])
        if not (0 <= dslm <= road_age):
            raise ValueError(f"days_since_last_maintenance out of range for key {key}")
        if int(row["previous_repairs"]) < 0:
            raise ValueError(f"previous_repairs is negative for key {key}")
        heavy = float(row["heavy_vehicle_ratio"])
        if not math.isfinite(heavy) or not 0.0 <= heavy <= 1.0:
            raise ValueError(f"heavy_vehicle_ratio invalid for key {key}")
        rainfall = float(row["rainfall_mm"])
        if not math.isfinite(rainfall) or rainfall < 0:
            raise ValueError(f"rainfall_mm invalid for key {key}")
        for column, minimum, maximum in (
            ("temperature", -50.0, 60.0),
            ("humidity", 0.0, 100.0),
        ):
            value = float(row[column])
            if not math.isfinite(value) or not (minimum <= value <= maximum):
                raise ValueError(f"{column} invalid for key {key}")
        for column in (
            "road_condition_score",
            "marking_condition_score",
            "guardrail_condition_score",
            "sign_condition_score",
        ):
            score = int(row[column])
            if not 1 <= score <= 100:
                raise ValueError(f"{column} out of range for key {key}")
        count_30d = int(row["accident_count_30d"])
        count_365d = int(row["accident_count_365d"])
        if count_30d < 0 or count_365d < 0 or count_30d > count_365d:
            raise ValueError(f"accident counts invalid for key {key}")


__all__ = [
    "CORRUPTION_RNG_NAMESPACE",
    "DUPLICATE_RATE_DEFAULT",
    "MISSING_RATE_DEFAULT",
    "MAX_CORRUPTION_RATE",
    "OUTLIER_COLUMNS",
    "OUTLIER_MULTIPLIER_DEFAULT",
    "OUTLIER_RATE_DEFAULT",
    "WEATHER_COLUMNS",
    "CorruptionEntry",
    "CorruptionManifest",
    "inject_observation_corruption",
]
