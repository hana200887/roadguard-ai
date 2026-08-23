"""Phase 3 observation-core synthetic generation.

``generate_observations`` builds the clean, complete, causally valid
``road_observations`` table: one start-of-day snapshot per (segment, month)
with traffic, heavy-vehicle, weather, road-age, maintenance-history,
condition-score, and exact 30/365-day accident-window columns. No targets,
cost, materials, anomalies, missingness, or engineered features are
produced; no cleaning or imputation happens here.

Methodology (authoritative Phase 3 addendum)
--------------------------------------------
- Observation rows are labelled at as-of month-start dates ``t``. Traffic
  and weather represent the most recently completed monthly aggregate
  available at ``t``; seasonality follows the as-of observation month
  ``m``, trend follows the zero-based observation index ``k``.
- Traffic: ``base * trend * season * log-normal noise``, rounded with
  ``np.rint`` to a non-negative integer. Heavy-vehicle ratio is centred on
  its segment base with small seasonal and noise terms, clipped to [0, 1].
- Weather: rainfall from ``BASE_RAINFALL_MM * weather_exposure * season *
  log-normal noise``; temperature and humidity follow the documented
  formulas, with humidity positively dependent on rainfall.
- Condition scores: the latent condition state replays the Phase 2
  ``month_transition``/``decay_condition`` chain from the configured
  pre-period, capturing the pre-transition state for each observation month
  (an event on ``t`` cannot improve the score at ``t``). The four scores
  share the latent condition but use distinct documented modifiers and
  noise, keeping them correlated yet distinct.
- Accident windows: monthly counts are deterministically expanded into
  dated occurrences (dedicated per-(segment, month) RNG namespace), and the
  literal half-open windows ``[t - 30d, t)`` and ``[t - 365d, t)`` are
  counted exactly. Missing required history raises instead of being zero-
  filled.
- Maintenance history: only events strictly before ``t``; never-maintained
  segments use ``min(road_age_days, NEVER_MAINTAINED_DAYS_CAP)``.

RNG contract
------------
Per segment: ``SeedSequence([seed, segment_key, OBSERVATION_RNG_NAMESPACE,
STREAM])`` with ``segment_key =
int.from_bytes(segment_id.encode("ascii"), "big")`` and
streams TRAFFIC_STREAM, WEATHER_STREAM, CONDITION_STREAM. Accident-day
expansion adds ``(year, month)`` to the entropy tuple. Draws are consumed
per observation date in the documented order; no global NumPy state, no
Python ``hash()``, and no Phase 2 RNG objects are used, so segment and
row-order changes never alter results.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from decimal import Decimal
from numbers import Rational
from typing import Any, Final

import numpy as np
import pandas as pd

from roadguard.contracts import V1_OBSERVATION_START, DatasetSpec
from roadguard.events import (
    _check_construction_dates,
    _segment_masters,
    month_transition,
    observation_dates,
)
from roadguard.segments import SEGMENT_COLUMNS, SEGMENT_ID_PATTERN, SegmentMaster

OBSERVATION_COLUMNS: Final[tuple[str, ...]] = (
    "segment_id",
    "date",
    "traffic_volume",
    "heavy_vehicle_ratio",
    "road_age_days",
    "rainfall_mm",
    "temperature",
    "humidity",
    "days_since_last_maintenance",
    "previous_repairs",
    "road_condition_score",
    "marking_condition_score",
    "guardrail_condition_score",
    "sign_condition_score",
    "accident_count_30d",
    "accident_count_365d",
)

OBSERVATION_RNG_NAMESPACE: Final[int] = 0x524733
TRAFFIC_STREAM: Final[int] = 0
WEATHER_STREAM: Final[int] = 1
CONDITION_STREAM: Final[int] = 2
ACCIDENT_DAY_STREAM: Final[int] = 3

TRAFFIC_MONTHLY_TREND: Final[float] = 0.003
TRAFFIC_SEASONAL_AMPLITUDE: Final[float] = 0.08
TRAFFIC_LOG_NOISE_SIGMA: Final[float] = 0.06

HEAVY_SEASONAL_AMPLITUDE: Final[float] = 0.02
HEAVY_NOISE_SIGMA: Final[float] = 0.012

BASE_RAINFALL_MM: Final[float] = 140.0
RAINFALL_SEASONAL_AMPLITUDE: Final[float] = 0.65
RAINFALL_LOG_NOISE_SIGMA: Final[float] = 0.22

BASE_TEMPERATURE_C: Final[float] = 27.0
TEMPERATURE_SEASONAL_AMPLITUDE: Final[float] = 4.0
TEMPERATURE_EXPOSURE_COEFFICIENT: Final[float] = -0.8
TEMPERATURE_NOISE_SIGMA: Final[float] = 0.8

BASE_HUMIDITY_PERCENT: Final[float] = 60.0
HUMIDITY_RAINFALL_COEFFICIENT: Final[float] = 0.075
HUMIDITY_EXPOSURE_COEFFICIENT: Final[float] = 7.0
HUMIDITY_NOISE_SIGMA: Final[float] = 2.0

ROAD_CONDITION_NOISE_SIGMA: Final[float] = 1.25
COMPONENT_CONDITION_NOISE_SIGMA: Final[float] = 1.5

NEVER_MAINTAINED_DAYS_CAP: Final[int] = 3650

ACCIDENT_WINDOW_30D: Final[int] = 30
ACCIDENT_WINDOW_365D: Final[int] = 365
MAX_ACCIDENTS_PER_SEGMENT_MONTH: Final[int] = 10_000

_EPOCH: Final[date] = date(1970, 1, 1)
_INT_LATENT_COLUMNS: Final[tuple[str, ...]] = ("traffic_base", "initial_condition")
_FLOAT_LATENT_COLUMNS: Final[tuple[str, ...]] = (
    "road_length_km",
    "heavy_vehicle_ratio_base",
    "weather_exposure",
    "deterioration_rate",
    "accident_propensity",
)
_INT_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
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
_FLOAT_OUTPUT_COLUMNS: Final[tuple[str, ...]] = (
    "heavy_vehicle_ratio",
    "rainfall_mm",
    "temperature",
    "humidity",
)
_MSL_UNINITIALIZED: Final[int] = 10_000


def generate_observations(
    segments: pd.DataFrame,
    maintenance_events: pd.DataFrame,
    accident_timeline: pd.DataFrame,
    spec: DatasetSpec,
    seed: int,
    *,
    start_date: date = V1_OBSERVATION_START,
    pre_period_months: int = 24,
) -> pd.DataFrame:
    """Generate the clean observation-core table.

    Returns exactly the documented 16 columns, one row per (segment,
    observation date), sorted by ``segment_id`` and ``date``. Inputs are
    validated at the boundary and never mutated.
    """
    _validate_seed(seed)
    if start_date.day != 1:
        raise ValueError("observation start_date must be the first day of a month")
    if isinstance(pre_period_months, bool) or not isinstance(pre_period_months, int):
        raise ValueError("pre_period_months must be a positive integer")
    if pre_period_months < 1:
        raise ValueError("pre_period_months must be a positive integer")
    missing_columns = [column for column in SEGMENT_COLUMNS if column not in segments.columns]
    if missing_columns:
        raise ValueError(f"segments frame is missing columns: {missing_columns}")
    _validate_segment_ids(segments)
    _validate_latent_fields(segments)
    masters = _segment_masters(segments, spec.dataset_segments)
    _check_construction_dates(masters, start_date)

    events_by_segment = _validate_maintenance_events(maintenance_events, masters)
    timeline_by_segment = _validate_accident_timeline(
        accident_timeline, masters, spec, start_date, pre_period_months
    )

    obs_dates = [
        value.date() for value in observation_dates(spec.dataset_months_per_segment, start_date)
    ]
    last_obs = obs_dates[-1]

    records: list[dict[str, Any]] = []
    for master in sorted(masters, key=lambda item: item.segment_id):
        key = _segment_key(master.segment_id)
        traffic_rng = _stream_rng(seed, key, TRAFFIC_STREAM)
        weather_rng = _stream_rng(seed, key, WEATHER_STREAM)
        condition_rng = _stream_rng(seed, key, CONDITION_STREAM)
        event_dates = sorted(events_by_segment[master.segment_id])
        event_months = {(value.year, value.month) for value in event_dates}
        timeline_months = timeline_by_segment[master.segment_id]
        accident_days = _expand_accidents(master, timeline_months, seed, key, last_obs, start_date)

        simulation_start = _add_months(start_date, -pre_period_months)
        construction_month = date(master.construction_date.year, master.construction_date.month, 1)
        month = max(simulation_start, construction_month)
        condition = float(master.initial_condition)
        months_since_last_event = _MSL_UNINITIALIZED
        accident_window: tuple[int, ...] = ()

        for k, t in enumerate(obs_dates):
            while month < t:
                months_since_last_event, condition, accident_window = _transition(
                    master,
                    month,
                    event_months,
                    timeline_months,
                    months_since_last_event,
                    condition,
                    accident_window,
                )
                month = _add_months(month, 1)

            m = t.month
            traffic_noise = traffic_rng.normal(
                -0.5 * TRAFFIC_LOG_NOISE_SIGMA**2, TRAFFIC_LOG_NOISE_SIGMA
            )
            heavy_noise = traffic_rng.normal(0.0, HEAVY_NOISE_SIGMA)
            rain_noise = weather_rng.normal(
                -0.5 * RAINFALL_LOG_NOISE_SIGMA**2, RAINFALL_LOG_NOISE_SIGMA
            )
            temperature_noise = weather_rng.normal(0.0, TEMPERATURE_NOISE_SIGMA)
            humidity_noise = weather_rng.normal(0.0, HUMIDITY_NOISE_SIGMA)
            road_noise = condition_rng.normal(0.0, ROAD_CONDITION_NOISE_SIGMA)
            marking_noise = condition_rng.normal(0.0, COMPONENT_CONDITION_NOISE_SIGMA)
            guardrail_noise = condition_rng.normal(0.0, COMPONENT_CONDITION_NOISE_SIGMA)
            sign_noise = condition_rng.normal(0.0, COMPONENT_CONDITION_NOISE_SIGMA)

            traffic_volume = _traffic_volume(master.traffic_base, k, m, traffic_noise)
            heavy_vehicle_ratio = _heavy_vehicle_ratio(
                master.heavy_vehicle_ratio_base, m, heavy_noise
            )
            rainfall_mm = _rainfall_mm(master.weather_exposure, m, rain_noise)
            temperature = _temperature(master.weather_exposure, m, temperature_noise)
            humidity = _humidity(master.weather_exposure, rainfall_mm, humidity_noise)

            road_age_days = (t - master.construction_date).days
            past_events = [value for value in event_dates if value < t]
            previous_repairs = len(past_events)
            if past_events:
                days_since_last_maintenance = (t - past_events[-1]).days
            else:
                days_since_last_maintenance = min(road_age_days, NEVER_MAINTAINED_DAYS_CAP)

            count_30d, count_365d = _window_counts(accident_days, t)

            road_condition_score = _road_score(condition, road_noise)
            marking_condition_score = _marking_score(
                condition, traffic_volume, rainfall_mm, marking_noise
            )
            guardrail_condition_score = _guardrail_score(
                condition, heavy_vehicle_ratio, count_365d, guardrail_noise
            )
            sign_condition_score = _sign_score(condition, humidity, rainfall_mm, sign_noise)

            records.append(
                {
                    "segment_id": master.segment_id,
                    "date": t,
                    "traffic_volume": traffic_volume,
                    "heavy_vehicle_ratio": heavy_vehicle_ratio,
                    "road_age_days": road_age_days,
                    "rainfall_mm": rainfall_mm,
                    "temperature": temperature,
                    "humidity": humidity,
                    "days_since_last_maintenance": days_since_last_maintenance,
                    "previous_repairs": previous_repairs,
                    "road_condition_score": road_condition_score,
                    "marking_condition_score": marking_condition_score,
                    "guardrail_condition_score": guardrail_condition_score,
                    "sign_condition_score": sign_condition_score,
                    "accident_count_30d": count_30d,
                    "accident_count_365d": count_365d,
                }
            )

            if t != last_obs:
                months_since_last_event, condition, accident_window = _transition(
                    master,
                    t,
                    event_months,
                    timeline_months,
                    months_since_last_event,
                    condition,
                    accident_window,
                )
                month = _add_months(t, 1)

    frame = pd.DataFrame(records, columns=list(OBSERVATION_COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"])
    _validate_derived_values(frame)
    for column in _INT_OUTPUT_COLUMNS:
        frame[column] = frame[column].astype("int64")
    for column in _FLOAT_OUTPUT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    frame["segment_id"] = frame["segment_id"].astype(object)
    return frame.sort_values(["segment_id", "date"], kind="stable").reset_index(drop=True)


def _transition(
    master: SegmentMaster,
    month: date,
    event_months: set[tuple[int, int]],
    timeline_months: dict[date, int],
    months_since_last_event: int,
    condition: float,
    accident_window: tuple[int, ...],
) -> tuple[int, float, tuple[int, ...]]:
    return month_transition(
        master,
        event_occurred=(month.year, month.month) in event_months,
        condition=condition,
        months_since_last_event=months_since_last_event,
        accident_window=accident_window,
        new_accident_count=timeline_months[month],
    )


def _traffic_volume(traffic_base: int, k: int, m: int, traffic_noise: float) -> int:
    trend = np.float64(1.0) + np.float64(TRAFFIC_MONTHLY_TREND) * np.float64(k)
    season = np.float64(1.0) + np.float64(TRAFFIC_SEASONAL_AMPLITUDE) * np.sin(
        np.float64(2.0) * np.pi * np.float64(m - 1) / np.float64(12.0)
    )
    raw = np.float64(traffic_base) * trend * season * np.exp(np.float64(traffic_noise))
    return int(max(np.float64(0.0), np.rint(raw)))


def _heavy_vehicle_ratio(heavy_vehicle_ratio_base: float, m: int, heavy_noise: float) -> float:
    season = np.float64(HEAVY_SEASONAL_AMPLITUDE) * np.sin(
        np.float64(2.0) * np.pi * np.float64(m - 2) / np.float64(12.0)
    )
    value = np.clip(
        np.float64(heavy_vehicle_ratio_base) + season + np.float64(heavy_noise),
        np.float64(0.0),
        np.float64(1.0),
    )
    return float(np.round(value, 4))


def _rainfall_mm(weather_exposure: float, m: int, rain_noise: float) -> float:
    season = np.float64(1.0) + np.float64(RAINFALL_SEASONAL_AMPLITUDE) * np.sin(
        np.float64(2.0) * np.pi * np.float64(m - 5) / np.float64(12.0)
    )
    value = (
        np.float64(BASE_RAINFALL_MM)
        * np.float64(weather_exposure)
        * season
        * np.exp(np.float64(rain_noise))
    )
    return float(np.round(np.clip(value, np.float64(0.0), np.inf), 1))


def _temperature(weather_exposure: float, m: int, temperature_noise: float) -> float:
    value = (
        np.float64(BASE_TEMPERATURE_C)
        + np.float64(TEMPERATURE_SEASONAL_AMPLITUDE)
        * np.sin(np.float64(2.0) * np.pi * np.float64(m - 1) / np.float64(12.0))
        + np.float64(TEMPERATURE_EXPOSURE_COEFFICIENT)
        * (np.float64(weather_exposure) - np.float64(1.0))
        + np.float64(temperature_noise)
    )
    return float(np.round(np.clip(value, np.float64(-50.0), np.float64(60.0)), 1))


def _humidity(weather_exposure: float, rainfall_mm: float, humidity_noise: float) -> float:
    value = (
        np.float64(BASE_HUMIDITY_PERCENT)
        + np.float64(HUMIDITY_RAINFALL_COEFFICIENT) * np.float64(rainfall_mm)
        + np.float64(HUMIDITY_EXPOSURE_COEFFICIENT)
        * (np.float64(weather_exposure) - np.float64(1.0))
        + np.float64(humidity_noise)
    )
    return float(np.round(np.clip(value, np.float64(0.0), np.float64(100.0)), 1))


def _road_score(condition: float, road_noise: float) -> int:
    return _bounded_score(np.float64(condition) + np.float64(road_noise))


def _marking_score(
    condition: float, traffic_volume: int, rainfall_mm: float, marking_noise: float
) -> int:
    return _bounded_score(
        np.float64(condition)
        - np.float64(4.0)
        - np.float64(0.00015) * np.float64(traffic_volume)
        - np.float64(0.008) * np.float64(rainfall_mm)
        + np.float64(marking_noise)
    )


def _guardrail_score(
    condition: float, heavy_vehicle_ratio: float, accidents_365d: int, guardrail_noise: float
) -> int:
    return _bounded_score(
        np.float64(condition)
        - np.float64(2.0)
        - np.float64(8.0) * np.float64(heavy_vehicle_ratio)
        - np.float64(2.0) * np.log1p(np.float64(accidents_365d))
        + np.float64(guardrail_noise)
    )


def _sign_score(condition: float, humidity: float, rainfall_mm: float, sign_noise: float) -> int:
    return _bounded_score(
        np.float64(condition)
        - np.float64(3.0)
        - np.float64(0.04) * np.maximum(np.float64(humidity) - np.float64(60.0), np.float64(0.0))
        - np.float64(0.004) * np.float64(rainfall_mm)
        + np.float64(sign_noise)
    )


def _bounded_score(value: np.float64) -> int:
    return int(np.clip(np.rint(value), np.float64(1.0), np.float64(100.0)))


def _expand_accidents(
    master: SegmentMaster,
    timeline_months: dict[date, int],
    seed: int,
    key: int,
    last_obs: date,
    first_obs: date,
) -> np.ndarray:
    days: list[int] = []
    construction_month = date(master.construction_date.year, master.construction_date.month, 1)
    expansion_floor = _floor_month(first_obs - timedelta(days=ACCIDENT_WINDOW_365D))
    for month in sorted(timeline_months):
        if month < expansion_floor:
            continue
        if month > last_obs:
            continue
        count = timeline_months[month]
        if count == 0:
            continue
        days_in_month = calendar.monthrange(month.year, month.month)[1]
        low_offset = master.construction_date.day - 1 if month == construction_month else 0
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, key, OBSERVATION_RNG_NAMESPACE, ACCIDENT_DAY_STREAM, month.year, month.month]
            )
        )
        offsets = rng.integers(low=low_offset, high=days_in_month, size=count)
        base_day = (month - _EPOCH).days
        days.extend(int(base_day + offset) for offset in offsets)
    result = np.array(days, dtype=np.int64)
    result.sort()
    return result


def _window_counts(accident_days: np.ndarray, t: date) -> tuple[int, int]:
    t_day = (t - _EPOCH).days
    right_30 = int(np.searchsorted(accident_days, t_day, side="left"))
    left_30 = int(np.searchsorted(accident_days, t_day - ACCIDENT_WINDOW_30D, side="left"))
    right_365 = int(np.searchsorted(accident_days, t_day, side="left"))
    left_365 = int(np.searchsorted(accident_days, t_day - ACCIDENT_WINDOW_365D, side="left"))
    return right_30 - left_30, right_365 - left_365


def _validate_derived_values(frame: pd.DataFrame) -> None:
    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max
    for column in _INT_OUTPUT_COLUMNS:
        for value in frame[column]:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"derived column {column} must be an integer, got {value!r}")
            if not (int64_min <= value <= int64_max):
                raise ValueError(f"derived column {column} value {value!r} overflows int64")
    for column in _FLOAT_OUTPUT_COLUMNS:
        for value in frame[column]:
            if not isinstance(value, (float, np.floating, int, np.integer)):
                raise ValueError(f"derived column {column} must be numeric, got {value!r}")
            if not np.isfinite(value):
                raise ValueError(f"derived column {column} contains a non-finite value: {value!r}")


def _validate_seed(seed: object) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 1:
        raise ValueError("seed must be a positive non-boolean integer")


def _validate_segment_ids(segments: pd.DataFrame) -> None:
    for value in segments["segment_id"]:
        if not isinstance(value, str) or re.fullmatch(SEGMENT_ID_PATTERN, value) is None:
            raise ValueError(
                f"malformed segment_id {value!r}; expected pattern {SEGMENT_ID_PATTERN}"
            )
        _segment_key(value)


def _segment_key(segment_id: str) -> int:
    try:
        return int.from_bytes(segment_id.encode("ascii"), "big", signed=False)
    except UnicodeEncodeError:
        raise ValueError(f"segment_id is not ASCII-encodable: {segment_id!r}") from None


def _validate_latent_fields(segments: pd.DataFrame) -> None:
    for column in _INT_LATENT_COLUMNS:
        _validate_int_latent_column(segments, column)
    for column in _FLOAT_LATENT_COLUMNS:
        _validate_float_latent_column(segments, column)


def _validate_int_latent_column(segments: pd.DataFrame, column: str) -> None:
    int64_min = np.iinfo(np.int64).min
    int64_max = np.iinfo(np.int64).max
    for value in segments[column]:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"segments column {column} must be numeric, got {value!r}")
        if isinstance(value, str):
            raise ValueError(f"segments column {column} must be numeric, got {value!r}")
        if isinstance(value, (int, np.integer)):
            as_int = int(value)
            if not (int64_min <= as_int <= int64_max):
                raise ValueError(f"segments column {column} value {value!r} overflows int64")
            continue
        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                raise ValueError(f"segments column {column} contains a non-finite value: {value!r}")
            as_float = float(value)
            if not as_float.is_integer():
                raise ValueError(f"segments column {column} must be an integer, got {value!r}")
            if not (int64_min <= as_float <= int64_max):
                raise ValueError(f"segments column {column} value {value!r} overflows int64")
            continue
        if isinstance(value, Decimal):
            if not value.is_finite():
                raise ValueError(f"segments column {column} contains a non-finite value: {value!r}")
            if value != value.to_integral_value():
                raise ValueError(f"segments column {column} must be an integer, got {value!r}")
            if not (int64_min <= value <= int64_max):
                raise ValueError(f"segments column {column} value {value!r} overflows int64")
            continue
        if isinstance(value, Rational):
            if value.denominator != 1:
                raise ValueError(f"segments column {column} must be an integer, got {value!r}")
            as_int = int(value.numerator)
            if not (int64_min <= as_int <= int64_max):
                raise ValueError(f"segments column {column} value {value!r} overflows int64")
            continue
        raise ValueError(f"segments column {column} contains unsupported value {value!r}")


def _validate_float_latent_column(segments: pd.DataFrame, column: str) -> None:
    for value in segments[column]:
        if isinstance(value, (bool, np.bool_, str)):
            raise ValueError(f"segments column {column} must be numeric, got {value!r}")
        try:
            as_float = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"segments column {column} must be numeric, got {value!r}") from None
        if not np.isfinite(as_float):
            raise ValueError(f"segments column {column} contains a non-finite value: {value!r}")


def _validate_maintenance_events(
    maintenance_events: pd.DataFrame, masters: list[SegmentMaster]
) -> dict[str, list[date]]:
    required = {"segment_id", "maintenance_date"}
    if not required.issubset(maintenance_events.columns):
        raise ValueError(f"maintenance_events must contain columns: {sorted(required)}")
    construction_by_id = {master.segment_id: master.construction_date for master in masters}
    events_by_segment: dict[str, list[date]] = {master.segment_id: [] for master in masters}
    seen_keys: set[tuple[str, date]] = set()
    month_counts: dict[tuple[str, int, int], int] = {}
    for _, row in maintenance_events.iterrows():
        segment_id = str(row["segment_id"])
        if segment_id not in construction_by_id:
            raise ValueError(f"maintenance event references unknown segment {segment_id!r}")
        event_date = _parse_date(row["maintenance_date"], "maintenance_date", segment_id)
        if event_date < construction_by_id[segment_id]:
            raise ValueError(f"maintenance event before construction for segment {segment_id!r}")
        key = (segment_id, event_date)
        if key in seen_keys:
            raise ValueError(f"duplicate maintenance event key {key}")
        seen_keys.add(key)
        month_key = (segment_id, event_date.year, event_date.month)
        month_counts[month_key] = month_counts.get(month_key, 0) + 1
        if month_counts[month_key] > 1:
            raise ValueError(f"more than one maintenance event in the same segment-month: {key}")
        events_by_segment[segment_id].append(event_date)
    return events_by_segment


def _validate_accident_timeline(
    accident_timeline: pd.DataFrame,
    masters: list[SegmentMaster],
    spec: DatasetSpec,
    start_date: date,
    pre_period_months: int,
) -> dict[str, dict[date, int]]:
    required = {"segment_id", "month", "accident_count"}
    if not required.issubset(accident_timeline.columns):
        raise ValueError(f"accident_timeline must contain columns: {sorted(required)}")
    construction_by_id = {master.segment_id: master.construction_date for master in masters}
    timeline_by_segment: dict[str, dict[date, int]] = {master.segment_id: {} for master in masters}
    seen_months: set[tuple[str, date]] = set()
    for _, row in accident_timeline.iterrows():
        segment_id = str(row["segment_id"])
        if segment_id not in construction_by_id:
            raise ValueError(f"accident timeline references unknown segment {segment_id!r}")
        month = _parse_date(row["month"], "month", segment_id)
        if month.day != 1:
            raise ValueError(f"accident timeline month must be a month start: {month}")
        construction_month = date(
            construction_by_id[segment_id].year, construction_by_id[segment_id].month, 1
        )
        if month < construction_month:
            raise ValueError(
                f"accident bucket before construction month for segment {segment_id!r}"
            )
        count = row["accident_count"]
        if isinstance(count, (bool, np.bool_)):
            raise ValueError(
                f"accident_count for segment {segment_id!r} at month {month} "
                f"must be numeric, got {count!r}"
            )
        if isinstance(count, str):
            raise ValueError(
                f"accident_count for segment {segment_id!r} at month {month} "
                f"must be numeric, got {count!r}"
            )
        if isinstance(count, (int, np.integer)):
            as_int = int(count)
        elif isinstance(count, (float, np.floating)):
            if not np.isfinite(count):
                raise ValueError(
                    f"accident_count for segment {segment_id!r} at month {month} "
                    f"must be finite, got {count!r}"
                )
            as_float = float(count)
            if not as_float.is_integer():
                raise ValueError(
                    f"accident_count for segment {segment_id!r} at month {month} "
                    f"must be an integer, got {count!r}"
                )
            as_int = int(as_float)
        elif isinstance(count, Decimal):
            if not count.is_finite():
                raise ValueError(
                    f"accident_count for segment {segment_id!r} at month {month} "
                    f"must be finite, got {count!r}"
                )
            if count != count.to_integral_value():
                raise ValueError(
                    f"accident_count for segment {segment_id!r} at month {month} "
                    f"must be an integer, got {count!r}"
                )
            if count < 0:
                raise ValueError(
                    f"accident_count for segment {segment_id!r} at month {month} "
                    f"must be non-negative, got {count!r}"
                )
            if count > MAX_ACCIDENTS_PER_SEGMENT_MONTH:
                raise ValueError(
                    f"accident_count {count!r} for segment {segment_id!r} at month {month} "
                    f"exceeds MAX_ACCIDENTS_PER_SEGMENT_MONTH "
                    f"({MAX_ACCIDENTS_PER_SEGMENT_MONTH})"
                )
            as_int = int(count)
        elif isinstance(count, Rational):
            if count.denominator != 1:
                raise ValueError(
                    f"accident_count for segment {segment_id!r} at month {month} "
                    f"must be an integer, got {count!r}"
                )
            as_int = int(count.numerator)
        else:
            raise ValueError(
                f"accident_count for segment {segment_id!r} at month {month} "
                f"contains unsupported value {count!r}"
            )
        if as_int < 0:
            raise ValueError(
                f"accident_count for segment {segment_id!r} at month {month} "
                f"must be non-negative, got {count!r}"
            )
        if as_int > MAX_ACCIDENTS_PER_SEGMENT_MONTH:
            raise ValueError(
                f"accident_count {count!r} for segment {segment_id!r} at month {month} "
                f"exceeds MAX_ACCIDENTS_PER_SEGMENT_MONTH "
                f"({MAX_ACCIDENTS_PER_SEGMENT_MONTH})"
            )
        month_key = (segment_id, month)
        if month_key in seen_months:
            raise ValueError(f"duplicate accident timeline month {month_key}")
        seen_months.add(month_key)
        timeline_by_segment[segment_id][month] = as_int

    _check_history_coverage(timeline_by_segment, masters, spec, start_date, pre_period_months)
    return timeline_by_segment


def _parse_date(value: Any, column: str, segment_id: str) -> date:
    if value is None or pd.isna(value):
        raise ValueError(f"{column} must not be missing (NaT) for segment {segment_id!r}")
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        raise ValueError(f"malformed {column} value {value!r} for segment {segment_id!r}") from None


def _check_history_coverage(
    timeline_by_segment: dict[str, dict[date, int]],
    masters: list[SegmentMaster],
    spec: DatasetSpec,
    start_date: date,
    pre_period_months: int,
) -> None:
    simulation_start = _add_months(start_date, -pre_period_months)
    first_obs = start_date
    last_obs = _add_months(start_date, spec.dataset_months_per_segment - 1)
    window_floor = _floor_month(first_obs - timedelta(days=ACCIDENT_WINDOW_365D))
    last_required_month = _add_months(last_obs, -1)
    for master in masters:
        months = set(timeline_by_segment[master.segment_id])
        construction_month = date(master.construction_date.year, master.construction_date.month, 1)
        required_start = max(
            construction_month,
            min(simulation_start, window_floor),
        )
        month = required_start
        while month <= last_required_month:
            if month not in months:
                raise ValueError(
                    f"missing accident bucket {month} for segment {master.segment_id!r}"
                )
            month = _add_months(month, 1)


def _stream_rng(seed: int, key: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([seed, key, OBSERVATION_RNG_NAMESPACE, stream])
    )


def _add_months(value: date, months: int) -> date:
    total = value.year * 12 + (value.month - 1) + months
    year, month_zero = divmod(total, 12)
    month = month_zero + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _floor_month(value: date) -> date:
    return date(value.year, value.month, 1)


__all__ = [
    "ACCIDENT_WINDOW_30D",
    "ACCIDENT_WINDOW_365D",
    "MAX_ACCIDENTS_PER_SEGMENT_MONTH",
    "NEVER_MAINTAINED_DAYS_CAP",
    "OBSERVATION_COLUMNS",
    "generate_observations",
]
