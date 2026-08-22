"""Maintenance-event and accident-history simulation for the V1 synthetic network.

Methodology
-----------
Maintenance events are simulated per segment as a monthly Bernoulli process
(at most one event per segment-month) over ``pre_period_months`` of
pre-window history, the full observation window, and a future buffer after
the final observation date. The monthly event probability is the pure
function :func:`monthly_hazard`, the product of documented latent drivers:

- asset age (hazard grows with age since ``construction_date``);
- traffic exposure (``traffic_base``);
- heavy-vehicle exposure (``heavy_vehicle_ratio_base``);
- weather exposure (``weather_exposure`` with a calendar-month seasonal
  cycle);
- condition deterioration (a renewal proxy that decays each month by
  ``deterioration_rate`` via :func:`decay_condition` and jumps toward
  ``initial_condition`` after each event);
- trailing accident history (accidents in the 12 months strictly before the
  current month, drawn from the segment's accident stream);
- previous maintenance (a renewal factor that suppresses hazard for 12
  months after an event, applied to the months *after* the event month).

Accident history is real simulated state: :func:`generate_accident_timeline`
produces deterministic monthly accident counts per segment (Poisson with
rate ``accident_propensity * (traffic_base / 10000)**0.5 * 0.05``), reusable
by later phases. Hazard at month ``t`` uses only accident months strictly
before ``t``; future accidents never influence hazard.

Simulation never runs before ``construction_date``; construction dates after
the observation start are rejected. Months are always simulated through the
future-buffer horizon ``sim_end`` (so buffer length controls the retained
horizon and shorter buffers are prefixes of longer ones); if no event falls
after the final observation date by ``sim_end``, the same hazard and
accident processes continue until the next event, with a documented safety
cap and an explicit :class:`GenerationError` failure.

Targets are NOT generated here; future targets never influence event
generation.

Determinism and row-order independence: each segment draws randomness from
its own stream ``SeedSequence([seed, int.from_bytes(segment_id)])`` (no
Python ``hash()``); child stream 0 is accidents, child stream 1 is
maintenance events. Segment master uses ``SeedSequence(seed)`` child stream
0. Shuffling the segment table therefore never changes the output frames.
"""

from __future__ import annotations

import calendar
import math
from datetime import date, timedelta
from typing import Any, Final, cast

import numpy as np
import pandas as pd

from roadguard.contracts import V1_OBSERVATION_START, DatasetSpec
from roadguard.segments import SEGMENT_COLUMNS, SegmentMaster

DEFAULT_PRE_PERIOD_MONTHS: Final[int] = 24
DEFAULT_FUTURE_BUFFER_MONTHS: Final[int] = 24
DEFAULT_BASE_RATE: Final[float] = 0.15
DEFAULT_MAX_MONTHS_PER_SEGMENT: Final[int] = 600
MIN_HAZARD: Final[float] = 0.02
MAX_HAZARD: Final[float] = 0.9
ACCIDENT_WINDOW_MONTHS: Final[int] = 12

EVENT_COLUMNS: Final[tuple[str, ...]] = ("segment_id", "maintenance_date")
ACCIDENT_COLUMNS: Final[tuple[str, ...]] = ("segment_id", "month", "accident_count")


class GenerationError(Exception):
    """Raised when event simulation cannot satisfy its guarantees."""


def observation_dates(month_count: int, start_date: date | None = None) -> pd.DatetimeIndex:
    """Return the monthly observation calendar for ``month_count`` months."""
    if month_count < 1:
        raise ValueError("month_count must be a positive integer")
    start = start_date or V1_OBSERVATION_START
    return pd.date_range(start=start, periods=month_count, freq="MS")


def decay_condition(condition: float, deterioration_rate: float) -> float:
    """Return the condition score after one month of deterioration.

    The score decays by ``deterioration_rate * 0.6`` per month and never
    falls below 20.
    """
    return max(20.0, condition - deterioration_rate * 0.6)


def month_transition(
    master: SegmentMaster,
    *,
    event_occurred: bool,
    condition: float,
    months_since_last_event: int,
    accident_window: tuple[int, ...],
    new_accident_count: int,
) -> tuple[int, float, tuple[int, ...]]:
    """Return the state after one simulated month.

    Returns ``(months_since_last_event, condition, accident_window)``. An
    event occurring during the month becomes historical state only here,
    i.e. it never affects the start-of-day snapshot of its own month: the
    renewal reset, the condition bump, and the new accident count all apply
    from the following month onward.
    """
    if event_occurred:
        months_since_last_event = 0
        condition = min(float(master.initial_condition), condition + 25.0)
    else:
        months_since_last_event = months_since_last_event + 1
    condition = decay_condition(condition, master.deterioration_rate)
    window = (accident_window + (new_accident_count,))[-ACCIDENT_WINDOW_MONTHS:]
    return months_since_last_event, condition, window


def monthly_hazard(
    master: SegmentMaster,
    month_date: date,
    months_since_last_event: int,
    condition: float,
    trailing_accidents: int,
    base_rate: float = DEFAULT_BASE_RATE,
) -> float:
    """Return the pure monthly maintenance probability for one segment-month.

    The hazard is the product of documented latent drivers, clamped to
    ``[MIN_HAZARD, MAX_HAZARD]``:

    - base rate: ``base_rate`` (validated: not boolean, finite, positive);
    - age factor: ``0.7 + 0.025 * age_years`` clamped to [0.7, 1.8];
    - traffic factor: ``(traffic_base / 10000) ** 0.2``;
    - heavy-vehicle factor: ``1 + (ratio - 0.25) * 2``;
    - weather factor: ``weather_exposure`` times a calendar-month seasonal
      cycle;
    - condition factor: ``1 + (60 - condition) / 40`` clamped to [0.5, 2.0];
    - accident factor: ``1 + 0.3 * min(trailing_accidents, 5)``;
    - renewal factor: 0.3 / 0.6 / 1.0 for ``months_since_last_event``
      < 6 / < 12 / >= 12.
    """
    if isinstance(base_rate, bool):
        raise ValueError("base_rate must be a numeric value, not a boolean")
    if not math.isfinite(base_rate) or base_rate <= 0:
        raise ValueError("base_rate must be finite and positive")
    age_years = (month_date - master.construction_date).days / 365.25
    seasonal = 1.0 + 0.25 * math.sin(2.0 * math.pi * (float(month_date.month) - 3.0) / 12.0)
    age_factor = min(max(0.7 + age_years * 0.025, 0.7), 1.8)
    traffic_factor = math.pow(master.traffic_base / 10_000.0, 0.2)
    heavy_factor = 1.0 + (master.heavy_vehicle_ratio_base - 0.25) * 2.0
    weather_factor = master.weather_exposure * seasonal
    condition_factor = min(max(1.0 + (60.0 - condition) / 40.0, 0.5), 2.0)
    accident_factor = 1.0 + 0.3 * min(max(trailing_accidents, 0), 5)
    if months_since_last_event < 6:
        renewal_factor = 0.3
    elif months_since_last_event < 12:
        renewal_factor = 0.6
    else:
        renewal_factor = 1.0
    hazard = (
        base_rate
        * age_factor
        * traffic_factor
        * heavy_factor
        * weather_factor
        * condition_factor
        * accident_factor
        * renewal_factor
    )
    return min(max(hazard, MIN_HAZARD), MAX_HAZARD)


def generate_accident_timeline(
    segments: pd.DataFrame,
    spec: DatasetSpec,
    seed: int,
    *,
    start_date: date | None = None,
    pre_period_months: int = DEFAULT_PRE_PERIOD_MONTHS,
    future_buffer_months: int = DEFAULT_FUTURE_BUFFER_MONTHS,
) -> pd.DataFrame:
    """Generate deterministic monthly accident counts per segment.

    Returns ``(segment_id, month, accident_count)`` for every simulated
    month from the segment's simulation start (never before its
    construction month) through the future buffer end. The counts come from
    the same per-segment accident stream used by
    :func:`generate_maintenance_events`, so the two are consistent.
    """
    if seed < 1:
        raise ValueError("seed must be a positive integer")
    if pre_period_months < 1:
        raise ValueError("pre_period_months must be a positive integer")
    if future_buffer_months < 1:
        raise ValueError("future_buffer_months must be a positive integer")
    masters = _segment_masters(segments, spec.dataset_segments)
    start = start_date or V1_OBSERVATION_START
    sim_start, sim_end = _simulation_range(start, spec, pre_period_months, future_buffer_months)
    _check_construction_dates(masters, start)

    records: list[dict[str, Any]] = []
    for master in masters:
        rng = _segment_rng(seed, master.segment_id, child=0)
        month = _first_sim_month(master, sim_start)
        while month <= sim_end:
            count = int(rng.poisson(_accident_rate(master)))
            records.append(
                {"segment_id": master.segment_id, "month": month, "accident_count": count}
            )
            month = _add_months(month, 1)
    frame = pd.DataFrame(records, columns=list(ACCIDENT_COLUMNS))
    frame["month"] = pd.to_datetime(frame["month"])
    return frame.sort_values(list(ACCIDENT_COLUMNS), kind="stable").reset_index(drop=True)


def generate_maintenance_events(
    segments: pd.DataFrame,
    spec: DatasetSpec,
    seed: int,
    *,
    start_date: date | None = None,
    pre_period_months: int = DEFAULT_PRE_PERIOD_MONTHS,
    future_buffer_months: int = DEFAULT_FUTURE_BUFFER_MONTHS,
    base_rate: float = DEFAULT_BASE_RATE,
    max_months_per_segment: int = DEFAULT_MAX_MONTHS_PER_SEGMENT,
) -> pd.DataFrame:
    """Generate deterministic maintenance events for the segment master.

    The returned frame has columns ``segment_id`` and ``maintenance_date``,
    sorted by segment and date. ``seed`` must be a positive integer. Months
    are simulated from the pre-period start through ``sim_end`` (the future
    buffer horizon), so a longer ``future_buffer_months`` extends the
    retained history while preserving shorter-buffer histories as prefixes.
    If no event falls after the final observation date by ``sim_end``, the
    same hazard and accident processes continue until the first such event,
    failing explicitly with :class:`GenerationError` if
    ``max_months_per_segment`` is exceeded.
    """
    if seed < 1:
        raise ValueError("seed must be a positive integer")
    if pre_period_months < 1:
        raise ValueError("pre_period_months must be a positive integer")
    if future_buffer_months < 1:
        raise ValueError("future_buffer_months must be a positive integer")
    if base_rate <= 0 or isinstance(base_rate, bool) or not math.isfinite(base_rate):
        raise ValueError("base_rate must be a finite positive number")
    if max_months_per_segment < 1:
        raise ValueError("max_months_per_segment must be a positive integer")
    masters = _segment_masters(segments, spec.dataset_segments)
    start = start_date or V1_OBSERVATION_START
    sim_start, sim_end = _simulation_range(start, spec, pre_period_months, future_buffer_months)
    final_obs: date = observation_dates(spec.dataset_months_per_segment, start)[-1].date()
    _check_construction_dates(masters, start)

    records: list[dict[str, Any]] = []
    for master in masters:
        event_rng = _segment_rng(seed, master.segment_id, child=1)
        accident_rng = _segment_rng(seed, master.segment_id, child=0)
        events: list[date] = []
        months_since_last_event = 10_000
        condition = float(master.initial_condition)
        accident_window: tuple[int, ...] = ()
        construction_month = date(master.construction_date.year, master.construction_date.month, 1)
        month = _first_sim_month(master, sim_start)
        months_simulated = 0
        while True:
            hazard = monthly_hazard(
                master,
                month,
                months_since_last_event,
                condition,
                sum(accident_window),
                base_rate,
            )
            days_in_month = calendar.monthrange(month.year, month.month)[1]
            months_simulated += 1
            if months_simulated > max_months_per_segment:
                raise GenerationError(
                    f"no maintenance event after the final observation date for "
                    f"segment {master.segment_id} within {max_months_per_segment} months"
                )
            event_occurred = event_rng.random() < hazard
            if event_occurred:
                first_day = master.construction_date.day - 1 if month == construction_month else 0
                day = first_day + int(event_rng.integers(0, days_in_month - first_day))
                events.append(month + timedelta(days=day))
            accident = int(accident_rng.poisson(_accident_rate(master)))
            months_since_last_event, condition, accident_window = month_transition(
                master,
                event_occurred=event_occurred,
                condition=condition,
                months_since_last_event=months_since_last_event,
                accident_window=accident_window,
                new_accident_count=accident,
            )
            if month >= sim_end and events and events[-1] > final_obs:
                break
            month = _add_months(month, 1)
        for event in events:
            records.append({"segment_id": master.segment_id, "maintenance_date": event})

    frame = pd.DataFrame(records, columns=list(EVENT_COLUMNS))
    frame["maintenance_date"] = pd.to_datetime(frame["maintenance_date"])
    return frame.sort_values(list(EVENT_COLUMNS), kind="stable").reset_index(drop=True)


def _segment_masters(segments: pd.DataFrame, expected: int) -> list[SegmentMaster]:
    missing = [column for column in SEGMENT_COLUMNS if column not in segments.columns]
    if missing:
        raise ValueError(f"segments frame is missing columns: {missing}")
    if len(segments) != expected:
        raise ValueError(f"expected {expected} segments, got {len(segments)}")
    if segments["segment_id"].duplicated().any():
        raise ValueError("segments frame contains duplicate segment_id values")
    records = cast("list[dict[str, Any]]", segments.to_dict(orient="records"))
    return [SegmentMaster.from_record(record) for record in records]


def _check_construction_dates(masters: list[SegmentMaster], start: date) -> None:
    for master in masters:
        if master.construction_date > start:
            raise ValueError(
                f"construction_date {master.construction_date} after observation "
                f"start {start} is not supported for segment {master.segment_id}"
            )


def _simulation_range(
    start: date,
    spec: DatasetSpec,
    pre_period_months: int,
    future_buffer_months: int,
) -> tuple[date, date]:
    final_obs = _add_months(start, spec.dataset_months_per_segment - 1)
    sim_start = _add_months(start, -pre_period_months)
    sim_end = _add_months(final_obs, future_buffer_months)
    return sim_start, sim_end


def _first_sim_month(master: SegmentMaster, sim_start: date) -> date:
    construction_month = date(master.construction_date.year, master.construction_date.month, 1)
    return max(sim_start, construction_month)


def _accident_rate(master: SegmentMaster) -> float:
    return master.accident_propensity * math.sqrt(master.traffic_base / 10_000.0) * 0.05


def _segment_rng(seed: int, segment_id: str, child: int) -> np.random.Generator:
    key = int.from_bytes(segment_id.encode("ascii"), "big")
    stream = np.random.SeedSequence([seed, key]).spawn(2)[child]
    return np.random.default_rng(stream)


def _add_months(value: date, months: int) -> date:
    total = value.year * 12 + (value.month - 1) + months
    year, month_zero = divmod(total, 12)
    month = month_zero + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


__all__ = [
    "ACCIDENT_COLUMNS",
    "ACCIDENT_WINDOW_MONTHS",
    "DEFAULT_BASE_RATE",
    "DEFAULT_FUTURE_BUFFER_MONTHS",
    "DEFAULT_MAX_MONTHS_PER_SEGMENT",
    "DEFAULT_PRE_PERIOD_MONTHS",
    "EVENT_COLUMNS",
    "GenerationError",
    "decay_condition",
    "generate_accident_timeline",
    "generate_maintenance_events",
    "month_transition",
    "monthly_hazard",
    "observation_dates",
]
