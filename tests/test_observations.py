"""Tests for Phase 3 observation-core synthetic generation."""

from __future__ import annotations

import calendar
import dataclasses
from datetime import date, timedelta
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from roadguard import (
    DatasetSpec,
    SegmentMaster,
    generate_accident_timeline,
    generate_maintenance_events,
    generate_observations,
    generate_segments,
    observation_dates,
)
from roadguard.observations import (
    _guardrail_score,
    _heavy_vehicle_ratio,
    _humidity,
    _marking_score,
    _rainfall_mm,
    _road_score,
    _segment_key,
    _sign_score,
    _temperature,
    _traffic_volume,
)
from roadguard.segments import SEGMENT_COLUMNS

SPEC = DatasetSpec(dataset_segments=10, dataset_months_per_segment=12, dataset_observations=120)
ONE_SPEC = DatasetSpec(dataset_segments=1, dataset_months_per_segment=12, dataset_observations=12)
V1_SPEC = DatasetSpec(
    dataset_segments=300, dataset_months_per_segment=48, dataset_observations=14_400
)
START = date(2022, 1, 1)

OBSERVATION_COLUMNS = (
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
INT_COLUMNS = (
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
FLOAT_COLUMNS = (
    "heavy_vehicle_ratio",
    "rainfall_mm",
    "temperature",
    "humidity",
)

BASE_MASTER = SegmentMaster(
    segment_id="QL01-KM10-20",
    province="NA",
    road_type="national",
    construction_date=date(2015, 1, 1),
    road_length_km=10.0,
    traffic_base=10_000,
    heavy_vehicle_ratio_base=0.25,
    weather_exposure=1.0,
    deterioration_rate=0.8,
    accident_propensity=1.0,
    initial_condition=80,
)


def _master_frame(master: SegmentMaster) -> pd.DataFrame:
    return pd.DataFrame([dataclasses.asdict(master)], columns=list(SEGMENT_COLUMNS))


def _frames(spec: DatasetSpec = SPEC, seed: int = 42, start: date = START, **kwargs: object):
    segments = generate_segments(spec, seed, observation_start=start)
    events = generate_maintenance_events(segments, spec, seed, start_date=start, **kwargs)
    timeline = generate_accident_timeline(segments, spec, seed, start_date=start, **kwargs)
    return segments, events, timeline


def _observations(
    spec: DatasetSpec = SPEC,
    seed: int = 42,
    start: date = START,
    **kwargs: object,
):
    segments, events, timeline = _frames(spec, seed, start, **kwargs)
    observations = generate_observations(
        segments, events, timeline, spec, seed, start_date=start, **kwargs
    )
    return segments, events, timeline, observations


def _add_months(value: date, months: int) -> date:
    total = value.year * 12 + (value.month - 1) + months
    year, month_zero = divmod(total, 12)
    month = month_zero + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _month_range(start: date, months: int) -> list[date]:
    return [_add_months(start, i) for i in range(months)]


def _crafted_timeline(
    segment_ids: list[str],
    first_month: date,
    last_month: date,
    counts: dict[tuple[str, date], int] | None = None,
) -> pd.DataFrame:
    counts = counts or {}
    rows = []
    month = first_month
    while month <= last_month:
        for sid in segment_ids:
            rows.append(
                {"segment_id": sid, "month": month, "accident_count": counts.get((sid, month), 0)}
            )
        month = _add_months(month, 1)
    frame = pd.DataFrame(rows)
    frame["month"] = pd.to_datetime(frame["month"])
    return frame


def _empty_events(segment_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"segment_id": pd.Series(dtype=object), "maintenance_date": pd.Series(dtype=object)}
    )


class _TrapDecimal(Decimal):
    def __int__(self) -> int:
        raise AssertionError("__int__ must not be invoked on an over-cap Decimal")


def _expansion_dates(
    seed: int,
    segment_id: str,
    year: int,
    month: int,
    count: int,
    construction_day_offset: int = 0,
) -> list[date]:
    key = int.from_bytes(segment_id.encode("ascii"), "big", signed=False)
    rng = np.random.default_rng(np.random.SeedSequence([seed, key, 0x524733, 3, year, month]))
    days_in_month = calendar.monthrange(year, month)[1]
    offsets = rng.integers(low=construction_day_offset, high=days_in_month, size=count)
    base = date(year, month, 1)
    return sorted(base + timedelta(days=int(offset)) for offset in offsets)


class TestV1Shape:
    def test_exactly_14400_rows(self) -> None:
        _, _, _, observations = _observations(spec=V1_SPEC)
        assert len(observations) == 14_400

    def test_exactly_300_segments_and_48_dates(self) -> None:
        _, _, _, observations = _observations(spec=V1_SPEC)
        assert observations["segment_id"].nunique() == 300
        assert observations["date"].nunique() == 48
        assert (observations["date"].dt.day == 1).all()

    def test_complete_unique_sorted_grid(self) -> None:
        _, _, _, observations = _observations(spec=V1_SPEC)
        keys = list(zip(observations["segment_id"], observations["date"], strict=True))
        assert len(keys) == len(set(keys))
        assert keys == sorted(keys)
        per_segment = observations.groupby("segment_id")["date"].count()
        assert (per_segment == 48).all()


class TestSchemaAndDtypes:
    def test_exact_column_order(self) -> None:
        _, _, _, observations = _observations()
        assert list(observations.columns) == list(OBSERVATION_COLUMNS)

    def test_stable_dtypes(self) -> None:
        _, _, _, observations = _observations()
        assert observations["segment_id"].dtype == object
        assert all(isinstance(value, str) for value in observations["segment_id"])
        assert str(observations["date"].dtype) == "datetime64[ns]"
        for column in INT_COLUMNS:
            assert observations[column].dtype == "int64", column
        for column in FLOAT_COLUMNS:
            assert observations[column].dtype == "float64", column

    def test_no_targets_cost_materials_or_flags(self) -> None:
        _, _, _, observations = _observations()
        assert list(observations.columns) == list(OBSERVATION_COLUMNS)
        forbidden = {
            "maintenance_within_30_days",
            "days_until_maintenance",
            "next_maintenance_date",
            "maintenance_cost",
            "thermoplastic_paint_kg",
            "reflective_sheet_m2",
            "guardrail_meter",
            "traffic_sign_quantity",
        }
        assert not forbidden.intersection(observations.columns)


class TestDeterminism:
    def test_same_seed_identical_output(self) -> None:
        _, _, _, first = _observations()
        _, _, _, second = _observations()
        assert_frame_equal(first, second)

    def test_different_seed_changes_stochastic_not_keys(self) -> None:
        segments, events, timeline, first = _observations(seed=42)
        second = generate_observations(segments, events, timeline, SPEC, 43, start_date=START)
        assert first["segment_id"].equals(second["segment_id"])
        assert first["date"].equals(second["date"])
        assert first["road_age_days"].equals(second["road_age_days"])
        assert not first["traffic_volume"].equals(second["traffic_volume"])
        assert not first["rainfall_mm"].equals(second["rainfall_mm"])
        assert not first["road_condition_score"].equals(second["road_condition_score"])

    def test_segment_streams_are_isolated(self) -> None:
        single_segments, single_events, single_timeline = _frames()
        lone = single_segments.iloc[[0]]
        lone_spec = DatasetSpec(
            dataset_segments=1, dataset_months_per_segment=12, dataset_observations=12
        )
        lone_events = single_events[single_events["segment_id"] == lone.iloc[0]["segment_id"]]
        lone_timeline = single_timeline[single_timeline["segment_id"] == lone.iloc[0]["segment_id"]]
        lone_obs = generate_observations(
            lone, lone_events, lone_timeline, lone_spec, 42, start_date=START
        )
        _, _, _, full_obs = _observations()
        target = full_obs[full_obs["segment_id"] == lone.iloc[0]["segment_id"]]
        assert_frame_equal(lone_obs.reset_index(drop=True), target.reset_index(drop=True))

    def test_phase2_streams_not_altered(self) -> None:
        _, events_before, timeline_before = _frames()
        _, _, _, _ = _observations()
        _, events_after, timeline_after = _frames()
        assert_frame_equal(events_before, events_after)
        assert_frame_equal(timeline_before, timeline_after)


class TestRowOrderIndependence:
    def test_shuffled_segments_identical(self) -> None:
        segments, events, timeline, observations = _observations()
        shuffled = segments.sample(frac=1.0, random_state=7)
        rerun = generate_observations(shuffled, events, timeline, SPEC, 42, start_date=START)
        assert_frame_equal(observations, rerun)

    def test_shuffled_events_identical(self) -> None:
        segments, events, timeline, observations = _observations()
        shuffled = events.sample(frac=1.0, random_state=7)
        rerun = generate_observations(segments, shuffled, timeline, SPEC, 42, start_date=START)
        assert_frame_equal(observations, rerun)

    def test_shuffled_timeline_identical(self) -> None:
        segments, events, timeline, observations = _observations()
        shuffled = timeline.sample(frac=1.0, random_state=7)
        rerun = generate_observations(segments, events, shuffled, SPEC, 42, start_date=START)
        assert_frame_equal(observations, rerun)


class TestNoMutation:
    def test_inputs_not_mutated(self) -> None:
        segments, events, timeline, _ = _observations()
        segments_copy = segments.copy(deep=True)
        events_copy = events.copy(deep=True)
        timeline_copy = timeline.copy(deep=True)
        generate_observations(segments, events, timeline, SPEC, 42, start_date=START)
        assert_frame_equal(segments, segments_copy)
        assert_frame_equal(events, events_copy)
        assert_frame_equal(timeline, timeline_copy)


class TestRanges:
    def test_no_nulls_and_contract_ranges(self) -> None:
        _, _, _, observations = _observations()
        assert not observations.isna().any().any()
        assert (observations["traffic_volume"] >= 0).all()
        assert observations["heavy_vehicle_ratio"].between(0.0, 1.0).all()
        assert (observations["road_age_days"] >= 0).all()
        assert (observations["rainfall_mm"] >= 0).all()
        assert observations["temperature"].between(-50.0, 60.0).all()
        assert observations["humidity"].between(0.0, 100.0).all()
        assert (observations["days_since_last_maintenance"] >= 0).all()
        assert (observations["previous_repairs"] >= 0).all()
        score_columns = (
            "road_condition_score",
            "marking_condition_score",
            "guardrail_condition_score",
            "sign_condition_score",
        )
        for column in score_columns:
            assert observations[column].between(1, 100).all()
        assert (observations["accident_count_30d"] >= 0).all()
        assert (observations["accident_count_365d"] >= 0).all()

    def test_30d_never_exceeds_365d(self) -> None:
        _, _, _, observations = _observations()
        assert (observations["accident_count_30d"] <= observations["accident_count_365d"]).all()


class TestRoadAge:
    def test_road_age_is_exact_date_difference(self) -> None:
        segments, _, _, observations = _observations()
        merged = observations.merge(segments[["segment_id", "construction_date"]], on="segment_id")
        expected = (merged["date"] - pd.to_datetime(merged["construction_date"])).dt.days
        assert (merged["road_age_days"] == expected).all()


class TestMaintenanceHistory:
    def _single(self, master: SegmentMaster, events: pd.DataFrame, timeline: pd.DataFrame):
        spec = DatasetSpec(
            dataset_segments=1, dataset_months_per_segment=12, dataset_observations=12
        )
        segments = _master_frame(master)
        return generate_observations(segments, events, timeline, spec, 42, start_date=START)

    def test_previous_repairs_uses_only_events_before_t(self) -> None:
        events = pd.DataFrame(
            {
                "segment_id": ["QL01-KM10-20"] * 3,
                "maintenance_date": pd.to_datetime(["2021-06-15", "2021-11-10", "2022-03-01"]),
            }
        )
        timeline = generate_accident_timeline(
            _master_frame(BASE_MASTER),
            ONE_SPEC,
            42,
            start_date=START,
        )
        observations = self._single(BASE_MASTER, events, timeline)
        assert (
            observations.loc[observations["date"] == "2022-01-01", "previous_repairs"].iloc[0] == 2
        )
        assert (
            observations.loc[observations["date"] == "2022-04-01", "previous_repairs"].iloc[0] == 3
        )

    def test_days_since_uses_latest_event_before_t(self) -> None:
        events = pd.DataFrame(
            {
                "segment_id": ["QL01-KM10-20"] * 3,
                "maintenance_date": pd.to_datetime(["2021-06-15", "2021-11-10", "2022-03-01"]),
            }
        )
        timeline = generate_accident_timeline(
            _master_frame(BASE_MASTER),
            ONE_SPEC,
            42,
            start_date=START,
        )
        observations = self._single(BASE_MASTER, events, timeline)
        assert (
            observations.loc[
                observations["date"] == "2022-01-01", "days_since_last_maintenance"
            ].iloc[0]
            == (date(2022, 1, 1) - date(2021, 11, 10)).days
        )
        assert (
            observations.loc[
                observations["date"] == "2022-04-01", "days_since_last_maintenance"
            ].iloc[0]
            == (date(2022, 4, 1) - date(2022, 3, 1)).days
        )

    def test_never_maintained_cap_below(self) -> None:
        master = dataclasses.replace(BASE_MASTER, construction_date=date(2021, 1, 15))
        events = _empty_events([master.segment_id])
        timeline = generate_accident_timeline(
            _master_frame(master),
            ONE_SPEC,
            42,
            start_date=START,
        )
        observations = self._single(master, events, timeline)
        row = observations[observations["date"] == "2022-01-01"].iloc[0]
        expected = min((date(2022, 1, 1) - date(2021, 1, 15)).days, 3650)
        assert row["days_since_last_maintenance"] == expected == 351
        assert row["previous_repairs"] == 0

    def test_never_maintained_cap_above(self) -> None:
        master = dataclasses.replace(BASE_MASTER, construction_date=date(2000, 1, 1))
        events = _empty_events([master.segment_id])
        timeline = generate_accident_timeline(
            _master_frame(master),
            ONE_SPEC,
            42,
            start_date=START,
        )
        observations = self._single(master, events, timeline)
        row = observations[observations["date"] == "2022-01-01"].iloc[0]
        assert (date(2022, 1, 1) - date(2000, 1, 1)).days > 3650
        assert row["days_since_last_maintenance"] == 3650

    def test_event_on_t_excluded_at_t_included_at_t1(self) -> None:
        events = pd.DataFrame(
            {
                "segment_id": ["QL01-KM10-20"] * 2,
                "maintenance_date": pd.to_datetime(["2021-12-10", "2022-02-01"]),
            }
        )
        timeline = generate_accident_timeline(
            _master_frame(BASE_MASTER),
            ONE_SPEC,
            42,
            start_date=START,
        )
        observations = self._single(BASE_MASTER, events, timeline)
        at_t = observations[observations["date"] == "2022-02-01"].iloc[0]
        assert at_t["previous_repairs"] == 1
        assert at_t["days_since_last_maintenance"] == (date(2022, 2, 1) - date(2021, 12, 10)).days
        at_next = observations[observations["date"] == "2022-03-01"].iloc[0]
        assert at_next["previous_repairs"] == 2
        assert at_next["days_since_last_maintenance"] == (date(2022, 3, 1) - date(2022, 2, 1)).days


class TestConditionReplay:
    def _single_with_events(self, master: SegmentMaster, event_dates: list[date]):
        events = pd.DataFrame(
            {
                "segment_id": [master.segment_id] * len(event_dates),
                "maintenance_date": pd.to_datetime(event_dates),
            }
        )
        timeline = generate_accident_timeline(
            _master_frame(master),
            ONE_SPEC,
            42,
            start_date=START,
        )
        return self._single(master, events, timeline)

    def _single(self, master: SegmentMaster, events: pd.DataFrame, timeline: pd.DataFrame):
        spec = DatasetSpec(
            dataset_segments=1, dataset_months_per_segment=12, dataset_observations=12
        )
        return generate_observations(
            _master_frame(master), events, timeline, spec, 42, start_date=START
        )

    def test_same_day_maintenance_cannot_improve_condition_at_t(self) -> None:
        fast = dataclasses.replace(BASE_MASTER, deterioration_rate=1.8, initial_condition=80)
        with_event = self._single_with_events(fast, [date(2022, 2, 1)])
        without_event = self._single_with_events(fast, [])
        at_t_with = with_event[with_event["date"] == "2022-02-01"].iloc[0]
        at_t_without = without_event[without_event["date"] == "2022-02-01"].iloc[0]
        assert at_t_with["road_condition_score"] == at_t_without["road_condition_score"]
        at_next_with = with_event[with_event["date"] == "2022-03-01"].iloc[0]
        at_next_without = without_event[without_event["date"] == "2022-03-01"].iloc[0]
        assert at_next_with["road_condition_score"] > at_next_without["road_condition_score"]

    def test_historical_maintenance_improves_following_snapshot(self) -> None:
        fast = dataclasses.replace(BASE_MASTER, deterioration_rate=1.8)
        with_event = self._single_with_events(fast, [date(2022, 1, 10)])
        without_event = self._single_with_events(fast, [])
        at_t_with = with_event[with_event["date"] == "2022-02-01"].iloc[0]
        at_t_without = without_event[without_event["date"] == "2022-02-01"].iloc[0]
        assert at_t_with["road_condition_score"] > at_t_without["road_condition_score"]

    def test_faster_deterioration_worsens_condition(self) -> None:
        fast = dataclasses.replace(BASE_MASTER, deterioration_rate=1.8)
        slow = dataclasses.replace(BASE_MASTER, deterioration_rate=0.3)
        fast_obs = self._single_with_events(fast, [])
        slow_obs = self._single_with_events(slow, [])
        fast_road = fast_obs["road_condition_score"].mean()
        slow_road = slow_obs["road_condition_score"].mean()
        assert fast_road < slow_road


class TestComponentScores:
    def test_correlated_but_not_identical(self) -> None:
        _, _, _, observations = _observations(spec=V1_SPEC)
        score_columns = (
            "road_condition_score",
            "marking_condition_score",
            "guardrail_condition_score",
            "sign_condition_score",
        )
        for i, left in enumerate(score_columns):
            for right in score_columns[i + 1 :]:
                assert observations[left].corr(observations[right]) > 0.5
                assert not observations[left].equals(observations[right])
        assert observations["road_condition_score"].nunique() > 1

    def test_modifiers_drive_distinct_component_behavior(self) -> None:
        assert _road_score(80.0, 0.0) == 80
        assert _marking_score(80.0, 0, 0.0, 0.0) == 76
        assert _guardrail_score(80.0, 0.0, 0, 0.0) == 78
        assert _sign_score(80.0, 60.0, 0.0, 0.0) == 77
        assert _marking_score(80.0, 10_000, 0.0, 0.0) < _marking_score(80.0, 0, 0.0, 0.0)
        assert _guardrail_score(80.0, 0.5, 0, 0.0) < _guardrail_score(80.0, 0.0, 0, 0.0)
        assert _guardrail_score(80.0, 0.0, 30, 0.0) < _guardrail_score(80.0, 0.0, 0, 0.0)
        assert _sign_score(80.0, 90.0, 0.0, 0.0) < _sign_score(80.0, 60.0, 0.0, 0.0)

    def test_accident_modifier_changes_only_guardrail_integration(self) -> None:
        clean_timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
        )
        risky_timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): 25},
        )
        clean = TestAccidentWindows()._single_crafted(timeline=clean_timeline)
        risky = TestAccidentWindows()._single_crafted(timeline=risky_timeline)
        assert not clean["guardrail_condition_score"].equals(risky["guardrail_condition_score"])
        assert clean["road_condition_score"].equals(risky["road_condition_score"])
        assert clean["marking_condition_score"].equals(risky["marking_condition_score"])
        assert clean["sign_condition_score"].equals(risky["sign_condition_score"])


class TestAccidentWindows:
    def _single_crafted(
        self,
        master: SegmentMaster = BASE_MASTER,
        timeline: pd.DataFrame | None = None,
        events: pd.DataFrame | None = None,
        months: int = 12,
        start: date = START,
        pre_period_months: int = 24,
    ):
        if timeline is None:
            timeline = generate_accident_timeline(
                _master_frame(master),
                DatasetSpec(
                    dataset_segments=1,
                    dataset_months_per_segment=months,
                    dataset_observations=months,
                ),
                42,
                start_date=start,
            )
        if events is None:
            events = _empty_events([master.segment_id])
        spec = DatasetSpec(
            dataset_segments=1,
            dataset_months_per_segment=months,
            dataset_observations=months,
        )
        return generate_observations(
            _master_frame(master),
            events,
            timeline,
            spec,
            42,
            start_date=start,
            pre_period_months=pre_period_months,
        )

    def test_monthly_expansion_preserves_every_source_count(self) -> None:
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {
                (BASE_MASTER.segment_id, date(2021, 6, 1)): 5,
                (BASE_MASTER.segment_id, date(2021, 7, 1)): 3,
            },
        )
        observations = self._single_crafted(timeline=timeline)
        at_jan = observations[observations["date"] == "2022-01-01"].iloc[0]
        assert at_jan["accident_count_365d"] == 8
        assert at_jan["accident_count_30d"] == 0
        at_jul = observations[observations["date"] == "2022-07-01"].iloc[0]
        assert at_jul["accident_count_365d"] == 3

    def test_30d_window_is_exact_not_calendar_month(self) -> None:
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {
                (BASE_MASTER.segment_id, date(2022, 1, 1)): 300,
                (BASE_MASTER.segment_id, date(2022, 2, 1)): 4,
            },
        )
        observations = self._single_crafted(timeline=timeline)
        jan_dates = _expansion_dates(42, BASE_MASTER.segment_id, 2022, 1, 300)
        feb_dates = _expansion_dates(42, BASE_MASTER.segment_id, 2022, 2, 4)
        t = date(2022, 3, 1)
        expected = sum(1 for d in jan_dates + feb_dates if t - timedelta(days=30) <= d < t)
        previous_calendar_month_count = len(feb_dates)
        assert expected > previous_calendar_month_count
        actual = observations[observations["date"] == "2022-03-01"].iloc[0]["accident_count_30d"]
        assert actual == expected

    def test_365d_window_leap_year_year_boundary(self) -> None:
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2024, 12, 1),
            {
                (BASE_MASTER.segment_id, date(2023, 2, 1)): 1,
                (BASE_MASTER.segment_id, date(2023, 3, 1)): 1,
                (BASE_MASTER.segment_id, date(2024, 1, 1)): 1,
                (BASE_MASTER.segment_id, date(2024, 2, 1)): 1,
            },
        )
        observations = self._single_crafted(timeline=timeline, months=27)
        t = date(2024, 3, 1)
        assert t - timedelta(days=365) == date(2023, 3, 2)
        assert _add_months(t, -12) == date(2023, 3, 1)
        assert date(2023, 3, 2) != date(2023, 3, 1)
        expected = 0
        for year, month, count in [
            (2023, 2, 1),
            (2023, 3, 1),
            (2024, 1, 1),
            (2024, 2, 1),
        ]:
            dates = _expansion_dates(42, BASE_MASTER.segment_id, year, month, count)
            expected += sum(1 for d in dates if t - timedelta(days=365) <= d < t)
        assert expected == 3
        actual = observations[observations["date"] == "2024-03-01"].iloc[0]["accident_count_365d"]
        assert actual == expected
        expected_30d = sum(
            1
            for d in _expansion_dates(42, BASE_MASTER.segment_id, 2024, 2, 1)
            if t - timedelta(days=30) <= d < t
        )
        actual_30d = observations[observations["date"] == "2024-03-01"].iloc[0][
            "accident_count_30d"
        ]
        assert actual_30d == expected_30d

    def test_accident_on_t_and_future_excluded_at_t(self) -> None:
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {
                (BASE_MASTER.segment_id, date(2022, 1, 1)): 3,
                (BASE_MASTER.segment_id, date(2022, 2, 1)): 2,
                (BASE_MASTER.segment_id, date(2022, 3, 1)): 4,
            },
        )
        observations = self._single_crafted(timeline=timeline)
        jan_dates = _expansion_dates(42, BASE_MASTER.segment_id, 2022, 1, 3)
        feb_dates = _expansion_dates(42, BASE_MASTER.segment_id, 2022, 2, 2)
        mar_dates = _expansion_dates(42, BASE_MASTER.segment_id, 2022, 3, 4)
        at_feb = observations[observations["date"] == "2022-02-01"].iloc[0]
        expected_30d = sum(
            1 for d in jan_dates if date(2022, 2, 1) - timedelta(days=30) <= d < date(2022, 2, 1)
        )
        assert at_feb["accident_count_30d"] == expected_30d
        assert at_feb["accident_count_365d"] == 3
        at_mar = observations[observations["date"] == "2022-03-01"].iloc[0]
        expected_30d_mar = sum(
            1
            for d in jan_dates + feb_dates
            if date(2022, 3, 1) - timedelta(days=30) <= d < date(2022, 3, 1)
        )
        assert at_mar["accident_count_30d"] == expected_30d_mar
        assert at_mar["accident_count_365d"] == 5
        at_apr = observations[observations["date"] == "2022-04-01"].iloc[0]
        assert at_apr["accident_count_365d"] == 9
        assert all(d >= date(2022, 3, 1) for d in mar_dates)

    def test_missing_required_history_raises(self) -> None:
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2021, 6, 1),
            date(2022, 12, 1),
        )
        with pytest.raises(ValueError):
            self._single_crafted(timeline=timeline)

    def test_construction_inside_first_365_days_is_valid(self) -> None:
        master = dataclasses.replace(BASE_MASTER, construction_date=date(2021, 6, 15))
        timeline = _crafted_timeline(
            [master.segment_id],
            date(2021, 6, 1),
            date(2022, 12, 1),
            {(master.segment_id, date(2021, 7, 1)): 2},
        )
        observations = self._single_crafted(master=master, timeline=timeline)
        assert len(observations) == 12
        row = observations[observations["date"] == "2022-01-01"].iloc[0]
        assert row["accident_count_365d"] == 2

    def test_old_road_insufficient_pre_period_fails(self) -> None:
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2021, 7, 1),
            date(2022, 12, 1),
        )
        with pytest.raises(ValueError):
            self._single_crafted(timeline=timeline, months=12, pre_period_months=6)

    def test_recent_construction_short_pre_period_missing_months_fails(self) -> None:
        master = dataclasses.replace(BASE_MASTER, construction_date=date(2021, 6, 15))
        timeline = _crafted_timeline(
            [master.segment_id],
            date(2021, 10, 1),
            date(2022, 12, 1),
        )
        with pytest.raises(ValueError):
            self._single_crafted(master=master, timeline=timeline, months=12, pre_period_months=3)

    def test_final_observation_month_bucket_not_required(self) -> None:
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 11, 1),
            {(BASE_MASTER.segment_id, date(2022, 6, 1)): 3},
        )
        observations = self._single_crafted(timeline=timeline)
        assert len(observations) == 12
        assert observations["accident_count_365d"].iloc[-1] == 3

    def test_gap_in_timeline_raises(self) -> None:
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
        )
        timeline = timeline[timeline["month"] != pd.Timestamp("2021-06-01")]
        with pytest.raises(ValueError):
            self._single_crafted(timeline=timeline)

    def test_non_month_start_timeline_bucket_raises(self) -> None:
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
        )
        timeline = timeline.copy()
        timeline.loc[0, "month"] = pd.Timestamp("2021-06-15")
        with pytest.raises(ValueError):
            self._single_crafted(timeline=timeline)


class TestCausalDrivers:
    def test_traffic_base_increase_raises_expected_traffic(self) -> None:
        low = dataclasses.replace(BASE_MASTER, traffic_base=1_000)
        high = dataclasses.replace(BASE_MASTER, traffic_base=10_000)
        low_obs = _observations_crafted(low)
        high_obs = _observations_crafted(high)
        assert high_obs["traffic_volume"].mean() > 2 * low_obs["traffic_volume"].mean()

    def test_heavy_ratio_centered_on_base(self) -> None:
        low = dataclasses.replace(BASE_MASTER, heavy_vehicle_ratio_base=0.10)
        high = dataclasses.replace(BASE_MASTER, heavy_vehicle_ratio_base=0.40)
        low_obs = _observations_crafted(low)
        high_obs = _observations_crafted(high)
        assert abs(low_obs["heavy_vehicle_ratio"].mean() - 0.10) < 0.01
        assert abs(high_obs["heavy_vehicle_ratio"].mean() - 0.40) < 0.01

    def test_weather_exposure_increases_expected_rainfall(self) -> None:
        dry = dataclasses.replace(BASE_MASTER, weather_exposure=0.6)
        wet = dataclasses.replace(BASE_MASTER, weather_exposure=1.6)
        dry_obs = _observations_crafted(dry)
        wet_obs = _observations_crafted(wet)
        assert wet_obs["rainfall_mm"].mean() > 2 * dry_obs["rainfall_mm"].mean()

    def test_rainfall_humidity_temperature_relationships(self) -> None:
        _, _, _, observations = _observations()
        rainy = observations[observations["date"].dt.month.isin([6, 7, 8, 9])]
        dry_months = observations[observations["date"].dt.month.isin([12, 1, 2, 3])]
        assert rainy["rainfall_mm"].mean() > dry_months["rainfall_mm"].mean()
        warm = observations[observations["date"].dt.month.isin([3, 4, 5])]
        cool = observations[observations["date"].dt.month.isin([9, 10, 11])]
        assert warm["temperature"].mean() > cool["temperature"].mean()
        humidity_corr = observations["humidity"].corr(observations["rainfall_mm"])
        assert humidity_corr > 0.3


def _observations_crafted(master: SegmentMaster) -> pd.DataFrame:
    spec = ONE_SPEC
    segments = _master_frame(master)
    events = _empty_events([master.segment_id])
    timeline = generate_accident_timeline(segments, spec, 42, start_date=START)
    return generate_observations(segments, events, timeline, spec, 42, start_date=START)


class TestInvalidInputs:
    def _base(self):
        return _frames()

    @pytest.mark.parametrize("seed", [0, -1, True, 1.5])
    def test_invalid_seed_rejected(self, seed: object) -> None:
        segments, events, timeline = self._base()
        with pytest.raises(ValueError):
            generate_observations(segments, events, timeline, SPEC, seed, start_date=START)

    def test_wrong_segment_count_rejected(self) -> None:
        segments, events, timeline = self._base()
        wrong = DatasetSpec(
            dataset_segments=9, dataset_months_per_segment=12, dataset_observations=108
        )
        with pytest.raises(ValueError):
            generate_observations(segments, events, timeline, wrong, 42, start_date=START)

    def test_missing_segment_columns_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.drop(columns=["traffic_base"])
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_duplicate_segment_ids_rejected(self) -> None:
        segments, events, timeline = self._base()
        duplicated = segments.copy()
        duplicated.loc[1, "segment_id"] = duplicated.loc[0, "segment_id"]
        with pytest.raises(ValueError):
            generate_observations(duplicated, events, timeline, SPEC, 42, start_date=START)

    def test_unknown_segment_in_events_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad_events = events.copy()
        bad_events.loc[0, "segment_id"] = "QL99-KM999-1000"
        with pytest.raises(ValueError):
            generate_observations(segments, bad_events, timeline, SPEC, 42, start_date=START)

    def test_duplicate_event_key_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad_events = events.copy()
        bad_events = pd.concat([bad_events, bad_events.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError):
            generate_observations(segments, bad_events, timeline, SPEC, 42, start_date=START)

    def test_multiple_events_same_segment_month_rejected(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = pd.DataFrame(
            {
                "segment_id": ["QL01-KM10-20"] * 2,
                "maintenance_date": pd.to_datetime(["2021-03-05", "2021-03-20"]),
            }
        )
        timeline = generate_accident_timeline(
            segments,
            ONE_SPEC,
            42,
            start_date=START,
        )
        with pytest.raises(ValueError):
            generate_observations(
                segments,
                events,
                timeline,
                DatasetSpec(
                    dataset_segments=1, dataset_months_per_segment=12, dataset_observations=12
                ),
                42,
                start_date=START,
            )

    def test_event_before_construction_rejected(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = pd.DataFrame(
            {"segment_id": ["QL01-KM10-20"], "maintenance_date": pd.to_datetime(["2014-06-01"])}
        )
        timeline = generate_accident_timeline(
            segments,
            ONE_SPEC,
            42,
            start_date=START,
        )
        with pytest.raises(ValueError):
            generate_observations(
                segments,
                events,
                timeline,
                DatasetSpec(
                    dataset_segments=1, dataset_months_per_segment=12, dataset_observations=12
                ),
                42,
                start_date=START,
            )

    def test_unknown_segment_in_timeline_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad_timeline = timeline.copy()
        bad_timeline.loc[0, "segment_id"] = "QL99-KM999-1000"
        with pytest.raises(ValueError):
            generate_observations(segments, events, bad_timeline, SPEC, 42, start_date=START)

    def test_duplicate_timeline_month_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad_timeline = pd.concat([timeline, timeline.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError):
            generate_observations(segments, events, bad_timeline, SPEC, 42, start_date=START)

    @pytest.mark.parametrize("count", [-1, 2.5, float("nan"), True, "oops"])
    def test_invalid_accident_count_rejected(self, count: object) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): count},
        )
        with pytest.raises(ValueError):
            generate_observations(
                segments,
                events,
                timeline,
                DatasetSpec(
                    dataset_segments=1, dataset_months_per_segment=12, dataset_observations=12
                ),
                42,
                start_date=START,
            )

    def test_np_bool_accident_count_rejected(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): np.bool_(True)},
        )
        with pytest.raises(ValueError):
            generate_observations(segments, events, timeline, ONE_SPEC, 42, start_date=START)

    @pytest.mark.parametrize(
        "count",
        [1.5, Decimal("1.5"), Fraction(3, 2)],
    )
    def test_fractional_accident_count_rejected(self, count: object) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): count},
        )
        with pytest.raises(ValueError):
            generate_observations(segments, events, timeline, ONE_SPEC, 42, start_date=START)

    def test_large_int_accident_count_not_silently_rounded(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): 2**53 + 1},
        )
        with pytest.raises(ValueError):
            generate_observations(segments, events, timeline, ONE_SPEC, 42, start_date=START)

    def test_max_accidents_per_segment_month_accepted(self) -> None:
        from roadguard.observations import MAX_ACCIDENTS_PER_SEGMENT_MONTH

        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): MAX_ACCIDENTS_PER_SEGMENT_MONTH},
        )
        observations = generate_observations(
            segments, events, timeline, ONE_SPEC, 42, start_date=START
        )
        row = observations[observations["date"] == "2022-01-01"].iloc[0]
        assert row["accident_count_365d"] == MAX_ACCIDENTS_PER_SEGMENT_MONTH

    def test_max_plus_one_accident_count_rejected_before_expansion(self) -> None:
        from roadguard.observations import MAX_ACCIDENTS_PER_SEGMENT_MONTH

        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): MAX_ACCIDENTS_PER_SEGMENT_MONTH + 1},
        )
        with pytest.raises(ValueError, match="MAX_ACCIDENTS_PER_SEGMENT_MONTH"):
            generate_observations(segments, events, timeline, ONE_SPEC, 42, start_date=START)

    def test_accident_count_error_includes_segment_and_month(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): Fraction(3, 2)},
        )
        with pytest.raises(ValueError, match="QL01-KM10-20.*2021-06-01"):
            generate_observations(segments, events, timeline, ONE_SPEC, 42, start_date=START)

    def test_decimal_nan_accident_count_rejected(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): Decimal("NaN")},
        )
        with pytest.raises(ValueError):
            generate_observations(segments, events, timeline, ONE_SPEC, 42, start_date=START)

    def test_negative_decimal_accident_count_rejected(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): Decimal("-5")},
        )
        with pytest.raises(ValueError):
            generate_observations(segments, events, timeline, ONE_SPEC, 42, start_date=START)

    def test_decimal_integral_accident_count_accepted(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): Decimal("5")},
        )
        observations = generate_observations(
            segments, events, timeline, ONE_SPEC, 42, start_date=START
        )
        row = observations[observations["date"] == "2022-01-01"].iloc[0]
        assert row["accident_count_365d"] == 5

    def test_fraction_integral_accident_count_accepted(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): Fraction(5, 1)},
        )
        observations = generate_observations(
            segments, events, timeline, ONE_SPEC, 42, start_date=START
        )
        row = observations[observations["date"] == "2022-01-01"].iloc[0]
        assert row["accident_count_365d"] == 5

    def test_unsupported_accident_count_rejected(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): [1]},
        )
        with pytest.raises(ValueError):
            generate_observations(segments, events, timeline, ONE_SPEC, 42, start_date=START)

    def test_over_cap_trap_decimal_rejected_before_int_conversion(self) -> None:
        from roadguard.observations import MAX_ACCIDENTS_PER_SEGMENT_MONTH

        count = _TrapDecimal(MAX_ACCIDENTS_PER_SEGMENT_MONTH + 1)
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            {(BASE_MASTER.segment_id, date(2021, 6, 1)): count},
        )
        with pytest.raises(ValueError, match="MAX_ACCIDENTS_PER_SEGMENT_MONTH"):
            generate_observations(segments, events, timeline, ONE_SPEC, 42, start_date=START)

    def test_accident_bucket_before_construction_rejected(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2014, 1, 1),
            date(2022, 12, 1),
        )
        with pytest.raises(ValueError):
            generate_observations(
                segments,
                events,
                timeline,
                DatasetSpec(
                    dataset_segments=1, dataset_months_per_segment=12, dataset_observations=12
                ),
                42,
                start_date=START,
            )

    def test_observation_start_not_first_of_month_rejected(self) -> None:
        segments, events, timeline = self._base()
        with pytest.raises(ValueError):
            generate_observations(
                segments, events, timeline, SPEC, 42, start_date=date(2022, 1, 15)
            )

    def test_construction_after_observation_start_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad.loc[0, "construction_date"] = pd.Timestamp("2022-06-01")
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_non_finite_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad.loc[0, "weather_exposure"] = float("nan")
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_boolean_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = True
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_string_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["weather_exposure"] = bad["weather_exposure"].astype(object)
        bad.loc[0, "weather_exposure"] = "1.0"
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_none_float_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["weather_exposure"] = bad["weather_exposure"].astype(object)
        bad.loc[0, "weather_exposure"] = None
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_none_integer_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = None
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_nan_integer_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = float("nan")
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_nan_via_non_float_integer_latent_rejected(self) -> None:
        from decimal import Decimal

        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = Decimal("NaN")
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_integral_float_integer_latent_accepted(self) -> None:
        segments, events, timeline = self._base()
        adjusted = segments.copy()
        adjusted["traffic_base"] = adjusted["traffic_base"].astype(object)
        adjusted.loc[0, "traffic_base"] = 1500.0
        observations = generate_observations(adjusted, events, timeline, SPEC, 42, start_date=START)
        assert (observations["traffic_volume"] >= 0).all()

    def test_np_float32_fractional_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = np.float32(1500.5)
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_decimal_fractional_latent_rejected(self) -> None:
        from decimal import Decimal

        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = Decimal("1500.5")
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_np_bool_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = np.bool_(True)
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_np_int64_max_latent_not_falsely_overflowed(self) -> None:
        from roadguard.observations import _validate_int_latent_column

        frame = _master_frame(BASE_MASTER)
        frame["traffic_base"] = frame["traffic_base"].astype(object)
        frame.loc[0, "traffic_base"] = np.int64(np.iinfo(np.int64).max)
        _validate_int_latent_column(frame, "traffic_base")

    def test_decimal_overflow_latent_rejected(self) -> None:
        from decimal import Decimal

        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = Decimal(10) ** 40
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_fraction_overflow_latent_rejected(self) -> None:
        from fractions import Fraction

        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = Fraction(2**63, 1)
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_fraction_three_halves_latent_rejected(self) -> None:
        from fractions import Fraction

        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = Fraction(3, 2)
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_string_integer_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = "5000"
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_float_integer_latent_overflow_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = 2.0**63
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_decimal_integral_latent_accepted(self) -> None:
        from decimal import Decimal

        segments, events, timeline = self._base()
        adjusted = segments.copy()
        adjusted["traffic_base"] = adjusted["traffic_base"].astype(object)
        adjusted.loc[0, "traffic_base"] = Decimal("1500")
        observations = generate_observations(adjusted, events, timeline, SPEC, 42, start_date=START)
        assert (observations["traffic_volume"] >= 0).all()

    def test_fraction_integral_latent_accepted(self) -> None:
        from fractions import Fraction

        segments, events, timeline = self._base()
        adjusted = segments.copy()
        adjusted["traffic_base"] = adjusted["traffic_base"].astype(object)
        adjusted.loc[0, "traffic_base"] = Fraction(1500, 1)
        observations = generate_observations(adjusted, events, timeline, SPEC, 42, start_date=START)
        assert (observations["traffic_volume"] >= 0).all()

    def test_derived_validation_guards(self) -> None:
        from roadguard.observations import (
            _FLOAT_OUTPUT_COLUMNS,
            _INT_OUTPUT_COLUMNS,
            _validate_derived_values,
        )

        def _frame() -> pd.DataFrame:
            data = {column: [1] for column in _INT_OUTPUT_COLUMNS}
            data.update({column: [1.0] for column in _FLOAT_OUTPUT_COLUMNS})
            return pd.DataFrame(data)

        bad_int = _frame()
        bad_int["traffic_volume"] = bad_int["traffic_volume"].astype(object)
        bad_int.loc[0, "traffic_volume"] = "x"
        with pytest.raises(ValueError):
            _validate_derived_values(bad_int)
        bad_float = _frame()
        bad_float["humidity"] = bad_float["humidity"].astype(object)
        bad_float.loc[0, "humidity"] = float("inf")
        with pytest.raises(ValueError):
            _validate_derived_values(bad_float)
        bad_float = _frame()
        bad_float["rainfall_mm"] = bad_float["rainfall_mm"].astype(object)
        bad_float.loc[0, "rainfall_mm"] = "x"
        with pytest.raises(ValueError):
            _validate_derived_values(bad_float)
        _validate_derived_values(_frame())

    def test_segment_key_rejects_non_ascii(self) -> None:
        with pytest.raises(ValueError):
            _segment_key("QL01-KM134-135é")

    def test_fractional_integer_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = 1500.5
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_int64_overflow_latent_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = 2**63
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_missing_segment_id_column_rejected(self) -> None:
        segments, events, timeline = self._base()
        with pytest.raises(ValueError):
            generate_observations(
                segments.drop(columns=["segment_id"]), events, timeline, SPEC, 42, start_date=START
            )

    def test_nat_maintenance_date_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad_events = events.copy()
        bad_events["maintenance_date"] = bad_events["maintenance_date"].astype(object)
        bad_events.loc[0, "maintenance_date"] = pd.NaT
        with pytest.raises(ValueError):
            generate_observations(segments, bad_events, timeline, SPEC, 42, start_date=START)

    def test_malformed_maintenance_date_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad_events = events.copy()
        bad_events["maintenance_date"] = bad_events["maintenance_date"].astype(object)
        bad_events.loc[0, "maintenance_date"] = "not-a-date"
        with pytest.raises(ValueError):
            generate_observations(segments, bad_events, timeline, SPEC, 42, start_date=START)

    def test_malformed_timeline_month_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad_timeline = timeline.copy()
        bad_timeline["month"] = bad_timeline["month"].astype(object)
        bad_timeline.loc[0, "month"] = "not-a-date"
        with pytest.raises(ValueError):
            generate_observations(segments, events, bad_timeline, SPEC, 42, start_date=START)

    @pytest.mark.parametrize("value", [True, 1.5, "3", 0])
    def test_invalid_pre_period_months_rejected(self, value: object) -> None:
        segments, events, timeline = self._base()
        with pytest.raises(ValueError):
            generate_observations(
                segments, events, timeline, SPEC, 42, start_date=START, pre_period_months=value
            )

    def test_derived_traffic_overflow_raises(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad["traffic_base"] = bad["traffic_base"].astype(object)
        bad.loc[0, "traffic_base"] = 9_000_000_000_000_000_000
        with pytest.raises(ValueError, match="traffic_volume"):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_derived_non_finite_weather_raises(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad.loc[0, "weather_exposure"] = np.finfo(np.float64).max
        with pytest.raises(ValueError, match="rainfall_mm"):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_non_ascii_segment_id_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad.loc[0, "segment_id"] = "QL01-KM134-135é"
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    @pytest.mark.parametrize(
        "segment_id",
        [
            "QL01KM134-135",
            "ql01-KM134-135",
            "QL1-KM134-135",
            "QL01-KM134135",
            "ZZ01-KM134-135",
            "QL01-KM134-135-extra",
        ],
    )
    def test_malformed_segment_id_rejected(self, segment_id: str) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad.loc[0, "segment_id"] = segment_id
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_rng_keys_are_collision_free(self) -> None:
        segments = generate_segments(V1_SPEC, 42, observation_start=START)
        keys = [_segment_key(sid) for sid in segments["segment_id"]]
        assert len(keys) == len(set(keys)) == 300

    def test_empty_segment_id_rejected(self) -> None:
        segments, events, timeline = self._base()
        bad = segments.copy()
        bad.loc[0, "segment_id"] = ""
        with pytest.raises(ValueError):
            generate_observations(bad, events, timeline, SPEC, 42, start_date=START)

    def test_pre_period_months_must_be_positive(self) -> None:
        segments, events, timeline = self._base()
        with pytest.raises(ValueError):
            generate_observations(
                segments, events, timeline, SPEC, 42, start_date=START, pre_period_months=0
            )

    def test_maintenance_missing_columns_rejected(self) -> None:
        segments, events, timeline = self._base()
        with pytest.raises(ValueError):
            generate_observations(
                segments, events[["segment_id"]], timeline, SPEC, 42, start_date=START
            )

    def test_timeline_missing_columns_rejected(self) -> None:
        segments, events, timeline = self._base()
        with pytest.raises(ValueError):
            generate_observations(
                segments, events, timeline[["segment_id", "month"]], SPEC, 42, start_date=START
            )


class TestExpansionScope:
    def _craft(self, counts: dict[tuple[str, date], object]) -> pd.DataFrame:
        return _crafted_timeline(
            [BASE_MASTER.segment_id],
            date(2020, 1, 1),
            date(2022, 12, 1),
            counts,
        )

    def test_irrelevant_old_bucket_not_expanded_and_no_effect(self) -> None:
        from roadguard.observations import _expand_accidents, _segment_key

        old_timeline = self._craft({(BASE_MASTER.segment_id, date(2020, 3, 1)): 10_000})
        base_timeline = self._craft({})
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        with_old = generate_observations(
            segments, events, old_timeline, ONE_SPEC, 42, start_date=START
        )
        without_old = generate_observations(
            segments, events, base_timeline, ONE_SPEC, 42, start_date=START
        )
        assert_frame_equal(with_old, without_old)
        key = _segment_key(BASE_MASTER.segment_id)
        months = {
            date(2020, 3, 1): 10_000,
            date(2021, 6, 1): 0,
            date(2022, 11, 1): 0,
        }
        dates = _expand_accidents(BASE_MASTER, months, 42, key, date(2022, 12, 1), date(2022, 1, 1))
        assert len(dates) == 0

    def test_floor_month_bucket_with_boundary_occurrence_eligible(self) -> None:
        timeline = self._craft({(BASE_MASTER.segment_id, date(2021, 1, 1)): 300})
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        observations = generate_observations(
            segments, events, timeline, ONE_SPEC, 42, start_date=START
        )
        jan_dates = _expansion_dates(42, BASE_MASTER.segment_id, 2021, 1, 300)
        boundary = date(2022, 1, 1) - timedelta(days=365)
        assert boundary == date(2021, 1, 1)
        assert boundary in jan_dates
        expected = sum(1 for d in jan_dates if boundary <= d < date(2022, 1, 1))
        row = observations[observations["date"] == "2022-01-01"].iloc[0]
        assert row["accident_count_365d"] == expected


class TestPrefixInvariance:
    def test_altering_future_maintenance_events_keeps_prefix_identical(self) -> None:
        segments, events, timeline, observations = _observations(spec=V1_SPEC)
        cutoff = date(2022, 7, 1)
        future_events = events[events["maintenance_date"] >= pd.Timestamp(cutoff)]
        assert len(future_events) > 0
        trimmed = events[events["maintenance_date"] < pd.Timestamp(cutoff)]
        assert len(trimmed) < len(events)
        rerun = generate_observations(segments, trimmed, timeline, V1_SPEC, 42, start_date=START)
        prefix = observations[observations["date"] <= pd.Timestamp(cutoff)].reset_index(drop=True)
        prefix_rerun = rerun[rerun["date"] <= pd.Timestamp(cutoff)].reset_index(drop=True)
        assert_frame_equal(prefix, prefix_rerun)

    def test_altering_future_accident_buckets_keeps_prefix_identical(self) -> None:
        segments, events, timeline, observations = _observations()
        cutoff = date(2022, 7, 1)
        altered = timeline.copy()
        future = altered["month"] >= pd.Timestamp(cutoff)
        altered.loc[future, "accident_count"] = altered.loc[future, "accident_count"] + 7
        rerun = generate_observations(segments, events, altered, SPEC, 42, start_date=START)
        prefix = observations[observations["date"] <= pd.Timestamp(cutoff)].reset_index(drop=True)
        prefix_rerun = rerun[rerun["date"] <= pd.Timestamp(cutoff)].reset_index(drop=True)
        assert_frame_equal(prefix, prefix_rerun)
        suffix = observations[observations["date"] > pd.Timestamp(cutoff)]
        suffix_rerun = rerun[rerun["date"] > pd.Timestamp(cutoff)]
        assert not suffix.equals(suffix_rerun)


class TestObservationCalendar:
    def test_all_dates_are_first_of_month_observations(self) -> None:
        _, _, _, observations = _observations(spec=V1_SPEC)
        expected = set(observation_dates(48, START))
        assert set(observations["date"]) == set(expected)


class TestFormulaContract:
    def test_traffic_monthly_trend(self) -> None:
        assert _traffic_volume(10_000, 0, 1, 0.0) == 10_000
        assert _traffic_volume(10_000, 12, 1, 0.0) == 10_360
        assert _traffic_volume(10_000, 24, 1, 0.0) == 10_720

    def test_traffic_seasonality_and_phase(self) -> None:
        assert _traffic_volume(10_000, 0, 4, 0.0) == 10_800
        assert _traffic_volume(10_000, 0, 10, 0.0) == 9_200
        assert _traffic_volume(10_000, 0, 1, 0.0) == 10_000
        assert _traffic_volume(10_000, 0, 7, 0.0) == 10_000

    def test_heavy_vehicle_formula(self) -> None:
        assert _heavy_vehicle_ratio(0.25, 2, 0.0) == 0.25
        assert _heavy_vehicle_ratio(0.25, 5, 0.0) == 0.27
        assert _heavy_vehicle_ratio(0.25, 11, 0.0) == 0.23
        assert _heavy_vehicle_ratio(0.0, 2, -0.5) == 0.0
        assert _heavy_vehicle_ratio(0.99, 2, 0.5) == 1.0

    def test_rainfall_formula(self) -> None:
        assert _rainfall_mm(1.0, 5, 0.0) == 140.0
        assert _rainfall_mm(1.0, 8, 0.0) == 231.0
        assert _rainfall_mm(1.6, 5, 0.0) == 224.0

    def test_temperature_formula(self) -> None:
        assert _temperature(1.0, 1, 0.0) == 27.0
        assert _temperature(1.0, 4, 0.0) == 31.0
        assert _temperature(1.0, 10, 0.0) == 23.0
        assert _temperature(1.6, 1, 0.0) == 26.5
        assert _temperature(1.0, 4, 100.0) == 60.0
        assert _temperature(1.0, 10, -100.0) == -50.0

    def test_humidity_formula_and_positive_rainfall_coefficient(self) -> None:
        assert _humidity(1.0, 140.0, 0.0) == 70.5
        assert _humidity(1.0, 280.0, 0.0) == 81.0
        assert _humidity(1.0, 0.0, 0.0) == 60.0
        assert _humidity(1.6, 0.0, 0.0) == 64.2
        assert _humidity(1.0, 5000.0, 0.0) == 100.0
        assert _humidity(1.0, 140.0, 0.0) < _humidity(1.0, 280.0, 0.0)

    def test_condition_score_formulas(self) -> None:
        assert _road_score(80.0, 0.0) == 80
        assert _road_score(100.5, 0.0) == 100
        assert _road_score(0.2, 0.0) == 1
        assert _marking_score(80.0, 0, 0.0, 0.0) == 76
        assert _guardrail_score(80.0, 0.25, 0, 0.0) == 76
        assert _guardrail_score(80.0, 0.25, 10, 0.0) == 71
        assert _sign_score(80.0, 60.0, 0.0, 0.0) == 77
        assert _sign_score(80.0, 80.0, 0.0, 0.0) == 76

    def test_marking_rainfall_term_crosses_rounding_boundary(self) -> None:
        assert _marking_score(80.0, 0, 100.0, 0.0) == 75
        assert _marking_score(80.0, 0, 200.0, 0.0) == 74
        assert _marking_score(80.0, 0, 100.0, 0.0) > _marking_score(80.0, 0, 200.0, 0.0)

    def test_sign_rainfall_term_crosses_rounding_boundary(self) -> None:
        assert _sign_score(80.0, 60.0, 300.0, 0.0) == 76
        assert _sign_score(80.0, 60.0, 400.0, 0.0) == 75
        assert _sign_score(80.0, 60.0, 300.0, 0.0) > _sign_score(80.0, 60.0, 400.0, 0.0)


class TestRngDrawOrder:
    def test_draw_order_is_traffic_heavy_then_weather_then_condition(self) -> None:
        segments = _master_frame(BASE_MASTER)
        events = _empty_events([BASE_MASTER.segment_id])
        timeline = generate_accident_timeline(segments, ONE_SPEC, 42, start_date=START)
        observations = generate_observations(
            segments, events, timeline, ONE_SPEC, 42, start_date=START
        )

        key = int.from_bytes(BASE_MASTER.segment_id.encode("ascii"), "big", signed=False)
        traffic_rng = np.random.default_rng(np.random.SeedSequence([42, key, 0x524733, 0]))
        weather_rng = np.random.default_rng(np.random.SeedSequence([42, key, 0x524733, 1]))
        condition_rng = np.random.default_rng(np.random.SeedSequence([42, key, 0x524733, 2]))
        first_month = date(2020, 1, 1)
        for k, row in enumerate(observations.itertuples(index=False)):
            traffic_noise = traffic_rng.normal(-0.5 * 0.06**2, 0.06)
            heavy_noise = traffic_rng.normal(0.0, 0.012)
            rain_noise = weather_rng.normal(-0.5 * 0.22**2, 0.22)
            temperature_noise = weather_rng.normal(0.0, 0.8)
            humidity_noise = weather_rng.normal(0.0, 2.0)
            road_noise = condition_rng.normal(0.0, 1.25)
            marking_noise = condition_rng.normal(0.0, 1.5)
            guardrail_noise = condition_rng.normal(0.0, 1.5)
            sign_noise = condition_rng.normal(0.0, 1.5)
            m = row.date.month
            assert row.traffic_volume == _traffic_volume(
                BASE_MASTER.traffic_base, k, m, traffic_noise
            )
            assert row.heavy_vehicle_ratio == _heavy_vehicle_ratio(
                BASE_MASTER.heavy_vehicle_ratio_base, m, heavy_noise
            )
            assert row.rainfall_mm == _rainfall_mm(BASE_MASTER.weather_exposure, m, rain_noise)
            assert row.temperature == _temperature(
                BASE_MASTER.weather_exposure, m, temperature_noise
            )
            assert row.humidity == _humidity(
                BASE_MASTER.weather_exposure, row.rainfall_mm, humidity_noise
            )
            t = row.date.date()
            months_since_first = (t.year * 12 + t.month) - (
                first_month.year * 12 + first_month.month
            )
            condition = float(BASE_MASTER.initial_condition) - 0.6 * (
                BASE_MASTER.deterioration_rate * months_since_first
            )
            assert row.road_condition_score == _road_score(condition, road_noise)
            assert row.marking_condition_score == _marking_score(
                condition, row.traffic_volume, row.rainfall_mm, marking_noise
            )
            assert row.guardrail_condition_score == _guardrail_score(
                condition, row.heavy_vehicle_ratio, row.accident_count_365d, guardrail_noise
            )
            assert row.sign_condition_score == _sign_score(
                condition, row.humidity, row.rainfall_mm, sign_noise
            )
