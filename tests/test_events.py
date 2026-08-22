"""Tests for the maintenance-event simulation engine."""

from __future__ import annotations

import dataclasses
from datetime import date

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from roadguard import (
    DatasetSpec,
    SegmentMaster,
    days_until_maintenance,
    decay_condition,
    generate_accident_timeline,
    generate_maintenance_events,
    generate_segments,
    maintenance_within_30_days,
    month_transition,
    monthly_hazard,
    observation_dates,
)
from roadguard.events import GenerationError
from roadguard.segments import SEGMENT_COLUMNS

SPEC = DatasetSpec(dataset_segments=60, dataset_months_per_segment=24, dataset_observations=1_440)
V1_SPEC = DatasetSpec(
    dataset_segments=300, dataset_months_per_segment=48, dataset_observations=14_400
)
START = date(2021, 1, 1)

BASE_MASTER = SegmentMaster(
    segment_id="QL01-KM10-20",
    province="NA",
    road_type="national",
    construction_date=date(2005, 1, 1),
    road_length_km=10.0,
    traffic_base=10_000,
    heavy_vehicle_ratio_base=0.25,
    weather_exposure=1.0,
    deterioration_rate=0.8,
    accident_propensity=1.0,
    initial_condition=80,
)
HAZARD_DATE = date(2021, 6, 1)


def _frames(seed: int = 42, spec: DatasetSpec = SPEC, **kwargs: object):
    segments = generate_segments(spec, seed)
    events = generate_maintenance_events(segments, spec, seed, start_date=START, **kwargs)
    return segments, events


def _master_frame(master: SegmentMaster) -> pd.DataFrame:
    return pd.DataFrame([dataclasses.asdict(master)], columns=list(SEGMENT_COLUMNS))


class TestCalendar:
    def test_monthly_cadence(self) -> None:
        dates = observation_dates(24, START)
        assert len(dates) == 24
        assert dates[0].date() == START
        assert (dates.day == 1).all()
        previous = dates[:-1]
        following = dates[1:]
        assert (following.month == (previous.month % 12) + 1).all()
        assert (following.year == previous.year + (previous.month == 12)).all()

    def test_last_observation_date(self) -> None:
        dates = observation_dates(24, START)
        assert dates[-1].date() == date(2022, 12, 1)

    def test_rejects_non_positive_month_count(self) -> None:
        with pytest.raises(ValueError):
            observation_dates(0, START)


class TestDeterminism:
    def test_same_seed_gives_identical_events(self) -> None:
        segments, events = _frames()
        segments_b, events_b = _frames()
        assert_frame_equal(segments, segments_b)
        assert_frame_equal(events, events_b)

    def test_different_seeds_change_stochastic_values(self) -> None:
        _, events_a = _frames(seed=42)
        _, events_b = _frames(seed=43)
        assert not events_a.equals(events_b)
        assert not events_a["maintenance_date"].equals(events_b["maintenance_date"])


class TestEventStructure:
    def test_columns_are_key_only(self) -> None:
        _, events = _frames()
        assert list(events.columns) == ["segment_id", "maintenance_date"]

    def test_events_are_chronologically_ordered_per_segment(self) -> None:
        _, events = _frames()
        for _, group in events.groupby("segment_id", sort=False):
            dates = group["maintenance_date"].tolist()
            assert dates == sorted(dates)

    def test_no_duplicate_segment_date_keys(self) -> None:
        _, events = _frames()
        keys = list(zip(events["segment_id"], events["maintenance_date"], strict=True))
        assert len(keys) == len(set(keys))

    def test_events_cover_observation_period(self) -> None:
        _, events = _frames()
        assert (events["maintenance_date"] >= pd.Timestamp(START)).any()
        assert len(events) > len(events["segment_id"].unique())

    def test_no_targets_or_phase3_artifacts(self) -> None:
        _, events = _frames()
        assert list(events.columns) == ["segment_id", "maintenance_date"]


class TestTailEvents:
    def test_every_segment_has_event_after_final_observation(self) -> None:
        segments, events = _frames()
        final_obs = observation_dates(SPEC.dataset_months_per_segment, START)[-1]
        max_dates = events.groupby("segment_id")["maintenance_date"].max()
        assert set(max_dates.index) == set(segments["segment_id"])
        assert (max_dates > pd.Timestamp(final_obs)).all()

    def test_every_v1_segment_has_event_after_final_observation(self) -> None:
        segments, events = _frames(spec=V1_SPEC)
        final_obs = observation_dates(V1_SPEC.dataset_months_per_segment, START)[-1]
        max_dates = events.groupby("segment_id")["maintenance_date"].max()
        assert len(segments) == 300
        assert set(max_dates.index) == set(segments["segment_id"])
        assert (max_dates > pd.Timestamp(final_obs)).all()

    def test_tail_guarantee_with_very_low_hazard(self) -> None:
        segments, events = _frames(base_rate=0.02, future_buffer_months=6)
        final_obs = observation_dates(SPEC.dataset_months_per_segment, START)[-1]
        max_dates = events.groupby("segment_id")["maintenance_date"].max()
        assert set(max_dates.index) == set(segments["segment_id"])
        assert (max_dates > pd.Timestamp(final_obs)).all()

    def test_low_base_rate_really_low_hazard(self) -> None:
        _, low = _frames(base_rate=0.02)
        _, high = _frames(base_rate=0.5)
        final_obs = observation_dates(SPEC.dataset_months_per_segment, START)[-1]
        window = pd.Timestamp(final_obs)
        counts_low = low[low["maintenance_date"] <= window].groupby("segment_id").size()
        counts_high = high[high["maintenance_date"] <= window].groupby("segment_id").size()
        assert counts_low.sum() < counts_high.sum()
        assert counts_low.mean() < counts_high.mean()

    def test_base_rate_changes_deterministic_event_history(self) -> None:
        _, low = _frames(base_rate=0.02)
        _, high = _frames(base_rate=0.5)
        assert not low.equals(high)
        assert len(low) < len(high)

    def test_safety_cap_fails_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("roadguard.events.monthly_hazard", lambda *args: 0.0)
        segments = generate_segments(SPEC, 42)
        with pytest.raises(GenerationError):
            generate_maintenance_events(
                segments,
                SPEC,
                42,
                start_date=START,
                max_months_per_segment=6,
            )


class TestFutureBufferSemantics:
    def _buffer_frames(self, buffer: int):
        return _frames(future_buffer_months=buffer)

    def test_longer_buffer_extends_horizon(self) -> None:
        _, short = self._buffer_frames(6)
        _, long = self._buffer_frames(24)
        assert not short.equals(long)
        final_obs = observation_dates(SPEC.dataset_months_per_segment, START)[-1]
        sim_end_short = final_obs + pd.DateOffset(months=6)
        extended = long[long["maintenance_date"] > sim_end_short]
        assert len(extended) > 0
        max_short = short.groupby("segment_id")["maintenance_date"].max()
        max_long = long.groupby("segment_id")["maintenance_date"].max()
        assert (max_long >= max_short).all()

    def test_shorter_buffer_history_is_prefix_of_longer(self) -> None:
        _, short = self._buffer_frames(6)
        _, long = self._buffer_frames(24)
        short_by_segment = {
            sid: group["maintenance_date"].tolist()
            for sid, group in short.groupby("segment_id", sort=False)
        }
        long_by_segment = {
            sid: group["maintenance_date"].tolist()
            for sid, group in long.groupby("segment_id", sort=False)
        }
        assert set(short_by_segment) == set(long_by_segment)
        for sid, short_dates in short_by_segment.items():
            long_dates = long_by_segment[sid]
            assert set(short_dates) <= set(long_dates)
            assert short_dates == [d for d in long_dates if d <= short_dates[-1]]
        short_total = sum(len(dates) for dates in short_by_segment.values())
        long_total = sum(len(dates) for dates in long_by_segment.values())
        assert long_total > short_total

    def test_every_segment_has_post_final_event_for_any_buffer(self) -> None:
        for buffer in (6, 24):
            segments, events = self._buffer_frames(buffer)
            final_obs = observation_dates(SPEC.dataset_months_per_segment, START)[-1]
            max_dates = events.groupby("segment_id")["maintenance_date"].max()
            assert (max_dates > pd.Timestamp(final_obs)).all()

    def test_accident_timeline_extends_with_buffer(self) -> None:
        segments = generate_segments(SPEC, 42)
        short = generate_accident_timeline(
            segments, SPEC, 42, start_date=START, future_buffer_months=6
        )
        long = generate_accident_timeline(
            segments, SPEC, 42, start_date=START, future_buffer_months=24
        )
        final_obs = observation_dates(SPEC.dataset_months_per_segment, START)[-1]
        sim_end_short = final_obs + pd.DateOffset(months=6)
        long_prefix = long[long["month"] <= sim_end_short]
        assert_frame_equal(short, long_prefix.reset_index(drop=True))


class TestAccidentTimeline:
    def test_structure(self) -> None:
        segments = generate_segments(SPEC, 42)
        timeline = generate_accident_timeline(segments, SPEC, 42, start_date=START)
        assert list(timeline.columns) == ["segment_id", "month", "accident_count"]
        assert (timeline["accident_count"] >= 0).all()

    def test_same_seed_identical(self) -> None:
        segments = generate_segments(SPEC, 42)
        first = generate_accident_timeline(segments, SPEC, 42, start_date=START)
        second = generate_accident_timeline(segments, SPEC, 42, start_date=START)
        assert_frame_equal(first, second)

    def test_different_seeds_differ(self) -> None:
        segments = generate_segments(SPEC, 42)
        first = generate_accident_timeline(segments, SPEC, 42, start_date=START)
        second = generate_accident_timeline(segments, SPEC, 43, start_date=START)
        assert not first.equals(second)

    def test_covers_observation_window_and_history(self) -> None:
        segments = generate_segments(SPEC, 42)
        timeline = generate_accident_timeline(segments, SPEC, 42, start_date=START)
        months = timeline["month"].unique()
        assert pd.Timestamp(date(2020, 1, 1)) in months
        assert pd.Timestamp(date(2022, 12, 1)) in months

    def test_shuffled_rows_identical_timeline(self) -> None:
        segments = generate_segments(SPEC, 42)
        shuffled = segments.sample(frac=1.0, random_state=7)
        first = generate_accident_timeline(segments, SPEC, 42, start_date=START)
        second = generate_accident_timeline(shuffled, SPEC, 42, start_date=START)
        assert_frame_equal(first, second)

    def test_validation(self) -> None:
        segments = generate_segments(SPEC, 42)
        with pytest.raises(ValueError):
            generate_accident_timeline(segments, SPEC, 0, start_date=START)
        with pytest.raises(ValueError):
            generate_accident_timeline(segments, SPEC, 42, start_date=START, pre_period_months=0)
        with pytest.raises(ValueError):
            generate_accident_timeline(segments, SPEC, 42, start_date=START, future_buffer_months=0)


class TestRowOrderIndependence:
    def test_shuffled_segment_rows_give_identical_events(self) -> None:
        segments = generate_segments(SPEC, 42)
        events_original = generate_maintenance_events(segments, SPEC, 42, start_date=START)
        shuffled = segments.sample(frac=1.0, random_state=7)
        events_shuffled = generate_maintenance_events(shuffled, SPEC, 42, start_date=START)
        assert_frame_equal(events_original, events_shuffled)

    def test_reversed_segment_rows_give_identical_events(self) -> None:
        segments = generate_segments(SPEC, 42)
        events_original = generate_maintenance_events(segments, SPEC, 42, start_date=START)
        reversed_frame = segments.iloc[::-1].reset_index(drop=True)
        events_reversed = generate_maintenance_events(reversed_frame, SPEC, 42, start_date=START)
        assert_frame_equal(events_original, events_reversed)


class TestConstructionInvariants:
    def test_construction_after_observation_start_rejected(self) -> None:
        master = dataclasses.replace(BASE_MASTER, construction_date=date(2022, 6, 15))
        frame = _master_frame(master)
        one = DatasetSpec(
            dataset_segments=1, dataset_months_per_segment=24, dataset_observations=24
        )
        with pytest.raises(ValueError):
            generate_maintenance_events(frame, one, 42, start_date=START)

    def test_events_never_precede_construction_date(self) -> None:
        segments, events = _frames()
        merged = events.merge(segments[["segment_id", "construction_date"]], on="segment_id")
        constructions = pd.to_datetime(merged["construction_date"])
        assert (merged["maintenance_date"] >= constructions).all()

    def test_mid_month_construction_first_event_after_construction(self) -> None:
        master = dataclasses.replace(BASE_MASTER, construction_date=date(2020, 6, 15))
        frame = _master_frame(master)
        one = DatasetSpec(
            dataset_segments=1, dataset_months_per_segment=24, dataset_observations=24
        )
        events = generate_maintenance_events(frame, one, 42, start_date=START)
        assert (events["maintenance_date"] >= pd.Timestamp("2020-06-15")).all()


class TestMonthlyHazardSensitivity:
    def _hazard(self, master: SegmentMaster) -> float:
        return monthly_hazard(
            master,
            HAZARD_DATE,
            months_since_last_event=10_000,
            condition=60.0,
            trailing_accidents=1,
        )

    def test_asset_age_increases_hazard(self) -> None:
        older = dataclasses.replace(BASE_MASTER, construction_date=date(1995, 1, 1))
        newer = dataclasses.replace(BASE_MASTER, construction_date=date(2015, 1, 1))
        assert self._hazard(older) > self._hazard(newer)

    def test_traffic_exposure_increases_hazard(self) -> None:
        high = dataclasses.replace(BASE_MASTER, traffic_base=20_000)
        low = dataclasses.replace(BASE_MASTER, traffic_base=1_000)
        assert self._hazard(high) > self._hazard(low)

    def test_heavy_vehicle_exposure_increases_hazard(self) -> None:
        high = dataclasses.replace(BASE_MASTER, heavy_vehicle_ratio_base=0.5)
        low = dataclasses.replace(BASE_MASTER, heavy_vehicle_ratio_base=0.05)
        assert self._hazard(high) > self._hazard(low)

    def test_weather_exposure_increases_hazard(self) -> None:
        wet = dataclasses.replace(BASE_MASTER, weather_exposure=1.6)
        dry = dataclasses.replace(BASE_MASTER, weather_exposure=0.6)
        assert self._hazard(wet) > self._hazard(dry)

    def test_deterioration_increases_hazard_through_condition(self) -> None:
        fast = dataclasses.replace(BASE_MASTER, deterioration_rate=1.8)
        slow = dataclasses.replace(BASE_MASTER, deterioration_rate=0.3)
        condition = 80.0
        for _ in range(36):
            condition = decay_condition(condition, fast.deterioration_rate)
        condition_fast = condition
        condition = 80.0
        for _ in range(36):
            condition = decay_condition(condition, slow.deterioration_rate)
        condition_slow = condition
        assert condition_fast < condition_slow
        hazard_fast = monthly_hazard(
            fast, HAZARD_DATE, 10_000, condition_fast, trailing_accidents=1
        )
        hazard_slow = monthly_hazard(
            slow, HAZARD_DATE, 10_000, condition_slow, trailing_accidents=1
        )
        assert hazard_fast > hazard_slow

    def test_trailing_accident_history_increases_hazard(self) -> None:
        none = monthly_hazard(BASE_MASTER, HAZARD_DATE, 10_000, 60.0, trailing_accidents=0)
        some = monthly_hazard(BASE_MASTER, HAZARD_DATE, 10_000, 60.0, trailing_accidents=3)
        many = monthly_hazard(BASE_MASTER, HAZARD_DATE, 10_000, 60.0, trailing_accidents=10)
        assert none < some < many

    def test_previous_maintenance_suppresses_hazard(self) -> None:
        recent = monthly_hazard(BASE_MASTER, HAZARD_DATE, 1, 60.0, 1)
        medium = monthly_hazard(BASE_MASTER, HAZARD_DATE, 6, 60.0, 1)
        distant = monthly_hazard(BASE_MASTER, HAZARD_DATE, 24, 60.0, 1)
        assert recent < medium < distant

    def test_condition_decay_floor(self) -> None:
        assert decay_condition(22.0, 1.8) >= 20.0
        assert decay_condition(20.0, 0.0) == 20.0

    def test_base_rate_controls_hazard_level(self) -> None:
        low = monthly_hazard(BASE_MASTER, HAZARD_DATE, 10_000, 60.0, 1, base_rate=0.02)
        high = monthly_hazard(BASE_MASTER, HAZARD_DATE, 10_000, 60.0, 1, base_rate=0.5)
        assert low < high
        assert monthly_hazard(BASE_MASTER, HAZARD_DATE, 10_000, 60.0, 1) == monthly_hazard(
            BASE_MASTER, HAZARD_DATE, 10_000, 60.0, 1, base_rate=0.15
        )

    @pytest.mark.parametrize(
        "bad",
        [True, False, 0.0, -1.0, float("nan"), float("inf")],
    )
    def test_invalid_base_rate_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError):
            monthly_hazard(BASE_MASTER, HAZARD_DATE, 10_000, 60.0, 1, base_rate=bad)


class TestSameDaySnapshotSemantics:
    def test_event_on_observation_date_is_next_event(self) -> None:
        segments, events = _frames(base_rate=0.9)
        obs_dates = {d.date() for d in observation_dates(SPEC.dataset_months_per_segment, START)}
        event_dates = set(
            zip(events["segment_id"], events["maintenance_date"].dt.date, strict=True)
        )
        same_day = [pair for pair in event_dates if pair[1] in obs_dates]
        assert len(same_day) > 0
        known_ids = set(segments["segment_id"])
        for segment_id, event_date in same_day:
            assert segment_id in known_ids
            assert days_until_maintenance(event_date, event_date) == 0
            assert maintenance_within_30_days(0) is True

    def test_event_strictly_before_observation_is_past_history(self) -> None:
        segments, events = _frames()
        early = events[events["maintenance_date"] < pd.Timestamp(START)]
        assert len(early) > 0
        for _, row in early.head(10).iterrows():
            with pytest.raises(ValueError):
                days_until_maintenance(START, row["maintenance_date"].date())

    def test_forced_event_on_observation_date_transitions_state_after_snapshot(
        self,
    ) -> None:
        observation = date(2022, 6, 1)
        following_month = date(2022, 7, 1)
        window_pre = (1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0)
        months_since_pre = 8
        condition_pre = 60.0
        base_rate = 0.15
        hazard_at_snapshot = monthly_hazard(
            BASE_MASTER,
            observation,
            months_since_pre,
            condition_pre,
            sum(window_pre),
            base_rate,
        )
        months_since_post, condition_post, window_post = month_transition(
            BASE_MASTER,
            event_occurred=True,
            condition=condition_pre,
            months_since_last_event=months_since_pre,
            accident_window=window_pre,
            new_accident_count=3,
        )
        assert months_since_post == 0
        assert condition_post == decay_condition(
            min(float(BASE_MASTER.initial_condition), condition_pre + 25.0),
            BASE_MASTER.deterioration_rate,
        )
        assert window_post == window_pre[1:] + (3,)
        hazard_following = monthly_hazard(
            BASE_MASTER,
            following_month,
            months_since_post,
            condition_post,
            sum(window_post),
            base_rate,
        )
        assert hazard_following < hazard_at_snapshot

    def test_integration_real_event_hazard_snapshot_via_spy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import roadguard.events as events_module

        segments = generate_segments(SPEC, 42)
        timeline = generate_accident_timeline(segments, SPEC, 42, start_date=START)
        calls: list[dict[str, object]] = []
        real_hazard = events_module.monthly_hazard

        def spy(
            master: SegmentMaster,
            month_date: date,
            months_since_last_event: int,
            condition: float,
            trailing_accidents: int,
            base_rate: float = 0.15,
        ) -> float:
            hazard = real_hazard(
                master,
                month_date,
                months_since_last_event,
                condition,
                trailing_accidents,
                base_rate,
            )
            calls.append(
                {
                    "segment_id": master.segment_id,
                    "month_date": month_date,
                    "months_since_last_event": months_since_last_event,
                    "condition": condition,
                    "trailing_accidents": trailing_accidents,
                    "hazard": hazard,
                }
            )
            return hazard

        monkeypatch.setattr(events_module, "monthly_hazard", spy)
        events = generate_maintenance_events(segments, SPEC, 42, start_date=START, base_rate=0.9)
        assert len(calls) > 0

        obs_dates = {d.date() for d in observation_dates(SPEC.dataset_months_per_segment, START)}
        same_day = events[events["maintenance_date"].dt.date.isin(obs_dates)]
        assert len(same_day) > 0
        row = same_day.iloc[0]
        segment_id = row["segment_id"]
        event_date = row["maintenance_date"].date()
        assert event_date in obs_dates

        by_month = {(c["segment_id"], c["month_date"]): c for c in calls}
        recorded_at_t = by_month[(segment_id, event_date)]
        assert 0.0 <= float(recorded_at_t["hazard"]) <= 1.0

        master_record = segments[segments["segment_id"] == segment_id].iloc[0]
        master = SegmentMaster.from_record(master_record.to_dict())

        segment_events = events[events["segment_id"] == segment_id]
        event_months = {
            date(d.year, d.month, 1) for d in segment_events["maintenance_date"].dt.date
        }
        segment_timeline = timeline[timeline["segment_id"] == segment_id]
        prior_months = [m.date() for m in segment_timeline["month"] if m.date() < event_date]

        months_since = 10_000
        condition = float(master.initial_condition)
        window: tuple[int, ...] = ()
        for month in prior_months:
            accident_count = int(
                segment_timeline.loc[
                    segment_timeline["month"] == pd.Timestamp(month), "accident_count"
                ].iloc[0]
            )
            months_since, condition, window = month_transition(
                master,
                event_occurred=month in event_months,
                condition=condition,
                months_since_last_event=months_since,
                accident_window=window,
                new_accident_count=accident_count,
            )

        assert recorded_at_t["month_date"] == event_date
        assert recorded_at_t["months_since_last_event"] == months_since
        assert recorded_at_t["condition"] == condition
        assert recorded_at_t["trailing_accidents"] == sum(window)

        accident_at_t = int(
            segment_timeline.loc[
                segment_timeline["month"] == pd.Timestamp(event_date), "accident_count"
            ].iloc[0]
        )
        months_since_post, condition_post, window_post = month_transition(
            master,
            event_occurred=True,
            condition=condition,
            months_since_last_event=months_since,
            accident_window=window,
            new_accident_count=accident_at_t,
        )
        next_month = date(
            event_date.year + (event_date.month == 12),
            (event_date.month % 12) + 1,
            1,
        )
        recorded_next = by_month[(segment_id, next_month)]
        assert recorded_next["months_since_last_event"] == 0
        assert recorded_next["condition"] == condition_post
        assert recorded_next["trailing_accidents"] == sum(window_post)
        assert recorded_next["trailing_accidents"] != recorded_at_t["trailing_accidents"]


class TestValidation:
    def test_seed_must_be_positive(self) -> None:
        segments = generate_segments(SPEC, 42)
        with pytest.raises(ValueError):
            generate_maintenance_events(segments, SPEC, 0, start_date=START)

    def test_segment_count_must_match_spec(self) -> None:
        segments = generate_segments(SPEC, 42)
        wrong = DatasetSpec(
            dataset_segments=59, dataset_months_per_segment=24, dataset_observations=1_416
        )
        with pytest.raises(ValueError):
            generate_maintenance_events(segments, wrong, 42, start_date=START)

    def test_missing_columns_rejected(self) -> None:
        segments = generate_segments(SPEC, 42).drop(columns=["traffic_base"])
        with pytest.raises(ValueError):
            generate_maintenance_events(segments, SPEC, 42, start_date=START)

    def test_duplicate_segment_ids_rejected(self) -> None:
        segments = generate_segments(SPEC, 42)
        duplicated = segments.copy()
        duplicated.loc[1, "segment_id"] = duplicated.loc[0, "segment_id"]
        with pytest.raises(ValueError):
            generate_maintenance_events(duplicated, SPEC, 42, start_date=START)

    def test_non_positive_base_rate_rejected(self) -> None:
        segments = generate_segments(SPEC, 42)
        with pytest.raises(ValueError):
            generate_maintenance_events(segments, SPEC, 42, start_date=START, base_rate=0.0)

    def test_non_positive_future_buffer_rejected(self) -> None:
        segments = generate_segments(SPEC, 42)
        with pytest.raises(ValueError):
            generate_maintenance_events(
                segments, SPEC, 42, start_date=START, future_buffer_months=0
            )

    def test_non_positive_pre_period_rejected(self) -> None:
        segments = generate_segments(SPEC, 42)
        with pytest.raises(ValueError):
            generate_maintenance_events(segments, SPEC, 42, start_date=START, pre_period_months=0)

    def test_non_positive_cap_rejected(self) -> None:
        segments = generate_segments(SPEC, 42)
        with pytest.raises(ValueError):
            generate_maintenance_events(
                segments, SPEC, 42, start_date=START, max_months_per_segment=0
            )
