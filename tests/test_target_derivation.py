"""Tests for Phase 4 event-derived supervised target derivation."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from roadguard import (
    DatasetSpec,
    derive_observation_targets,
    generate_accident_timeline,
    generate_maintenance_events,
    generate_observations,
    generate_segments,
)
from roadguard.targets import TARGET_COLUMNS

V1_SPEC = DatasetSpec(
    dataset_segments=300, dataset_months_per_segment=48, dataset_observations=14_400
)
START = date(2022, 1, 1)


def _obs(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    data = [
        {"segment_id": sid, "date": pd.Timestamp(day), "traffic_volume": vol}
        for sid, day, vol in rows
    ]
    return pd.DataFrame(data)


def _events(
    rows: list[tuple[str, str]],
    extra: dict[str, object] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "segment_id": pd.Series([sid for sid, _ in rows], dtype=object),
            "maintenance_date": pd.to_datetime([day for _, day in rows]),
        }
    )
    if extra:
        for name, value in extra.items():
            frame[name] = value
    return frame


def _reference_targets(observations: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Independent naive reference: per-segment min over events on-or-after t."""
    events_by_segment = events.groupby("segment_id")["maintenance_date"].apply(
        lambda series: list(series)
    )
    records = []
    for _, row in observations.iterrows():
        sid = row["segment_id"]
        t = pd.Timestamp(row["date"])
        candidates = [d for d in events_by_segment.get(sid, []) if pd.Timestamp(d) >= t]
        assert len(candidates) > 0, f"no next event for {sid} at {t}"
        next_date = min(candidates)
        days = (pd.Timestamp(next_date) - t).days
        records.append(
            {
                "segment_id": sid,
                "date": t,
                "days_until_maintenance": days,
                "maintenance_within_30_days": int(0 <= days <= 30),
            }
        )
    frame = pd.DataFrame(records, columns=list(TARGET_COLUMNS))
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["segment_id", "date"]).reset_index(drop=True)


def _v1_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    segments = generate_segments(V1_SPEC, 42, observation_start=START)
    events = generate_maintenance_events(segments, V1_SPEC, 42, start_date=START)
    timeline = generate_accident_timeline(segments, V1_SPEC, 42, start_date=START)
    observations = generate_observations(segments, events, timeline, V1_SPEC, 42, start_date=START)
    return observations, events


class TestSchema:
    def test_exact_column_order_and_dtypes(self) -> None:
        observations = _obs(
            [
                ("QL01-KM10-20", "2022-02-01", 1000),
                ("QL01-KM10-20", "2022-01-01", 900),
            ]
        )
        events = _events(
            [
                ("QL01-KM10-20", "2022-03-01"),
                ("QL01-KM10-20", "2022-01-15"),
            ]
        )
        targets = derive_observation_targets(observations, events)
        assert list(targets.columns) == list(TARGET_COLUMNS)
        assert targets["segment_id"].dtype == object
        assert all(isinstance(value, str) for value in targets["segment_id"])
        assert str(targets["date"].dtype) == "datetime64[ns]"
        assert targets["date"].dt.tz is None
        assert targets["days_until_maintenance"].dtype == "int64"
        assert targets["maintenance_within_30_days"].dtype == "int64"
        assert set(targets["maintenance_within_30_days"]) <= {0, 1}

    def test_no_future_date_material_cost_or_feature_columns(self) -> None:
        observations = _obs([("QL01-KM10-20", "2022-01-01", 1000)])
        events = _events(
            [("QL01-KM10-20", "2022-03-01")],
            extra={"maintenance_cost": 1_000_000, "thermoplastic_paint_kg": 5.0},
        )
        targets = derive_observation_targets(observations, events)
        assert list(targets.columns) == list(TARGET_COLUMNS)


class TestBoundaries:
    def _single(self, event_dates: list[str], observation: str = "2022-01-01"):
        observations = _obs([("QL01-KM10-20", observation, 1000)])
        events = _events([("QL01-KM10-20", day) for day in event_dates])
        return derive_observation_targets(observations, events).iloc[0]

    def test_same_day_event_days_zero_class_one(self) -> None:
        row = self._single(["2022-01-01", "2022-03-01"])
        assert row["days_until_maintenance"] == 0
        assert row["maintenance_within_30_days"] == 1

    def test_thirty_day_event_class_one(self) -> None:
        row = self._single(["2022-01-31", "2022-03-01"])
        assert row["days_until_maintenance"] == 30
        assert row["maintenance_within_30_days"] == 1

    def test_thirty_one_day_event_class_zero(self) -> None:
        row = self._single(["2022-02-01", "2022-03-01"])
        assert row["days_until_maintenance"] == 31
        assert row["maintenance_within_30_days"] == 0

    def test_past_events_ignored(self) -> None:
        row = self._single(["2021-11-20", "2021-12-05", "2022-01-10"])
        assert row["days_until_maintenance"] == 9
        assert row["maintenance_within_30_days"] == 1

    def test_earliest_on_or_after_event_selected(self) -> None:
        row = self._single(["2022-01-20", "2022-01-06", "2022-02-15"])
        assert row["days_until_maintenance"] == 5
        assert row["maintenance_within_30_days"] == 1

    def test_final_observation_labeled_from_post_window_event(self) -> None:
        observations = _obs([("QL01-KM10-20", "2022-12-01", 1000)])
        events = _events([("QL01-KM10-20", "2023-01-15")])
        row = derive_observation_targets(observations, events).iloc[0]
        assert row["days_until_maintenance"] == 45
        assert row["maintenance_within_30_days"] == 0

    def test_missing_future_event_raises_contextual_value_error(self) -> None:
        observations = _obs([("QL01-KM10-20", "2022-03-01", 1000)])
        events = _events([("QL01-KM10-20", "2022-01-15")])
        with pytest.raises(ValueError, match="QL01-KM10-20.*2022-03-01"):
            derive_observation_targets(observations, events)


class TestReferenceMatch:
    def test_matches_independent_reference_implementation(self) -> None:
        observations = _obs(
            [
                ("QL01-KM10-20", "2022-01-01", 1000),
                ("QL01-KM10-20", "2022-02-01", 1100),
                ("QL01-KM10-20", "2022-03-01", 1200),
                ("QL14-KM27-36", "2022-01-01", 500),
                ("QL14-KM27-36", "2022-02-01", 600),
                ("QL01-KM10-20", "2022-04-01", 1300),
            ]
        )
        events = _events(
            [
                ("QL01-KM10-20", "2021-12-10"),
                ("QL01-KM10-20", "2022-01-25"),
                ("QL01-KM10-20", "2022-03-20"),
                ("QL01-KM10-20", "2022-06-01"),
                ("QL14-KM27-36", "2022-01-15"),
                ("QL14-KM27-36", "2022-05-30"),
            ]
        )
        actual = derive_observation_targets(observations, events)
        expected = _reference_targets(observations, events)
        assert_frame_equal(actual, expected)


class TestV1:
    def test_exactly_14400_rows_and_unique_keys(self) -> None:
        observations, events = _v1_pipeline()
        targets = derive_observation_targets(observations, events)
        assert len(targets) == 14_400
        keys = list(zip(targets["segment_id"], targets["date"], strict=True))
        assert len(keys) == len(set(keys))

    def test_contains_both_positive_and_negative_labels(self) -> None:
        observations, events = _v1_pipeline()
        targets = derive_observation_targets(observations, events)
        assert 0 in set(targets["maintenance_within_30_days"])
        assert 1 in set(targets["maintenance_within_30_days"])

    def test_label_consistency_invariant(self) -> None:
        observations, events = _v1_pipeline()
        targets = derive_observation_targets(observations, events)
        expected = (
            (targets["days_until_maintenance"] >= 0) & (targets["days_until_maintenance"] <= 30)
        ).astype(int)
        assert (targets["maintenance_within_30_days"] == expected).all()


class TestRowOrderInvariance:
    def test_shuffled_observations_identical_output(self) -> None:
        observations = _obs(
            [
                ("QL01-KM10-20", "2022-01-01", 1000),
                ("QL01-KM10-20", "2022-02-01", 1100),
                ("QL14-KM27-36", "2022-01-01", 500),
                ("QL14-KM27-36", "2022-02-01", 600),
            ]
        )
        events = _events(
            [
                ("QL01-KM10-20", "2022-01-20"),
                ("QL01-KM10-20", "2022-03-01"),
                ("QL14-KM27-36", "2022-02-10"),
            ]
        )
        base = derive_observation_targets(observations, events)
        shuffled = observations.sample(frac=1.0, random_state=3)
        assert_frame_equal(base, derive_observation_targets(shuffled, events))

    def test_shuffled_events_identical_output(self) -> None:
        observations = _obs(
            [
                ("QL01-KM10-20", "2022-01-01", 1000),
                ("QL01-KM10-20", "2022-02-01", 1100),
                ("QL14-KM27-36", "2022-01-01", 500),
                ("QL14-KM27-36", "2022-02-01", 600),
            ]
        )
        events = _events(
            [
                ("QL01-KM10-20", "2022-01-20"),
                ("QL01-KM10-20", "2022-03-01"),
                ("QL14-KM27-36", "2022-02-10"),
            ]
        )
        base = derive_observation_targets(observations, events)
        shuffled = events.sample(frac=1.0, random_state=5)
        assert_frame_equal(base, derive_observation_targets(observations, shuffled))

    def test_inputs_not_mutated(self) -> None:
        observations = _obs(
            [("QL01-KM10-20", "2022-01-01", 1000), ("QL01-KM10-20", "2022-02-01", 1100)]
        )
        events = _events([("QL01-KM10-20", "2022-01-20"), ("QL01-KM10-20", "2022-03-01")])
        observations_copy = observations.copy(deep=True)
        events_copy = events.copy(deep=True)
        derive_observation_targets(observations, events)
        assert_frame_equal(observations, observations_copy)
        assert_frame_equal(events, events_copy)


class TestValidation:
    def _base_frames(self):
        observations = _obs(
            [("QL01-KM10-20", "2022-01-01", 1000), ("QL01-KM10-20", "2022-02-01", 1100)]
        )
        events = _events([("QL01-KM10-20", "2022-01-20"), ("QL01-KM10-20", "2022-03-01")])
        return observations, events

    def test_duplicate_observation_keys_rejected(self) -> None:
        observations, events = self._base_frames()
        duplicated = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError):
            derive_observation_targets(duplicated, events)

    def test_duplicate_event_keys_rejected(self) -> None:
        observations, events = self._base_frames()
        duplicated = pd.concat([events, events.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError):
            derive_observation_targets(observations, duplicated)

    def test_missing_observation_columns_rejected(self) -> None:
        observations, events = self._base_frames()
        with pytest.raises(ValueError):
            derive_observation_targets(observations.drop(columns=["date"]), events)

    def test_missing_event_columns_rejected(self) -> None:
        observations, events = self._base_frames()
        with pytest.raises(ValueError):
            derive_observation_targets(observations, events.drop(columns=["segment_id"]))

    def test_nat_dates_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = bad["date"].astype(object)
        bad.loc[0, "date"] = pd.NaT
        with pytest.raises(ValueError):
            derive_observation_targets(bad, events)

    def test_nat_in_datetime64_column_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = pd.to_datetime(bad["date"])
        bad.loc[0, "date"] = pd.NaT
        with pytest.raises(ValueError):
            derive_observation_targets(bad, events)

    def test_object_date_column_accepted(self) -> None:
        observations, events = self._base_frames()
        obj = observations.copy()
        obj["date"] = obj["date"].astype(object)
        targets = derive_observation_targets(obj, events)
        assert len(targets) == 2

    def test_timezone_aware_object_dates_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = bad["date"].astype(object)
        bad.loc[0, "date"] = pd.Timestamp("2022-01-01", tz="UTC")
        with pytest.raises(ValueError):
            derive_observation_targets(bad, events)

    def test_non_midnight_datetime64_observation_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = pd.to_datetime(bad["date"])
        bad.loc[0, "date"] = pd.Timestamp("2022-01-01 12:00:00")
        with pytest.raises(ValueError, match="non-midnight"):
            derive_observation_targets(bad, events)

    def test_non_midnight_datetime64_event_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = events.copy()
        bad["maintenance_date"] = pd.to_datetime(bad["maintenance_date"])
        bad.loc[0, "maintenance_date"] = pd.Timestamp("2022-01-20 08:30:00")
        with pytest.raises(ValueError, match="non-midnight"):
            derive_observation_targets(observations, bad)

    def test_nanosecond_non_midnight_datetime64_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = pd.to_datetime(bad["date"])
        bad.loc[0, "date"] = pd.Timestamp("2022-01-01 00:00:00.000000001")
        with pytest.raises(ValueError, match="non-midnight"):
            derive_observation_targets(bad, events)

    def test_non_midnight_object_datetime_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = bad["date"].astype(object)
        bad.loc[0, "date"] = datetime(2022, 1, 1, 12, 0)
        with pytest.raises(ValueError, match="non-midnight"):
            derive_observation_targets(bad, events)
        bad_events = events.copy()
        bad_events["maintenance_date"] = bad_events["maintenance_date"].astype(object)
        bad_events.loc[0, "maintenance_date"] = pd.Timestamp("2022-01-20 00:00:01")
        with pytest.raises(ValueError, match="non-midnight"):
            derive_observation_targets(observations, bad_events)

    def test_midnight_values_accepted(self) -> None:
        observations, events = self._base_frames()
        obj = observations.copy()
        obj["date"] = obj["date"].astype(object)
        obj.loc[0, "date"] = datetime(2022, 1, 1, 0, 0)
        targets = derive_observation_targets(obj, events)
        assert len(targets) == 2

    def test_same_calendar_day_event_cannot_become_same_day_target(self) -> None:
        observations = _obs([("QL01-KM10-20", "2022-01-01 12:00:00", 1000)])
        events = _events([("QL01-KM10-20", "2022-01-01 00:00:00")])
        with pytest.raises(ValueError, match="non-midnight"):
            derive_observation_targets(observations, events)

    def test_out_of_ns_range_date_in_observations_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = bad["date"].astype(object)
        bad.loc[0, "date"] = date(9999, 1, 1)
        with pytest.raises(ValueError):
            derive_observation_targets(bad, events)

    def test_out_of_ns_range_date_in_events_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = events.copy()
        bad["maintenance_date"] = bad["maintenance_date"].astype(object)
        bad.loc[0, "maintenance_date"] = date(9998, 12, 31)
        with pytest.raises(ValueError):
            derive_observation_targets(observations, bad)

    def test_out_of_ns_datetime64_seconds_observation_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = pd.Series(["9999-01-01", "2022-02-01"], dtype="datetime64[s]")
        with pytest.raises(ValueError, match="outside datetime64"):
            derive_observation_targets(bad, events)

    def test_out_of_ns_datetime64_seconds_event_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = events.copy()
        bad["maintenance_date"] = pd.Series(["9999-01-01", "2022-03-01"], dtype="datetime64[s]")
        with pytest.raises(ValueError, match="outside datetime64"):
            derive_observation_targets(observations, bad)

    def test_out_of_ns_datetime64_microseconds_event_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = events.copy()
        bad["maintenance_date"] = pd.Series(["9999-01-01", "2022-03-01"], dtype="datetime64[us]")
        with pytest.raises(ValueError, match="outside datetime64"):
            derive_observation_targets(observations, bad)

    def test_out_of_ns_object_timestamp_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = bad["date"].astype(object)
        bad.loc[0, "date"] = pd.Timestamp("9999-01-01")
        with pytest.raises(ValueError, match="outside datetime64"):
            derive_observation_targets(bad, events)

    def test_valid_lower_resolution_datetime64_converted_exactly(self) -> None:
        observations = _obs([("QL01-KM10-20", "2022-01-01", 1000)])
        observations["date"] = pd.Series(["2022-01-01"], dtype="datetime64[s]")
        events = _events([("QL01-KM10-20", "2022-01-10")])
        targets = derive_observation_targets(observations, events)
        assert targets.loc[0, "date"] == pd.Timestamp("2022-01-01")
        assert targets.loc[0, "days_until_maintenance"] == 9

    def test_container_object_date_rejected_contextually(self) -> None:
        observations, events = self._base_frames()
        for container in ([date(2022, 1, 1)], {"day": 1}):
            bad = observations.copy()
            bad["date"] = pd.Series([container, pd.Timestamp("2022-02-01")], dtype=object)
            with pytest.raises(ValueError, match="unsupported"):
                derive_observation_targets(bad, events)

    def test_duplicate_required_column_labels_rejected(self) -> None:
        observations, events = self._base_frames()
        dup_obs = observations.copy()
        dup_obs.columns = ["segment_id", "date", "date"]
        with pytest.raises(ValueError, match="duplicate"):
            derive_observation_targets(dup_obs, events)
        dup_events = pd.DataFrame(
            [["QL01-KM10-20", "QL01-KM10-20", pd.Timestamp("2022-01-20")]],
            columns=["segment_id", "segment_id", "maintenance_date"],
        )
        with pytest.raises(ValueError, match="duplicate"):
            derive_observation_targets(observations, dup_events)

    def test_malformed_date_strings_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = bad["date"].astype(object)
        bad.loc[0, "date"] = "not-a-date"
        with pytest.raises(ValueError):
            derive_observation_targets(bad, events)

    def test_incompatible_date_dtype_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = bad["date"].astype("int64")
        with pytest.raises(ValueError):
            derive_observation_targets(bad, events)

    def test_timezone_aware_dates_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["date"] = pd.to_datetime(bad["date"]).dt.tz_localize("UTC")
        with pytest.raises(ValueError):
            derive_observation_targets(bad, events)

    def test_malformed_segment_id_rejected(self) -> None:
        observations, events = self._base_frames()
        bad = observations.copy()
        bad["segment_id"] = bad["segment_id"].astype(object)
        bad.loc[0, "segment_id"] = "NOT-A-KEY"
        with pytest.raises(ValueError):
            derive_observation_targets(bad, events)

    def test_event_segment_absent_from_observations_rejected(self) -> None:
        observations, events = self._base_frames()
        extra = pd.DataFrame(
            {
                "segment_id": ["QL14-KM27-36"],
                "maintenance_date": pd.to_datetime(["2022-03-01"]),
            }
        )
        with pytest.raises(ValueError):
            derive_observation_targets(observations, pd.concat([events, extra], ignore_index=True))

    def test_observation_segment_without_event_history_rejected(self) -> None:
        observations = _obs([("QL14-KM27-36", "2022-01-01", 500)])
        events = _events([("QL01-KM10-20", "2022-01-20")])
        with pytest.raises(ValueError):
            derive_observation_targets(observations, events)

    def test_segment_with_no_events_at_all_rejected(self) -> None:
        observations = _obs([("QL01-KM10-20", "2022-01-01", 1000)])
        events = _events([])
        with pytest.raises(ValueError, match="QL01-KM10-20.*2022-01-01"):
            derive_observation_targets(observations, events)


class TestEventModificationSensitivity:
    def test_changing_only_cost_material_columns_does_not_change_targets(self) -> None:
        observations = _obs(
            [("QL01-KM10-20", "2022-01-01", 1000), ("QL01-KM10-20", "2022-02-01", 1100)]
        )
        base_events = _events(
            [("QL01-KM10-20", "2022-01-20"), ("QL01-KM10-20", "2022-03-01")],
            extra={
                "maintenance_cost": [1_000_000, 2_000_000],
                "thermoplastic_paint_kg": [5.0, 8.0],
            },
        )
        changed_events = _events(
            [("QL01-KM10-20", "2022-01-20"), ("QL01-KM10-20", "2022-03-01")],
            extra={
                "maintenance_cost": [9_000_000, 9_000_000],
                "thermoplastic_paint_kg": [99.0, 99.0],
            },
        )
        assert not base_events.equals(changed_events)
        first = derive_observation_targets(observations, base_events)
        second = derive_observation_targets(observations, changed_events)
        assert_frame_equal(first, second)

    def test_later_event_cannot_change_existing_targets(self) -> None:
        observations = _obs(
            [("QL01-KM10-20", "2022-01-01", 1000), ("QL01-KM10-20", "2022-02-01", 1100)]
        )
        base_events = _events([("QL01-KM10-20", "2022-01-20"), ("QL01-KM10-20", "2022-03-01")])
        extra_events = _events(
            [
                ("QL01-KM10-20", "2022-01-20"),
                ("QL01-KM10-20", "2022-03-01"),
                ("QL01-KM10-20", "2022-05-01"),
            ]
        )
        assert len(extra_events) > len(base_events)
        first = derive_observation_targets(observations, base_events)
        second = derive_observation_targets(observations, extra_events)
        assert_frame_equal(first, second)

    def test_inserting_earlier_event_changes_only_affected_rows(self) -> None:
        observations = _obs(
            [
                ("QL01-KM10-20", "2022-01-01", 1000),
                ("QL01-KM10-20", "2022-01-15", 1000),
                ("QL01-KM10-20", "2022-01-25", 1000),
            ]
        )
        base_events = _events([("QL01-KM10-20", "2022-02-01")])
        earlier_events = _events([("QL01-KM10-20", "2022-01-10"), ("QL01-KM10-20", "2022-02-01")])
        first = derive_observation_targets(observations, base_events)
        second = derive_observation_targets(observations, earlier_events)
        changed = first["days_until_maintenance"] != second["days_until_maintenance"]
        assert changed.sum() == 1
        assert second.loc[0, "days_until_maintenance"] == 9
        assert first.loc[0, "days_until_maintenance"] == 31
