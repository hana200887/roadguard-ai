"""Phase 5 cleaning and full-pipeline integration tests (coherent mini-spec frames)."""

from __future__ import annotations

from datetime import date

import numpy as np
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
from roadguard.data_quality import (
    PUBLIC_SEGMENT_COLUMNS,
    TRAFFIC_VOLUME_OUTLIER_MAX,
    clean_raw_dataset,
    inject_observation_corruption,
    validate_cleaned_dataset,
    validate_raw_dataset,
)
from roadguard.observations import OBSERVATION_COLUMNS

SPEC = DatasetSpec(dataset_segments=10, dataset_months_per_segment=12, dataset_observations=120)
MINI_SPEC = DatasetSpec(dataset_segments=2, dataset_months_per_segment=12, dataset_observations=24)
V1_SPEC = DatasetSpec(
    dataset_segments=300, dataset_months_per_segment=48, dataset_observations=14_400
)
START = date(2022, 1, 1)


def _pipeline(spec: DatasetSpec = SPEC, seed: int = 42):
    segments = generate_segments(spec, seed, observation_start=START)
    events = generate_maintenance_events(segments, spec, seed, start_date=START)
    timeline = generate_accident_timeline(segments, spec, seed, start_date=START)
    observations = generate_observations(segments, events, timeline, spec, seed, start_date=START)
    targets = derive_observation_targets(observations, events)
    public_segments = segments[list(PUBLIC_SEGMENT_COLUMNS)].copy()
    public_segments["construction_date"] = pd.to_datetime(public_segments["construction_date"])
    return public_segments, observations, targets, events


def _mini(seed: int = 42):
    return _pipeline(spec=MINI_SPEC, seed=seed)


def _crafted_raw_mini():
    segments, observations, targets, events = _mini()
    raw = observations.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    segment_a = sorted(raw["segment_id"].unique())[0]
    rows = sorted(raw[raw["segment_id"] == segment_a].index)
    first, second, third = rows[0], rows[1], rows[2]
    raw.loc[first, "traffic_volume"] = TRAFFIC_VOLUME_OUTLIER_MAX + 50_000
    raw.loc[first, "rainfall_mm"] = 2_000.0
    raw.loc[second, "rainfall_mm"] = np.nan
    raw.loc[second, "humidity"] = np.nan
    raw.loc[third, "rainfall_mm"] = np.nan
    raw.loc[third, "temperature"] = np.nan
    raw.loc[third, "humidity"] = np.nan
    duplicate = raw.loc[first].copy()
    raw = pd.concat([raw, pd.DataFrame([duplicate])], ignore_index=True)
    return segments, raw, targets, events


class TestCleaning:
    def test_exact_duplicates_removed_deterministically(self) -> None:
        segments, raw, targets, events = _crafted_raw_mini()
        result = clean_raw_dataset(segments, raw, targets, events, MINI_SPEC)
        keys = list(
            zip(result.observations["segment_id"], result.observations["date"], strict=True)
        )
        assert len(keys) == len(set(keys))
        assert len(result.observations) == MINI_SPEC.dataset_observations

    def test_conflicting_duplicate_keys_fail(self) -> None:
        segments, raw, targets, events = _crafted_raw_mini()
        conflicting = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        conflicting.loc[len(conflicting) - 1, "traffic_volume"] = 1
        with pytest.raises(ValueError):
            clean_raw_dataset(segments, conflicting, targets, events, MINI_SPEC)

    def test_weather_filled_only_from_same_segment_past(self) -> None:
        segments, observations, targets, events = _mini()
        raw = observations.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        segment_a = sorted(raw["segment_id"].unique())[0]
        segment_b = sorted(raw["segment_id"].unique())[1]
        a_rows = sorted(raw[raw["segment_id"] == segment_a].index)
        b_rows = sorted(raw[raw["segment_id"] == segment_b].index)
        raw.loc[a_rows[0], "rainfall_mm"] = 111.0
        raw.loc[a_rows[1], "rainfall_mm"] = np.nan
        raw.loc[a_rows[2], "rainfall_mm"] = np.nan
        raw.loc[b_rows[0], "rainfall_mm"] = 222.0
        raw.loc[b_rows[1], "rainfall_mm"] = np.nan
        result = clean_raw_dataset(segments, raw, targets, events, MINI_SPEC)
        filled = result.observations.set_index(["segment_id", "date"])
        a_dates = result.observations[result.observations["segment_id"] == segment_a]["date"]
        b_dates = result.observations[result.observations["segment_id"] == segment_b]["date"]
        assert filled.loc[(segment_a, a_dates.iloc[1]), "rainfall_mm"] == 111.0
        assert filled.loc[(segment_a, a_dates.iloc[2]), "rainfall_mm"] == 111.0
        assert filled.loc[(segment_b, b_dates.iloc[1]), "rainfall_mm"] == 222.0

    def test_first_row_missingness_fails(self) -> None:
        segments, observations, targets, events = _mini()
        raw = observations.copy()
        raw["date"] = pd.to_datetime(raw["date"])
        raw.loc[0, "rainfall_mm"] = np.nan
        with pytest.raises(ValueError, match="no earlier valid"):
            clean_raw_dataset(segments, raw, targets, events, MINI_SPEC)

    def test_future_rows_cannot_alter_earlier_prefix(self) -> None:
        segments, observations, targets, events = _mini()
        base_raw = observations.copy()
        base_raw["date"] = pd.to_datetime(base_raw["date"])
        base_raw.loc[0, "rainfall_mm"] = 10.0
        base_raw.loc[1, "rainfall_mm"] = np.nan
        variant_raw = base_raw.copy()
        variant_raw.loc[len(variant_raw) - 1, "rainfall_mm"] = 999.0
        base_result = clean_raw_dataset(segments, base_raw, targets, events, MINI_SPEC)
        variant_result = clean_raw_dataset(segments, variant_raw, targets, events, MINI_SPEC)
        base_prefix = base_result.observations.iloc[:-1].reset_index(drop=True)
        variant_prefix = variant_result.observations.iloc[:-1].reset_index(drop=True)
        assert_frame_equal(base_prefix, variant_prefix)

    def test_other_segments_cannot_alter_existing_results(self) -> None:
        segments, observations, targets, events = _mini()
        segment_a = sorted(observations["segment_id"].unique())[0]
        segment_b = sorted(observations["segment_id"].unique())[1]
        base_raw = observations.copy()
        base_raw["date"] = pd.to_datetime(base_raw["date"])
        base_raw.loc[0, "rainfall_mm"] = 50.0
        base_raw.loc[1, "rainfall_mm"] = np.nan
        variant_raw = base_raw.copy()
        b_mask = variant_raw["segment_id"] == segment_b
        variant_raw.loc[b_mask, "rainfall_mm"] = variant_raw.loc[b_mask, "rainfall_mm"] + 100.0
        base_result = clean_raw_dataset(segments, base_raw, targets, events, MINI_SPEC)
        variant_result = clean_raw_dataset(segments, variant_raw, targets, events, MINI_SPEC)
        expected = base_result.observations[
            base_result.observations["segment_id"] == segment_a
        ].reset_index(drop=True)
        actual = variant_result.observations[
            variant_result.observations["segment_id"] == segment_a
        ].reset_index(drop=True)
        assert_frame_equal(expected, actual)

    def test_outliers_preserved_not_clipped(self) -> None:
        segments, raw, targets, events = _crafted_raw_mini()
        result = clean_raw_dataset(segments, raw, targets, events, MINI_SPEC)
        row = result.observations.iloc[0]
        assert row["traffic_volume"] == TRAFFIC_VOLUME_OUTLIER_MAX + 50_000
        assert row["rainfall_mm"] == 2_000.0
        assert result.report.warning_count > 0
        assert "operational_outlier" in result.report.counts_by_code
        assert result.report.error_count == 0

    def test_targets_and_events_remain_unchanged_and_unaliased(self) -> None:
        segments, raw, targets, events = _crafted_raw_mini()
        targets_copy = targets.copy(deep=True)
        events_copy = events.copy(deep=True)
        result = clean_raw_dataset(segments, raw, targets, events, MINI_SPEC)
        assert_frame_equal(result.targets, targets_copy)
        assert_frame_equal(result.maintenance_events, events_copy)
        assert result.targets is not targets
        assert result.maintenance_events is not events

    def test_cleaned_schemas_ordering_and_dtypes_exact(self) -> None:
        segments, raw, targets, events = _crafted_raw_mini()
        result = clean_raw_dataset(segments, raw, targets, events, MINI_SPEC)
        assert list(result.segments.columns) == list(PUBLIC_SEGMENT_COLUMNS)
        assert list(result.observations.columns) == list(OBSERVATION_COLUMNS)
        keys = list(
            zip(result.observations["segment_id"], result.observations["date"], strict=True)
        )
        assert keys == sorted(keys)
        assert str(result.observations["date"].dtype) == "datetime64[ns]"
        assert result.observations["traffic_volume"].dtype == "int64"
        assert result.observations["rainfall_mm"].dtype == "float64"

    def test_inputs_remain_unchanged(self) -> None:
        segments, raw, targets, events = _crafted_raw_mini()
        copies = [frame.copy(deep=True) for frame in (segments, raw, targets, events)]
        clean_raw_dataset(segments, raw, targets, events, MINI_SPEC)
        for original, copy in zip((segments, raw, targets, events), copies, strict=True):
            assert_frame_equal(original, copy)

    def test_clean_rejects_conflicting_keys(self) -> None:
        segments, observations, targets, events = _mini()
        raw = observations.copy()
        raw.loc[1, "segment_id"] = raw.loc[0, "segment_id"]
        raw.loc[1, "date"] = raw.loc[0, "date"]
        raw.loc[1, "traffic_volume"] = raw.loc[0, "traffic_volume"] + 1
        with pytest.raises(ValueError):
            clean_raw_dataset(segments, raw, targets, events, MINI_SPEC)

    def test_clean_fails_on_incomplete_grid(self) -> None:
        segments, observations, targets, events = _mini()
        raw = observations[observations["date"] != "2022-06-01"].reset_index(drop=True)
        with pytest.raises(ValueError):
            clean_raw_dataset(segments, raw, targets, events, MINI_SPEC)

    def test_clean_fails_on_misaligned_targets(self) -> None:
        segments, observations, targets, events = _mini()
        bad_targets = targets.iloc[:-1].reset_index(drop=True)
        with pytest.raises(ValueError):
            clean_raw_dataset(segments, observations, bad_targets, events, MINI_SPEC)

    def test_clean_fails_on_target_event_inconsistency(self) -> None:
        segments, observations, targets, events = _mini()
        extra_event = pd.DataFrame(
            {
                "segment_id": [events.iloc[0]["segment_id"]],
                "maintenance_date": [pd.Timestamp("2023-01-01")],
            }
        )
        changed_events = pd.concat([events, extra_event], ignore_index=True)
        with pytest.raises(ValueError):
            clean_raw_dataset(segments, observations, targets, changed_events, MINI_SPEC)

    def test_corrupted_pipeline_clean_and_revalidate(self) -> None:
        segments, observations, targets, events = _mini()
        corrupted, _ = inject_observation_corruption(
            observations,
            7,
            missing_rate=0.05,
            outlier_rate=0.05,
            duplicate_rate=0.1,
        )
        result = clean_raw_dataset(segments, corrupted, targets, events, MINI_SPEC)
        assert (
            not result.observations[["rainfall_mm", "temperature", "humidity"]].isna().any().any()
        )
        cleaned_report = validate_cleaned_dataset(
            segments, result.observations, targets, events, MINI_SPEC
        )
        assert cleaned_report.error_count == 0
        assert cleaned_report.is_valid


class TestV1Integration:
    def test_full_v1_pipeline_corrupt_validate_clean(self) -> None:
        segments, observations, targets, events = _pipeline(spec=V1_SPEC)
        assert len(observations) == 14_400
        corrupted, manifest = inject_observation_corruption(observations, 42)
        assert len(manifest.entries) > 0
        raw_report = validate_raw_dataset(segments, corrupted, targets, events, V1_SPEC)
        assert raw_report.error_count == 0
        assert raw_report.warning_count > 0
        result = clean_raw_dataset(segments, corrupted, targets, events, V1_SPEC)
        cleaned = result.observations
        keys = list(zip(cleaned["segment_id"], cleaned["date"], strict=True))
        assert len(keys) == 14_400
        assert len(set(keys)) == 14_400
        target_keys = set(zip(targets["segment_id"], targets["date"], strict=True))
        assert set(keys) == target_keys
        recomputed = derive_observation_targets(cleaned, events)
        assert_frame_equal(recomputed, targets.reset_index(drop=True))
        assert len(result.segments) == 300
        assert list(result.segments.columns) == list(PUBLIC_SEGMENT_COLUMNS)
        assert not {"traffic_base", "weather_exposure"}.intersection(result.segments.columns)
        target_names = {"days_until_maintenance", "maintenance_within_30_days"}
        assert not target_names.intersection(cleaned.columns)
        assert not cleaned[["rainfall_mm", "temperature", "humidity"]].isna().any().any()
        cleaned_report = validate_cleaned_dataset(
            result.segments, cleaned, targets, events, V1_SPEC
        )
        assert cleaned_report.error_count == 0
        assert cleaned_report.is_valid
        assert result.targets.equals(targets)
        assert result.maintenance_events.equals(events)


class TestReviewCleaning:
    def _string_dates_segments(self) -> pd.DataFrame:
        segments, _, _, _ = _mini()
        bad = segments.copy()
        bad["construction_date"] = pd.Series(
            [d.strftime("%Y-%m-%d") for d in segments["construction_date"]],
            dtype=object,
        )
        return bad

    def test_validate_raw_reports_invalid_dtype_for_string_dates(self) -> None:
        segments, observations, targets, events = _mini()
        bad = self._string_dates_segments()
        report = validate_raw_dataset(bad, observations, targets, events, MINI_SPEC)
        assert "invalid_dtype" in report.counts_by_code

    def test_clean_rejects_string_construction_dates(self) -> None:
        segments, observations, targets, events = _mini()
        with pytest.raises(ValueError):
            clean_raw_dataset(
                self._string_dates_segments(), observations, targets, events, MINI_SPEC
            )

    def test_clean_accepts_valid_dates_and_normalizes(self) -> None:
        _, observations, targets, events = _mini()
        raw_segments = generate_segments(MINI_SPEC, 42, observation_start=START)
        raw_segments["construction_date"] = pd.to_datetime(raw_segments["construction_date"])
        result = clean_raw_dataset(raw_segments, observations, targets, events, MINI_SPEC)
        assert str(result.segments["construction_date"].dtype) == "datetime64[ns]"
        assert list(result.segments.columns) == list(PUBLIC_SEGMENT_COLUMNS)
        assert not {"traffic_base", "weather_exposure"}.intersection(result.segments.columns)

    def test_clean_rejects_event_before_construction(self) -> None:
        segments, observations, targets, events = _mini()
        sid = sorted(observations["segment_id"].unique())[0]
        bad_segments = segments.copy()
        bad_segments.loc[bad_segments["segment_id"] == sid, "construction_date"] = pd.Timestamp(
            "2021-06-01"
        )
        bad_events = pd.DataFrame(
            {
                "segment_id": pd.Series([sid], dtype=object),
                "maintenance_date": pd.to_datetime(["2021-05-31"]),
            }
        )
        with pytest.raises(ValueError):
            clean_raw_dataset(bad_segments, observations, targets, bad_events, MINI_SPEC)

    def test_clean_rejects_same_month_conflicting_events(self) -> None:
        segments, observations, targets, events = _mini()
        sid = sorted(observations["segment_id"].unique())[0]
        bad_events = pd.DataFrame(
            {
                "segment_id": pd.Series([sid, sid], dtype=object),
                "maintenance_date": pd.to_datetime(["2022-01-05", "2022-01-20"]),
            }
        )
        with pytest.raises(ValueError):
            clean_raw_dataset(segments, observations, targets, bad_events, MINI_SPEC)


class TestReviewCanonicalPipeline:
    def test_direct_generator_to_cleaner_pipeline(self) -> None:
        segments = generate_segments(MINI_SPEC, 42, observation_start=START)
        events = generate_maintenance_events(segments, MINI_SPEC, 42, start_date=START)
        timeline = generate_accident_timeline(segments, MINI_SPEC, 42, start_date=START)
        observations = generate_observations(
            segments, events, timeline, MINI_SPEC, 42, start_date=START
        )
        targets = derive_observation_targets(observations, events)
        result = clean_raw_dataset(segments, observations, targets, events, MINI_SPEC)
        assert str(result.segments["construction_date"].dtype) == "datetime64[ns]"
        assert list(result.segments.columns) == list(PUBLIC_SEGMENT_COLUMNS)

    @pytest.mark.parametrize(
        "unit",
        ["datetime64[s]", "datetime64[ms]", "datetime64[us]", "datetime64[ns]"],
    )
    def test_construction_date_units_normalized_to_ns(self, unit: str) -> None:
        segments, observations, targets, events = _mini()
        segments = segments.copy()
        segments["construction_date"] = pd.to_datetime(segments["construction_date"]).astype(unit)
        result = clean_raw_dataset(segments, observations, targets, events, MINI_SPEC)
        assert str(result.segments["construction_date"].dtype) == "datetime64[ns]"

    def test_out_of_ns_range_construction_date_fails_contextually(self) -> None:
        segments, observations, targets, events = _mini()
        segments = segments.copy()
        segments["construction_date"] = pd.Series(
            np.array(["2015-01-01", "9999-01-01"], dtype="datetime64[s]")
        )
        with pytest.raises(ValueError):
            clean_raw_dataset(segments, observations, targets, events, MINI_SPEC)
