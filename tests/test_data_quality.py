"""Tests for Phase 5 raw-data corruption, validation and safe cleaning."""

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
    MAX_CORRUPTION_RATE,
    PUBLIC_SEGMENT_COLUMNS,
    clean_raw_dataset,
    inject_observation_corruption,
    validate_cleaned_dataset,
    validate_raw_dataset,
)
from roadguard.observations import OBSERVATION_COLUMNS

SPEC = DatasetSpec(dataset_segments=10, dataset_months_per_segment=12, dataset_observations=120)
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


def _corrupt(
    observations: pd.DataFrame,
    seed: int = 7,
    *,
    missing_rate: float = 0.05,
    outlier_rate: float = 0.05,
    duplicate_rate: float = 0.1,
):
    return inject_observation_corruption(
        observations,
        seed,
        missing_rate=missing_rate,
        outlier_rate=outlier_rate,
        duplicate_rate=duplicate_rate,
    )


class TestCorruption:
    def _base(self):
        _, observations, _, _ = _pipeline()
        return observations, _corrupt(observations)

    def test_missing_only_in_approved_weather_columns(self) -> None:
        observations, (corrupted, manifest) = self._base()
        assert corrupted.isna().any().any()
        assert all(
            entry.kind == "missing" and entry.column in ("rainfall_mm", "temperature", "humidity")
            for entry in manifest.entries
            if entry.kind == "missing"
        )
        for column in OBSERVATION_COLUMNS:
            if column in ("rainfall_mm", "temperature", "humidity"):
                continue
            assert not corrupted[column].isna().any(), column

    def test_outliers_only_in_allowed_columns(self) -> None:
        observations, (corrupted, manifest) = self._base()
        assert any(e.kind == "outlier" for e in manifest.entries)
        assert all(
            e.column in ("traffic_volume", "rainfall_mm")
            for e in manifest.entries
            if e.kind == "outlier"
        )
        deduped = corrupted.drop_duplicates(
            subset=["segment_id", "date"], keep="first"
        ).reset_index(drop=True)
        original_sorted = observations.sort_values(["segment_id", "date"]).reset_index(drop=True)
        allowed_changes = {"traffic_volume", "rainfall_mm", "temperature", "humidity"}
        for column in OBSERVATION_COLUMNS:
            if column in allowed_changes:
                continue
            assert deduped[column].equals(original_sorted[column]), column

    def test_keys_never_change(self) -> None:
        observations, (corrupted, _) = self._base()
        original_keys = set(zip(observations["segment_id"], observations["date"], strict=True))
        corrupted_keys = set(zip(corrupted["segment_id"], corrupted["date"], strict=True))
        assert original_keys == corrupted_keys

    def test_same_seed_identical(self) -> None:
        _, observations, _, _ = _pipeline()
        first = _corrupt(observations)
        second = _corrupt(observations)
        assert_frame_equal(first[0], second[0])
        assert first[1] == second[1]

    def test_different_seed_changes_affected_keys(self) -> None:
        _, observations, _, _ = _pipeline()
        first = _corrupt(observations, seed=7)
        second = _corrupt(observations, seed=8)
        assert not first[0].equals(second[0])
        assert first[1] != second[1]

    def test_shuffled_input_invariant(self) -> None:
        _, observations, _, _ = _pipeline()
        base = _corrupt(observations)
        shuffled = observations.sample(frac=1.0, random_state=11)
        rerun = _corrupt(shuffled)
        assert_frame_equal(base[0], rerun[0])
        assert base[1] == rerun[1]

    def test_manifest_exactly_matches_changed_cells_and_duplicates(self) -> None:
        _, raw, _, _ = _pipeline()
        corrupted, manifest = _corrupt(raw)
        original_by_key = raw.set_index(["segment_id", "date"])
        for entry in manifest.entries:
            mask = (corrupted["segment_id"] == entry.segment_id) & (
                corrupted["date"] == pd.Timestamp(entry.date)
            )
            if entry.kind == "duplicate":
                continue
            original = original_by_key.loc[
                (entry.segment_id, pd.Timestamp(entry.date)), entry.column
            ]
            value = corrupted.loc[mask, entry.column].iloc[0]
            if entry.kind == "missing":
                assert pd.isna(original) is False
                assert pd.isna(value)
            else:
                assert value != original
        duplicates = [e for e in manifest.entries if e.kind == "duplicate"]
        assert len(duplicates) > 0
        assert len(corrupted) == len(raw) + len(duplicates)

    def test_inputs_not_mutated(self) -> None:
        _, observations, _, _ = _pipeline()
        copy = observations.copy(deep=True)
        _corrupt(observations)
        assert_frame_equal(observations, copy)

    @pytest.mark.parametrize(
        "rate",
        [True, "0.02", float("nan"), float("inf"), -0.01, MAX_CORRUPTION_RATE + 0.01],
    )
    def test_invalid_rates_rejected(self, rate: object) -> None:
        _, observations, _, _ = _pipeline()
        with pytest.raises(ValueError):
            _corrupt(observations, missing_rate=rate)

    def test_invalid_outlier_multiplier_rejected(self) -> None:
        _, observations, _, _ = _pipeline()
        for multiplier in (True, "8", float("nan"), 0.5, 1.0):
            with pytest.raises(ValueError):
                inject_observation_corruption(observations, 7, outlier_multiplier=multiplier)

    def test_outlier_overflow_rejected(self) -> None:
        _, observations, _, _ = _pipeline()
        bad = observations.copy()
        bad["traffic_volume"] = 2**62
        with pytest.raises(ValueError, match="overflows int64"):
            inject_observation_corruption(
                bad, 7, missing_rate=0.0, outlier_rate=0.25, duplicate_rate=0.0
            )

    def test_zero_outlier_is_not_recorded(self) -> None:
        _, observations, _, _ = _pipeline()
        frame = observations.copy()
        frame.loc[0, "traffic_volume"] = 0
        frame.loc[0, "rainfall_mm"] = 0.0
        corrupted, manifest = inject_observation_corruption(
            frame, 7, missing_rate=0.0, outlier_rate=0.25, duplicate_rate=0.0
        )
        key = (frame.loc[0, "segment_id"], frame.loc[0, "date"])
        entries = [
            e for e in manifest.entries if e.segment_id == key[0] and e.date == key[1].date()
        ]
        assert not any(e.kind == "outlier" for e in entries)
        row = corrupted.loc[
            (corrupted["segment_id"] == key[0]) & (corrupted["date"] == key[1])
        ].iloc[0]
        assert row["traffic_volume"] == 0
        assert row["rainfall_mm"] == 0.0

    def test_manifest_is_exact_diff_both_directions(self) -> None:
        _, raw, _, _ = _pipeline()
        corrupted, manifest = _corrupt(raw)
        original_by_key = raw.set_index(["segment_id", "date"])
        for entry in manifest.entries:
            if entry.kind == "duplicate":
                continue
            original = original_by_key.loc[
                (entry.segment_id, pd.Timestamp(entry.date)), entry.column
            ]
            mask = (corrupted["segment_id"] == entry.segment_id) & (
                corrupted["date"] == pd.Timestamp(entry.date)
            )
            value = corrupted.loc[mask, entry.column].iloc[0]
            if entry.kind == "missing":
                assert not pd.isna(original)
                assert pd.isna(value)
            else:
                assert not pd.isna(original)
                assert value != original
        duplicates = [e for e in manifest.entries if e.kind == "duplicate"]
        assert len(corrupted) == len(raw) + len(duplicates)
        observed_changes: set[tuple[str, str, str, str]] = set()
        for _, row in raw.iterrows():
            key = (str(row["segment_id"]), pd.Timestamp(row["date"]))
            original = original_by_key.loc[key]
            mask = (corrupted["segment_id"] == key[0]) & (corrupted["date"] == key[1])
            candidates = corrupted.loc[mask]
            for column in ("rainfall_mm", "temperature", "humidity"):
                values = candidates[column]
                if values.isna().any() and not pd.isna(original[column]):
                    observed_changes.add(("missing", key[0], str(key[1].date()), column))
            for column in ("traffic_volume", "rainfall_mm"):
                values = candidates[column].dropna()
                if (values != original[column]).any():
                    observed_changes.add(("outlier", key[0], str(key[1].date()), column))
            for _ in range(len(candidates) - 1):
                observed_changes.add(("duplicate", key[0], str(key[1].date()), ""))
        expected_set = {
            (e.kind, e.segment_id, str(e.date), e.column or "") for e in manifest.entries
        }
        assert observed_changes == expected_set


class TestRawValidation:
    def test_clean_data_passes(self) -> None:
        segments, observations, targets, events = _pipeline()
        report = validate_raw_dataset(segments, observations, targets, events, SPEC)
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.is_valid

    def test_corrupted_raw_has_warnings_but_no_errors(self) -> None:
        segments, observations, targets, events = _pipeline()
        corrupted, _ = _corrupt(observations)
        report = validate_raw_dataset(segments, corrupted, targets, events, SPEC)
        assert report.error_count == 0
        assert report.warning_count > 0
        assert report.is_valid
        codes = set(report.counts_by_code)
        assert {"missing_value", "duplicate_row", "operational_outlier"} <= codes

    def test_missing_required_columns_fail(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.drop(columns=["rainfall_mm"])
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "schema_missing_columns" in report.counts_by_code

    def test_duplicate_column_labels_fail(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad.columns = ["segment_id", "date", "date"] + list(observations.columns[3:])
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "schema_duplicate_column" in report.counts_by_code

    def test_malformed_segment_id_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["segment_id"] = bad["segment_id"].astype(object)
        bad.loc[0, "segment_id"] = "NOT-A-KEY"
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "invalid_segment_id" in report.counts_by_code

    def test_malformed_dates_fail(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["date"] = pd.to_datetime(bad["date"])
        bad.loc[0, "date"] = pd.Timestamp("2022-01-01 12:00:00")
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "non_midnight_date" in report.counts_by_code

    def test_nat_dates_fail(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["date"] = pd.to_datetime(bad["date"])
        bad.loc[0, "date"] = pd.NaT
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "invalid_date" in report.counts_by_code

    def test_wrong_dtypes_fail(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["traffic_volume"] = bad["traffic_volume"].astype("float64")
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "invalid_dtype" in report.counts_by_code

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("traffic_volume", "x"),
            ("heavy_vehicle_ratio", True),
            ("road_condition_score", 3.5),
            ("temperature", float("inf")),
            ("heavy_vehicle_ratio", float("nan")),
        ],
    )
    def test_adversarial_values_fail(self, column: str, value: object) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad[column] = bad[column].astype(object)
        bad.loc[0, column] = value
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert report.error_count > 0

    def test_orphan_segment_reference_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad.loc[0, "segment_id"] = "QL99-KM999-1000"
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "foreign_key" in report.counts_by_code

    def test_invalid_grid_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations[observations["date"] != "2022-03-01"].reset_index(drop=True)
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "observation_date_count" in report.counts_by_code

    def test_invalid_cadence_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["date"] = pd.to_datetime(bad["date"])
        bad.loc[bad["date"] == "2022-03-01", "date"] = pd.Timestamp("2022-03-15")
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "invalid_cadence" in report.counts_by_code

    def test_road_age_mismatch_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad.loc[0, "road_age_days"] = bad.loc[0, "road_age_days"] + 1
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "road_age_mismatch" in report.counts_by_code

    def test_30d_gt_365d_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["accident_count_30d"] = bad["accident_count_365d"] + 1
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "accident_count_inconsistent" in report.counts_by_code

    def test_condition_range_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad.loc[0, "road_condition_score"] = 101
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "condition_score_range" in report.counts_by_code

    def test_target_mismatch_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad_targets = targets.copy()
        bad_targets.loc[0, "maintenance_within_30_days"] = (
            1 - bad_targets.loc[0, "maintenance_within_30_days"]
        )
        report = validate_raw_dataset(segments, observations, bad_targets, events, SPEC)
        assert "target_label_mismatch" in report.counts_by_code

    def test_event_derived_target_mismatch_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        extra_event = pd.DataFrame(
            {
                "segment_id": [events.iloc[0]["segment_id"]],
                "maintenance_date": [pd.Timestamp("2023-01-01")],
            }
        )
        changed_events = pd.concat([events, extra_event], ignore_index=True)
        report = validate_raw_dataset(segments, observations, targets, changed_events, SPEC)
        assert "target_event_inconsistency" in report.counts_by_code

    def test_future_target_column_in_observations_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["days_until_maintenance"] = 0
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "schema_extra_columns" in report.counts_by_code

    def test_conflicting_observation_keys_are_errors(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad.loc[1, "segment_id"] = bad.loc[0, "segment_id"]
        bad.loc[1, "date"] = bad.loc[0, "date"]
        bad.loc[1, "traffic_volume"] = bad.loc[0, "traffic_volume"] + 1
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "conflicting_key" in report.counts_by_code

    def test_exact_duplicates_are_warnings(self) -> None:
        segments, observations, targets, events = _pipeline()
        duplicated = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
        report = validate_raw_dataset(segments, duplicated, targets, events, SPEC)
        assert report.error_count == 0
        assert report.counts_by_code["duplicate_row"] == 1

    def test_issue_counts_codes_and_keys_are_exact(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad.loc[0, "road_condition_score"] = 101
        bad.loc[5, "sign_condition_score"] = 0
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        score_issues = [issue for issue in report.issues if issue.code == "condition_score_range"]
        assert len(score_issues) == 2
        keys = {issue.row_key for issue in score_issues}
        assert keys == {
            f"{bad.loc[0, 'segment_id']}|{bad.loc[0, 'date'].date()}",
            f"{bad.loc[5, 'segment_id']}|{bad.loc[5, 'date'].date()}",
        }
        assert all(issue.severity == "error" for issue in score_issues)


class TestReportAndManifestStructures:
    def test_report_grouped_counts(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad.loc[0, "road_condition_score"] = 101
        bad.loc[1, "road_condition_score"] = 0
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert report.counts_by_code["condition_score_range"] == 2
        assert report.counts_by_table["observations"] == 2
        assert report.counts_by_column["road_condition_score"] == 2
        assert report.is_valid is False

    def test_manifest_counts_by_kind(self) -> None:
        _, observations, _, _ = _pipeline()
        _, manifest = _corrupt(observations)
        counts = manifest.counts_by_kind()
        assert set(counts) == {"missing", "outlier", "duplicate"}
        assert counts["duplicate"] == len([e for e in manifest.entries if e.kind == "duplicate"])

    def test_corruption_rejects_bad_seed_and_rates(self) -> None:
        _, observations, _, _ = _pipeline()
        with pytest.raises(ValueError):
            inject_observation_corruption(observations, True)
        with pytest.raises(ValueError):
            inject_observation_corruption(observations, 0)
        with pytest.raises(ValueError):
            inject_observation_corruption(observations, 7, missing_rate="x")


class TestMoreValidation:
    def _report(self, observations, targets=None, events=None, segments=None):
        base = _pipeline()
        return validate_raw_dataset(
            segments if segments is not None else base[0],
            observations,
            targets if targets is not None else base[2],
            events if events is not None else base[3],
            SPEC,
        )

    def test_segment_schema_extra_columns_fail(self) -> None:
        _, observations, targets, events = _pipeline()
        segments, _, _, _ = _pipeline()
        full = generate_segments(SPEC, 42, observation_start=START)
        report = validate_raw_dataset(full, observations, targets, events, SPEC)
        assert "schema_extra_columns" in report.counts_by_code

    def test_segment_count_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        report = validate_raw_dataset(segments.iloc[:-1], observations, targets, events, SPEC)
        assert "segment_count" in report.counts_by_code

    def test_invalid_categories_fail(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = segments.copy()
        bad.loc[0, "province"] = "XX"
        bad.loc[1, "road_type"] = "maglev"
        report = validate_raw_dataset(bad, observations, targets, events, SPEC)
        assert report.counts_by_code["invalid_category"] == 2

    def test_road_length_range_fails(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = segments.copy()
        bad.loc[0, "road_length_km"] = 0.0
        report = validate_raw_dataset(bad, observations, targets, events, SPEC)
        assert "road_length_range" in report.counts_by_code

    def test_dslm_range_fails(self) -> None:
        observations = _pipeline()[1].copy()
        observations.loc[0, "days_since_last_maintenance"] = (
            observations.loc[0, "road_age_days"] + 1
        )
        report = self._report(observations)
        assert "days_since_maintenance_range" in report.counts_by_code

    def test_previous_repairs_negative_fails(self) -> None:
        observations = _pipeline()[1].copy()
        observations.loc[0, "previous_repairs"] = -1
        report = self._report(observations)
        assert "previous_repairs_negative" in report.counts_by_code

    def test_traffic_negative_fails(self) -> None:
        observations = _pipeline()[1].copy()
        observations.loc[0, "traffic_volume"] = -5
        report = self._report(observations)
        assert "traffic_volume_negative" in report.counts_by_code

    def test_rainfall_negative_fails(self) -> None:
        observations = _pipeline()[1].copy()
        observations.loc[0, "rainfall_mm"] = -1.0
        report = self._report(observations)
        assert "rainfall_negative" in report.counts_by_code

    def test_temperature_and_humidity_range_fail(self) -> None:
        observations = _pipeline()[1].copy()
        observations.loc[0, "temperature"] = 61.0
        observations.loc[1, "humidity"] = 101.0
        report = self._report(observations)
        assert report.counts_by_code["temperature_range"] == 1
        assert report.counts_by_code["humidity_range"] == 1

    def test_accident_negative_fails(self) -> None:
        observations = _pipeline()[1].copy()
        observations.loc[0, "accident_count_365d"] = -2
        report = self._report(observations)
        assert "accident_count_negative" in report.counts_by_code

    def test_target_days_negative_fails(self) -> None:
        targets = _pipeline()[2].copy()
        targets.loc[0, "days_until_maintenance"] = -1
        report = self._report(_pipeline()[1], targets=targets)
        assert "target_days_negative" in report.counts_by_code

    def test_target_label_invalid_fails(self) -> None:
        targets = _pipeline()[2].copy()
        targets.loc[0, "maintenance_within_30_days"] = 7
        report = self._report(_pipeline()[1], targets=targets)
        assert "target_label_invalid" in report.counts_by_code

    def test_duplicate_target_keys_fail(self) -> None:
        targets = pd.concat([_pipeline()[2], _pipeline()[2].iloc[[0]]], ignore_index=True)
        report = self._report(_pipeline()[1], targets=targets)
        assert "duplicate_key" in report.counts_by_code

    def test_duplicate_event_keys_fail(self) -> None:
        events = pd.concat([_pipeline()[3], _pipeline()[3].iloc[[0]]], ignore_index=True)
        report = self._report(_pipeline()[1], events=events)
        assert "duplicate_key" in report.counts_by_code

    def test_target_orphan_segment_fails(self) -> None:
        targets = _pipeline()[2].copy()
        targets.loc[0, "segment_id"] = "QL99-KM999-1000"
        report = self._report(_pipeline()[1], targets=targets)
        assert "foreign_key" in report.counts_by_code

    def test_timezone_aware_observation_date_fails(self) -> None:
        observations = _pipeline()[1].copy()
        observations["date"] = pd.to_datetime(observations["date"]).dt.tz_localize("UTC")
        report = self._report(observations)
        assert "timezone_aware_date" in report.counts_by_code

    def test_cleaned_mode_escalates_duplicates_and_missing(self) -> None:
        segments, observations, targets, events = _pipeline()
        duplicated = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
        raw_report = validate_raw_dataset(segments, duplicated, targets, events, SPEC)
        assert raw_report.error_count == 0
        cleaned_report = validate_cleaned_dataset(segments, duplicated, targets, events, SPEC)
        assert "duplicate_key" in cleaned_report.counts_by_code
        missing = observations.copy()
        missing.loc[0, "rainfall_mm"] = np.nan
        cleaned_missing = validate_cleaned_dataset(segments, missing, targets, events, SPEC)
        assert "missing_value" in cleaned_missing.counts_by_code


class TestCleaningRejectsInvalidInput:
    def test_missing_public_segment_columns_fail(self) -> None:
        segments, observations, targets, events = _pipeline()
        with pytest.raises(ValueError):
            clean_raw_dataset(
                segments.drop(columns=["road_length_km"]), observations, targets, events, SPEC
            )

    def test_duplicate_public_segment_labels_fail(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = segments.copy()
        bad.columns = ["segment_id", "province", "road_type", "construction_date", "segment_id"]
        with pytest.raises(ValueError):
            clean_raw_dataset(bad, observations, targets, events, SPEC)

    def test_missing_observation_columns_fail(self) -> None:
        segments, observations, targets, events = _pipeline()
        raw = observations.drop(columns=["humidity"])
        with pytest.raises(ValueError):
            clean_raw_dataset(segments, raw, targets, events, SPEC)


class TestNonFiniteAndCategoricalFailClosed:
    def _report(self, observations=None, segments=None):
        base = _pipeline()
        return validate_raw_dataset(
            segments if segments is not None else base[0],
            observations if observations is not None else base[1],
            base[2],
            base[3],
            SPEC,
        )

    def test_infinite_rainfall_is_error_not_warning(self) -> None:
        observations = _pipeline()[1].copy()
        observations.loc[0, "rainfall_mm"] = float("inf")
        report = self._report(observations)
        assert "non_finite" in report.counts_by_code
        assert "operational_outlier" not in report.counts_by_code
        assert report.is_valid is False

    def test_negative_infinite_temperature_is_error(self) -> None:
        observations = _pipeline()[1].copy()
        observations.loc[0, "temperature"] = float("-inf")
        report = self._report(observations)
        assert "non_finite" in report.counts_by_code

    def test_nan_heavy_ratio_is_non_finite_error(self) -> None:
        observations = _pipeline()[1].copy()
        observations.loc[0, "heavy_vehicle_ratio"] = float("nan")
        report = self._report(observations)
        assert "non_finite" in report.counts_by_code

    def test_clean_rejects_infinite_rainfall(self) -> None:
        segments, observations, targets, events = _pipeline(
            spec=DatasetSpec(
                dataset_segments=10, dataset_months_per_segment=12, dataset_observations=120
            )
        )
        bad = observations.copy()
        bad.loc[0, "rainfall_mm"] = float("inf")
        from roadguard.data_quality import clean_raw_dataset

        with pytest.raises(ValueError):
            clean_raw_dataset(segments, bad, targets, events, SPEC)

    @pytest.mark.parametrize(
        "value",
        [pd.NA, None, [1], {"a": 1}, np.array([1]), 123.5],
    )
    def test_categorical_non_scalar_fails_closed(self, value: object) -> None:
        segments, observations, targets, events = _pipeline()
        bad = segments.copy()
        bad["province"] = pd.Series(
            [value] + list(segments["province"].iloc[1:]),
            dtype=object,
            index=segments.index,
        )
        report = validate_raw_dataset(bad, observations, targets, events, SPEC)
        assert "invalid_category" in report.counts_by_code

    @pytest.mark.parametrize(
        "value",
        [pd.NA, None, ["national"], 5],
    )
    def test_road_type_non_scalar_fails_closed(self, value: object) -> None:
        segments, observations, targets, events = _pipeline()
        bad = segments.copy()
        bad["road_type"] = pd.Series(
            [value] + list(segments["road_type"].iloc[1:]),
            dtype=object,
            index=segments.index,
        )
        report = validate_raw_dataset(bad, observations, targets, events, SPEC)
        assert "invalid_category" in report.counts_by_code


class TestCorruptionInputStrictness:
    def _base(self):
        _, observations, _, _ = _pipeline()
        return observations

    def test_extra_columns_rejected(self) -> None:
        bad = self._base().copy()
        bad["days_until_maintenance"] = 0
        with pytest.raises(ValueError, match="extra"):
            inject_observation_corruption(bad, 7)

    def test_object_dates_rejected(self) -> None:
        bad = self._base().copy()
        bad["date"] = bad["date"].astype(object)
        with pytest.raises(ValueError, match="date"):
            inject_observation_corruption(bad, 7)

    def test_fractional_integer_column_rejected(self) -> None:
        bad = self._base().copy()
        bad["traffic_volume"] = bad["traffic_volume"].astype("float64")
        bad.loc[0, "traffic_volume"] = 3.5
        with pytest.raises(ValueError, match="dtype"):
            inject_observation_corruption(bad, 7)

    def test_numeric_string_rejected(self) -> None:
        bad = self._base().copy()
        bad["traffic_volume"] = bad["traffic_volume"].astype(object)
        bad.loc[0, "traffic_volume"] = "1000"
        with pytest.raises(ValueError, match="dtype"):
            inject_observation_corruption(bad, 7)

    def test_boolean_value_rejected(self) -> None:
        bad = self._base().copy()
        bad["heavy_vehicle_ratio"] = bad["heavy_vehicle_ratio"].astype(object)
        bad.loc[0, "heavy_vehicle_ratio"] = True
        with pytest.raises(ValueError, match="dtype"):
            inject_observation_corruption(bad, 7)

    def test_nan_value_rejected(self) -> None:
        bad = self._base().copy()
        bad["heavy_vehicle_ratio"] = bad["heavy_vehicle_ratio"].astype(object)
        bad.loc[0, "heavy_vehicle_ratio"] = float("nan")
        with pytest.raises(ValueError):
            inject_observation_corruption(bad, 7)

    def test_duplicate_keys_rejected(self) -> None:
        bad = pd.concat([self._base(), self._base().iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            inject_observation_corruption(bad, 7)

    def test_conflicting_keys_rejected(self) -> None:
        bad = self._base().copy()
        bad.loc[1, "segment_id"] = bad.loc[0, "segment_id"]
        bad.loc[1, "date"] = bad.loc[0, "date"]
        bad.loc[1, "traffic_volume"] = bad.loc[0, "traffic_volume"] + 1
        with pytest.raises(ValueError, match="conflicting"):
            inject_observation_corruption(bad, 7)

    def test_non_midnight_date_rejected(self) -> None:
        bad = self._base().copy()
        bad["date"] = pd.to_datetime(bad["date"])
        bad.loc[0, "date"] = pd.Timestamp("2022-01-01 12:00:00")
        with pytest.raises(ValueError, match="midnight"):
            inject_observation_corruption(bad, 7)

    def test_missing_column_rejected(self) -> None:
        bad = self._base().drop(columns=["rainfall_mm"])
        with pytest.raises(ValueError, match="missing"):
            inject_observation_corruption(bad, 7)


MINI_SPEC = DatasetSpec(dataset_segments=2, dataset_months_per_segment=12, dataset_observations=24)


def _mini(seed: int = 42):
    segments = generate_segments(MINI_SPEC, seed, observation_start=START)
    events = generate_maintenance_events(segments, MINI_SPEC, seed, start_date=START)
    timeline = generate_accident_timeline(segments, MINI_SPEC, seed, start_date=START)
    observations = generate_observations(
        segments, events, timeline, MINI_SPEC, seed, start_date=START
    )
    targets = derive_observation_targets(observations, events)
    public_segments = segments[list(PUBLIC_SEGMENT_COLUMNS)].copy()
    public_segments["construction_date"] = pd.to_datetime(public_segments["construction_date"])
    return public_segments, observations, targets, events


class TestReviewTrafficArithmetic:
    def _frame_with(self, value: int) -> pd.DataFrame:
        observations = _mini()[1].copy()
        observations["traffic_volume"] = value
        return observations

    def test_values_above_2_53_retain_exact_arithmetic(self) -> None:
        base = 2**53 + 10
        corrupted, _ = inject_observation_corruption(
            self._frame_with(base),
            7,
            missing_rate=0.0,
            outlier_rate=0.25,
            duplicate_rate=0.0,
            outlier_multiplier=2.0,
        )
        changed = corrupted[corrupted["traffic_volume"] != base]
        assert len(changed) > 0
        assert set(changed["traffic_volume"]) == {2**54 + 20}
        assert (corrupted["traffic_volume"] >= 0).all()

    def test_near_int64_max_does_not_wrap_negative(self) -> None:
        base = np.iinfo(np.int64).max - 2049
        multiplier = np.nextafter(1.0, 2.0)
        corrupted, manifest = inject_observation_corruption(
            self._frame_with(base),
            2,
            missing_rate=0.0,
            outlier_rate=0.25,
            duplicate_rate=0.0,
            outlier_multiplier=multiplier,
        )
        changed = corrupted[corrupted["traffic_volume"] != base]
        assert len(changed) > 0
        assert set(changed["traffic_volume"]) == {np.iinfo(np.int64).max - 1}
        assert (corrupted["traffic_volume"] >= 0).all()
        assert all(
            entry.column == "traffic_volume" and entry.kind == "outlier"
            for entry in manifest.entries
        )

    def test_genuine_positive_overflow_rejected(self) -> None:
        with pytest.raises(ValueError, match="overflows int64"):
            inject_observation_corruption(
                self._frame_with(2**62),
                7,
                missing_rate=0.0,
                outlier_rate=0.25,
                duplicate_rate=0.0,
            )

    def test_no_wraparound_can_occur(self) -> None:
        for base in (np.iinfo(np.int64).max - 1, 2**53 + 1, 2**62):
            try:
                corrupted, _ = inject_observation_corruption(
                    self._frame_with(base),
                    3,
                    missing_rate=0.0,
                    outlier_rate=0.25,
                    duplicate_rate=0.0,
                )
            except ValueError:
                continue
            assert (corrupted["traffic_volume"] >= 0).all()


class TestReviewUnicodeIds:
    UNICODE_DIGIT_ID = "QL٠١-KM١-١"

    def test_unicode_digit_id_rejected_by_validation(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["segment_id"] = pd.Series(
            [self.UNICODE_DIGIT_ID] + list(observations["segment_id"].iloc[1:]),
            dtype=object,
        )
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert "invalid_segment_id" in report.counts_by_code

    def test_unicode_digit_id_corruption_raises_contextual_value_error(self) -> None:
        observations = _pipeline()[1].copy()
        observations["segment_id"] = pd.Series(
            [self.UNICODE_DIGIT_ID] + list(observations["segment_id"].iloc[1:]),
            dtype=object,
        )
        with pytest.raises(ValueError, match="malformed"):
            inject_observation_corruption(observations, 7)

    def test_normal_ids_remain_valid(self) -> None:
        segments, observations, targets, events = _pipeline()
        report = validate_raw_dataset(segments, observations, targets, events, SPEC)
        assert report.error_count == 0


class TestReviewEventInvariants:
    def _events_frame(self, rows: list[tuple[str, str]]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "segment_id": pd.Series([sid for sid, _ in rows], dtype=object),
                "maintenance_date": pd.to_datetime([day for _, day in rows]),
            }
        )

    def test_event_one_day_before_construction_rejected(self) -> None:
        segments, observations, targets, events = _mini()
        sid = sorted(observations["segment_id"].unique())[0]
        bad_segments = segments.copy()
        bad_segments.loc[bad_segments["segment_id"] == sid, "construction_date"] = pd.Timestamp(
            "2021-06-01"
        )
        bad_events = self._events_frame([(sid, "2021-05-31")])
        report = validate_raw_dataset(bad_segments, observations, targets, bad_events, MINI_SPEC)
        assert "event_before_construction" in report.counts_by_code

    def test_two_distinct_dates_same_segment_month_rejected(self) -> None:
        segments, observations, targets, events = _mini()
        sid = sorted(observations["segment_id"].unique())[0]
        bad_events = self._events_frame([(sid, "2022-01-05"), (sid, "2022-01-20")])
        report = validate_raw_dataset(segments, observations, targets, bad_events, MINI_SPEC)
        assert "event_month_conflict" in report.counts_by_code

    def test_exact_duplicate_event_key_rejected(self) -> None:
        segments, observations, targets, events = _mini()
        sid = sorted(observations["segment_id"].unique())[0]
        bad_events = self._events_frame([(sid, "2022-01-05"), (sid, "2022-01-05")])
        report = validate_raw_dataset(segments, observations, targets, bad_events, MINI_SPEC)
        assert "duplicate_key" in report.counts_by_code

    def test_valid_events_across_different_months_pass(self) -> None:
        segments, observations, targets, events = _mini()
        sid = sorted(observations["segment_id"].unique())[0]
        ok_events = self._events_frame([(sid, "2022-01-05"), (sid, "2022-02-05")])
        report = validate_raw_dataset(segments, observations, targets, ok_events, MINI_SPEC)
        assert "event_before_construction" not in report.counts_by_code
        assert "event_month_conflict" not in report.counts_by_code

    def test_cleaned_validation_reports_event_errors(self) -> None:
        segments, observations, targets, events = _mini()
        sid = sorted(observations["segment_id"].unique())[0]
        bad_events = self._events_frame([(sid, "2022-01-05"), (sid, "2022-01-20")])
        report = validate_cleaned_dataset(segments, observations, targets, bad_events, MINI_SPEC)
        assert "event_month_conflict" in report.counts_by_code


class TestReviewPreEpoch:
    def _shifted(self, target_first: str) -> pd.DataFrame:
        observations = _mini()[1].copy()
        offset_days = (pd.Timestamp("2022-01-01") - pd.Timestamp(target_first)).days
        observations["date"] = pd.to_datetime(observations["date"]) - pd.offsets.Day(offset_days)
        observations["road_age_days"] = 4000
        return observations

    def test_pre_epoch_1960_succeeds(self) -> None:
        corrupted, manifest = inject_observation_corruption(
            self._shifted("1960-01-01"),
            7,
            missing_rate=0.05,
            outlier_rate=0.05,
            duplicate_rate=0.1,
        )
        assert len(corrupted) > 0
        assert len(manifest.entries) > 0

    def test_epoch_1970_succeeds(self) -> None:
        corrupted, _ = inject_observation_corruption(
            self._shifted("1970-01-01"),
            7,
            missing_rate=0.05,
            outlier_rate=0.05,
            duplicate_rate=0.1,
        )
        assert len(corrupted) > 0

    def test_equal_distance_pre_and_post_epoch_streams_differ(self) -> None:
        before = self._shifted("1969-12-31")
        after = self._shifted("1970-01-02")
        differing = False
        for seed in range(1, 8):
            b_frame, _ = inject_observation_corruption(
                before, seed, missing_rate=0.25, outlier_rate=0.0, duplicate_rate=0.0
            )
            a_frame, _ = inject_observation_corruption(
                after, seed, missing_rate=0.25, outlier_rate=0.0, duplicate_rate=0.0
            )
            row_diff = (b_frame["rainfall_mm"].isna() != a_frame["rainfall_mm"].isna()).sum()
            if row_diff > 0:
                differing = True
                break
        assert differing

    def test_shuffled_pre_epoch_input_invariant(self) -> None:
        frame = self._shifted("1960-01-01")
        base = inject_observation_corruption(
            frame, 7, missing_rate=0.05, outlier_rate=0.05, duplicate_rate=0.1
        )
        shuffled = frame.sample(frac=1.0, random_state=9)
        rerun = inject_observation_corruption(
            shuffled, 7, missing_rate=0.05, outlier_rate=0.05, duplicate_rate=0.1
        )
        assert_frame_equal(base[0], rerun[0])
        assert base[1] == rerun[1]

    def test_post_epoch_entropy_uses_plain_day_offset(self) -> None:
        observations = _mini()[1].copy()
        sid = sorted(observations["segment_id"].unique())[0]
        single = (
            observations[observations["segment_id"] == sid].iloc[:2].copy().reset_index(drop=True)
        )
        offset = (pd.Timestamp("2022-01-01") - pd.Timestamp("1970-01-01")).days
        single["date"] = pd.to_datetime(single["date"]) - pd.offsets.Day(offset)
        single["road_age_days"] = 4000
        corrupted, _ = inject_observation_corruption(
            single, 7, missing_rate=0.25, outlier_rate=0.0, duplicate_rate=0.0
        )
        second = corrupted.iloc[1]
        key = int.from_bytes(sid.encode("ascii"), "big", signed=False)
        day_second = (pd.Timestamp("1970-02-01").date() - date(1970, 1, 1)).days
        rng = np.random.default_rng(np.random.SeedSequence([7, key, 0x524735, day_second]))
        expected = {
            column for column in ("rainfall_mm", "temperature", "humidity") if rng.random() < 0.25
        }
        actual = {
            column
            for column in ("rainfall_mm", "temperature", "humidity")
            if pd.isna(second[column])
        }
        assert actual == expected


class TestReviewMultiplierTypes:
    def _corrupt(self, multiplier: object):
        observations = _mini()[1].copy()
        observations["traffic_volume"] = 1000
        return inject_observation_corruption(
            observations,
            7,
            missing_rate=0.0,
            outlier_rate=0.25,
            duplicate_rate=0.0,
            outlier_multiplier=multiplier,
        )

    @pytest.mark.parametrize(
        "multiplier",
        [2, 2.0, np.int8(2), np.int64(2), np.uint64(2), np.float32(2.0), np.float64(2.0)],
    )
    def test_valid_multiplier_types_accepted(self, multiplier: object) -> None:
        corrupted, manifest = self._corrupt(multiplier)
        assert any(entry.kind == "outlier" for entry in manifest.entries)
        changed = corrupted[corrupted["traffic_volume"] != 1000]
        assert len(changed) > 0
        assert set(changed["traffic_volume"]) == {2000}

    @pytest.mark.parametrize(
        "multiplier",
        [True, np.bool_(True), "2", float("nan"), float("inf")],
    )
    def test_invalid_multiplier_types_rejected(self, multiplier: object) -> None:
        with pytest.raises(ValueError):
            self._corrupt(multiplier)

    def test_excessively_large_integer_multiplier_overflows_contextually(self) -> None:
        with pytest.raises(ValueError, match="overflows int64"):
            self._corrupt(2**70)


class TestReviewCategoryDeterminism:
    @pytest.mark.parametrize(
        "value",
        [{"NA", "TH"}, frozenset({"NA"}), {"a": 1}, [1], np.array([1]), pd.NA, None],
    )
    def test_set_valued_categories_fail_closed_deterministically(self, value: object) -> None:
        segments, observations, targets, events = _pipeline()
        bad = segments.copy()
        bad["province"] = pd.Series([value] + list(segments["province"].iloc[1:]), dtype=object)
        report = validate_raw_dataset(bad, observations, targets, events, SPEC)
        issues = [issue for issue in report.issues if issue.code == "invalid_category"]
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "unsupported value of type" in issues[0].message

    def test_identical_reports_across_hash_seeds(self, tmp_path) -> None:
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import sys
            from datetime import date

            import pandas as pd

            from roadguard import (
                DatasetSpec,
                derive_observation_targets,
                generate_accident_timeline,
                generate_maintenance_events,
                generate_observations,
                generate_segments,
            )
            from roadguard.data_quality import PUBLIC_SEGMENT_COLUMNS, validate_raw_dataset

            spec = DatasetSpec(
                dataset_segments=10,
                dataset_months_per_segment=12,
                dataset_observations=120,
            )
            start = date(2022, 1, 1)
            segments = generate_segments(spec, 42, observation_start=start)
            events = generate_maintenance_events(segments, spec, 42, start_date=start)
            timeline = generate_accident_timeline(segments, spec, 42, start_date=start)
            observations = generate_observations(
                segments, events, timeline, spec, 42, start_date=start
            )
            targets = derive_observation_targets(observations, events)
            public = segments[list(PUBLIC_SEGMENT_COLUMNS)].copy()
            public["construction_date"] = pd.to_datetime(public["construction_date"])
            public["province"] = pd.Series(
                [{"NA", "TH"}] + list(public["province"].iloc[1:]), dtype=object
            )
            report = validate_raw_dataset(public, observations, targets, events, spec)
            rows = [(i.severity, i.code, i.column, i.row_key, i.message) for i in report.issues]
            with open(sys.argv[1], "w", encoding="utf-8") as handle:
                handle.write(repr(rows))
            """
        )
        outputs = []
        for seed in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outfile = tmp_path / f"hashseed_{seed}.txt"
            subprocess.run(
                [sys.executable, "-c", script, str(outfile)],
                env=env,
                check=True,
            )
            outputs.append(outfile.read_text(encoding="utf-8"))
        assert outputs[0] == outputs[1]


class TestReviewMixedTypeSchemaLabels:
    def test_mixed_str_int_extra_columns_do_not_raise(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["extra"] = 1
        bad[123] = 2
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert report.counts_by_code["schema_extra_columns"] == 2
        assert all(isinstance(issue.column, str) for issue in report.issues)

    def test_mixed_schema_labels_report_is_deterministic(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["extra"] = 1
        bad[123] = 2
        first = validate_raw_dataset(segments, bad, targets, events, SPEC)
        second = validate_raw_dataset(segments, bad, targets, events, SPEC)
        assert first.issues == second.issues
        extras = [issue for issue in first.issues if issue.code == "schema_extra_columns"]
        assert any(
            issue.column == f"extra[{list(bad.columns).index('extra')}]:str:extra"
            for issue in extras
        )
        assert any(issue.column == f"extra[{list(bad.columns).index(123)}]:int" for issue in extras)

    def test_pd_na_extra_column_label(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad[pd.NA] = 1
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        extras = [issue for issue in report.issues if issue.code == "schema_extra_columns"]
        assert len(extras) == 1
        assert extras[0].column == f"extra[{len(bad.columns) - 1}]:NAType"

    def test_canonical_label_collision_pairs_are_distinct(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad["123"] = 1
        bad[123] = 2
        bad["<frozenset>"] = 3
        bad[frozenset({"x"})] = 4
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        extras = sorted(
            issue.column for issue in report.issues if issue.code == "schema_extra_columns"
        )
        base = len(bad.columns) - 4
        assert extras == [
            f"extra[{base}]:str:123",
            f"extra[{base + 1}]:int",
            f"extra[{base + 2}]:str:<frozenset>",
            f"extra[{base + 3}]:frozenset",
        ]

    def test_set_label_extra_column_reports_stable_type(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad[frozenset({"x", "y"})] = 1
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        extras = [issue for issue in report.issues if issue.code == "schema_extra_columns"]
        assert len(extras) == 1
        assert extras[0].column == f"extra[{len(bad.columns) - 1}]:frozenset"
        assert "{" not in extras[0].message


class TestReviewExtremeNumericArguments:
    def test_huge_integer_missing_rate_raises_contextual_value_error(self) -> None:
        observations = _mini()[1].copy()
        with pytest.raises(ValueError, match="MAX_CORRUPTION_RATE"):
            inject_observation_corruption(
                observations,
                7,
                missing_rate=10**1000,
                outlier_rate=0.0,
                duplicate_rate=0.0,
            )

    def test_huge_integer_outlier_multiplier_rainfall_raises_contextual_value_error(
        self,
    ) -> None:
        observations = _mini()[1].copy()
        observations["traffic_volume"] = 1000
        with pytest.raises(ValueError, match="multiplier is too large"):
            inject_observation_corruption(
                observations,
                1,
                missing_rate=0.0,
                outlier_rate=0.25,
                duplicate_rate=0.0,
                outlier_multiplier=10**1000,
            )

    def test_huge_integer_outlier_multiplier_traffic_overflows_contextually(self) -> None:
        observations = _mini()[1].copy()
        observations["traffic_volume"] = 2**62
        with pytest.raises(ValueError, match="overflows int64"):
            inject_observation_corruption(
                observations,
                2,
                missing_rate=0.0,
                outlier_rate=0.25,
                duplicate_rate=0.0,
                outlier_multiplier=2**100,
            )


class TestReviewExactIndexResults:
    def test_bool_index_rate_rejected_before_comparisons(self) -> None:
        with pytest.raises(ValueError, match="finite number"):
            inject_observation_corruption(
                _mini()[1].copy(),
                7,
                missing_rate=_BoolIndexInt(2),
                outlier_rate=0.0,
                duplicate_rate=0.0,
            )

    def test_bool_index_seed_rejected(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            inject_observation_corruption(_mini()[1].copy(), _BoolIndexInt(2))

    def test_bool_index_multiplier_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite number"):
            inject_observation_corruption(
                _mini()[1].copy(),
                7,
                missing_rate=0.0,
                outlier_rate=0.25,
                duplicate_rate=0.0,
                outlier_multiplier=_BoolIndexInt(2),
            )

    def test_int_subclass_index_rate_comparisons_do_not_escape(self) -> None:
        with pytest.raises(ValueError, match="finite number"):
            inject_observation_corruption(
                _mini()[1].copy(),
                7,
                missing_rate=_ComparisonsRaiseIndexInt(2),
                outlier_rate=0.0,
                duplicate_rate=0.0,
            )

    def test_int_subclass_index_multiplier_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite number"):
            inject_observation_corruption(
                _mini()[1].copy(),
                7,
                missing_rate=0.0,
                outlier_rate=0.25,
                duplicate_rate=0.0,
                outlier_multiplier=_ComparisonsRaiseIndexInt(2),
            )

    def test_int_subclass_index_seed_comparisons_do_not_escape(self) -> None:
        with pytest.raises(ValueError, match="seed"):
            inject_observation_corruption(_mini()[1].copy(), _ComparisonsRaiseIndexInt(2))


class TestReviewExactStringBoundary:
    def test_hostile_id_hash_never_called_validation_returns_report(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = segments.copy()
        bad["segment_id"] = pd.Series(
            [_HostileId("QL01-KM1-1")] + list(bad["segment_id"].iloc[1:]), dtype=object
        )
        report = validate_raw_dataset(bad, observations, targets, events, SPEC)
        issues = [issue for issue in report.issues if issue.code == "invalid_segment_id"]
        assert len(issues) == 1
        assert issues[0].row_key == "row[0]:_HostileId"

    def test_hostile_category_eq_never_called_validation_returns_report(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = segments.copy()
        bad["province"] = pd.Series(
            [_HostileCategory("NA")] + list(bad["province"].iloc[1:]), dtype=object
        )
        bad["road_type"] = pd.Series(
            [_HostileCategory("primary")] + list(bad["road_type"].iloc[1:]), dtype=object
        )
        report = validate_raw_dataset(bad, observations, targets, events, SPEC)
        issues = [issue for issue in report.issues if issue.code == "invalid_category"]
        assert len(issues) == 2
        assert all(
            "unsupported value of type _HostileCategory" in issue.message for issue in issues
        )

    def test_hostile_id_corruption_raises_contextual_value_error(self) -> None:
        observations = _mini()[1].copy()
        observations["segment_id"] = pd.Series(
            [_HostileId("QL01-KM1-1")] + list(observations["segment_id"].iloc[1:]),
            dtype=object,
        )
        with pytest.raises(ValueError, match="malformed segment_id"):
            inject_observation_corruption(observations, 7)


class _BoolIndexInt(int):
    def __index__(self) -> int:
        return True


class _ComparisonsRaise(int):
    def __lt__(self, other: object) -> bool:
        raise RuntimeError("hostile lt")

    def __gt__(self, other: object) -> bool:
        raise RuntimeError("hostile gt")

    def __eq__(self, other: object) -> bool:
        raise RuntimeError("hostile eq")


class _ComparisonsRaiseIndexInt(int):
    def __index__(self) -> int:
        return _ComparisonsRaise(1)


class _HostileId(str):
    def __hash__(self) -> int:
        raise RuntimeError("hostile id hash")


class _HostileCategory(str):
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("hostile category eq")


class TestReviewCanonicalArgumentNormalization:
    def test_liar_float_rate_does_not_create_full_corruption(self) -> None:
        corrupted, manifest = inject_observation_corruption(
            _mini()[1].copy(),
            7,
            missing_rate=_LiarFloat(1.0),
            outlier_rate=0.0,
            duplicate_rate=0.0,
        )
        assert sum(1 for e in manifest.entries if e.kind == "missing") == 0
        assert corrupted["rainfall_mm"].isna().sum() == 0

    def test_hostile_comparisons_do_not_escape(self) -> None:
        corrupted, manifest = inject_observation_corruption(
            _mini()[1].copy(),
            7,
            missing_rate=_HostileCmp(0.1),
            outlier_rate=0.0,
            duplicate_rate=0.0,
        )
        assert sum(1 for e in manifest.entries if e.kind == "missing") > 0

    def test_split_int_uses_single_canonical_multiplier(self) -> None:
        base = _mini()[1].copy()
        corrupted, manifest = inject_observation_corruption(
            base,
            2,
            missing_rate=0.0,
            outlier_rate=0.25,
            duplicate_rate=0.0,
            outlier_multiplier=_SplitInt(2),
        )
        traffic_mask = corrupted["traffic_volume"] != base["traffic_volume"]
        rain_mask = corrupted["rainfall_mm"] != base["rainfall_mm"]
        assert len(corrupted[traffic_mask]) > 0
        assert len(corrupted[rain_mask]) > 0
        assert set(corrupted.loc[traffic_mask, "traffic_volume"]) == set(
            base.loc[traffic_mask, "traffic_volume"] * 3
        )
        assert set(corrupted.loc[rain_mask, "rainfall_mm"]) == set(
            (base.loc[rain_mask, "rainfall_mm"] * 3).round(1)
        )

    def test_hostile_seed_subclass_rejected_contextually(self) -> None:
        with pytest.raises(ValueError):
            inject_observation_corruption(_mini()[1].copy(), _HostileSeed(5))


class TestReviewCorruptionSchemaSafety:
    def test_extra_pd_na_column_rejected_contextually(self) -> None:
        observations = _mini()[1].copy()
        observations[pd.NA] = 1
        with pytest.raises(ValueError, match="extra"):
            inject_observation_corruption(observations, 7)

    def test_extra_np_int64_labels_distinct_locators(self) -> None:
        observations = _mini()[1].copy()
        observations[np.int64(123)] = 1
        observations[np.int64(456)] = 2
        with pytest.raises(ValueError) as excinfo:
            inject_observation_corruption(observations, 7)
        message = str(excinfo.value)
        base = len(observations.columns) - 2
        assert f"extra[{base}]:int64" in message
        assert f"extra[{base + 1}]:int64" in message


class TestReviewIssueLocators:
    def test_extra_label_locators_position_based(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad[pd.Timestamp("2020-01-01")] = 1
        bad[("a", "b")] = 2
        bad[frozenset({"x"})] = 3
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        extras = [issue.column for issue in report.issues if issue.code == "schema_extra_columns"]
        base = len(bad.columns) - 3
        assert extras == [
            f"extra[{base}]:Timestamp",
            f"extra[{base + 1}]:tuple",
            f"extra[{base + 2}]:frozenset",
        ]

    def test_hostile_str_subclass_label_no_eq_called(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.copy()
        bad[_HostileStr("evil")] = 1
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        extras = [issue for issue in report.issues if issue.code == "schema_extra_columns"]
        assert len(extras) == 1
        assert extras[0].column == f"extra[{len(bad.columns) - 1}]:_HostileStr"

    def test_malformed_numpy_int_ids_with_nat_dates_have_position_keys(self) -> None:
        segments, observations, targets, events = _pipeline()
        bad = observations.iloc[:2].copy()
        bad["segment_id"] = pd.Series([np.int64(123), np.int64(456)], dtype=object)
        bad["date"] = pd.to_datetime([pd.NaT, pd.NaT])
        report = validate_raw_dataset(segments, bad, targets, events, SPEC)
        id_issues = [issue for issue in report.issues if issue.code == "invalid_segment_id"]
        date_issues = [issue for issue in report.issues if issue.code == "invalid_date"]
        assert [issue.row_key for issue in id_issues] == ["row[0]:int64", "row[1]:int64"]
        assert [issue.row_key for issue in date_issues] == [
            "row[0]:int64|NaT",
            "row[1]:int64|NaT",
        ]


class _LiarFloat(float):
    def __float__(self) -> float:
        return 0.0


class _HostileCmp(float):
    def __lt__(self, other: object) -> bool:
        raise RuntimeError("hostile lt")

    def __gt__(self, other: object) -> bool:
        raise RuntimeError("hostile gt")


class _SplitInt(int):
    def __index__(self) -> int:
        return 3


class _HostileSeed(int):
    def __index__(self) -> int:
        raise RuntimeError("hostile seed index")


class _HostileStr(str):
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("hostile eq")

    def __hash__(self) -> int:
        return hash("hostile")


class TestReviewRowKeyDeterminism:
    def test_observation_set_id_na_date_reports_across_hash_seeds(self, tmp_path) -> None:
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import sys
            from datetime import date

            import pandas as pd

            from roadguard import (
                DatasetSpec,
                derive_observation_targets,
                generate_accident_timeline,
                generate_maintenance_events,
                generate_observations,
                generate_segments,
            )
            from roadguard.data_quality import PUBLIC_SEGMENT_COLUMNS, validate_raw_dataset

            spec = DatasetSpec(
                dataset_segments=10,
                dataset_months_per_segment=12,
                dataset_observations=120,
            )
            start = date(2022, 1, 1)
            segments = generate_segments(spec, 42, observation_start=start)
            events = generate_maintenance_events(segments, spec, 42, start_date=start)
            timeline = generate_accident_timeline(segments, spec, 42, start_date=start)
            observations = generate_observations(
                segments, events, timeline, spec, 42, start_date=start
            )
            targets = derive_observation_targets(observations, events)
            public = segments[list(PUBLIC_SEGMENT_COLUMNS)].copy()
            public["construction_date"] = pd.to_datetime(public["construction_date"])
            observations["segment_id"] = pd.Series(
                [{"QL01-KM1-1"}] + list(observations["segment_id"].iloc[1:]),
                dtype=object,
            )
            observations.loc[0, "date"] = pd.NaT
            report = validate_raw_dataset(public, observations, targets, events, spec)
            rows = [(i.severity, i.code, i.column, i.row_key, i.message) for i in report.issues]
            assert sum(1 for r in rows if r[1] == "invalid_segment_id") > 0
            assert sum(1 for r in rows if r[1] == "invalid_date") > 0
            assert any(r[3] == "row[0]:set" for r in rows)
            assert any(r[3] == "row[0]:set|NaT" for r in rows)
            with open(sys.argv[1], "w", encoding="utf-8") as handle:
                handle.write(repr(rows))
            """
        )
        outputs = []
        for seed in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outfile = tmp_path / f"rowkey_hashseed_{seed}.txt"
            subprocess.run(
                [sys.executable, "-c", script, str(outfile)],
                env=env,
                check=True,
            )
            outputs.append(outfile.read_text(encoding="utf-8"))
        assert outputs[0] == outputs[1]


class TestReviewNumericNormalization:
    def test_np_float64_multiplier_rainfall_overflow_contextual(self) -> None:
        import warnings

        observations = _mini()[1].copy()
        observations["traffic_volume"] = 1000
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(ValueError):
                inject_observation_corruption(
                    observations,
                    1,
                    missing_rate=0.0,
                    outlier_rate=0.25,
                    duplicate_rate=0.0,
                    outlier_multiplier=np.float64(1e308),
                )

    def test_hostile_int_subclass_rate_rejected_contextually(self) -> None:
        observations = _mini()[1].copy()
        with pytest.raises(ValueError):
            inject_observation_corruption(
                observations,
                7,
                missing_rate=_HostileInt(0),
                outlier_rate=0.0,
                duplicate_rate=0.0,
            )

    def test_hostile_float_subclass_multiplier_rejected_contextually(self) -> None:
        observations = _mini()[1].copy()
        with pytest.raises(ValueError):
            inject_observation_corruption(
                observations,
                7,
                missing_rate=0.0,
                outlier_rate=0.25,
                duplicate_rate=0.0,
                outlier_multiplier=_HostileFloat(2.0),
            )

    def test_hostile_int_subclass_multiplier_rejected_contextually(self) -> None:
        observations = _mini()[1].copy()
        with pytest.raises(ValueError):
            inject_observation_corruption(
                observations,
                7,
                missing_rate=0.0,
                outlier_rate=0.25,
                duplicate_rate=0.0,
                outlier_multiplier=_HostileInt(2),
            )


class _HostileInt(int):
    def __index__(self) -> int:
        raise RuntimeError("hostile int conversion")


class _HostileFloat(float):
    def __float__(self) -> float:
        raise RuntimeError("hostile float conversion")


class TestReviewMalformedIdDeterminism:
    @pytest.mark.parametrize(
        "value",
        [{"QL01-KM1-1"}, frozenset({"QL01-KM1-1"}), {"a": 1}, [1]],
    )
    def test_set_valued_ids_fail_closed_deterministically(self, value: object) -> None:
        segments, observations, targets, events = _pipeline()
        bad = segments.copy()
        bad["segment_id"] = pd.Series([value] + list(segments["segment_id"].iloc[1:]), dtype=object)
        report = validate_raw_dataset(bad, observations, targets, events, SPEC)
        issues = [issue for issue in report.issues if issue.code == "invalid_segment_id"]
        assert len(issues) == 1
        assert issues[0].row_key == f"row[0]:{type(value).__name__}"
        assert "{" not in issues[0].message
        assert "{" not in issues[0].row_key

    def test_identical_id_reports_across_hash_seeds(self, tmp_path) -> None:
        import os
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent(
            """
            import sys
            from datetime import date

            import pandas as pd

            from roadguard import (
                DatasetSpec,
                derive_observation_targets,
                generate_accident_timeline,
                generate_maintenance_events,
                generate_observations,
                generate_segments,
            )
            from roadguard.data_quality import PUBLIC_SEGMENT_COLUMNS, validate_raw_dataset

            spec = DatasetSpec(
                dataset_segments=10,
                dataset_months_per_segment=12,
                dataset_observations=120,
            )
            start = date(2022, 1, 1)
            segments = generate_segments(spec, 42, observation_start=start)
            events = generate_maintenance_events(segments, spec, 42, start_date=start)
            timeline = generate_accident_timeline(segments, spec, 42, start_date=start)
            observations = generate_observations(
                segments, events, timeline, spec, 42, start_date=start
            )
            targets = derive_observation_targets(observations, events)
            public = segments[list(PUBLIC_SEGMENT_COLUMNS)].copy()
            public["construction_date"] = pd.to_datetime(public["construction_date"])
            public["segment_id"] = pd.Series(
                [{"QL01-KM1-1"}] + list(public["segment_id"].iloc[1:]), dtype=object
            )
            report = validate_raw_dataset(public, observations, targets, events, spec)
            rows = [(i.severity, i.code, i.column, i.row_key, i.message) for i in report.issues]
            assert sum(1 for r in rows if r[1] == "invalid_segment_id") > 0
            assert any(r[3] == "row[0]:set" for r in rows)
            with open(sys.argv[1], "w", encoding="utf-8") as handle:
                handle.write(repr(rows))
            """
        )
        outputs = []
        for seed in ("0", "12345"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outfile = tmp_path / f"ids_hashseed_{seed}.txt"
            subprocess.run(
                [sys.executable, "-c", script, str(outfile)],
                env=env,
                check=True,
            )
            outputs.append(outfile.read_text(encoding="utf-8"))
        assert outputs[0] == outputs[1]
