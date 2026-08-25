"""Phase 9 train-only EDA and deterministic data card contract tests."""

from __future__ import annotations

import dataclasses
import decimal
import hashlib
import json
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import roadguard
from roadguard import (
    DatasetSpec,
    RepositoryExport,
    clean_raw_dataset,
    derive_observation_targets,
    generate_accident_timeline,
    generate_maintenance_events,
    generate_observations,
    generate_segments,
)
from roadguard.eda import (
    CategoricalLevel,
    CategoricalSummary,
    ClassificationBalance,
    DataQualitySummary,
    DateSummary,
    EDAError,
    EDAReport,
    NumericSummary,
    SplitInventory,
    TargetCorrelation,
    build_eda_report,
    render_data_card,
)
from roadguard.features import (
    FEATURE_FRAME_COLUMNS,
    FEATURE_KEY_COLUMNS,
    FEATURE_REGISTRY,
    build_feature_frame,
)
from roadguard.preprocessing import ChronologicalSplit, split_chronologically
from roadguard.targets import TARGET_COLUMNS

MINI_SPEC = DatasetSpec(dataset_segments=3, dataset_months_per_segment=48, dataset_observations=144)
ONE_SPEC = DatasetSpec(dataset_segments=1, dataset_months_per_segment=48, dataset_observations=48)
START = date(2022, 1, 1)

NUMERIC_FEATURE_COLUMNS = tuple(
    definition.name for definition in FEATURE_REGISTRY if definition.kind == "numeric"
)
CATEGORICAL_FEATURE_COLUMNS = tuple(
    definition.name for definition in FEATURE_REGISTRY if definition.kind == "categorical"
)
DATETIME_FEATURE_COLUMNS = tuple(
    definition.name for definition in FEATURE_REGISTRY if definition.kind == "datetime"
)
TARGET_VALUE_COLUMNS = TARGET_COLUMNS[2:]

CONTRACT_VERSION = "roadguard.phase9.v1"
PHASE9_PUBLIC_NAMES = (
    "EDAError",
    "EDAReport",
    "SplitInventory",
    "DataQualitySummary",
    "NumericSummary",
    "CategoricalLevel",
    "CategoricalSummary",
    "DateSummary",
    "ClassificationBalance",
    "TargetCorrelation",
    "build_eda_report",
    "render_data_card",
)


def _export(spec: DatasetSpec, seed: int = 42) -> RepositoryExport:
    segments = generate_segments(spec, seed, observation_start=START)
    events = generate_maintenance_events(segments, spec, seed, start_date=START)
    timeline = generate_accident_timeline(segments, spec, seed, start_date=START)
    observations = generate_observations(segments, events, timeline, spec, seed, start_date=START)
    targets = derive_observation_targets(observations, events)
    cleaned = clean_raw_dataset(segments, observations, targets, events, spec)
    return RepositoryExport(
        segments=cleaned.segments,
        observations=cleaned.observations,
        targets=cleaned.targets,
        maintenance_events=cleaned.maintenance_events,
    )


def _canonical_split(dataset: RepositoryExport, spec: DatasetSpec) -> ChronologicalSplit:
    frame = build_feature_frame(dataset, spec)
    return split_chronologically(frame, spec)


@pytest.fixture(scope="module")
def dataset() -> RepositoryExport:
    return _export(MINI_SPEC)


@pytest.fixture(scope="module")
def split(dataset: RepositoryExport) -> ChronologicalSplit:
    return _canonical_split(dataset, MINI_SPEC)


@pytest.fixture(scope="module")
def report(dataset: RepositoryExport, split: ChronologicalSplit) -> EDAReport:
    return build_eda_report(dataset, split, MINI_SPEC)


class TestPublicSurface:
    def test_eda_module_all_exact(self) -> None:
        import roadguard.eda as eda

        assert tuple(eda.__all__) == PHASE9_PUBLIC_NAMES

    def test_package_root_exports_phase9_symbols(self) -> None:
        for name in PHASE9_PUBLIC_NAMES:
            assert hasattr(roadguard, name), f"missing public attribute: {name}"

    def test_phase1_to_8_exports_preserved(self) -> None:
        expected = (
            "CleaningResult",
            "CLASSIFICATION_FEATURE_COLUMNS",
            "ConfigError",
            "CorruptionEntry",
            "CorruptionManifest",
            "ENV_PREFIX",
            "FEATURE_COLUMNS",
            "FEATURE_FRAME_COLUMNS",
            "FEATURE_KEY_COLUMNS",
            "FEATURE_REGISTRY",
            "DatasetSpec",
            "DatabaseConfigurationError",
            "DatabaseUnavailableError",
            "GenerationError",
            "FeatureDefinition",
            "FeatureInputError",
            "LoadReport",
            "PersistenceConflict",
            "PersistenceError",
            "PostgresRepository",
            "PreprocessingError",
            "PreprocessorFit",
            "RiskBand",
            "RiskBands",
            "REGRESSION_FEATURE_COLUMNS",
            "RoadGuardConfig",
            "RepositoryExport",
            "RepositoryInputError",
            "SegmentHistory",
            "SegmentMaster",
            "TARGET_COLUMNS",
            "TransformedData",
            "V1Contract",
            "ValidationIssue",
            "ValidationReport",
            "build_feature_frame",
            "clean_raw_dataset",
            "create_database_engine",
            "days_until_maintenance",
            "decay_condition",
            "derive_observation_targets",
            "fit_preprocessor",
            "generate_accident_timeline",
            "generate_maintenance_events",
            "generate_observations",
            "generate_segments",
            "inject_observation_corruption",
            "initialize_database",
            "load_config",
            "load_cleaning_result",
            "maintenance_within_30_days",
            "month_transition",
            "monthly_hazard",
            "observation_dates",
            "risk_score_from_probability",
            "split_chronologically",
            "transform",
            "validate_cleaned_dataset",
            "validate_raw_dataset",
        )
        for name in expected:
            assert hasattr(roadguard, name), f"missing locked export: {name}"

    @pytest.mark.parametrize(
        ("cls", "fields"),
        [
            (
                SplitInventory,
                ("name", "row_count", "date_count", "first_date", "last_date"),
            ),
            (
                DataQualitySummary,
                (
                    "row_count",
                    "segment_count",
                    "date_count",
                    "duplicate_key_count",
                    "missing_cell_count",
                    "non_finite_numeric_count",
                ),
            ),
            (
                NumericSummary,
                (
                    "column",
                    "count",
                    "missing_count",
                    "mean",
                    "population_std",
                    "minimum",
                    "q1",
                    "median",
                    "q3",
                    "maximum",
                    "iqr_outlier_count",
                    "iqr_outlier_rate",
                    "zero_variance",
                ),
            ),
            (CategoricalLevel, ("value", "count", "proportion")),
            (
                CategoricalSummary,
                ("column", "count", "missing_count", "cardinality", "levels"),
            ),
            (
                DateSummary,
                ("column", "count", "missing_count", "unique_count", "minimum", "maximum"),
            ),
            (
                ClassificationBalance,
                ("column", "negative_count", "positive_count", "positive_rate"),
            ),
            (TargetCorrelation, ("feature", "target", "pearson_r")),
            (
                EDAReport,
                (
                    "contract_version",
                    "training_fingerprint",
                    "feature_columns",
                    "split_inventory",
                    "data_quality",
                    "numeric_features",
                    "categorical_features",
                    "datetime_features",
                    "regression_target",
                    "classification_target",
                    "target_correlations",
                ),
            ),
        ],
    )
    def test_dataclasses_frozen_with_exact_field_order(
        self, cls: type, fields: tuple[str, ...]
    ) -> None:
        assert dataclasses.is_dataclass(cls)
        assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]
        assert tuple(field.name for field in dataclasses.fields(cls)) == fields

    def test_report_stores_only_builtin_scalars_and_tuples(self, report: EDAReport) -> None:
        assert report.contract_version == CONTRACT_VERSION
        assert type(report.contract_version) is str
        assert type(report.training_fingerprint) is str
        assert type(report.feature_columns) is tuple
        assert all(type(value) is str for value in report.feature_columns)
        assert type(report.split_inventory) is tuple
        assert type(report.numeric_features) is tuple
        assert type(report.categorical_features) is tuple
        assert type(report.datetime_features) is tuple
        assert type(report.target_correlations) is tuple
        assert type(report.data_quality) is DataQualitySummary
        assert type(report.regression_target) is NumericSummary
        assert type(report.classification_target) is ClassificationBalance

        for summary in report.numeric_features:
            assert type(summary.column) is str
            assert type(summary.count) is int
            assert type(summary.missing_count) is int
            assert type(summary.mean) is float
            assert type(summary.population_std) is float
            assert type(summary.minimum) is float
            assert type(summary.q1) is float
            assert type(summary.median) is float
            assert type(summary.q3) is float
            assert type(summary.maximum) is float
            assert type(summary.iqr_outlier_count) is int
            assert type(summary.iqr_outlier_rate) is float
            assert type(summary.zero_variance) is bool
        for summary in report.categorical_features:
            assert type(summary.column) is str
            assert type(summary.count) is int
            assert type(summary.missing_count) is int
            assert type(summary.cardinality) is int
            assert type(summary.levels) is tuple
            for level in summary.levels:
                assert type(level.value) is str
                assert type(level.count) is int
                assert type(level.proportion) is float
        for summary in report.datetime_features:
            assert type(summary.minimum) is date
            assert type(summary.maximum) is date
        for inventory in report.split_inventory:
            assert type(inventory.row_count) is int
            assert type(inventory.date_count) is int
            assert type(inventory.first_date) is date
            assert type(inventory.last_date) is date
        for correlation in report.target_correlations:
            assert type(correlation.feature) is str
            assert type(correlation.target) is str
            assert correlation.pearson_r is None or type(correlation.pearson_r) is float


class TestInputValidation:
    def test_wrong_input_types_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        with pytest.raises(TypeError, match="RepositoryExport"):
            build_eda_report(split, split, MINI_SPEC)
        with pytest.raises(TypeError, match="ChronologicalSplit"):
            build_eda_report(dataset, dataset, MINI_SPEC)
        with pytest.raises(TypeError, match="DatasetSpec"):
            build_eda_report(dataset, split, "not-a-spec")

    def test_lookalike_objects_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        lookalike = SimpleNamespace(
            segments=dataset.segments,
            observations=dataset.observations,
            targets=dataset.targets,
            maintenance_events=dataset.maintenance_events,
        )
        with pytest.raises(TypeError, match="RepositoryExport"):
            build_eda_report(lookalike, split, MINI_SPEC)

    def test_mismatched_spec_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        with pytest.raises(EDAError):
            build_eda_report(dataset, split, ONE_SPEC)

    def test_forged_export_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        targets = dataset.targets.copy()
        targets.loc[0, "days_until_maintenance"] += 1
        forged = dataclasses.replace(dataset, targets=targets)
        with pytest.raises(EDAError, match="target_event_inconsistency"):
            build_eda_report(forged, split, MINI_SPEC)

    def test_extra_column_export_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        observations = dataset.observations.copy()
        observations["extra"] = 1
        forged = dataclasses.replace(dataset, observations=observations)
        with pytest.raises(EDAError):
            build_eda_report(forged, split, MINI_SPEC)

    def test_invalid_dtype_export_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        observations = dataset.observations.copy()
        observations["traffic_volume"] = observations["traffic_volume"].astype("float64")
        forged = dataclasses.replace(dataset, observations=observations)
        with pytest.raises(EDAError):
            build_eda_report(forged, split, MINI_SPEC)

    def test_duplicate_observation_key_export_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        observations = dataset.observations.copy()
        duplicate = observations.iloc[0].copy()
        duplicate["segment_id"] = observations["segment_id"].iloc[1]
        duplicate["date"] = observations["date"].iloc[1]
        forged = dataclasses.replace(
            dataset,
            observations=pd.concat([observations, pd.DataFrame([duplicate])], ignore_index=True),
        )
        with pytest.raises(EDAError):
            build_eda_report(forged, split, MINI_SPEC)

    def test_missing_target_row_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        targets = dataset.targets.iloc[:-1].copy()
        forged = dataclasses.replace(dataset, targets=targets)
        with pytest.raises(EDAError, match="target"):
            build_eda_report(forged, split, MINI_SPEC)

    def test_invalid_date_export_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        observations = dataset.observations.copy()
        observations["date"] = pd.to_datetime(observations["date"]).dt.tz_localize("UTC")
        forged = dataclasses.replace(dataset, observations=observations)
        with pytest.raises(EDAError):
            build_eda_report(forged, split, MINI_SPEC)

    def test_supplied_split_value_mismatch_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        train = split.train.copy()
        train.loc[0, "rainfall_mm"] = train["rainfall_mm"].iloc[0] + 1.0
        bad_split = dataclasses.replace(split, train=train)
        with pytest.raises(EDAError, match="split"):
            build_eda_report(dataset, bad_split, MINI_SPEC)

    def test_supplied_split_date_provenance_mismatch_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        dates = split.train_dates[:-1] + (date(2024, 12, 31),)
        bad_split = dataclasses.replace(split, train_dates=dates)
        with pytest.raises(EDAError, match="split"):
            build_eda_report(dataset, bad_split, MINI_SPEC)

    def test_supplied_split_schema_mismatch_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        train = split.train.copy()
        train["extra"] = 1
        bad_split = dataclasses.replace(split, train=train)
        with pytest.raises(EDAError, match="split"):
            build_eda_report(dataset, bad_split, MINI_SPEC)

    def test_caller_frames_unchanged_on_success_and_failure(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        before_segments = dataset.segments.copy(deep=True)
        before_observations = dataset.observations.copy(deep=True)
        before_targets = dataset.targets.copy(deep=True)
        before_events = dataset.maintenance_events.copy(deep=True)
        before_train = split.train.copy(deep=True)
        before_validation = split.validation.copy(deep=True)
        before_test = split.test.copy(deep=True)

        build_eda_report(dataset, split, MINI_SPEC)
        bad_split = dataclasses.replace(
            split,
            train=split.train.copy().assign(rainfall_mm=split.train["rainfall_mm"] + 5),
        )
        with pytest.raises(EDAError):
            build_eda_report(dataset, bad_split, MINI_SPEC)

        assert_frame_equal(dataset.segments, before_segments)
        assert_frame_equal(dataset.observations, before_observations)
        assert_frame_equal(dataset.targets, before_targets)
        assert_frame_equal(dataset.maintenance_events, before_events)
        assert_frame_equal(split.train, before_train)
        assert_frame_equal(split.validation, before_validation)
        assert_frame_equal(split.test, before_test)


class TestTemporalLeakage:
    def test_validation_test_values_cannot_change_summaries(
        self, dataset: RepositoryExport
    ) -> None:
        baseline = build_eda_report(dataset, _canonical_split(dataset, MINI_SPEC), MINI_SPEC)
        mutated = _mutated_validation_test(dataset)
        altered = build_eda_report(mutated, _canonical_split(mutated, MINI_SPEC), MINI_SPEC)

        assert baseline.split_inventory == altered.split_inventory
        assert baseline.data_quality == altered.data_quality
        assert baseline.numeric_features == altered.numeric_features
        assert baseline.categorical_features == altered.categorical_features
        assert baseline.datetime_features == altered.datetime_features
        assert baseline.regression_target == altered.regression_target
        assert baseline.classification_target == altered.classification_target
        assert baseline.target_correlations == altered.target_correlations

    def test_validation_test_target_values_cannot_change_summaries(
        self, dataset: RepositoryExport
    ) -> None:
        baseline = build_eda_report(dataset, _canonical_split(dataset, MINI_SPEC), MINI_SPEC)
        mutated = self._shifted_later_partition_event(dataset, _canonical_split(dataset, MINI_SPEC))
        assert not mutated.targets.equals(dataset.targets)
        altered = build_eda_report(mutated, _canonical_split(mutated, MINI_SPEC), MINI_SPEC)

        assert baseline.numeric_features == altered.numeric_features
        assert baseline.regression_target == altered.regression_target
        assert baseline.classification_target == altered.classification_target
        assert baseline.target_correlations == altered.target_correlations
        assert baseline.training_fingerprint == altered.training_fingerprint
        assert render_data_card(baseline) == render_data_card(altered)

    @staticmethod
    def _shifted_later_partition_event(
        dataset: RepositoryExport, split: ChronologicalSplit
    ) -> RepositoryExport:
        """Shift one validation/test maintenance event forward one month.

        The event's previous event lies after the last training date, so no
        training target or feature changes; validation/test targets change.
        """
        last_train = split.train_dates[-1]
        window_start = split.validation_dates[0]
        window_end = split.test_dates[-1]
        events = dataset.maintenance_events.copy()
        observations = dataset.observations.copy()
        for segment_id in sorted(events["segment_id"].unique()):
            segment_events = events[events["segment_id"] == segment_id].sort_values(
                "maintenance_date"
            )
            dates = [pd.Timestamp(value).date() for value in segment_events["maintenance_date"]]
            for position, event_date in enumerate(dates):
                if not (window_start <= event_date <= window_end):
                    continue
                if position == 0 or position == len(dates) - 1:
                    continue
                previous_date = dates[position - 1]
                following_date = dates[position + 1]
                if previous_date <= last_train:
                    continue
                shifted_timestamp = pd.Timestamp(event_date) + pd.DateOffset(months=1)
                shifted_date = shifted_timestamp.date()
                if shifted_date >= following_date:
                    continue
                mutated_events = events.copy()
                mutated_events.loc[segment_events.index[position], "maintenance_date"] = (
                    shifted_timestamp
                )
                mutated_observations = observations.copy()
                after_mask = (
                    (mutated_observations["segment_id"] == segment_id)
                    & (mutated_observations["date"] >= pd.Timestamp(event_date))
                    & (mutated_observations["date"] < shifted_timestamp)
                )
                for index in mutated_observations.index[after_mask]:
                    row_date = mutated_observations.at[index, "date"]
                    mutated_observations.at[index, "previous_repairs"] = (
                        int(mutated_observations.at[index, "previous_repairs"]) - 1
                    )
                    mutated_observations.at[index, "days_since_last_maintenance"] = (
                        row_date - pd.Timestamp(previous_date)
                    ).days
                later_mask = (
                    (mutated_observations["segment_id"] == segment_id)
                    & (mutated_observations["date"] >= shifted_timestamp)
                    & (mutated_observations["date"] < pd.Timestamp(following_date))
                )
                for index in mutated_observations.index[later_mask]:
                    row_date = mutated_observations.at[index, "date"]
                    mutated_observations.at[index, "days_since_last_maintenance"] = (
                        row_date - shifted_timestamp
                    ).days
                mutated_targets = derive_observation_targets(mutated_observations, mutated_events)
                return dataclasses.replace(
                    dataset,
                    observations=mutated_observations,
                    targets=mutated_targets,
                    maintenance_events=mutated_events,
                )
        raise AssertionError("no shiftable later-partition maintenance event found")

    def test_validation_test_values_cannot_change_fingerprint(
        self, dataset: RepositoryExport
    ) -> None:
        baseline = build_eda_report(dataset, _canonical_split(dataset, MINI_SPEC), MINI_SPEC)
        mutated = _mutated_validation_test(dataset)
        altered = build_eda_report(mutated, _canonical_split(mutated, MINI_SPEC), MINI_SPEC)
        assert altered.training_fingerprint == baseline.training_fingerprint

    def test_shuffled_upstream_input_produces_equal_report(self, dataset: RepositoryExport) -> None:
        shuffled = dataclasses.replace(
            dataset,
            segments=dataset.segments.sample(frac=1.0, random_state=3).reset_index(drop=True),
            observations=dataset.observations.sample(frac=1.0, random_state=7).reset_index(
                drop=True
            ),
            targets=dataset.targets.sample(frac=1.0, random_state=11).reset_index(drop=True),
            maintenance_events=dataset.maintenance_events.sample(
                frac=1.0, random_state=13
            ).reset_index(drop=True),
        )
        baseline = build_eda_report(dataset, _canonical_split(dataset, MINI_SPEC), MINI_SPEC)
        altered = build_eda_report(shuffled, _canonical_split(shuffled, MINI_SPEC), MINI_SPEC)

        assert altered == baseline
        assert render_data_card(altered) == render_data_card(baseline)

    def test_split_inventory_covers_only_metadata(self, report: EDAReport) -> None:
        assert [item.name for item in report.split_inventory] == ["train", "validation", "test"]
        for item in report.split_inventory:
            assert item.row_count > 0
            assert item.date_count > 0
            assert item.first_date <= item.last_date


class TestDescriptiveCalculations:
    def test_numeric_feature_order_exact(self, report: EDAReport) -> None:
        assert tuple(item.column for item in report.numeric_features) == NUMERIC_FEATURE_COLUMNS
        assert len(report.numeric_features) == 15

    def test_categorical_feature_order_exact(self, report: EDAReport) -> None:
        assert (
            tuple(item.column for item in report.categorical_features)
            == CATEGORICAL_FEATURE_COLUMNS
        )

    def test_datetime_feature_order_exact(self, report: EDAReport) -> None:
        assert tuple(item.column for item in report.datetime_features) == DATETIME_FEATURE_COLUMNS

    def test_data_quality_summary_on_training_join(self, report: EDAReport) -> None:
        assert report.data_quality.row_count == 34 * 3
        assert report.data_quality.segment_count == 3
        assert report.data_quality.date_count == 34
        assert report.data_quality.duplicate_key_count == 0
        assert report.data_quality.missing_cell_count == 0
        assert report.data_quality.non_finite_numeric_count == 0

    def test_numeric_summary_counts_match_row_count(self, report: EDAReport) -> None:
        for summary in report.numeric_features:
            assert summary.count == report.data_quality.row_count
            assert summary.missing_count == 0

    def test_population_std_ddof_zero(self) -> None:
        from roadguard.eda import _numeric_statistics

        values = [float(value) for value in range(1, 9)]
        stats = _numeric_statistics(values)
        assert stats.mean == 4.5
        assert stats.population_std == pytest.approx(
            np.sqrt(sum((value - 4.5) ** 2 for value in values) / 8)
        )
        assert stats.zero_variance is False

    def test_linear_quartiles_known_vectors(self) -> None:
        from roadguard.eda import _numeric_statistics

        stats = _numeric_statistics([1.0, 2.0, 3.0, 4.0, 5.0])
        assert stats.minimum == 1.0
        assert stats.q1 == 2.0
        assert stats.median == 3.0
        assert stats.q3 == 4.0
        assert stats.maximum == 5.0

        stats = _numeric_statistics([1.0, 2.0, 3.0, 4.0])
        assert stats.q1 == 1.75
        assert stats.median == 2.5
        assert stats.q3 == 3.25

    def test_iqr_outlier_strict_boundaries(self) -> None:
        from roadguard.eda import _numeric_statistics

        q1, q3 = 2.0, 8.0
        values = [2.0, 8.0, q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1), 5.0]
        stats = _numeric_statistics(values)
        assert stats.q1 == 2.0
        assert stats.q3 == 8.0
        assert stats.iqr_outlier_count == 0

        values = [
            2.0,
            8.0,
            q1 - 1.5 * (q3 - q1) - 1.0,
            q3 + 1.5 * (q3 - q1) + 1.0,
            5.0,
        ]
        stats = _numeric_statistics(values)
        assert stats.iqr_outlier_count == 2
        assert stats.iqr_outlier_rate == pytest.approx(2 / 5)

    def test_constant_column_exact_zero_std_and_zero_variance_flag(self) -> None:
        from roadguard.eda import _numeric_statistics

        stats = _numeric_statistics([1000.0, 1000.0, 1000.0, 1000.0])
        assert stats.mean == 1000.0
        assert stats.population_std == 0.0
        assert stats.minimum == 1000.0
        assert stats.maximum == 1000.0
        assert stats.q1 == 1000.0
        assert stats.median == 1000.0
        assert stats.q3 == 1000.0
        assert stats.iqr_outlier_count == 0
        assert stats.iqr_outlier_rate == 0.0
        assert stats.zero_variance is True

    def test_constant_feature_column_in_full_report(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        observations = dataset.observations.copy()
        observations["traffic_volume"] = 777
        forged = dataclasses.replace(dataset, observations=observations)
        report = build_eda_report(forged, _canonical_split(forged, MINI_SPEC), MINI_SPEC)
        summary = next(item for item in report.numeric_features if item.column == "traffic_volume")
        assert summary.zero_variance is True
        assert summary.population_std == 0.0
        assert summary.mean == 777.0

    def test_non_constant_computed_zero_variance_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from roadguard.eda import _numeric_statistics

        def zero_variance(
            values: tuple[decimal.Decimal, ...], mean: decimal.Decimal
        ) -> decimal.Decimal:
            return decimal.Decimal("0")

        monkeypatch.setattr("roadguard.eda._population_variance", zero_variance)
        with pytest.raises(EDAError, match="zero variance"):
            _numeric_statistics([1.0, 2.0, 3.0])

    def test_classification_balance(self, report: EDAReport) -> None:
        assert report.classification_target.column == "maintenance_within_30_days"
        total = (
            report.classification_target.negative_count
            + report.classification_target.positive_count
        )
        assert total == report.data_quality.row_count
        assert report.classification_target.positive_rate == pytest.approx(
            report.classification_target.positive_count / total
        )

    def test_regression_target_column(self, report: EDAReport) -> None:
        assert report.regression_target.column == "days_until_maintenance"
        assert report.regression_target.count == report.data_quality.row_count

    def test_correlation_cardinality_order_and_uniqueness(self, report: EDAReport) -> None:
        assert len(report.target_correlations) == 15 * 2
        pairs = [(item.feature, item.target) for item in report.target_correlations]
        expected = [
            (feature, target)
            for feature in NUMERIC_FEATURE_COLUMNS
            for target in TARGET_VALUE_COLUMNS
        ]
        assert pairs == expected
        assert len(set(pairs)) == len(pairs)

    def test_pearson_known_vectors(self) -> None:
        from roadguard.eda import _pearson_correlation

        assert _pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
        assert _pearson_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
        value = _pearson_correlation([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 1.0, 2.0])
        assert value is not None
        assert -1.0 <= value <= 1.0

    def test_constant_sequence_pearson_is_none(self) -> None:
        from roadguard.eda import _pearson_correlation

        assert _pearson_correlation([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) is None
        assert _pearson_correlation([1.0, 2.0, 3.0], [7.0, 7.0, 7.0]) is None

    def test_categorical_levels_ordered_by_descending_count(self, report: EDAReport) -> None:
        for summary in report.categorical_features:
            counts = [level.count for level in summary.levels]
            assert counts == sorted(counts, reverse=True)
            assert sum(counts) == report.data_quality.row_count
            assert summary.cardinality == len(summary.levels)
            for level in summary.levels:
                assert level.proportion == pytest.approx(
                    level.count / report.data_quality.row_count
                )

    def test_categorical_levels_ties_by_ascending_value(self) -> None:
        from roadguard.eda import _categorical_levels

        levels = _categorical_levels({"NA": 3, "TH": 1, "QB": 3}, total=7)
        assert [level.value for level in levels] == ["NA", "QB", "TH"]

    def test_datetime_summary(self, report: EDAReport) -> None:
        summary = report.datetime_features[0]
        assert summary.column == "construction_date"
        assert summary.count == report.data_quality.row_count
        assert summary.missing_count == 0
        assert summary.unique_count >= 1
        assert summary.minimum <= summary.maximum

    def test_categorical_levels_descending_count_then_ascending_value(
        self, report: EDAReport
    ) -> None:
        for summary in report.categorical_features:
            keys = [(-level.count, level.value) for level in summary.levels]
            assert keys == sorted(keys)


class TestAdversarialNumeric:
    def test_huge_finite_constant_column(self) -> None:
        from roadguard.eda import _numeric_statistics

        huge = np.finfo(np.float64).max
        stats = _numeric_statistics([huge, huge, huge, huge])
        assert stats.population_std == 0.0
        assert stats.zero_variance is True
        assert stats.mean == huge
        assert np.isfinite(stats.mean)
        assert np.isfinite(stats.maximum)

    def test_huge_finite_non_constant_extrema(self) -> None:
        from roadguard.eda import _numeric_statistics

        huge = np.finfo(np.float64).max
        stats = _numeric_statistics([huge, np.nextafter(huge, 0.0), huge, np.nextafter(huge, 0.0)])
        assert stats.zero_variance is False
        assert np.isfinite(stats.mean)
        assert np.isfinite(stats.population_std)
        assert np.isfinite(stats.minimum)
        assert np.isfinite(stats.maximum)

    def test_unrepresentable_float_raises(self) -> None:
        from roadguard.eda import _to_float

        with pytest.raises(EDAError):
            _to_float(decimal.Decimal("1e400"))

    def test_huge_constant_in_full_report(self, dataset: RepositoryExport) -> None:
        observations = dataset.observations.copy()
        observations["rainfall_mm"] = np.finfo(np.float64).max
        forged = dataclasses.replace(dataset, observations=observations)
        report = build_eda_report(forged, _canonical_split(forged, MINI_SPEC), MINI_SPEC)
        summary = next(item for item in report.numeric_features if item.column == "rainfall_mm")
        assert summary.population_std == 0.0
        assert summary.zero_variance is True
        assert summary.mean == np.finfo(np.float64).max


class TestFingerprint:
    def test_fingerprint_is_lowercase_sha256_hex(self, report: EDAReport) -> None:
        assert len(report.training_fingerprint) == 64
        assert report.training_fingerprint == report.training_fingerprint.lower()
        int(report.training_fingerprint, 16)

    def test_known_vector_digest(self) -> None:
        dataset = _export(ONE_SPEC)
        split = _canonical_split(dataset, ONE_SPEC)
        report = build_eda_report(dataset, split, ONE_SPEC)

        frame = build_feature_frame(dataset, ONE_SPEC)
        join = frame.merge(dataset.targets, on=list(FEATURE_KEY_COLUMNS), how="inner")
        join = join[join["date"].dt.date.isin(split.train_dates)]
        columns = list(FEATURE_FRAME_COLUMNS) + list(TARGET_VALUE_COLUMNS)
        rows: list[list[object]] = []
        for _, row in join.sort_values(list(FEATURE_KEY_COLUMNS), kind="stable").iterrows():
            rows.append([_canonical_scalar(row[column]) for column in columns])
        payload = {
            "columns": columns,
            "contract": CONTRACT_VERSION,
            "spec": {
                "dataset_months_per_segment": ONE_SPEC.dataset_months_per_segment,
                "dataset_observations": ONE_SPEC.dataset_observations,
                "dataset_segments": ONE_SPEC.dataset_segments,
            },
            "split": {
                "test": [value.isoformat() for value in split.test_dates],
                "train": [value.isoformat() for value in split.train_dates],
                "validation": [value.isoformat() for value in split.validation_dates],
            },
            "train_rows": rows,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        expected = hashlib.sha256(canonical).hexdigest()
        assert report.training_fingerprint == expected
        assert (
            report.training_fingerprint
            == "0f124527fd119b18cf5be029fdce520aa7a81e3145c9fe7be5bb51204cf3d922"
        )

    def test_negative_zero_normalized(self) -> None:
        from roadguard.eda import _canonical_scalar

        assert _canonical_scalar(-0.0) == "0x0.0p+0"
        assert _canonical_scalar(0.0) == "0x0.0p+0"

    def test_float_hex_normalization(self) -> None:
        from roadguard.eda import _canonical_scalar

        assert _canonical_scalar(1.5) == (1.5).hex()
        assert _canonical_scalar(0.1).startswith("0x")
        assert _canonical_scalar(0.1) == _canonical_scalar(0.1)

    def test_validation_test_rows_not_hashed(self, dataset: RepositoryExport) -> None:
        baseline = build_eda_report(dataset, _canonical_split(dataset, MINI_SPEC), MINI_SPEC)
        mutated = _mutated_validation_test(dataset)
        altered = build_eda_report(mutated, _canonical_split(mutated, MINI_SPEC), MINI_SPEC)
        assert altered.training_fingerprint == baseline.training_fingerprint


class TestRenderer:
    def test_known_vector_markdown(self) -> None:
        dataset = _export(ONE_SPEC)
        split = _canonical_split(dataset, ONE_SPEC)
        report = build_eda_report(dataset, split, ONE_SPEC)
        rendered = render_data_card(report)
        assert rendered == (
            "# RoadGuard AI - Phase 9 Train-Only Data Card\n"
            "\n"
            "## Scope and leakage guard\n"
            "\n"
            "- Statistics and correlations use only the canonical 34-date training partition.\n"
            "- Validation and test are represented only by row counts, date counts, and date boundaries.\n"  # noqa: E501
            "- No preprocessing was fit or applied, and no model was trained, selected, or evaluated.\n"  # noqa: E501
            "\n"
            "## Provenance\n"
            "\n"
            "- Contract: `roadguard.phase9.v1`\n"
            "- Training fingerprint: `0f124527fd119b18cf5be029fdce520aa7a81e3145c9fe7be5bb51204cf3d922`\n"  # noqa: E501
            "- Feature columns: `province`, `road_type`, `construction_date`, `road_length_km`, `traffic_volume`, `heavy_vehicle_ratio`, `road_age_days`, `rainfall_mm`, `temperature`, `humidity`, `days_since_last_maintenance`, `previous_repairs`, `road_condition_score`, `marking_condition_score`, `guardrail_condition_score`, `sign_condition_score`, `accident_count_30d`, `accident_count_365d`\n"  # noqa: E501
            "\n"
            "## Split inventory\n"
            "\n"
            "| Partition | Rows | Dates | First date | Last date |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| train | 34 | 34 | 2022-01-01 | 2024-10-01 |\n"
            "| validation | 7 | 7 | 2024-11-01 | 2025-05-01 |\n"
            "| test | 7 | 7 | 2025-06-01 | 2025-12-01 |\n"
            "\n"
            "## Training data quality\n"
            "\n"
            "| Rows | Segments | Dates | Duplicate keys | Missing cells | Non-finite numeric cells |\n"  # noqa: E501
            "| --- | --- | --- | --- | --- | --- |\n"
            "| 34 | 1 | 34 | 0 | 0 | 0 |\n"
            "\n"
            "## Training feature summaries\n"
            "\n"
            "### Numeric features\n"
            "\n"
            "| Column | Count | Missing | Mean | Population std | Min | Q1 | Median | Q3 | Max | IQR outliers | IQR outlier rate | Zero variance |\n"  # noqa: E501
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| road_length_km | 34 | 0 | 4.000000 | 0.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 4.000000 | 0 | 0.000000 | true |\n"  # noqa: E501
            "| traffic_volume | 34 | 0 | 19536.264706 | 1452.378400 | 16610.000000 | 18560.750000 | 19418.000000 | 20380.250000 | 23169.000000 | 1 | 0.029412 | false |\n"  # noqa: E501
            "| heavy_vehicle_ratio | 34 | 0 | 0.461800 | 0.020776 | 0.402300 | 0.447250 | 0.463750 | 0.474725 | 0.509200 | 1 | 0.029412 | false |\n"  # noqa: E501
            "| road_age_days | 34 | 0 | 6865.294118 | 298.680976 | 6364.000000 | 6614.500000 | 6864.500000 | 7117.250000 | 7368.000000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "| rainfall_mm | 34 | 0 | 207.523529 | 105.131551 | 54.700000 | 109.600000 | 194.500000 | 294.275000 | 435.900000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "| temperature | 34 | 0 | 26.517647 | 3.192682 | 20.900000 | 23.650000 | 26.450000 | 29.725000 | 31.500000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "| humidity | 34 | 0 | 79.335294 | 8.473796 | 63.200000 | 72.225000 | 78.900000 | 85.650000 | 96.200000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "| days_since_last_maintenance | 34 | 0 | 141.147059 | 103.334446 | 1.000000 | 57.000000 | 117.500000 | 205.500000 | 392.000000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "| previous_repairs | 34 | 0 | 4.970588 | 1.773751 | 3.000000 | 3.000000 | 4.000000 | 6.000000 | 8.000000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "| road_condition_score | 34 | 0 | 93.970588 | 2.202743 | 88.000000 | 93.000000 | 94.000000 | 95.000000 | 98.000000 | 2 | 0.058824 | false |\n"  # noqa: E501
            "| marking_condition_score | 34 | 0 | 85.117647 | 2.867308 | 78.000000 | 83.000000 | 86.000000 | 87.000000 | 90.000000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "| guardrail_condition_score | 34 | 0 | 86.352941 | 2.167709 | 82.000000 | 85.000000 | 87.000000 | 88.000000 | 90.000000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "| sign_condition_score | 34 | 0 | 89.235294 | 2.532829 | 83.000000 | 87.250000 | 89.500000 | 91.000000 | 93.000000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "| accident_count_30d | 34 | 0 | 0.147059 | 0.354165 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 | 5 | 0.147059 | false |\n"  # noqa: E501
            "| accident_count_365d | 34 | 0 | 1.264706 | 0.917440 | 0.000000 | 1.000000 | 1.000000 | 2.000000 | 3.000000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "\n"
            "### Categorical features\n"
            "\n"
            "| Column | Level | Count | Proportion |\n"
            "| --- | --- | --- | --- |\n"
            "| province | NA | 34 | 1.000000 |\n"
            "| road_type | highway | 34 | 1.000000 |\n"
            "\n"
            "### Datetime features\n"
            "\n"
            "| Column | Count | Missing | Unique | Min | Max |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| construction_date | 34 | 0 | 1 | 2004-07-30 | 2004-07-30 |\n"
            "\n"
            "## Training target summaries\n"
            "\n"
            "### Regression target\n"
            "\n"
            "| Column | Count | Missing | Mean | Population std | Min | Q1 | Median | Q3 | Max | IQR outliers | IQR outlier rate | Zero variance |\n"  # noqa: E501
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| days_until_maintenance | 34 | 0 | 145.264706 | 104.114278 | 7.000000 | 63.250000 | 127.500000 | 200.750000 | 394.000000 | 0 | 0.000000 | false |\n"  # noqa: E501
            "\n"
            "### Classification target\n"
            "\n"
            "| Column | Negative | Positive | Positive rate |\n"
            "| --- | --- | --- | --- |\n"
            "| maintenance_within_30_days | 29 | 5 | 0.147059 |\n"
            "\n"
            "## Train-only target correlations\n"
            "\n"
            "| Feature | Target | Pearson r |\n"
            "| --- | --- | --- |\n"
            "| road_length_km | days_until_maintenance | not-defined |\n"
            "| road_length_km | maintenance_within_30_days | not-defined |\n"
            "| traffic_volume | days_until_maintenance | -0.311065 |\n"
            "| traffic_volume | maintenance_within_30_days | 0.441174 |\n"
            "| heavy_vehicle_ratio | days_until_maintenance | 0.005183 |\n"
            "| heavy_vehicle_ratio | maintenance_within_30_days | -0.007994 |\n"
            "| road_age_days | days_until_maintenance | 0.060048 |\n"
            "| road_age_days | maintenance_within_30_days | 0.131104 |\n"
            "| rainfall_mm | days_until_maintenance | 0.073205 |\n"
            "| rainfall_mm | maintenance_within_30_days | 0.183484 |\n"
            "| temperature | days_until_maintenance | -0.267460 |\n"
            "| temperature | maintenance_within_30_days | 0.125160 |\n"
            "| humidity | days_until_maintenance | 0.031677 |\n"
            "| humidity | maintenance_within_30_days | 0.179576 |\n"
            "| days_since_last_maintenance | days_until_maintenance | -0.463261 |\n"
            "| days_since_last_maintenance | maintenance_within_30_days | 0.195501 |\n"
            "| previous_repairs | days_until_maintenance | 0.256617 |\n"
            "| previous_repairs | maintenance_within_30_days | 0.006885 |\n"
            "| road_condition_score | days_until_maintenance | 0.315778 |\n"
            "| road_condition_score | maintenance_within_30_days | -0.296063 |\n"
            "| marking_condition_score | days_until_maintenance | 0.237335 |\n"
            "| marking_condition_score | maintenance_within_30_days | -0.017037 |\n"
            "| guardrail_condition_score | days_until_maintenance | 0.436026 |\n"
            "| guardrail_condition_score | maintenance_within_30_days | -0.297468 |\n"
            "| sign_condition_score | days_until_maintenance | 0.337264 |\n"
            "| sign_condition_score | maintenance_within_30_days | -0.366450 |\n"
            "| accident_count_30d | days_until_maintenance | -0.228382 |\n"
            "| accident_count_30d | maintenance_within_30_days | 0.296552 |\n"
            "| accident_count_365d | days_until_maintenance | -0.242140 |\n"
            "| accident_count_365d | maintenance_within_30_days | 0.332789 |\n"
            "\n"
            "## Limitations\n"
            "\n"
            "- This card is descriptive train-only evidence; it is not causal analysis or model-performance evidence.\n"  # noqa: E501
            "- Validation and test feature/target distributions were not summarized.\n"
            "- The SHA-256 fingerprint is an equality/integrity identifier, not anonymization, authentication, or a digital signature.\n"  # noqa: E501
        )

    def test_exactly_one_trailing_newline(self, report: EDAReport) -> None:
        rendered = render_data_card(report)
        assert rendered.endswith("\n")
        assert not rendered.endswith("\n\n")

    def test_no_blank_line_inside_tables(self, report: EDAReport) -> None:
        rendered = render_data_card(report)
        for block in rendered.split("\n\n"):
            assert not block.startswith("|") or "\n\n" not in block

    def test_undefined_correlation_renders_not_defined(self) -> None:
        from roadguard.eda import _pearson_correlation

        assert _pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None

    def test_float_formatting_round_half_even(self) -> None:
        from roadguard._eda_render import _format_float

        assert _format_float(0.1234567) == "0.123457"
        assert _format_float(0.0000005) == "0.000000"
        assert _format_float(0.0000015) == "0.000002"
        assert _format_float(-0.0) == "0.000000"
        assert _format_float(1.0) == "1.000000"

    def test_float_formatting_near_limits(self) -> None:
        from roadguard._eda_render import _format_float

        huge = np.finfo(np.float64).max
        assert _format_float(huge).startswith("17976931348623157")
        assert _format_float(5e-324) == "0.000000"

    def test_renderer_independent_of_global_decimal_context(self, report: EDAReport) -> None:
        old_prec = decimal.getcontext().prec
        old_rounding = decimal.getcontext().rounding
        try:
            decimal.getcontext().prec = 3
            decimal.getcontext().rounding = decimal.ROUND_FLOOR
            rendered = render_data_card(report)
        finally:
            decimal.getcontext().prec = old_prec
            decimal.getcontext().rounding = old_rounding
        assert rendered == render_data_card(report)

    def test_no_forbidden_content_in_markdown(self, report: EDAReport) -> None:
        rendered = render_data_card(report)
        assert "password" not in rendered.lower()
        assert "secret" not in rendered.lower()
        assert "C:\\" not in rendered
        assert "/tmp" not in rendered
        assert "QL01" not in rendered
        assert "timestamp" not in rendered.lower()

    def test_invalid_contract_version_rejected(self, report: EDAReport) -> None:
        forged = dataclasses.replace(report, contract_version="roadguard.phase9.v2")
        with pytest.raises(EDAError, match="version"):
            render_data_card(forged)

    def test_invalid_feature_columns_rejected(self, report: EDAReport) -> None:
        forged = dataclasses.replace(report, feature_columns=("province", "road_type"))
        with pytest.raises(EDAError, match="feature"):
            render_data_card(forged)

    def test_malformed_digest_rejected(self, report: EDAReport) -> None:
        forged = dataclasses.replace(report, training_fingerprint="not-a-digest")
        with pytest.raises(EDAError, match="fingerprint"):
            render_data_card(forged)

    def test_uppercase_digest_rejected(self, report: EDAReport) -> None:
        forged = dataclasses.replace(
            report, training_fingerprint=report.training_fingerprint.upper()
        )
        with pytest.raises(EDAError, match="fingerprint"):
            render_data_card(forged)

    def test_wrong_split_order_rejected(self, report: EDAReport) -> None:
        forged = dataclasses.replace(
            report, split_inventory=tuple(reversed(report.split_inventory))
        )
        with pytest.raises(EDAError, match="split"):
            render_data_card(forged)

    def test_forged_split_name_rejected(self, report: EDAReport) -> None:
        inventory = report.split_inventory[0]
        forged_inventory = dataclasses.replace(inventory, name="train<script>")
        forged = dataclasses.replace(
            report,
            split_inventory=(
                forged_inventory,
                report.split_inventory[1],
                report.split_inventory[2],
            ),
        )
        with pytest.raises(EDAError, match="split"):
            render_data_card(forged)

    def test_contradictory_data_quality_totals_rejected(self, report: EDAReport) -> None:
        forged = dataclasses.replace(
            report,
            data_quality=dataclasses.replace(report.data_quality, row_count=1),
        )
        with pytest.raises(EDAError):
            render_data_card(forged)

    def test_negative_counts_rejected(self, report: EDAReport) -> None:
        numeric = report.numeric_features[0]
        forged = dataclasses.replace(
            report,
            numeric_features=(dataclasses.replace(numeric, iqr_outlier_count=-1),)
            + report.numeric_features[1:],
        )
        with pytest.raises(EDAError):
            render_data_card(forged)

    def test_outlier_rate_out_of_range_rejected(self, report: EDAReport) -> None:
        numeric = report.numeric_features[0]
        forged = dataclasses.replace(
            report,
            numeric_features=(dataclasses.replace(numeric, iqr_outlier_rate=2.5),)
            + report.numeric_features[1:],
        )
        with pytest.raises(EDAError):
            render_data_card(forged)

    def test_zero_variance_contradiction_rejected(self, report: EDAReport) -> None:
        numeric = report.numeric_features[0]
        forged = dataclasses.replace(
            report,
            numeric_features=(dataclasses.replace(numeric, zero_variance=True, population_std=5.0),)
            + report.numeric_features[1:],
        )
        with pytest.raises(EDAError):
            render_data_card(forged)

    def test_forged_categorical_level_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        level = categorical.levels[0]
        forged_level = dataclasses.replace(level, value="NA<script>")
        forged = dataclasses.replace(
            report,
            categorical_features=(
                dataclasses.replace(categorical, levels=(forged_level,) + categorical.levels[1:]),
            )
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="level|category"):
            render_data_card(forged)

    def test_forged_categorical_level_pipe_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        level = categorical.levels[0]
        forged_level = dataclasses.replace(level, value="NA|evil")
        forged = dataclasses.replace(
            report,
            categorical_features=(
                dataclasses.replace(categorical, levels=(forged_level,) + categorical.levels[1:]),
            )
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="level|category"):
            render_data_card(forged)

    def test_forged_categorical_level_markdown_link_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        level = categorical.levels[0]
        forged_level = dataclasses.replace(level, value="[NA](https://evil.example)")
        forged = dataclasses.replace(
            report,
            categorical_features=(
                dataclasses.replace(categorical, levels=(forged_level,) + categorical.levels[1:]),
            )
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="level|category"):
            render_data_card(forged)

    def test_forged_categorical_level_line_break_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        level = categorical.levels[0]
        forged_level = dataclasses.replace(level, value="NA\n- fake bullet")
        forged = dataclasses.replace(
            report,
            categorical_features=(
                dataclasses.replace(categorical, levels=(forged_level,) + categorical.levels[1:]),
            )
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="level|category"):
            render_data_card(forged)

    def test_forged_classification_target_name_rejected(self, report: EDAReport) -> None:
        forged = dataclasses.replace(
            report,
            classification_target=dataclasses.replace(
                report.classification_target, column="maintenance_within_30_days<script>"
            ),
        )
        with pytest.raises(EDAError, match="target"):
            render_data_card(forged)

    def test_forged_regression_target_name_rejected(self, report: EDAReport) -> None:
        forged = dataclasses.replace(
            report,
            regression_target=dataclasses.replace(report.regression_target, column="evil"),
        )
        with pytest.raises(EDAError, match="target"):
            render_data_card(forged)

    def test_pearson_out_of_range_rejected(self, report: EDAReport) -> None:
        correlation = report.target_correlations[0]
        forged = dataclasses.replace(
            report,
            target_correlations=(dataclasses.replace(correlation, pearson_r=1.5),)
            + report.target_correlations[1:],
        )
        with pytest.raises(EDAError):
            render_data_card(forged)

    def test_missing_correlation_pair_rejected(self, report: EDAReport) -> None:
        forged = dataclasses.replace(report, target_correlations=report.target_correlations[:-1])
        with pytest.raises(EDAError, match="correlation"):
            render_data_card(forged)

    def test_wrong_correlation_target_name_rejected(self, report: EDAReport) -> None:
        correlation = report.target_correlations[0]
        forged = dataclasses.replace(
            report,
            target_correlations=(
                dataclasses.replace(correlation, target="days_until_maintenance|evil"),
            )
            + report.target_correlations[1:],
        )
        with pytest.raises(EDAError, match="correlation"):
            render_data_card(forged)


class TestRendererRejections:
    def _replace(self, report: EDAReport, **kwargs: object) -> EDAReport:
        return dataclasses.replace(report, **kwargs)

    def test_wrong_report_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="EDAReport"):
            render_data_card(object())

    def test_invalid_split_entry_type_rejected(self, report: EDAReport) -> None:
        forged = self._replace(report, split_inventory=(object(),) + report.split_inventory[1:])
        with pytest.raises(EDAError, match="split"):
            render_data_card(forged)

    def test_invalid_split_date_count_rejected(self, report: EDAReport) -> None:
        inventory = dataclasses.replace(report.split_inventory[0], date_count=5)
        forged = self._replace(report, split_inventory=(inventory,) + report.split_inventory[1:])
        with pytest.raises(EDAError, match="split"):
            render_data_card(forged)

    def test_inverted_split_boundaries_rejected(self, report: EDAReport) -> None:
        inventory = dataclasses.replace(
            report.split_inventory[0], first_date=date(2025, 1, 1), last_date=date(2022, 1, 1)
        )
        forged = self._replace(report, split_inventory=(inventory,) + report.split_inventory[1:])
        with pytest.raises(EDAError, match="split"):
            render_data_card(forged)

    def test_non_chronological_split_rejected(self, report: EDAReport) -> None:
        forged = self._replace(report, split_inventory=tuple(reversed(report.split_inventory)))
        with pytest.raises(EDAError, match="split"):
            render_data_card(forged)

    def test_invalid_data_quality_type_rejected(self, report: EDAReport) -> None:
        forged = self._replace(report, data_quality=object())
        with pytest.raises(EDAError, match="data quality"):
            render_data_card(forged)

    def test_nonzero_duplicate_keys_rejected(self, report: EDAReport) -> None:
        forged = self._replace(
            report,
            data_quality=dataclasses.replace(report.data_quality, duplicate_key_count=1),
        )
        with pytest.raises(EDAError, match="contradictory"):
            render_data_card(forged)

    def test_grid_contradiction_rejected(self, report: EDAReport) -> None:
        forged = self._replace(
            report,
            data_quality=dataclasses.replace(report.data_quality, segment_count=1),
        )
        with pytest.raises(EDAError, match="grid"):
            render_data_card(forged)

    def test_invalid_numeric_ordering_rejected(self, report: EDAReport) -> None:
        swapped = (
            report.numeric_features[1],
            report.numeric_features[0],
        ) + report.numeric_features[2:]
        forged = self._replace(report, numeric_features=swapped)
        with pytest.raises(EDAError, match="numeric"):
            render_data_card(forged)

    def test_invalid_numeric_entry_type_rejected(self, report: EDAReport) -> None:
        forged = self._replace(report, numeric_features=(object(),) + report.numeric_features[1:])
        with pytest.raises(EDAError, match="numeric"):
            render_data_card(forged)

    def test_invalid_categorical_ordering_rejected(self, report: EDAReport) -> None:
        swapped = (report.categorical_features[1], report.categorical_features[0])
        forged = self._replace(report, categorical_features=swapped)
        with pytest.raises(EDAError, match="categorical"):
            render_data_card(forged)

    def test_invalid_datetime_ordering_rejected(self, report: EDAReport) -> None:
        from roadguard.eda import DateSummary

        forged_summary = DateSummary(
            column="construction_date",
            count=1,
            missing_count=0,
            unique_count=1,
            minimum=date(2020, 1, 1),
            maximum=date(2020, 1, 1),
        )
        forged = self._replace(report, datetime_features=(forged_summary, object()))
        with pytest.raises(EDAError, match="datetime"):
            render_data_card(forged)

    def test_numeric_mean_not_float_rejected(self, report: EDAReport) -> None:
        numeric = report.numeric_features[0]
        forged = self._replace(
            report,
            numeric_features=(dataclasses.replace(numeric, mean="nan"),)
            + report.numeric_features[1:],
        )
        with pytest.raises(EDAError, match="non-finite"):
            render_data_card(forged)

    def test_numeric_mean_outside_summary_bounds_rejected(self, report: EDAReport) -> None:
        numeric = report.numeric_features[0]
        forged = self._replace(
            report,
            numeric_features=(
                dataclasses.replace(
                    numeric,
                    mean=numeric.maximum + max(abs(numeric.maximum), 1.0),
                ),
            )
            + report.numeric_features[1:],
        )
        with pytest.raises(EDAError, match="mean"):
            render_data_card(forged)

    def test_negative_population_std_rejected(self, report: EDAReport) -> None:
        numeric = report.numeric_features[0]
        forged = self._replace(
            report,
            numeric_features=(dataclasses.replace(numeric, population_std=-1.0),)
            + report.numeric_features[1:],
        )
        with pytest.raises(EDAError, match="std"):
            render_data_card(forged)

    def test_quartiles_not_ordered_rejected(self, report: EDAReport) -> None:
        numeric = report.numeric_features[0]
        forged = self._replace(
            report,
            numeric_features=(dataclasses.replace(numeric, q1=numeric.maximum + 10.0),)
            + report.numeric_features[1:],
        )
        with pytest.raises(EDAError, match="quartiles"):
            render_data_card(forged)

    def test_outlier_rate_contradiction_rejected(self, report: EDAReport) -> None:
        numeric = report.numeric_features[0]
        forged = self._replace(
            report,
            numeric_features=(dataclasses.replace(numeric, iqr_outlier_rate=0.5),)
            + report.numeric_features[1:],
        )
        with pytest.raises(EDAError, match="rate"):
            render_data_card(forged)

    def test_zero_variance_not_bool_rejected(self, report: EDAReport) -> None:
        numeric = report.numeric_features[0]
        forged = self._replace(
            report,
            numeric_features=(dataclasses.replace(numeric, zero_variance="yes"),)
            + report.numeric_features[1:],
        )
        with pytest.raises(EDAError, match="zero_variance"):
            render_data_card(forged)

    def test_zero_std_without_flag_rejected(self, report: EDAReport) -> None:
        numeric = report.numeric_features[0]
        forged = self._replace(
            report,
            numeric_features=(
                dataclasses.replace(
                    numeric,
                    mean=1.0,
                    population_std=0.0,
                    minimum=1.0,
                    maximum=1.0,
                    q1=1.0,
                    median=1.0,
                    q3=1.0,
                    zero_variance=False,
                ),
            )
            + report.numeric_features[1:],
        )
        with pytest.raises(EDAError, match="zero population std"):
            render_data_card(forged)

    def test_categorical_count_mismatch_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        forged = self._replace(
            report,
            categorical_features=(dataclasses.replace(categorical, count=1),)
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="categorical"):
            render_data_card(forged)

    def test_categorical_no_levels_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        forged = self._replace(
            report,
            categorical_features=(dataclasses.replace(categorical, levels=()),)
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="levels"):
            render_data_card(forged)

    def test_categorical_cardinality_mismatch_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        forged = self._replace(
            report,
            categorical_features=(dataclasses.replace(categorical, cardinality=99),)
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="cardinality"):
            render_data_card(forged)

    def test_categorical_level_type_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        forged = self._replace(
            report,
            categorical_features=(dataclasses.replace(categorical, levels=(object(),)),)
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="level"):
            render_data_card(forged)

    def test_categorical_nonpositive_level_count_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        level = categorical.levels[0]
        forged_level = dataclasses.replace(level, count=0)
        forged = self._replace(
            report,
            categorical_features=(dataclasses.replace(categorical, levels=(forged_level,)),)
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="count"):
            render_data_card(forged)

    def test_categorical_proportion_contradiction_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        level = categorical.levels[0]
        forged_level = dataclasses.replace(level, proportion=0.123456)
        forged = self._replace(
            report,
            categorical_features=(
                dataclasses.replace(categorical, levels=(forged_level,) + categorical.levels[1:]),
            )
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="proportion"):
            render_data_card(forged)

    def test_categorical_level_count_total_mismatch_rejected(self, report: EDAReport) -> None:
        categorical = report.categorical_features[0]
        level = categorical.levels[0]
        forged_level = dataclasses.replace(level, count=level.count + 1)
        forged = self._replace(
            report,
            categorical_features=(
                dataclasses.replace(categorical, levels=(forged_level,) + categorical.levels[1:]),
            )
            + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="proportion|counts"):
            render_data_card(forged)

    def test_categorical_level_order_rejected(self, report: EDAReport) -> None:
        from roadguard.eda import CategoricalLevel, CategoricalSummary

        row_count = report.data_quality.row_count
        forged_summary = CategoricalSummary(
            column="province",
            count=row_count,
            missing_count=0,
            cardinality=2,
            levels=(
                CategoricalLevel("NA", 10, 10 / row_count),
                CategoricalLevel("TH", row_count - 10, (row_count - 10) / row_count),
            ),
        )
        forged = self._replace(
            report,
            categorical_features=(forged_summary,) + report.categorical_features[1:],
        )
        with pytest.raises(EDAError, match="order"):
            render_data_card(forged)

    def test_datetime_summary_type_rejected(self, report: EDAReport) -> None:
        forged = self._replace(report, datetime_features=(object(),))
        with pytest.raises(EDAError, match="datetime"):
            render_data_card(forged)

    def test_datetime_count_mismatch_rejected(self, report: EDAReport) -> None:
        summary = report.datetime_features[0]
        forged = self._replace(
            report,
            datetime_features=(dataclasses.replace(summary, count=1),),
        )
        with pytest.raises(EDAError, match="datetime"):
            render_data_card(forged)

    def test_datetime_unique_out_of_range_rejected(self, report: EDAReport) -> None:
        summary = report.datetime_features[0]
        forged = self._replace(
            report,
            datetime_features=(dataclasses.replace(summary, unique_count=10_000),),
        )
        with pytest.raises(EDAError, match="unique"):
            render_data_card(forged)

    def test_datetime_boundaries_reversed_rejected(self, report: EDAReport) -> None:
        summary = report.datetime_features[0]
        forged = self._replace(
            report,
            datetime_features=(
                dataclasses.replace(summary, minimum=date(2030, 1, 1), maximum=date(2020, 1, 1)),
            ),
        )
        with pytest.raises(EDAError, match="ordered"):
            render_data_card(forged)

    def test_invalid_classification_counts_rejected(self, report: EDAReport) -> None:
        classification = report.classification_target
        forged = self._replace(
            report,
            classification_target=dataclasses.replace(
                classification, negative_count=classification.negative_count + 1
            ),
        )
        with pytest.raises(EDAError, match="classification"):
            render_data_card(forged)

    def test_positive_rate_out_of_range_rejected(self, report: EDAReport) -> None:
        classification = report.classification_target
        forged = self._replace(
            report,
            classification_target=dataclasses.replace(classification, positive_rate=1.5),
        )
        with pytest.raises(EDAError, match="rate"):
            render_data_card(forged)

    def test_non_float_positive_rate_rejected_as_eda_error(self, report: EDAReport) -> None:
        classification = report.classification_target
        forged = self._replace(
            report,
            classification_target=dataclasses.replace(
                classification,
                negative_count=report.data_quality.row_count,
                positive_count=0,
                positive_rate=0,
            ),
        )
        with pytest.raises(EDAError, match="classification positive rate"):
            render_data_card(forged)

    def test_invalid_correlation_entry_type_rejected(self, report: EDAReport) -> None:
        forged = self._replace(
            report, target_correlations=(object(),) + report.target_correlations[1:]
        )
        with pytest.raises(EDAError, match="correlation"):
            render_data_card(forged)

    def test_classification_balance_rejects_non_binary_values(self) -> None:
        from roadguard.eda import _classification_balance

        with pytest.raises(EDAError, match="0 and 1"):
            _classification_balance("maintenance_within_30_days", pd.Series([0, 1, 2]))

    def test_empty_numeric_statistics_rejected(self) -> None:
        from roadguard.eda import _numeric_statistics

        with pytest.raises(EDAError, match="at least one"):
            _numeric_statistics([])

    def test_paired_correlation_lengths_rejected(self) -> None:
        from roadguard.eda import _pearson_correlation

        with pytest.raises(EDAError, match="paired"):
            _pearson_correlation([1.0, 2.0], [1.0])

    def test_categorical_levels_empty_total_rejected(self) -> None:
        from roadguard.eda import _categorical_levels

        with pytest.raises(EDAError, match="at least one"):
            _categorical_levels({}, total=0)

    def test_canonical_scalar_datetime_and_unsupported(self) -> None:
        from roadguard.eda import _canonical_scalar

        assert _canonical_scalar(datetime(2022, 3, 4)) == "2022-03-04"
        assert _canonical_scalar(date(2022, 3, 4)) == "2022-03-04"
        with pytest.raises(EDAError, match="fingerprinted"):
            _canonical_scalar(decimal.Decimal("1.5"))

    def test_missing_training_target_row_rejected(self, dataset: RepositoryExport) -> None:
        targets = dataset.targets.copy()
        first_train_date = _canonical_split(dataset, MINI_SPEC).train_dates[0]
        targets = targets[targets["date"].dt.date != first_train_date]
        forged = dataclasses.replace(dataset, targets=targets)
        split = _canonical_split(dataset, MINI_SPEC)
        with pytest.raises(EDAError, match="target"):
            build_eda_report(forged, split, MINI_SPEC)

    def test_split_dtype_mismatch_rejected(
        self, dataset: RepositoryExport, split: ChronologicalSplit
    ) -> None:
        train = split.train.copy()
        train["rainfall_mm"] = train["rainfall_mm"].astype("int64")
        bad_split = dataclasses.replace(split, train=train)
        with pytest.raises(EDAError, match="dtypes"):
            build_eda_report(dataset, bad_split, MINI_SPEC)


class TestSideEffects:
    def test_no_filesystem_writes(
        self, dataset: RepositoryExport, split: ChronologicalSplit, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def forbidden_open(*args: object, **kwargs: object) -> None:
            raise AssertionError("filesystem write attempted")

        monkeypatch.setattr("builtins.open", forbidden_open)
        report = build_eda_report(dataset, split, MINI_SPEC)
        rendered = render_data_card(report)
        assert rendered.startswith("# RoadGuard AI")

    def test_no_timestamp_or_environment_path(self, report: EDAReport) -> None:
        rendered = render_data_card(report)
        for fragment in ("2026-", "datetime.now", "TEMP", "APPDATA", "USERNAME"):
            assert fragment not in rendered

    def test_v1_report_builds_and_renders(self) -> None:
        dataset = _export(
            V1_SPEC := DatasetSpec(
                dataset_segments=300, dataset_months_per_segment=48, dataset_observations=14_400
            )
        )
        split = _canonical_split(dataset, V1_SPEC)
        report = build_eda_report(dataset, split, V1_SPEC)
        assert report.data_quality.row_count == 10_200
        assert len(report.target_correlations) == 30
        rendered = render_data_card(report)
        assert rendered.count("| road_length_km |") == 3


def _mutated_validation_test(dataset: RepositoryExport) -> RepositoryExport:
    observations = dataset.observations.copy()
    split = _canonical_split(dataset, MINI_SPEC)
    validation_dates = split.validation["date"]
    test_dates = split.test["date"]
    mask = observations["date"].isin(pd.concat([validation_dates, test_dates]))
    observations.loc[mask, "rainfall_mm"] = 999.0
    observations.loc[mask, "temperature"] = -40.0
    return dataclasses.replace(dataset, observations=observations)


def _canonical_scalar(value: object) -> object:
    """Independent canonical scalar mirror used by the fingerprint test."""
    if isinstance(value, (datetime, date)):
        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number == 0.0:
            return "0x0.0p+0"
        return number.hex()
    raise AssertionError(f"unsupported scalar {value!r}")
