"""Phase 8 chronological split and train-only preprocessing contract tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

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
from roadguard.features import (
    FEATURE_COLUMNS,
    FEATURE_KEY_COLUMNS,
    build_feature_frame,
)
from roadguard.preprocessing import (
    CONSTRUCTION_DATE_DAY_COLUMN,
    TEST_DATE_COUNT,
    TRAIN_DATE_COUNT,
    V1_TEST_ROWS,
    V1_TRAIN_ROWS,
    V1_VALIDATION_ROWS,
    VALIDATION_DATE_COUNT,
    ChronologicalSplit,
    PreprocessingError,
    PreprocessorFit,
    TransformedData,
    fit_preprocessor,
    split_chronologically,
    transform,
)

MINI_SPEC = DatasetSpec(dataset_segments=3, dataset_months_per_segment=48, dataset_observations=144)
V1_SPEC = DatasetSpec(
    dataset_segments=300, dataset_months_per_segment=48, dataset_observations=14_400
)
START = date(2022, 1, 1)
DAY_EPOCH = date(1970, 1, 1)


def _feature_frame(spec: DatasetSpec, seed: int = 42) -> pd.DataFrame:
    segments = generate_segments(spec, seed, observation_start=START)
    events = generate_maintenance_events(segments, spec, seed, start_date=START)
    timeline = generate_accident_timeline(segments, spec, seed, start_date=START)
    observations = generate_observations(segments, events, timeline, spec, seed, start_date=START)
    targets = derive_observation_targets(observations, events)
    cleaned = clean_raw_dataset(segments, observations, targets, events, spec)
    return build_feature_frame(
        RepositoryExport(
            segments=cleaned.segments,
            observations=cleaned.observations,
            targets=cleaned.targets,
            maintenance_events=cleaned.maintenance_events,
        ),
        spec,
    )


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _feature_frame(MINI_SPEC)


@pytest.fixture(scope="module")
def split(frame: pd.DataFrame) -> ChronologicalSplit:
    return split_chronologically(frame, MINI_SPEC)


@pytest.fixture(scope="module")
def fit(frame: pd.DataFrame) -> PreprocessorFit:
    return fit_preprocessor(split_chronologically(frame, MINI_SPEC), MINI_SPEC)


class TestPublicSurface:
    def test_phase8_symbols_are_publicly_exported(self) -> None:
        assert roadguard.split_chronologically is split_chronologically
        assert roadguard.fit_preprocessor is fit_preprocessor
        assert roadguard.transform is transform
        assert roadguard.PreprocessorFit is PreprocessorFit
        assert roadguard.ChronologicalSplit is ChronologicalSplit
        assert roadguard.TransformedData is TransformedData
        assert roadguard.PreprocessingError is PreprocessingError

    def test_split_constants_are_exact(self) -> None:
        assert TRAIN_DATE_COUNT == 34
        assert VALIDATION_DATE_COUNT == 7
        assert TEST_DATE_COUNT == 7
        assert V1_TRAIN_ROWS == 10_200
        assert V1_VALIDATION_ROWS == 2_100
        assert V1_TEST_ROWS == 2_100
        assert CONSTRUCTION_DATE_DAY_COLUMN == "construction_date_days"


class TestChronologicalSplit:
    def test_exact_34_7_7_date_boundaries_and_rows(self, frame: pd.DataFrame) -> None:
        unique_dates = sorted(frame["date"].dt.date.unique())
        split = split_chronologically(frame, MINI_SPEC)

        assert len(unique_dates) == 48
        assert tuple(split.train_dates) == tuple(unique_dates[:34])
        assert tuple(split.validation_dates) == tuple(unique_dates[34:41])
        assert tuple(split.test_dates) == tuple(unique_dates[41:48])
        assert len(split.train) == 34 * 3
        assert len(split.validation) == 7 * 3
        assert len(split.test) == 7 * 3
        assert set(split.train["date"].dt.date.unique()) == set(split.train_dates)
        assert set(split.validation["date"].dt.date.unique()) == set(split.validation_dates)
        assert set(split.test["date"].dt.date.unique()) == set(split.test_dates)

    def test_v1_split_exact_counts(self) -> None:
        v1 = _feature_frame(V1_SPEC)
        split = split_chronologically(v1, V1_SPEC)

        assert len(v1) == 14_400
        assert v1["date"].nunique() == 48
        assert v1["segment_id"].nunique() == 300
        assert len(split.train) == V1_TRAIN_ROWS
        assert len(split.validation) == V1_VALIDATION_ROWS
        assert len(split.test) == V1_TEST_ROWS
        for partition in (split.train, split.validation, split.test):
            assert partition["segment_id"].nunique() == 300
            assert len(partition) == partition["date"].nunique() * 300

    def test_partitions_disjoint_contiguous_and_complete(self, frame: pd.DataFrame) -> None:
        split = split_chronologically(frame, MINI_SPEC)

        assert not set(split.train_dates).intersection(split.validation_dates)
        assert not set(split.train_dates).intersection(split.test_dates)
        assert not set(split.validation_dates).intersection(split.test_dates)
        assert (
            set(split.train_dates) | set(split.validation_dates) | set(split.test_dates)
        ) == set(frame["date"].dt.date.unique())

        combined = (
            pd.concat([split.train, split.validation, split.test], ignore_index=True)
            .sort_values(list(FEATURE_KEY_COLUMNS), kind="stable")
            .reset_index(drop=True)
        )
        expected = frame.sort_values(list(FEATURE_KEY_COLUMNS), kind="stable").reset_index(
            drop=True
        )
        assert_frame_equal(combined, expected)

    def test_partitions_canonically_sorted(self, split: ChronologicalSplit) -> None:
        for partition in (split.train, split.validation, split.test):
            assert_frame_equal(
                partition,
                partition.sort_values(list(FEATURE_KEY_COLUMNS), kind="stable").reset_index(
                    drop=True
                ),
            )

    def test_input_row_order_invariance(self, frame: pd.DataFrame) -> None:
        shuffled = frame.sample(frac=1.0, random_state=5).reset_index(drop=True)
        expected = split_chronologically(frame, MINI_SPEC)
        actual = split_chronologically(shuffled, MINI_SPEC)

        assert_frame_equal(actual.train, expected.train)
        assert_frame_equal(actual.validation, expected.validation)
        assert_frame_equal(actual.test, expected.test)

    def test_caller_frame_not_mutated(self, frame: pd.DataFrame) -> None:
        before = frame.copy(deep=True)
        split_chronologically(frame, MINI_SPEC)
        assert_frame_equal(frame, before)


class TestInputBoundary:
    def test_missing_column_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.drop(columns=["humidity"])
        with pytest.raises(PreprocessingError, match="column"):
            split_chronologically(bad, MINI_SPEC)

    def test_extra_column_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad["extra"] = 1
        with pytest.raises(PreprocessingError, match="column"):
            split_chronologically(bad, MINI_SPEC)

    def test_reordered_columns_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame[list(reversed(frame.columns))].copy()
        with pytest.raises(PreprocessingError, match="column"):
            split_chronologically(bad, MINI_SPEC)

    def test_duplicate_column_label_rejected(self, frame: pd.DataFrame) -> None:
        bad = pd.concat([frame.iloc[:, :2], frame[["temperature"]], frame.iloc[:, 2:]], axis=1)
        with pytest.raises(PreprocessingError, match="column"):
            split_chronologically(bad, MINI_SPEC)

    def test_duplicate_natural_keys_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        duplicate = bad.iloc[0].copy()
        duplicate["segment_id"] = bad["segment_id"].iloc[1]
        duplicate["date"] = bad["date"].iloc[1]
        bad.loc[len(bad)] = duplicate
        with pytest.raises(PreprocessingError, match="duplicate"):
            split_chronologically(bad, MINI_SPEC)

    def test_null_float_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad.loc[0, "rainfall_mm"] = np.nan
        with pytest.raises(PreprocessingError, match="null|missing"):
            split_chronologically(bad, MINI_SPEC)

    def test_null_object_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad.loc[0, "province"] = None
        with pytest.raises(PreprocessingError, match="null|missing"):
            split_chronologically(bad, MINI_SPEC)

    def test_infinite_float_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad.loc[0, "temperature"] = np.inf
        with pytest.raises(PreprocessingError, match="finite"):
            split_chronologically(bad, MINI_SPEC)

    def test_malformed_date_dtype_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad["date"] = pd.Series(bad["date"].astype(str), dtype=object)
        with pytest.raises(PreprocessingError, match="date"):
            split_chronologically(bad, MINI_SPEC)

    def test_nat_date_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad.loc[0, "date"] = pd.NaT
        with pytest.raises(PreprocessingError, match="null|missing|date"):
            split_chronologically(bad, MINI_SPEC)

    def test_timezone_aware_date_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad["date"] = pd.to_datetime(bad["date"]).dt.tz_localize("UTC")
        with pytest.raises(PreprocessingError, match="timezone"):
            split_chronologically(bad, MINI_SPEC)

    def test_non_midnight_date_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad["date"] = pd.to_datetime(bad["date"]) + pd.offsets.Hour(12)
        with pytest.raises(PreprocessingError, match="midnight"):
            split_chronologically(bad, MINI_SPEC)

    def test_invalid_dtype_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad["traffic_volume"] = bad["traffic_volume"].astype("float64")
        with pytest.raises(PreprocessingError, match="dtype"):
            split_chronologically(bad, MINI_SPEC)

    def test_incomplete_grid_missing_row_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.iloc[:-1].copy()
        with pytest.raises(PreprocessingError, match="grid|rows|date"):
            split_chronologically(bad, MINI_SPEC)

    def test_incomplete_grid_missing_date_rejected(self, frame: pd.DataFrame) -> None:
        last_date = frame["date"].max()
        bad = frame[frame["date"] != last_date].copy()
        with pytest.raises(PreprocessingError, match="grid|date"):
            split_chronologically(bad, MINI_SPEC)

    def test_spec_mismatch_rejected(self, frame: pd.DataFrame) -> None:
        with pytest.raises(PreprocessingError, match="grid|date|segment"):
            split_chronologically(frame, V1_SPEC)

    def test_spec_month_count_must_match_locked_48_date_split(self, frame: pd.DataFrame) -> None:
        wrong = DatasetSpec(
            dataset_segments=3,
            dataset_months_per_segment=47,
            dataset_observations=141,
        )
        with pytest.raises(PreprocessingError, match="48|month|date"):
            split_chronologically(frame, wrong)

    def test_monthly_calendar_gap_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        final_date = bad["date"].max()
        bad.loc[bad["date"] == final_date, "date"] = final_date + pd.DateOffset(months=2)
        with pytest.raises(PreprocessingError, match="calendar|month|date"):
            split_chronologically(bad, MINI_SPEC)

    def test_noncanonical_datetime_unit_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad["date"] = bad["date"].astype("datetime64[ms]")
        with pytest.raises(PreprocessingError, match=r"datetime64\[ns\]|dtype"):
            split_chronologically(bad, MINI_SPEC)

    def test_non_dataframe_rejected(self) -> None:
        with pytest.raises(PreprocessingError):
            split_chronologically([1, 2, 3], MINI_SPEC)

    def test_split_rejects_non_spec(self, frame: pd.DataFrame) -> None:
        with pytest.raises(TypeError, match="spec"):
            split_chronologically(frame, "not-a-spec")

    def test_fit_rejects_non_spec(self, split: ChronologicalSplit) -> None:
        with pytest.raises(TypeError, match="spec"):
            fit_preprocessor(split.train, "not-a-spec")

    def test_non_string_object_value_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        bad.loc[0, "province"] = 123
        with pytest.raises(PreprocessingError, match="string"):
            split_chronologically(bad, MINI_SPEC)

    def test_malformed_segment_id_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        segment_id = bad["segment_id"].iloc[0]
        bad.loc[bad["segment_id"] == segment_id, "segment_id"] = "not-a-road-id"
        with pytest.raises(PreprocessingError, match="segment_id|pattern"):
            split_chronologically(bad, MINI_SPEC)

    def test_static_segment_attribute_drift_rejected(self, frame: pd.DataFrame) -> None:
        bad = frame.copy()
        original = bad.loc[0, "province"]
        bad.loc[0, "province"] = next(value for value in ("NA", "TH") if value != original)
        with pytest.raises(PreprocessingError, match="static|invariant"):
            split_chronologically(bad, MINI_SPEC)


class TestTrainOnlyPreprocessing:
    def test_fit_requires_train_partition(
        self, frame: pd.DataFrame, split: ChronologicalSplit
    ) -> None:
        with pytest.raises(TypeError, match="ChronologicalSplit"):
            fit_preprocessor(split.validation, MINI_SPEC)
        with pytest.raises(TypeError, match="ChronologicalSplit"):
            fit_preprocessor(split.test, MINI_SPEC)

        future_34_dates = sorted(frame["date"].unique())[-TRAIN_DATE_COUNT:]
        future_contaminated = frame[frame["date"].isin(future_34_dates)].copy()
        with pytest.raises(TypeError, match="ChronologicalSplit"):
            fit_preprocessor(future_contaminated, MINI_SPEC)

    def test_transformed_output_excludes_keys(self, fit: PreprocessorFit) -> None:
        train = split_chronologically(_feature_frame(MINI_SPEC), MINI_SPEC).train
        result = transform(train, fit)

        assert tuple(result.keys.columns) == FEATURE_KEY_COLUMNS
        assert not set(FEATURE_KEY_COLUMNS).intersection(result.features.columns)
        assert tuple(result.features.columns) == fit.transformed_feature_columns

    def test_all_transformed_values_finite_float64(self, fit: PreprocessorFit) -> None:
        split = split_chronologically(_feature_frame(MINI_SPEC), MINI_SPEC)
        for partition in (split.train, split.validation, split.test):
            result = transform(partition, fit)
            assert all(str(dtype) == "float64" for dtype in result.features.dtypes), (
                f"non-float64 dtype: {dict(result.features.dtypes)}"
            )
            assert np.isfinite(result.features.to_numpy()).all()

    def test_one_hot_encoder_uses_train_categories_only(self, fit: PreprocessorFit) -> None:
        train = split_chronologically(_feature_frame(MINI_SPEC), MINI_SPEC).train
        assert fit.province_categories == tuple(sorted(train["province"].unique()))
        assert fit.road_type_categories == tuple(sorted(train["road_type"].unique()))
        province_columns = [
            column for column in fit.transformed_feature_columns if column.startswith("province_")
        ]
        road_type_columns = [
            column for column in fit.transformed_feature_columns if column.startswith("road_type_")
        ]
        assert province_columns == [f"province_{c}" for c in fit.province_categories]
        assert road_type_columns == [f"road_type_{c}" for c in fit.road_type_categories]

    def test_unseen_category_encoded_as_all_zeros(self, fit: PreprocessorFit) -> None:
        validation = split_chronologically(_feature_frame(MINI_SPEC), MINI_SPEC).validation
        extra = validation.iloc[:1].copy()
        extra["segment_id"] = "QL99-KM99-99"
        extra["province"] = "XX"
        extra["road_type"] = "expressway"
        extended = pd.concat([validation, extra], ignore_index=True)

        result = transform(extended, fit)
        row = result.features.iloc[-1]
        province_columns = [
            column for column in fit.transformed_feature_columns if column.startswith("province_")
        ]
        road_type_columns = [
            column for column in fit.transformed_feature_columns if column.startswith("road_type_")
        ]
        assert float(row[province_columns].sum()) == 0.0
        assert float(row[road_type_columns].sum()) == 0.0
        assert tuple(result.features.columns) == fit.transformed_feature_columns

    def test_transform_rejects_static_attribute_drift(
        self, fit: PreprocessorFit, split: ChronologicalSplit
    ) -> None:
        bad = split.validation.copy()
        bad.loc[0, "road_length_km"] = bad.loc[0, "road_length_km"] + 1.0
        with pytest.raises(PreprocessingError, match="static|invariant"):
            transform(bad, fit)

    def test_construction_date_day_representation_and_train_scaling(
        self, fit: PreprocessorFit
    ) -> None:
        train = split_chronologically(_feature_frame(MINI_SPEC), MINI_SPEC).train
        days = np.array(
            [(value.date() - DAY_EPOCH).days for value in train["construction_date"]],
            dtype="float64",
        )
        index = fit.scaled_columns.index(CONSTRUCTION_DATE_DAY_COLUMN)
        mean = float(np.mean(days))
        std = float(np.std(days))
        scale = std if std > 0 else 1.0
        expected = (days - mean) / scale

        result = transform(train, fit)
        np.testing.assert_allclose(
            result.features[CONSTRUCTION_DATE_DAY_COLUMN].to_numpy(), expected
        )
        assert fit.means[index] == pytest.approx(mean)
        assert fit.stds[index] == pytest.approx(std)

    def test_scaling_uses_train_statistics_only(self, fit: PreprocessorFit) -> None:
        train = split_chronologically(_feature_frame(MINI_SPEC), MINI_SPEC).train
        result = transform(train, fit)
        for position, column in enumerate(fit.scaled_columns):
            if column == CONSTRUCTION_DATE_DAY_COLUMN:
                raw = np.array(
                    [(value.date() - DAY_EPOCH).days for value in train["construction_date"]],
                    dtype="float64",
                )
            else:
                raw = train[column].to_numpy(dtype="float64")
            mean = fit.means[position]
            std = fit.stds[position]
            scale = std if std > 0 else 1.0
            np.testing.assert_allclose(result.features[column].to_numpy(), (raw - mean) / scale)

    def test_changing_validation_test_cannot_change_fitted_state(self) -> None:
        frame = _feature_frame(MINI_SPEC)
        split = split_chronologically(frame, MINI_SPEC)
        fit_a = fit_preprocessor(split, MINI_SPEC)

        mutated_validation = split.validation.copy()
        mutated_validation["rainfall_mm"] = mutated_validation["rainfall_mm"] * 100
        mutated_test = split.test.copy()
        mutated_test["rainfall_mm"] = mutated_test["rainfall_mm"] * 1.1
        changed_split = replace(
            split,
            validation=mutated_validation,
            test=mutated_test,
        )
        fit_b = fit_preprocessor(changed_split, MINI_SPEC)
        transform(mutated_validation, fit_b)
        transform(mutated_test, fit_b)

        assert fit_a == fit_b

    def test_changing_validation_test_cannot_change_transformed_train(self) -> None:
        frame = _feature_frame(MINI_SPEC)
        split = split_chronologically(frame, MINI_SPEC)
        fit_a = fit_preprocessor(split, MINI_SPEC)
        baseline = transform(split.train, fit_a).features

        mutated_validation = split.validation.copy()
        mutated_validation["temperature"] = 0.0
        mutated_test = split.test.copy()
        mutated_test["humidity"] = 50.0
        transform(mutated_validation, fit_a)
        transform(mutated_test, fit_a)
        after = transform(split.train, fit_a).features

        assert_frame_equal(after, baseline)

    def test_zero_variance_training_column_remains_finite(self) -> None:
        frame = _feature_frame(MINI_SPEC)
        train = split_chronologically(frame, MINI_SPEC).train.copy()
        train["humidity"] = 42.0
        fit = fit_preprocessor(
            replace(split_chronologically(frame, MINI_SPEC), train=train),
            MINI_SPEC,
        )
        result = transform(train, fit)

        assert np.isfinite(result.features["humidity"].to_numpy()).all()
        np.testing.assert_allclose(result.features["humidity"].to_numpy(), np.zeros(len(train)))

    def test_large_zero_variance_training_column_transforms_to_exact_zero(self) -> None:
        frame = _feature_frame(MINI_SPEC)
        frame["road_length_km"] = 1e100
        split = split_chronologically(frame, MINI_SPEC)
        fit = fit_preprocessor(split, MINI_SPEC)
        result = transform(split.train, fit)

        position = fit.scaled_columns.index("road_length_km")
        assert fit.means[position] == 1e100
        assert fit.stds[position] == 0.0
        np.testing.assert_array_equal(
            result.features["road_length_km"].to_numpy(),
            np.zeros(len(split.train)),
        )

    def test_finite_extreme_that_overflows_transform_is_rejected(
        self, fit: PreprocessorFit, split: ChronologicalSplit
    ) -> None:
        rainfall_position = fit.scaled_columns.index("rainfall_mm")
        tiny_stds = list(fit.stds)
        tiny_stds[rainfall_position] = float(np.nextafter(0.0, 1.0))
        forged_fit = replace(fit, stds=tuple(tiny_stds))
        with pytest.raises(PreprocessingError, match="non-finite"):
            transform(split.validation, forged_fit)

    def test_forged_fitted_state_is_rejected(
        self, fit: PreprocessorFit, split: ChronologicalSplit
    ) -> None:
        with pytest.raises(PreprocessingError, match="schema|statistics"):
            transform(split.validation, replace(fit, means=fit.means[:-1]))
        with pytest.raises(PreprocessingError, match="registries"):
            transform(split.validation, replace(fit, province_categories=("UNBOUNDED",)))
        with pytest.raises(PreprocessingError, match="immutable tuples"):
            transform(split.validation, replace(fit, means=list(fit.means)))
        with pytest.raises(PreprocessingError, match="sorted unique strings"):
            transform(split.validation, replace(fit, province_categories=()))

    def test_fit_is_exactly_row_order_invariant(self, split: ChronologicalSplit) -> None:
        shuffled = replace(
            split,
            train=split.train.sample(frac=1.0, random_state=9),
            validation=split.validation.sample(frac=1.0, random_state=10),
            test=split.test.sample(frac=1.0, random_state=11),
        )
        assert fit_preprocessor(shuffled, MINI_SPEC) == fit_preprocessor(split, MINI_SPEC)

    def test_deterministic_feature_names_and_dtypes(self, fit: PreprocessorFit) -> None:
        train = split_chronologically(_feature_frame(MINI_SPEC), MINI_SPEC).train
        result = transform(train, fit)
        expected_scaled = [
            column
            for column in FEATURE_COLUMNS
            if column not in ("province", "road_type", "construction_date")
        ] + [CONSTRUCTION_DATE_DAY_COLUMN]
        expected_one_hot = [f"province_{c}" for c in fit.province_categories] + [
            f"road_type_{c}" for c in fit.road_type_categories
        ]
        assert tuple(result.features.columns) == tuple(expected_scaled + expected_one_hot)

    def test_repeated_execution_identical(self, frame: pd.DataFrame) -> None:
        first = split_chronologically(frame, MINI_SPEC)
        second = split_chronologically(frame, MINI_SPEC)
        assert_frame_equal(first.train, second.train)
        assert_frame_equal(first.validation, second.validation)
        assert_frame_equal(first.test, second.test)

        fit = fit_preprocessor(first, MINI_SPEC)
        assert_frame_equal(
            transform(first.train, fit).features, transform(second.train, fit).features
        )

    def test_forbidden_fields_never_become_features(self, fit: PreprocessorFit) -> None:
        forbidden = {
            "days_until_maintenance",
            "maintenance_within_30_days",
            "maintenance_date",
            "maintenance_cost",
            "thermoplastic_paint_kg",
            "traffic_base",
        }
        assert not forbidden.intersection(fit.transformed_feature_columns)

    def test_transform_rejects_invalid_frame(self, fit: PreprocessorFit) -> None:
        train = split_chronologically(_feature_frame(MINI_SPEC), MINI_SPEC).train
        bad = train.drop(columns=["rainfall_mm"])
        with pytest.raises(PreprocessingError, match="column"):
            transform(bad, fit)

    def test_fit_and_transform_do_not_mutate_inputs(self) -> None:
        frame = _feature_frame(MINI_SPEC)
        split = split_chronologically(frame, MINI_SPEC)
        before_train = split.train.copy(deep=True)
        before_validation = split.validation.copy(deep=True)
        fit = fit_preprocessor(split, MINI_SPEC)
        transform(split.validation, fit)
        assert_frame_equal(split.train, before_train)
        assert_frame_equal(split.validation, before_validation)

    def test_v1_fit_transform_shapes(self) -> None:
        v1 = _feature_frame(V1_SPEC)
        split = split_chronologically(v1, V1_SPEC)
        fit = fit_preprocessor(split, V1_SPEC)
        for partition in (split.train, split.validation, split.test):
            result = transform(partition, fit)
            assert len(result.keys) == len(partition)
            assert tuple(result.features.columns) == fit.transformed_feature_columns
            assert np.isfinite(result.features.to_numpy()).all()
