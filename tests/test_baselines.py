"""Phase 10 baseline supervised evaluation contract tests.

The contract under test is docs/contracts.md section 17. Success-path and
known-vector fixtures are deterministic crafted exports (one segment, 48
months) whose events are fully controlled; one generated multi-segment export
and a complete V1 evaluation exercise the real pipeline. Private helpers
``_select_threshold``, ``_positive_probas`` and ``_regression_predictions``
are exercised directly because the locked dummy estimators always emit
constant probabilities, which makes threshold tie-break rules and
estimator-output validation unreachable through the public workflow alone.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.dummy import DummyClassifier, DummyRegressor  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

import roadguard
import roadguard.baselines as baselines_module
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
from roadguard.baselines import (
    BASELINE_CLASSIFIER_NAME,
    BASELINE_CONTRACT_VERSION,
    BASELINE_REGRESSOR_NAME,
    BaselineEvaluation,
    BaselineEvaluationError,
    ClassificationBaselineMetrics,
    RegressionBaselineMetrics,
    _fit_classifier,
    _fit_regressor,
    _positive_probas,
    _regression_predictions,
    _select_threshold,
    evaluate_baselines,
)
from roadguard.features import FEATURE_KEY_COLUMNS, build_feature_frame
from roadguard.preprocessing import (
    ChronologicalSplit,
    PreprocessorFit,
    fit_preprocessor,
    split_chronologically,
    transform,
)
from roadguard.targets import TARGET_COLUMNS

START = date(2022, 1, 1)
CRAFTED_SPEC = DatasetSpec(
    dataset_segments=1, dataset_months_per_segment=48, dataset_observations=48
)
GENERATED_SPEC = DatasetSpec(
    dataset_segments=5, dataset_months_per_segment=48, dataset_observations=240
)
V1_SPEC = DatasetSpec(
    dataset_segments=300, dataset_months_per_segment=48, dataset_observations=14_400
)

PHASE10_PUBLIC_NAMES = (
    "evaluate_baselines",
    "BaselineEvaluationError",
    "BaselineEvaluation",
    "ClassificationBaselineMetrics",
    "RegressionBaselineMetrics",
    "BASELINE_CONTRACT_VERSION",
    "BASELINE_CLASSIFIER_NAME",
    "BASELINE_REGRESSOR_NAME",
)

CLASSIFICATION_FIELDS = (
    "validation_pr_auc",
    "decision_threshold",
    "validation_f1",
    "validation_recall",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "test_confusion_matrix",
)
REGRESSION_FIELDS = ("validation_mae", "validation_rmse", "test_mae", "test_rmse", "test_r2")
EVALUATION_FIELDS = (
    "contract_version",
    "classifier_name",
    "regressor_name",
    "feature_columns",
    "train_rows",
    "validation_rows",
    "test_rows",
    "classification",
    "regression",
)

FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "segment_id",
        "date",
        "days_until_maintenance",
        "maintenance_within_30_days",
        "maintenance_date",
        "maintenance_cost",
        "thermoplastic_paint_kg",
        "reflective_sheet_m2",
        "guardrail_meter",
        "traffic_sign_quantity",
        "traffic_base",
        "heavy_vehicle_ratio_base",
        "weather_exposure",
        "deterioration_rate",
        "accident_propensity",
        "initial_condition",
    }
)


def _month_first(month_index: int) -> date:
    year = 2022 + (month_index - 1) // 12
    month = (month_index - 1) % 12 + 1
    return date(year, month, 1)


def _month_15(month_index: int) -> date:
    return _month_first(month_index).replace(day=15)


def _crafted_export(events: list[date]) -> RepositoryExport:
    """Build a fully valid one-segment export whose events are exactly ``events``."""
    construction = pd.Timestamp("2000-01-01")
    segments = pd.DataFrame(
        {
            "segment_id": pd.Series(["QL01-KM0-1"], dtype=object),
            "province": pd.Series(["NA"], dtype=object),
            "road_type": pd.Series(["national"], dtype=object),
            "construction_date": pd.to_datetime([construction]).astype("datetime64[ns]"),
            "road_length_km": pd.Series([1.0], dtype="float64"),
        }
    )
    event_frame = pd.DataFrame(
        {
            "segment_id": pd.Series(["QL01-KM0-1"] * len(events), dtype=object),
            "maintenance_date": pd.to_datetime(pd.Series(events)).astype("datetime64[ns]"),
        }
    )
    rows: list[dict[str, object]] = []
    for month_index in range(1, 49):
        t = _month_first(month_index)
        prior = [event for event in events if event < t]
        rows.append(
            {
                "segment_id": "QL01-KM0-1",
                "date": t,
                "traffic_volume": 1000,
                "heavy_vehicle_ratio": 0.3,
                "road_age_days": (t - construction.date()).days,
                "rainfall_mm": 100.0,
                "temperature": 27.0,
                "humidity": 60.0,
                "days_since_last_maintenance": (
                    (t - prior[-1]).days if prior else min((t - construction.date()).days, 3650)
                ),
                "previous_repairs": len(prior),
                "road_condition_score": 50,
                "marking_condition_score": 50,
                "guardrail_condition_score": 50,
                "sign_condition_score": 50,
                "accident_count_30d": 0,
                "accident_count_365d": 0,
            }
        )
    observations = pd.DataFrame(rows)
    observations["segment_id"] = observations["segment_id"].astype(object)
    observations["date"] = pd.to_datetime(observations["date"]).astype("datetime64[ns]")
    for column in (
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
    ):
        observations[column] = observations[column].astype("int64")
    for column in ("heavy_vehicle_ratio", "rainfall_mm", "temperature", "humidity"):
        observations[column] = observations[column].astype("float64")
    targets = derive_observation_targets(observations, event_frame)
    return RepositoryExport(
        segments=segments,
        observations=observations,
        targets=targets,
        maintenance_events=event_frame,
    )


def _export_a() -> RepositoryExport:
    """Monthly day-15 events from month 36: single-class train, constant test days."""
    return _crafted_export([_month_15(month_index) for month_index in range(36, 61)])


def _export_b() -> RepositoryExport:
    """Two early events plus monthly day-15 events from month 49: single-class
    validation and single-class test with distinct days."""
    return _crafted_export(
        [date(2022, 3, 20), date(2022, 10, 5)]
        + [_month_15(month_index) for month_index in range(49, 61)]
    )


def _export_c() -> RepositoryExport:
    """Two early events plus monthly day-15 events from months 36-44 and 46-60:
    both classes in every partition, with known vectors for every metric."""
    return _crafted_export(
        [date(2022, 3, 20), date(2022, 10, 5)]
        + [_month_15(month_index) for month_index in range(36, 45)]
        + [_month_15(month_index) for month_index in range(46, 61)]
    )


def _export_d() -> RepositoryExport:
    """Two early events plus monthly day-15 events from months 36-60: two-class
    train and validation, constant 14-day test regression targets."""
    return _crafted_export(
        [date(2022, 3, 20), date(2022, 10, 5)]
        + [_month_15(month_index) for month_index in range(36, 61)]
    )


def _export_e() -> RepositoryExport:
    """Two early events plus monthly day-15 events from months 36-41 and 49-60:
    two-class train and validation, single-class test with distinct days."""
    return _crafted_export(
        [date(2022, 3, 20), date(2022, 10, 5)]
        + [_month_15(month_index) for month_index in range(36, 42)]
        + [_month_15(month_index) for month_index in range(49, 61)]
    )


def _generated_export(spec: DatasetSpec, seed: int = 42) -> RepositoryExport:
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
    return split_chronologically(build_feature_frame(dataset, spec), spec)


def _canonical_fit(split: ChronologicalSplit, spec: DatasetSpec) -> PreprocessorFit:
    return fit_preprocessor(split, spec)


def _reference_evaluation(dataset: RepositoryExport, spec: DatasetSpec) -> dict[str, object]:
    """Independent evaluation built only from public Phase 1-9 APIs and direct
    scikit-learn calls, used as the reference for known-vector comparisons."""
    frame = build_feature_frame(dataset, spec)
    canonical = split_chronologically(frame, spec)
    refit = fit_preprocessor(canonical, spec)
    transformed = {
        name: transform(getattr(canonical, name), refit) for name in ("train", "validation", "test")
    }
    joined = {}
    for name in ("train", "validation", "test"):
        joined[name] = transformed[name].keys.merge(
            dataset.targets,
            how="left",
            on=list(FEATURE_KEY_COLUMNS),
            sort=False,
            validate="one_to_one",
        )
    y_class = {
        name: joined[name]["maintenance_within_30_days"].to_numpy(dtype="int64")
        for name in ("train", "validation", "test")
    }
    y_regression = {
        name: joined[name]["days_until_maintenance"].to_numpy(dtype="int64")
        for name in ("train", "validation", "test")
    }
    x = {
        name: transformed[name].features.to_numpy(dtype="float64")
        for name in ("train", "validation", "test")
    }
    classifier = DummyClassifier(strategy="prior")
    classifier.fit(x["train"], y_class["train"])
    regressor = DummyRegressor(strategy="median")
    regressor.fit(x["train"], y_regression["train"])
    validation_probs = [float(v) for v in classifier.predict_proba(x["validation"])[:, 1]]
    test_probs = [float(v) for v in classifier.predict_proba(x["test"])[:, 1]]
    validation_preds = [float(v) for v in regressor.predict(x["validation"])]
    test_preds = [float(v) for v in regressor.predict(x["test"])]
    candidates = sorted({0.0, 1.0, *validation_probs})
    threshold = max(
        candidates,
        key=lambda t: (
            f1_score(
                y_class["validation"],
                (np.asarray(validation_probs) >= t).astype(np.int64),
                zero_division=0,
            ),
            recall_score(
                y_class["validation"],
                (np.asarray(validation_probs) >= t).astype(np.int64),
                zero_division=0,
            ),
            t,
        ),
    )
    validation_hard = (np.asarray(validation_probs) >= threshold).astype(np.int64)
    test_hard = (np.asarray(test_probs) >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_class["test"], test_hard, labels=(0, 1))
    return {
        "threshold": float(threshold),
        "val_pr_auc": float(average_precision_score(y_class["validation"], validation_probs)),
        "val_f1": float(f1_score(y_class["validation"], validation_hard, zero_division=0)),
        "val_recall": float(recall_score(y_class["validation"], validation_hard, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_class["test"], test_hard)),
        "test_precision": float(precision_score(y_class["test"], test_hard, zero_division=0)),
        "test_recall": float(recall_score(y_class["test"], test_hard, zero_division=0)),
        "test_f1": float(f1_score(y_class["test"], test_hard, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_class["test"], test_probs)),
        "test_cm": ((int(matrix[0, 0]), int(matrix[0, 1])), (int(matrix[1, 0]), int(matrix[1, 1]))),
        "val_mae": float(mean_absolute_error(y_regression["validation"], validation_preds)),
        "val_rmse": float(root_mean_squared_error(y_regression["validation"], validation_preds)),
        "test_mae": float(mean_absolute_error(y_regression["test"], test_preds)),
        "test_rmse": float(root_mean_squared_error(y_regression["test"], test_preds)),
        "test_r2": float(r2_score(y_regression["test"], test_preds, force_finite=False)),
        "feature_columns": refit.transformed_feature_columns,
        "train_rows": len(canonical.train),
        "validation_rows": len(canonical.validation),
        "test_rows": len(canonical.test),
    }


def _assert_matches_reference(
    result: BaselineEvaluation, dataset: RepositoryExport, spec: DatasetSpec
) -> None:
    expected = _reference_evaluation(dataset, spec)
    classification = result.classification
    assert classification.validation_pr_auc == pytest.approx(float(expected["val_pr_auc"]))
    assert classification.decision_threshold == pytest.approx(float(expected["threshold"]))
    assert classification.validation_f1 == pytest.approx(float(expected["val_f1"]))
    assert classification.validation_recall == pytest.approx(float(expected["val_recall"]))
    assert classification.test_accuracy == pytest.approx(float(expected["test_accuracy"]))
    assert classification.test_precision == pytest.approx(float(expected["test_precision"]))
    assert classification.test_recall == pytest.approx(float(expected["test_recall"]))
    assert classification.test_f1 == pytest.approx(float(expected["test_f1"]))
    assert classification.test_roc_auc == pytest.approx(float(expected["test_roc_auc"]))
    assert classification.test_confusion_matrix == expected["test_cm"]
    regression = result.regression
    assert regression.validation_mae == pytest.approx(float(expected["val_mae"]))
    assert regression.validation_rmse == pytest.approx(float(expected["val_rmse"]))
    assert regression.test_mae == pytest.approx(float(expected["test_mae"]))
    assert regression.test_rmse == pytest.approx(float(expected["test_rmse"]))
    assert regression.test_r2 == pytest.approx(float(expected["test_r2"]))
    assert result.feature_columns == expected["feature_columns"]
    assert result.train_rows == expected["train_rows"]
    assert result.validation_rows == expected["validation_rows"]
    assert result.test_rows == expected["test_rows"]
    assert result.contract_version == BASELINE_CONTRACT_VERSION
    assert result.classifier_name == BASELINE_CLASSIFIER_NAME
    assert result.regressor_name == BASELINE_REGRESSOR_NAME


@pytest.fixture(scope="module")
def dataset_c() -> RepositoryExport:
    return _export_c()


@pytest.fixture(scope="module")
def split_c(dataset_c: RepositoryExport) -> ChronologicalSplit:
    return _canonical_split(dataset_c, CRAFTED_SPEC)


@pytest.fixture(scope="module")
def fit_c(split_c: ChronologicalSplit) -> PreprocessorFit:
    return _canonical_fit(split_c, CRAFTED_SPEC)


@pytest.fixture(scope="module")
def result_c(
    dataset_c: RepositoryExport,
    split_c: ChronologicalSplit,
    fit_c: PreprocessorFit,
) -> BaselineEvaluation:
    return evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)


class TestPublicSurface:
    def test_module_all_exact(self) -> None:
        import roadguard.baselines as baselines

        assert tuple(baselines.__all__) == PHASE10_PUBLIC_NAMES

    def test_package_root_exports_phase10_symbols(self) -> None:
        for name in PHASE10_PUBLIC_NAMES:
            assert hasattr(roadguard, name), f"missing public attribute: {name}"

    def test_constants_exact(self) -> None:
        assert BASELINE_CONTRACT_VERSION == "roadguard.phase10.v1"
        assert BASELINE_CLASSIFIER_NAME == "dummy_prior"
        assert BASELINE_REGRESSOR_NAME == "dummy_median"

    def test_evaluate_baselines_signature_exact_and_seed_free(self) -> None:
        signature = inspect.signature(evaluate_baselines)
        assert tuple(signature.parameters) == ("dataset", "split", "fit", "spec")
        assert "seed" not in signature.parameters
        assert "random_state" not in signature.parameters

    def test_no_public_frozen_test_evaluation_function(self) -> None:
        import roadguard.baselines as baselines

        for name in ("evaluate_test", "evaluate_test_partition", "predict"):
            assert not hasattr(baselines, name)


class TestFrozenSchema:
    def test_classification_metrics_fields_and_order(self) -> None:
        assert (
            tuple(field.name for field in dataclasses.fields(ClassificationBaselineMetrics))
            == CLASSIFICATION_FIELDS
        )

    def test_regression_metrics_fields_and_order(self) -> None:
        assert (
            tuple(field.name for field in dataclasses.fields(RegressionBaselineMetrics))
            == REGRESSION_FIELDS
        )

    def test_evaluation_fields_and_order(self) -> None:
        assert (
            tuple(field.name for field in dataclasses.fields(BaselineEvaluation))
            == EVALUATION_FIELDS
        )

    def test_result_types_are_frozen(self) -> None:
        for cls in (
            BaselineEvaluation,
            ClassificationBaselineMetrics,
            RegressionBaselineMetrics,
        ):
            assert dataclasses.is_dataclass(cls)
            assert cls.__dataclass_params__.frozen

    def test_result_contains_only_builtin_scalars_and_tuples(
        self, result_c: BaselineEvaluation
    ) -> None:
        classification = result_c.classification
        assert type(classification.validation_pr_auc) is float
        assert type(classification.decision_threshold) is float
        assert type(classification.validation_f1) is float
        assert type(classification.validation_recall) is float
        assert type(classification.test_accuracy) is float
        assert type(classification.test_precision) is float
        assert type(classification.test_recall) is float
        assert type(classification.test_f1) is float
        assert type(classification.test_roc_auc) is float
        assert type(classification.test_confusion_matrix) is tuple
        (tn, fp), (fn, tp) = classification.test_confusion_matrix
        assert (type(tn), type(fp), type(fn), type(tp)) == (int, int, int, int)
        regression = result_c.regression
        for value in (
            regression.validation_mae,
            regression.validation_rmse,
            regression.test_mae,
            regression.test_rmse,
            regression.test_r2,
        ):
            assert type(value) is float
        assert type(result_c.contract_version) is str
        assert type(result_c.classifier_name) is str
        assert type(result_c.regressor_name) is str
        assert type(result_c.feature_columns) is tuple
        assert all(type(column) is str for column in result_c.feature_columns)
        assert (
            type(result_c.train_rows),
            type(result_c.validation_rows),
            type(result_c.test_rows),
        ) == (
            int,
            int,
            int,
        )


class TestKnownVectors:
    def test_crafted_export_hand_known_metrics(self, result_c: BaselineEvaluation) -> None:
        classification = result_c.classification
        # Training has 34 rows with exactly two positives (19 and 4 days), so the
        # locked prior is 2/34 and the dummy emits constant probabilities 2/34.
        assert classification.decision_threshold == pytest.approx(2 / 34)
        assert classification.validation_pr_auc == pytest.approx(6 / 7)
        assert classification.validation_f1 == pytest.approx(12 / 13)
        assert classification.validation_recall == pytest.approx(1.0)
        # Test labels are [1, 1, 1, 0, 1, 1, 1]; every probability is >= the
        # frozen threshold, so hard labels are all positive.
        assert classification.test_accuracy == pytest.approx(6 / 7)
        assert classification.test_precision == pytest.approx(6 / 7)
        assert classification.test_recall == pytest.approx(1.0)
        assert classification.test_f1 == pytest.approx(12 / 13)
        assert classification.test_roc_auc == pytest.approx(0.5)
        assert classification.test_confusion_matrix == ((0, 1), (0, 6))

    def test_crafted_export_matches_direct_sklearn_reference(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        result = evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)
        _assert_matches_reference(result, dataset_c, CRAFTED_SPEC)

    def test_generated_export_matches_direct_sklearn_reference(self) -> None:
        dataset = _generated_export(GENERATED_SPEC)
        split = _canonical_split(dataset, GENERATED_SPEC)
        fit = _canonical_fit(split, GENERATED_SPEC)
        result = evaluate_baselines(dataset, split, fit, GENERATED_SPEC)
        _assert_matches_reference(result, dataset, GENERATED_SPEC)
        assert result.train_rows == 5 * 34
        assert result.validation_rows == 5 * 7
        assert result.test_rows == 5 * 7

    def test_threshold_equals_training_prior(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        train = split_c.train[["segment_id", "date"]].merge(
            dataset_c.targets, how="left", on=list(FEATURE_KEY_COLUMNS), validate="one_to_one"
        )
        prior = float(train["maintenance_within_30_days"].mean())
        result = evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)
        assert result.classification.decision_threshold == pytest.approx(prior)


class TestThresholdRanking:
    def test_f1_primary_ranking(self) -> None:
        y_true = np.array([1, 1, 0, 0, 1, 0], dtype=np.int64)
        probabilities = [0.95, 0.9, 0.85, 0.75, 0.7, 0.1]
        # Threshold 0.9 gives F1 = 0.8 (recall 2/3); threshold 0.7 gives F1 = 0.75
        # with recall 1.0. F1 is primary, so 0.9 must win.
        assert _select_threshold(y_true, probabilities) == pytest.approx(0.9)

    def test_recall_tie_break(self) -> None:
        y_true = np.array([1, 0, 0, 1], dtype=np.int64)
        probabilities = [0.9, 0.8, 0.7, 0.6]
        # 0.9 and 0.6 both reach F1 = 2/3; 0.6 has higher recall (1.0 vs 0.5).
        assert _select_threshold(y_true, probabilities) == pytest.approx(0.6)

    def test_higher_threshold_residual_tie_break(self) -> None:
        y_true = np.array([1, 0, 0, 0, 0], dtype=np.int64)
        probabilities = [0.9, 0.9, 0.9, 0.9, 0.9]
        # 0.0 and 0.9 produce identical hard labels (identical F1 and recall);
        # the higher threshold wins.
        assert _select_threshold(y_true, probabilities) == pytest.approx(0.9)

    def test_candidates_are_zero_one_and_distinct_probabilities(self) -> None:
        y_true = np.array([1, 0, 1, 0], dtype=np.int64)
        probabilities = [0.9, 0.8, 0.7, 0.6]
        expected = max(
            (0.0, 1.0, *probabilities),
            key=lambda t: (
                f1_score(
                    y_true,
                    (np.asarray(probabilities) >= t).astype(np.int64),
                    zero_division=0,
                ),
                recall_score(
                    y_true, (np.asarray(probabilities) >= t).astype(np.int64), zero_division=0
                ),
                t,
            ),
        )
        assert _select_threshold(y_true, probabilities) == pytest.approx(expected)


class TestEstimatorContracts:
    def test_classifier_is_prior_dummy(self, result_c: BaselineEvaluation) -> None:
        # Constant probabilities equal the training prior and the threshold is
        # the prior, which is exactly DummyClassifier(strategy="prior") behavior.
        assert result_c.classifier_name == "dummy_prior"
        assert result_c.classification.decision_threshold == pytest.approx(2 / 34)
        assert result_c.classification.validation_pr_auc == pytest.approx(6 / 7)

    def test_regressor_is_median_dummy(self, result_c: BaselineEvaluation) -> None:
        assert result_c.regressor_name == "dummy_median"
        expected = _reference_evaluation(_export_c(), CRAFTED_SPEC)
        assert result_c.regression.test_mae == pytest.approx(float(expected["test_mae"]))
        assert result_c.regression.test_rmse == pytest.approx(float(expected["test_rmse"]))
        assert result_c.regression.validation_mae == pytest.approx(float(expected["val_mae"]))
        assert result_c.regression.validation_rmse == pytest.approx(float(expected["val_rmse"]))

    def test_fit_and_test_stage_call_trace_enforces_temporal_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        events: list[tuple[str, int]] = []
        captured: dict[str, np.ndarray] = {}

        class ClassifierSpy:
            def __init__(self, *, strategy: str) -> None:
                assert strategy == "prior"
                self.delegate = DummyClassifier(strategy=strategy)

            def fit(self, x: np.ndarray, y: np.ndarray) -> ClassifierSpy:
                events.append(("classifier_fit", len(x)))
                captured["classifier_x"] = x.copy()
                captured["classifier_y"] = y.copy()
                self.delegate.fit(x, y)
                return self

            @property
            def classes_(self) -> np.ndarray:
                return cast(np.ndarray, self.delegate.classes_)

            @property
            def class_prior_(self) -> np.ndarray:
                return cast(np.ndarray, self.delegate.class_prior_)

            def predict_proba(self, x: np.ndarray) -> np.ndarray:
                events.append(("classifier_predict", len(x)))
                return cast(np.ndarray, self.delegate.predict_proba(x))

        class RegressorSpy:
            def __init__(self, *, strategy: str) -> None:
                assert strategy == "median"
                self.delegate = DummyRegressor(strategy=strategy)

            def fit(self, x: np.ndarray, y: np.ndarray) -> RegressorSpy:
                events.append(("regressor_fit", len(x)))
                captured["regressor_x"] = x.copy()
                captured["regressor_y"] = y.copy()
                self.delegate.fit(x, y)
                return self

            @property
            def constant_(self) -> np.ndarray:
                return cast(np.ndarray, self.delegate.constant_)

            def predict(self, x: np.ndarray) -> np.ndarray:
                events.append(("regressor_predict", len(x)))
                return cast(np.ndarray, self.delegate.predict(x))

        original_test_stage = baselines_module._evaluate_test_stage

        def traced_test_stage(**kwargs: Any) -> Any:
            events.append(("test_stage", len(kwargs["test_frame"])))
            return original_test_stage(**kwargs)

        monkeypatch.setattr(baselines_module, "DummyClassifier", ClassifierSpy)
        monkeypatch.setattr(baselines_module, "DummyRegressor", RegressorSpy)
        monkeypatch.setattr(baselines_module, "_evaluate_test_stage", traced_test_stage)

        evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)

        assert events == [
            ("classifier_fit", 34),
            ("regressor_fit", 34),
            ("classifier_predict", 7),
            ("regressor_predict", 7),
            ("test_stage", 7),
            ("classifier_predict", 7),
            ("regressor_predict", 7),
        ]
        transformed_train = transform(split_c.train, fit_c)
        expected_targets = transformed_train.keys.merge(
            dataset_c.targets,
            how="left",
            on=list(FEATURE_KEY_COLUMNS),
            sort=False,
            validate="one_to_one",
        )
        expected_x = transformed_train.features.to_numpy(dtype="float64")
        np.testing.assert_array_equal(captured["classifier_x"], expected_x)
        np.testing.assert_array_equal(captured["regressor_x"], expected_x)
        np.testing.assert_array_equal(
            captured["classifier_y"],
            expected_targets["maintenance_within_30_days"].to_numpy(dtype="int64"),
        )
        np.testing.assert_array_equal(
            captured["regressor_y"],
            expected_targets["days_until_maintenance"].to_numpy(dtype="int64"),
        )


class TestTemporalLeakage:
    def test_test_feature_mutation_changes_nothing(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        observations = dataset_c.observations.copy()
        mask = observations["date"].isin(pd.to_datetime(split_c.test_dates))
        observations.loc[mask, "rainfall_mm"] = observations.loc[mask, "rainfall_mm"] * 2.0
        mutated = replace(dataset_c, observations=observations)
        mutated_split = _canonical_split(mutated, CRAFTED_SPEC)
        assert fit_preprocessor(mutated_split, CRAFTED_SPEC) == fit_c
        baseline = evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)
        mutated_result = evaluate_baselines(mutated, mutated_split, fit_c, CRAFTED_SPEC)
        assert mutated_result == baseline

    def test_test_target_mutation_changes_only_test_metrics(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        event_dates = [value.date() for value in dataset_c.maintenance_events["maintenance_date"]]
        mutated = _crafted_export([value for value in event_dates if value != date(2025, 6, 15)])
        mutated_split = _canonical_split(mutated, CRAFTED_SPEC)
        assert_frame_equal(mutated_split.train, split_c.train)
        assert_frame_equal(mutated_split.validation, split_c.validation)
        assert not mutated_split.test.equals(split_c.test)
        assert mutated_split.train_dates == split_c.train_dates
        assert mutated_split.validation_dates == split_c.validation_dates
        assert mutated_split.test_dates == split_c.test_dates
        baseline = evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)
        mutated_result = evaluate_baselines(mutated, mutated_split, fit_c, CRAFTED_SPEC)
        test_dates = pd.to_datetime(split_c.test_dates)
        original_labels = dataset_c.targets.loc[
            dataset_c.targets["date"].isin(test_dates), "maintenance_within_30_days"
        ]
        mutated_labels = mutated.targets.loc[
            mutated.targets["date"].isin(test_dates), "maintenance_within_30_days"
        ]
        assert int((original_labels.to_numpy() != mutated_labels.to_numpy()).sum()) > 0
        assert (
            mutated_result.classification.validation_pr_auc
            == baseline.classification.validation_pr_auc
        )
        assert (
            mutated_result.classification.decision_threshold
            == baseline.classification.decision_threshold
        )
        assert mutated_result.classification.validation_f1 == baseline.classification.validation_f1
        assert (
            mutated_result.classification.validation_recall
            == baseline.classification.validation_recall
        )
        assert mutated_result.regression.validation_mae == baseline.regression.validation_mae
        assert mutated_result.regression.validation_rmse == baseline.regression.validation_rmse
        assert mutated_result.feature_columns == baseline.feature_columns
        assert (
            mutated_result.train_rows,
            mutated_result.validation_rows,
            mutated_result.test_rows,
        ) == (baseline.train_rows, baseline.validation_rows, baseline.test_rows)
        _assert_matches_reference(mutated_result, mutated, CRAFTED_SPEC)
        assert mutated_result.regression.test_mae != baseline.regression.test_mae

    def test_validation_feature_mutation_cannot_change_fitted_state(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        observations = dataset_c.observations.copy()
        mask = observations["date"].isin(pd.to_datetime(split_c.validation_dates))
        observations.loc[mask, "temperature"] = observations.loc[mask, "temperature"] + 1.0
        mutated = replace(dataset_c, observations=observations)
        mutated_split = _canonical_split(mutated, CRAFTED_SPEC)
        assert fit_preprocessor(mutated_split, CRAFTED_SPEC) == fit_c
        baseline = evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)
        mutated_result = evaluate_baselines(mutated, mutated_split, fit_c, CRAFTED_SPEC)
        assert mutated_result == baseline

    def test_validation_target_mutation_cannot_change_fitted_state(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        event_dates = [value.date() for value in dataset_c.maintenance_events["maintenance_date"]]
        mutated = _crafted_export([value for value in event_dates if value != date(2024, 12, 15)])
        mutated_split = _canonical_split(mutated, CRAFTED_SPEC)
        assert fit_preprocessor(mutated_split, CRAFTED_SPEC) == fit_c
        baseline = evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)
        mutated_result = evaluate_baselines(mutated, mutated_split, fit_c, CRAFTED_SPEC)
        validation_dates = pd.to_datetime(split_c.validation_dates)
        original_labels = dataset_c.targets.loc[
            dataset_c.targets["date"].isin(validation_dates), "maintenance_within_30_days"
        ]
        mutated_labels = mutated.targets.loc[
            mutated.targets["date"].isin(validation_dates), "maintenance_within_30_days"
        ]
        assert int((original_labels.to_numpy() != mutated_labels.to_numpy()).sum()) > 0
        assert (
            mutated_result.classification.decision_threshold
            == baseline.classification.decision_threshold
        )
        assert mutated_result.feature_columns == baseline.feature_columns
        assert (
            mutated_result.train_rows,
            mutated_result.validation_rows,
            mutated_result.test_rows,
        ) == (baseline.train_rows, baseline.validation_rows, baseline.test_rows)
        assert mutated_result.regression.validation_mae != baseline.regression.validation_mae
        _assert_matches_reference(mutated_result, mutated, CRAFTED_SPEC)

    def test_repeated_calls_identical_no_cross_call_state(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        first = evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)
        second = evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)
        assert second == first

    def test_shuffled_upstream_inputs_produce_equal_results(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        shuffled = replace(
            dataset_c,
            segments=dataset_c.segments.sample(frac=1.0, random_state=3).reset_index(drop=True),
            observations=dataset_c.observations.sample(frac=1.0, random_state=7).reset_index(
                drop=True
            ),
            targets=dataset_c.targets.sample(frac=1.0, random_state=11).reset_index(drop=True),
            maintenance_events=dataset_c.maintenance_events.sample(
                frac=1.0, random_state=13
            ).reset_index(drop=True),
        )
        shuffled_split = _canonical_split(shuffled, CRAFTED_SPEC)
        baseline = evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)
        shuffled_result = evaluate_baselines(shuffled, shuffled_split, fit_c, CRAFTED_SPEC)
        assert shuffled_result == baseline


class TestCallerImmutability:
    def test_caller_objects_unchanged(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        before_segments = dataset_c.segments.copy(deep=True)
        before_observations = dataset_c.observations.copy(deep=True)
        before_targets = dataset_c.targets.copy(deep=True)
        before_events = dataset_c.maintenance_events.copy(deep=True)
        before_train = split_c.train.copy(deep=True)
        before_validation = split_c.validation.copy(deep=True)
        before_test = split_c.test.copy(deep=True)
        before_fit = fit_c
        evaluate_baselines(dataset_c, split_c, fit_c, CRAFTED_SPEC)
        assert_frame_equal(dataset_c.segments, before_segments)
        assert_frame_equal(dataset_c.observations, before_observations)
        assert_frame_equal(dataset_c.targets, before_targets)
        assert_frame_equal(dataset_c.maintenance_events, before_events)
        assert_frame_equal(split_c.train, before_train)
        assert_frame_equal(split_c.validation, before_validation)
        assert_frame_equal(split_c.test, before_test)
        assert fit_c == before_fit

    def test_failed_call_does_not_mutate(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        before_segments = dataset_c.segments.copy(deep=True)
        before_observations = dataset_c.observations.copy(deep=True)
        forged = replace(split_c, validation=split_c.test)
        with pytest.raises(BaselineEvaluationError):
            evaluate_baselines(dataset_c, forged, fit_c, CRAFTED_SPEC)
        assert_frame_equal(dataset_c.segments, before_segments)
        assert_frame_equal(dataset_c.observations, before_observations)


class TestInputValidation:
    def test_wrong_top_level_types_raise_typeerror(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        with pytest.raises(TypeError, match="dataset"):
            evaluate_baselines(object(), split_c, fit_c, CRAFTED_SPEC)
        with pytest.raises(TypeError, match="split"):
            evaluate_baselines(dataset_c, object(), fit_c, CRAFTED_SPEC)
        with pytest.raises(TypeError, match="fit"):
            evaluate_baselines(dataset_c, split_c, object(), CRAFTED_SPEC)
        with pytest.raises(TypeError, match="spec"):
            evaluate_baselines(dataset_c, split_c, fit_c, object())

    def test_lookalike_objects_raise_typeerror(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        export_lookalike = SimpleNamespace(
            segments=dataset_c.segments,
            observations=dataset_c.observations,
            targets=dataset_c.targets,
            maintenance_events=dataset_c.maintenance_events,
        )
        with pytest.raises(TypeError, match="dataset"):
            evaluate_baselines(export_lookalike, split_c, fit_c, CRAFTED_SPEC)
        split_lookalike = SimpleNamespace(
            train=split_c.train,
            validation=split_c.validation,
            test=split_c.test,
            train_dates=split_c.train_dates,
            validation_dates=split_c.validation_dates,
            test_dates=split_c.test_dates,
        )
        with pytest.raises(TypeError, match="split"):
            evaluate_baselines(dataset_c, split_lookalike, fit_c, CRAFTED_SPEC)
        fit_lookalike = SimpleNamespace(
            scaled_columns=fit_c.scaled_columns,
            means=fit_c.means,
            stds=fit_c.stds,
            province_categories=fit_c.province_categories,
            road_type_categories=fit_c.road_type_categories,
        )
        with pytest.raises(TypeError, match="fit"):
            evaluate_baselines(dataset_c, split_c, fit_lookalike, CRAFTED_SPEC)
        spec_lookalike = SimpleNamespace(
            dataset_segments=1, dataset_months_per_segment=48, dataset_observations=48
        )
        with pytest.raises(TypeError, match="spec"):
            evaluate_baselines(dataset_c, split_c, fit_c, spec_lookalike)

    def test_subclass_instances_raise_typeerror(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        class ExportSubclass(RepositoryExport):
            pass

        subclass = ExportSubclass(
            segments=dataset_c.segments,
            observations=dataset_c.observations,
            targets=dataset_c.targets,
            maintenance_events=dataset_c.maintenance_events,
        )
        with pytest.raises(TypeError, match="dataset"):
            evaluate_baselines(subclass, split_c, fit_c, CRAFTED_SPEC)


class TestForgedInputs:
    def test_forged_repository_frame_type_rejected_contextually(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(dataset_c, targets=cast(Any, object()))
        with pytest.raises(BaselineEvaluationError, match="repository export") as excinfo:
            evaluate_baselines(forged, split_c, fit_c, CRAFTED_SPEC)
        assert excinfo.value.__cause__ is None

    def test_forged_split_frame_type_rejected_contextually(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(split_c, train=cast(Any, object()))
        with pytest.raises(BaselineEvaluationError, match="split") as excinfo:
            evaluate_baselines(dataset_c, forged, fit_c, CRAFTED_SPEC)
        assert excinfo.value.__cause__ is None

    def test_forged_fit_field_type_rejected_contextually(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(fit_c, means=cast(Any, np.asarray(fit_c.means)))
        with pytest.raises(BaselineEvaluationError, match="fit") as excinfo:
            evaluate_baselines(dataset_c, split_c, forged, CRAFTED_SPEC)
        assert excinfo.value.__cause__ is None

    def test_forged_split_partition_membership_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(split_c, validation=split_c.test)
        with pytest.raises(BaselineEvaluationError, match="split"):
            evaluate_baselines(dataset_c, forged, fit_c, CRAFTED_SPEC)

    def test_forged_split_dates_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(split_c, train_dates=split_c.train_dates[1:])
        with pytest.raises(BaselineEvaluationError, match="split"):
            evaluate_baselines(dataset_c, forged, fit_c, CRAFTED_SPEC)

    def test_forged_fit_state_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(fit_c, means=tuple(value + 1.0 for value in fit_c.means))
        with pytest.raises(BaselineEvaluationError, match="fit"):
            evaluate_baselines(dataset_c, split_c, forged, CRAFTED_SPEC)

    def test_mismatched_spec_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        with pytest.raises(BaselineEvaluationError):
            evaluate_baselines(dataset_c, split_c, fit_c, GENERATED_SPEC)


class TestTargetAlignment:
    def test_missing_target_row_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(dataset_c, targets=dataset_c.targets.iloc[:-1])
        with pytest.raises(BaselineEvaluationError):
            evaluate_baselines(forged, split_c, fit_c, CRAFTED_SPEC)

    def test_duplicate_target_key_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        targets = pd.concat([dataset_c.targets, dataset_c.targets.iloc[[0]]], ignore_index=True)
        forged = replace(dataset_c, targets=targets)
        with pytest.raises(BaselineEvaluationError):
            evaluate_baselines(forged, split_c, fit_c, CRAFTED_SPEC)

    def test_forged_target_value_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        targets = dataset_c.targets.copy()
        targets.loc[0, "days_until_maintenance"] = 999
        forged = replace(dataset_c, targets=targets)
        with pytest.raises(BaselineEvaluationError):
            evaluate_baselines(forged, split_c, fit_c, CRAFTED_SPEC)


class TestSanitizedErrors:
    def test_expected_failures_become_baseline_evaluation_error(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        targets = dataset_c.targets.copy()
        targets.loc[0, "days_until_maintenance"] = 999
        forged = replace(dataset_c, targets=targets)
        with pytest.raises(BaselineEvaluationError) as excinfo:
            evaluate_baselines(forged, split_c, fit_c, CRAFTED_SPEC)
        assert type(excinfo.value) is BaselineEvaluationError

    def test_error_message_is_fixed_and_contains_no_raw_values(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        targets = dataset_c.targets.copy()
        targets.loc[0, "days_until_maintenance"] = 999
        forged = replace(dataset_c, targets=targets)
        with pytest.raises(BaselineEvaluationError) as excinfo:
            evaluate_baselines(forged, split_c, fit_c, CRAFTED_SPEC)
        message = str(excinfo.value)
        assert message.startswith("evaluate_baselines failed: ")
        assert "999" not in message
        assert "target_event_inconsistency" not in message
        assert "days_until_maintenance" not in message
        assert "QL01" not in message
        assert "C:" not in message and "\\" not in message


class TestForbiddenFields:
    def test_feature_columns_match_transformed_features(
        self,
        fit_c: PreprocessorFit,
        result_c: BaselineEvaluation,
    ) -> None:
        assert result_c.feature_columns == fit_c.transformed_feature_columns

    def test_forbidden_names_never_model_features(self, result_c: BaselineEvaluation) -> None:
        assert not FORBIDDEN_FEATURE_NAMES.intersection(result_c.feature_columns)
        assert "segment_id" not in result_c.feature_columns
        assert "date" not in result_c.feature_columns
        for column in TARGET_COLUMNS[2:]:
            assert column not in result_c.feature_columns


class TestEstimatorOutputValidation:
    def _stub_classifier(self: Any, probabilities: Any, classes: Any = (0, 1)) -> Any:
        return SimpleNamespace(
            predict_proba=lambda _x: np.asarray(probabilities, dtype="float64"),
            classes_=np.asarray(classes),
        )

    def test_non_finite_probabilities_rejected(self) -> None:
        stub = self._stub_classifier([[0.5, np.nan], [0.5, np.inf]])
        with pytest.raises(BaselineEvaluationError, match="probabilit"):
            _positive_probas(stub, np.zeros((2, 3)))

    def test_out_of_range_probabilities_rejected(self) -> None:
        stub = self._stub_classifier([[0.5, 1.5]])
        with pytest.raises(BaselineEvaluationError, match="range"):
            _positive_probas(stub, np.zeros((1, 3)))
        stub = self._stub_classifier([[0.5, -0.1]])
        with pytest.raises(BaselineEvaluationError, match="range"):
            _positive_probas(stub, np.zeros((1, 3)))

    def test_malformed_probability_shape_rejected(self) -> None:
        stub = self._stub_classifier([[0.5]])
        with pytest.raises(BaselineEvaluationError, match="malformed"):
            _positive_probas(stub, np.zeros((1, 3)))
        stub = self._stub_classifier([[0.5, 0.4, 0.3]])
        with pytest.raises(BaselineEvaluationError, match="malformed"):
            _positive_probas(stub, np.zeros((1, 3)))

    def test_unexpected_class_order_rejected(self) -> None:
        stub = self._stub_classifier([[0.5, 0.5]], classes=(1, 0))
        with pytest.raises(BaselineEvaluationError, match="class"):
            _positive_probas(stub, np.zeros((1, 3)))

    def test_non_finite_regression_predictions_rejected(self) -> None:
        stub = SimpleNamespace(predict=lambda _x: np.array([1.0, np.nan]))
        with pytest.raises(BaselineEvaluationError, match="non-finite"):
            _regression_predictions(stub, np.zeros((2, 3)))

    def test_malformed_regression_predictions_rejected(self) -> None:
        stub = SimpleNamespace(predict=lambda _x: np.array([[1.0, 2.0]]))
        with pytest.raises(BaselineEvaluationError, match="malformed"):
            _regression_predictions(stub, np.zeros((1, 3)))
        stub = SimpleNamespace(predict=lambda _x: np.array([1.0]))
        with pytest.raises(BaselineEvaluationError, match="malformed"):
            _regression_predictions(stub, np.zeros((2, 3)))

    @pytest.mark.parametrize(
        ("fit_function", "estimator_name", "expected_context"),
        [
            (_fit_classifier, "DummyClassifier", "classifier"),
            (_fit_regressor, "DummyRegressor", "regressor"),
        ],
    )
    def test_estimator_fit_arithmetic_failure_is_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fit_function: Any,
        estimator_name: str,
        expected_context: str,
    ) -> None:
        marker = "sensitive-fit-marker"

        def fail_fit(_x: np.ndarray, _y: np.ndarray) -> None:
            raise ArithmeticError(marker)

        stub = SimpleNamespace(fit=fail_fit)
        monkeypatch.setattr(baselines_module, estimator_name, lambda **_kwargs: stub)
        with pytest.raises(BaselineEvaluationError, match=expected_context) as excinfo:
            fit_function(np.zeros((2, 3)), np.asarray([0, 1]))
        assert marker not in str(excinfo.value)
        assert excinfo.value.__cause__ is None

    @pytest.mark.parametrize(
        ("helper", "method_name", "expected_context"),
        [
            (_positive_probas, "predict_proba", "probabilit"),
            (_regression_predictions, "predict", "prediction"),
        ],
    )
    def test_estimator_prediction_arithmetic_failure_is_sanitized(
        self,
        helper: Any,
        method_name: str,
        expected_context: str,
    ) -> None:
        marker = "sensitive-predict-marker"

        def fail_prediction(_x: np.ndarray) -> np.ndarray:
            raise ArithmeticError(marker)

        attributes: dict[str, Any] = {method_name: fail_prediction}
        if method_name == "predict_proba":
            attributes["classes_"] = np.asarray([0, 1])
        stub = SimpleNamespace(**attributes)
        with pytest.raises(BaselineEvaluationError, match=expected_context) as excinfo:
            helper(stub, np.zeros((1, 3)))
        assert marker not in str(excinfo.value)
        assert excinfo.value.__cause__ is None

    def test_non_numeric_probability_scalar_is_sanitized(self) -> None:
        marker = "sensitive-probability-marker"
        stub = SimpleNamespace(
            predict_proba=lambda _x: np.asarray([[0.5, marker]], dtype=object),
            classes_=np.asarray([0, 1]),
        )
        with pytest.raises(BaselineEvaluationError, match="probabilit") as excinfo:
            _positive_probas(stub, np.zeros((1, 3)))
        assert marker not in str(excinfo.value)
        assert excinfo.value.__cause__ is None

    def test_non_numeric_regression_scalar_is_sanitized(self) -> None:
        marker = "sensitive-regression-marker"
        stub = SimpleNamespace(predict=lambda _x: np.asarray([marker], dtype=object))
        with pytest.raises(BaselineEvaluationError, match="prediction") as excinfo:
            _regression_predictions(stub, np.zeros((1, 3)))
        assert marker not in str(excinfo.value)
        assert excinfo.value.__cause__ is None

    def test_non_finite_classifier_fitted_state_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = SimpleNamespace(
            fit=lambda _x, _y: None,
            classes_=np.asarray([0, 1]),
            class_prior_=np.asarray([np.nan, np.nan]),
        )
        monkeypatch.setattr(baselines_module, "DummyClassifier", lambda **_kwargs: stub)
        with pytest.raises(BaselineEvaluationError, match="classifier"):
            _fit_classifier(np.zeros((2, 3)), np.asarray([0, 1]))

    def test_non_finite_regressor_fitted_state_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = SimpleNamespace(
            fit=lambda _x, _y: None,
            constant_=np.asarray([[np.nan]]),
        )
        monkeypatch.setattr(baselines_module, "DummyRegressor", lambda **_kwargs: stub)
        with pytest.raises(BaselineEvaluationError, match="regressor"):
            _fit_regressor(np.zeros((2, 3)), np.asarray([0, 1]))

    def test_select_threshold_rejects_non_finite_probabilities(self) -> None:
        y_true = np.array([0, 1], dtype=np.int64)
        with pytest.raises(BaselineEvaluationError, match="finite"):
            _select_threshold(y_true, [0.5, np.nan])
        with pytest.raises(BaselineEvaluationError, match="finite"):
            _select_threshold(y_true, [0.5, np.inf])


class TestDegeneratePartitions:
    def test_single_class_train_rejected(self) -> None:
        dataset = _export_a()
        split = _canonical_split(dataset, CRAFTED_SPEC)
        fit = _canonical_fit(split, CRAFTED_SPEC)
        with pytest.raises(BaselineEvaluationError, match="training"):
            evaluate_baselines(dataset, split, fit, CRAFTED_SPEC)

    def test_single_class_validation_rejected(self) -> None:
        dataset = _export_b()
        split = _canonical_split(dataset, CRAFTED_SPEC)
        fit = _canonical_fit(split, CRAFTED_SPEC)
        with pytest.raises(BaselineEvaluationError, match="validation"):
            evaluate_baselines(dataset, split, fit, CRAFTED_SPEC)

    def test_single_class_test_rejected(self) -> None:
        dataset = _export_e()
        split = _canonical_split(dataset, CRAFTED_SPEC)
        fit = _canonical_fit(split, CRAFTED_SPEC)
        with pytest.raises(BaselineEvaluationError, match="test"):
            evaluate_baselines(dataset, split, fit, CRAFTED_SPEC)

    def test_constant_test_regression_target_rejected(self) -> None:
        dataset = _export_d()
        split = _canonical_split(dataset, CRAFTED_SPEC)
        fit = _canonical_fit(split, CRAFTED_SPEC)
        with pytest.raises(BaselineEvaluationError, match="distinct"):
            evaluate_baselines(dataset, split, fit, CRAFTED_SPEC)


class TestV1Profile:
    def test_v1_profile_complete_evaluation(self) -> None:
        dataset = _generated_export(V1_SPEC)
        split = _canonical_split(dataset, V1_SPEC)
        fit = _canonical_fit(split, V1_SPEC)
        result = evaluate_baselines(dataset, split, fit, V1_SPEC)
        assert result.contract_version == BASELINE_CONTRACT_VERSION
        assert result.classifier_name == BASELINE_CLASSIFIER_NAME
        assert result.regressor_name == BASELINE_REGRESSOR_NAME
        assert result.feature_columns == fit.transformed_feature_columns
        assert (result.train_rows, result.validation_rows, result.test_rows) == (
            10_200,
            2_100,
            2_100,
        )
        classification = result.classification
        assert 0.0 <= classification.validation_pr_auc <= 1.0
        assert 0.0 <= classification.decision_threshold <= 1.0
        for value in (
            classification.validation_f1,
            classification.validation_recall,
            classification.test_accuracy,
            classification.test_precision,
            classification.test_recall,
            classification.test_f1,
            classification.test_roc_auc,
        ):
            assert 0.0 <= value <= 1.0
        (tn, fp), (fn, tp) = classification.test_confusion_matrix
        assert min(tn, fp, fn, tp) >= 0
        assert tn + fp + fn + tp == 2_100
        regression = result.regression
        assert regression.validation_mae >= 0.0
        assert regression.validation_rmse >= 0.0
        assert regression.test_mae >= 0.0
        assert regression.test_rmse >= 0.0
        assert math.isfinite(regression.test_r2)
        train = split.train[["segment_id", "date"]].merge(
            dataset.targets, how="left", on=list(FEATURE_KEY_COLUMNS), validate="one_to_one"
        )
        prior = float(train["maintenance_within_30_days"].mean())
        assert classification.decision_threshold == pytest.approx(prior)
        repeated = evaluate_baselines(dataset, split, fit, V1_SPEC)
        assert repeated == result
