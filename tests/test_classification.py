"""Phase 11 advanced-classification contract tests.

The contract under test is docs/contracts.md section 18. Success-path
fixtures are deterministic crafted exports (one segment, 48 months) whose
events are fully controlled; one generated multi-segment export and a
complete V1 evaluation exercise the real pipeline. Private helpers
``_select_threshold`` and ``_select_candidate_index`` are exercised directly
because real candidates almost never produce exact metric ties, which makes
the locked tie-break rules unreachable through the public workflow alone.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pydantic import SecretStr
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import roadguard
from roadguard import (
    DatasetSpec,
    RepositoryExport,
    RoadGuardConfig,
    clean_raw_dataset,
    derive_observation_targets,
    generate_accident_timeline,
    generate_maintenance_events,
    generate_observations,
    generate_segments,
)
from roadguard.classification import (
    ADVANCED_CLASSIFIER_CONTRACT_VERSION,
    ADVANCED_CLASSIFIER_RNG_NAMESPACE,
    CANDIDATE_CLASSIFIER_NAMES,
    AdvancedClassificationError,
    AdvancedClassificationEvaluation,
    CandidateValidationMetrics,
    _select_candidate_index,
    _select_threshold,
    evaluate_advanced_classifier,
)
from roadguard.classification import (
    TestClassificationMetrics as ClassificationMetricsSchema,
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

PHASE11_PUBLIC_NAMES = (
    "evaluate_advanced_classifier",
    "AdvancedClassificationError",
    "CandidateValidationMetrics",
    "TestClassificationMetrics",
    "AdvancedClassificationEvaluation",
    "ADVANCED_CLASSIFIER_CONTRACT_VERSION",
    "ADVANCED_CLASSIFIER_RNG_NAMESPACE",
    "CANDIDATE_CLASSIFIER_NAMES",
)

CANDIDATE_FIELDS = (
    "classifier_name",
    "validation_pr_auc",
    "decision_threshold",
    "validation_f1",
    "validation_recall",
)
TEST_FIELDS = ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix")
EVALUATION_FIELDS = (
    "contract_version",
    "selected_classifier_name",
    "feature_columns",
    "train_rows",
    "validation_rows",
    "test_rows",
    "candidates",
    "test",
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
    """Monthly day-15 events from month 36: single-class train."""
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
    both classes in every partition."""
    return _crafted_export(
        [date(2022, 3, 20), date(2022, 10, 5)]
        + [_month_15(month_index) for month_index in range(36, 45)]
        + [_month_15(month_index) for month_index in range(46, 61)]
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


def _derived_seed(config_seed: int, candidate_index: int) -> int:
    return int(
        np.random.SeedSequence(
            [config_seed, ADVANCED_CLASSIFIER_RNG_NAMESPACE, candidate_index]
        ).generate_state(1, dtype=np.uint32)[0]
    )


def _reference_evaluation(
    dataset: RepositoryExport, spec: DatasetSpec, config_seed: int
) -> dict[str, Any]:
    """Independent evaluation built only from public Phase 1-9 APIs and direct
    scikit-learn calls, used as the reference for known-vector comparisons."""
    frame = build_feature_frame(dataset, spec)
    canonical = split_chronologically(frame, spec)
    refit = fit_preprocessor(canonical, spec)
    transformed = {
        name: transform(getattr(canonical, name), refit) for name in ("train", "validation", "test")
    }
    y: dict[str, np.ndarray] = {}
    x: dict[str, np.ndarray] = {}
    for name in ("train", "validation", "test"):
        joined = transformed[name].keys.merge(
            dataset.targets[["segment_id", "date", "maintenance_within_30_days"]],
            how="left",
            on=list(FEATURE_KEY_COLUMNS),
            sort=False,
            validate="one_to_one",
        )
        y[name] = joined["maintenance_within_30_days"].to_numpy(dtype="int64")
        x[name] = transformed[name].features.to_numpy(dtype="float64")
    constructors = (
        lambda seed: LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            tol=1e-8,
            fit_intercept=True,
            class_weight=None,
            random_state=seed,
        ),
        lambda seed: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        ),
    )
    candidates: list[tuple[str, float, float, float, float]] = []
    fitted: list[Any] = []
    for index, name in enumerate(CANDIDATE_CLASSIFIER_NAMES):
        classifier = constructors[index](_derived_seed(config_seed, index))
        classifier.fit(x["train"], y["train"])
        probabilities = [float(v) for v in classifier.predict_proba(x["validation"])[:, 1]]
        thresholds = sorted({0.0, 1.0, *probabilities})
        threshold = max(
            thresholds,
            key=lambda t: (
                f1_score(
                    y["validation"],
                    (np.asarray(probabilities) >= t).astype(np.int64),
                    zero_division=0,
                ),
                recall_score(
                    y["validation"],
                    (np.asarray(probabilities) >= t).astype(np.int64),
                    zero_division=0,
                ),
                t,
            ),
        )
        hard = (np.asarray(probabilities) >= threshold).astype(np.int64)
        candidates.append(
            (
                name,
                float(average_precision_score(y["validation"], probabilities)),
                float(threshold),
                float(f1_score(y["validation"], hard, zero_division=0)),
                float(recall_score(y["validation"], hard, zero_division=0)),
            )
        )
        fitted.append(classifier)
    selected_index = 0
    for index in (1,):
        if (
            candidates[index][1],
            candidates[index][3],
            candidates[index][4],
        ) > (
            candidates[selected_index][1],
            candidates[selected_index][3],
            candidates[selected_index][4],
        ):
            selected_index = index
    selected_name, _, selected_threshold, _, _ = candidates[selected_index]
    test_probabilities = [float(v) for v in fitted[selected_index].predict_proba(x["test"])[:, 1]]
    test_hard = (np.asarray(test_probabilities) >= selected_threshold).astype(np.int64)
    matrix = confusion_matrix(y["test"], test_hard, labels=(0, 1))
    return {
        "candidates": candidates,
        "selected_index": selected_index,
        "selected_name": selected_name,
        "threshold": selected_threshold,
        "test_accuracy": float(accuracy_score(y["test"], test_hard)),
        "test_precision": float(precision_score(y["test"], test_hard, zero_division=0)),
        "test_recall": float(recall_score(y["test"], test_hard, zero_division=0)),
        "test_f1": float(f1_score(y["test"], test_hard, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y["test"], test_probabilities)),
        "test_cm": ((int(matrix[0, 0]), int(matrix[0, 1])), (int(matrix[1, 0]), int(matrix[1, 1]))),
        "feature_columns": refit.transformed_feature_columns,
        "train_rows": len(canonical.train),
        "validation_rows": len(canonical.validation),
        "test_rows": len(canonical.test),
    }


def _assert_matches_reference(
    result: AdvancedClassificationEvaluation,
    dataset: RepositoryExport,
    spec: DatasetSpec,
    config_seed: int,
) -> None:
    expected = _reference_evaluation(dataset, spec, config_seed)
    assert len(result.candidates) == 2
    for index, record in enumerate(result.candidates):
        name, pr_auc, threshold, f1, recall = expected["candidates"][index]
        assert record.classifier_name == name
        assert record.validation_pr_auc == pytest.approx(pr_auc)
        assert record.decision_threshold == pytest.approx(threshold)
        assert record.validation_f1 == pytest.approx(f1)
        assert record.validation_recall == pytest.approx(recall)
    assert result.selected_classifier_name == expected["selected_name"]
    assert result.test.accuracy == pytest.approx(float(expected["test_accuracy"]))
    assert result.test.precision == pytest.approx(float(expected["test_precision"]))
    assert result.test.recall == pytest.approx(float(expected["test_recall"]))
    assert result.test.f1 == pytest.approx(float(expected["test_f1"]))
    assert result.test.roc_auc == pytest.approx(float(expected["test_roc_auc"]))
    assert result.test.confusion_matrix == expected["test_cm"]
    assert result.feature_columns == expected["feature_columns"]
    assert result.train_rows == expected["train_rows"]
    assert result.validation_rows == expected["validation_rows"]
    assert result.test_rows == expected["test_rows"]
    assert result.contract_version == ADVANCED_CLASSIFIER_CONTRACT_VERSION


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
) -> AdvancedClassificationEvaluation:
    return evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())


class TestPublicSurface:
    def test_module_all_exact(self) -> None:
        import roadguard.classification as classification

        assert tuple(classification.__all__) == PHASE11_PUBLIC_NAMES

    def test_package_root_exports_phase11_symbols(self) -> None:
        for name in PHASE11_PUBLIC_NAMES:
            assert hasattr(roadguard, name), f"missing public attribute: {name}"

    def test_constants_exact(self) -> None:
        assert ADVANCED_CLASSIFIER_CONTRACT_VERSION == "roadguard.phase11.v1"
        assert ADVANCED_CLASSIFIER_RNG_NAMESPACE == 0x5247311
        assert CANDIDATE_CLASSIFIER_NAMES == ("logistic_l2", "hist_gradient_boosting")

    def test_evaluate_advanced_classifier_signature_exact(self) -> None:
        signature = inspect.signature(evaluate_advanced_classifier)
        assert tuple(signature.parameters) == ("dataset", "split", "fit", "spec", "config")
        assert "seed" not in signature.parameters

    def test_no_public_test_evaluation_function(self) -> None:
        import roadguard.classification as classification

        for name in ("evaluate_test", "evaluate_test_partition", "predict"):
            assert not hasattr(classification, name)


class TestFrozenSchema:
    def test_candidate_validation_metrics_fields_and_order(self) -> None:
        assert (
            tuple(field.name for field in dataclasses.fields(CandidateValidationMetrics))
            == CANDIDATE_FIELDS
        )

    def test_test_classification_metrics_fields_and_order(self) -> None:
        assert (
            tuple(field.name for field in dataclasses.fields(ClassificationMetricsSchema))
            == TEST_FIELDS
        )

    def test_evaluation_fields_and_order(self) -> None:
        assert (
            tuple(field.name for field in dataclasses.fields(AdvancedClassificationEvaluation))
            == EVALUATION_FIELDS
        )

    def test_result_types_are_frozen(self) -> None:
        for cls in (
            CandidateValidationMetrics,
            ClassificationMetricsSchema,
            AdvancedClassificationEvaluation,
        ):
            assert dataclasses.is_dataclass(cls)
            assert cls.__dataclass_params__.frozen

    def test_result_contains_only_builtin_scalars_and_tuples(
        self, result_c: AdvancedClassificationEvaluation
    ) -> None:
        assert type(result_c.contract_version) is str
        assert type(result_c.selected_classifier_name) is str
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
        assert type(result_c.candidates) is tuple
        assert len(result_c.candidates) == 2
        for record in result_c.candidates:
            assert type(record.classifier_name) is str
            assert type(record.validation_pr_auc) is float
            assert type(record.decision_threshold) is float
            assert type(record.validation_f1) is float
            assert type(record.validation_recall) is float
        test = result_c.test
        for value in (test.accuracy, test.precision, test.recall, test.f1, test.roc_auc):
            assert type(value) is float
        assert type(test.confusion_matrix) is tuple
        (tn, fp), (fn, tp) = test.confusion_matrix
        assert (type(tn), type(fp), type(fn), type(tp)) == (int, int, int, int)


class TestKnownVectors:
    def test_crafted_export_matches_direct_sklearn_reference(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        result = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        _assert_matches_reference(result, dataset_c, CRAFTED_SPEC, 42)
        assert result.candidates[0].classifier_name == CANDIDATE_CLASSIFIER_NAMES[0]
        assert result.candidates[1].classifier_name == CANDIDATE_CLASSIFIER_NAMES[1]

    def test_generated_export_matches_direct_sklearn_reference(self) -> None:
        dataset = _generated_export(GENERATED_SPEC)
        split = _canonical_split(dataset, GENERATED_SPEC)
        fit = _canonical_fit(split, GENERATED_SPEC)
        result = evaluate_advanced_classifier(
            dataset, split, fit, GENERATED_SPEC, RoadGuardConfig()
        )
        _assert_matches_reference(result, dataset, GENERATED_SPEC, 42)
        assert result.train_rows == 5 * 34
        assert result.validation_rows == 5 * 7
        assert result.test_rows == 5 * 7

    def test_selected_name_matches_exactly_one_candidate_record(
        self, result_c: AdvancedClassificationEvaluation
    ) -> None:
        matching = [
            record.classifier_name
            for record in result_c.candidates
            if record.classifier_name == result_c.selected_classifier_name
        ]
        assert len(matching) == 1


class TestThresholdRanking:
    def test_f1_primary_ranking(self) -> None:
        y_true = np.array([1, 1, 0, 0, 1, 0], dtype=np.int64)
        probabilities = [0.95, 0.9, 0.85, 0.75, 0.7, 0.1]
        assert _select_threshold(y_true, probabilities) == pytest.approx(0.9)

    def test_recall_tie_break(self) -> None:
        y_true = np.array([1, 0, 0, 1], dtype=np.int64)
        probabilities = [0.9, 0.8, 0.7, 0.6]
        assert _select_threshold(y_true, probabilities) == pytest.approx(0.6)

    def test_higher_threshold_residual_tie_break(self) -> None:
        y_true = np.array([1, 0, 0, 0, 0], dtype=np.int64)
        probabilities = [0.9, 0.9, 0.9, 0.9, 0.9]
        assert _select_threshold(y_true, probabilities) == pytest.approx(0.9)


class TestCandidateSelectionRanking:
    @staticmethod
    def _record(pr_auc: float, f1: float, recall: float) -> CandidateValidationMetrics:
        return CandidateValidationMetrics(
            classifier_name="candidate",
            validation_pr_auc=pr_auc,
            decision_threshold=0.5,
            validation_f1=f1,
            validation_recall=recall,
        )

    def test_pr_auc_primary(self) -> None:
        records = (self._record(0.9, 0.1, 0.1), self._record(0.8, 1.0, 1.0))
        assert _select_candidate_index(records) == 0

    def test_f1_tie_break(self) -> None:
        records = (self._record(0.7, 0.9, 0.2), self._record(0.7, 0.8, 1.0))
        assert _select_candidate_index(records) == 0

    def test_recall_tie_break(self) -> None:
        records = (self._record(0.7, 0.8, 0.9), self._record(0.7, 0.8, 0.3))
        assert _select_candidate_index(records) == 0

    def test_fixed_order_tie_break(self) -> None:
        records = (self._record(0.7, 0.8, 0.9), self._record(0.7, 0.8, 0.9))
        assert _select_candidate_index(records) == 0

    def test_earlier_position_wins_across_three_records(self) -> None:
        records = (
            self._record(0.5, 0.5, 0.5),
            self._record(0.5, 0.5, 0.5),
            self._record(0.5, 0.5, 0.5),
        )
        assert _select_candidate_index(records) == 0


class TestTemporalLeakage:
    def test_test_feature_mutation_cannot_change_candidates_or_selection(
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
        baseline = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        mutated_result = evaluate_advanced_classifier(
            mutated, mutated_split, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        assert mutated_result.candidates == baseline.candidates
        assert mutated_result.selected_classifier_name == baseline.selected_classifier_name
        assert mutated_result.feature_columns == baseline.feature_columns
        assert (
            mutated_result.train_rows,
            mutated_result.validation_rows,
            mutated_result.test_rows,
        ) == (baseline.train_rows, baseline.validation_rows, baseline.test_rows)
        _assert_matches_reference(mutated_result, mutated, CRAFTED_SPEC, 42)

    def test_test_target_mutation_changes_only_test_metrics(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        events = dataset_c.maintenance_events.copy()
        events.loc[
            events["maintenance_date"] == pd.Timestamp("2025-12-15"),
            "maintenance_date",
        ] = pd.Timestamp("2025-12-20")
        targets = derive_observation_targets(dataset_c.observations, events)
        mutated = replace(dataset_c, targets=targets, maintenance_events=events)
        mutated_split = _canonical_split(mutated, CRAFTED_SPEC)
        assert_frame_equal(mutated_split.train, split_c.train)
        assert_frame_equal(mutated_split.validation, split_c.validation)
        assert_frame_equal(mutated_split.test, split_c.test)
        baseline = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        mutated_result = evaluate_advanced_classifier(
            mutated, mutated_split, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        assert mutated_result.candidates == baseline.candidates
        assert mutated_result.selected_classifier_name == baseline.selected_classifier_name
        _assert_matches_reference(mutated_result, mutated, CRAFTED_SPEC, 42)

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
        assert_frame_equal(
            transform(mutated_split.train, fit_c).features,
            transform(split_c.train, fit_c).features,
        )
        result = evaluate_advanced_classifier(
            mutated, mutated_split, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        _assert_matches_reference(result, mutated, CRAFTED_SPEC, 42)

    def test_repeated_calls_identical_no_cross_call_state(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        first = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        second = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
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
        baseline = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        shuffled_result = evaluate_advanced_classifier(
            shuffled, shuffled_split, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        assert shuffled_result == baseline


class TestCallerImmutability:
    def test_caller_objects_unchanged(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        config = RoadGuardConfig(seed=42)
        before_segments = dataset_c.segments.copy(deep=True)
        before_observations = dataset_c.observations.copy(deep=True)
        before_targets = dataset_c.targets.copy(deep=True)
        before_events = dataset_c.maintenance_events.copy(deep=True)
        before_train = split_c.train.copy(deep=True)
        before_validation = split_c.validation.copy(deep=True)
        before_test = split_c.test.copy(deep=True)
        evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, config)
        assert_frame_equal(dataset_c.segments, before_segments)
        assert_frame_equal(dataset_c.observations, before_observations)
        assert_frame_equal(dataset_c.targets, before_targets)
        assert_frame_equal(dataset_c.maintenance_events, before_events)
        assert_frame_equal(split_c.train, before_train)
        assert_frame_equal(split_c.validation, before_validation)
        assert_frame_equal(split_c.test, before_test)
        assert fit_c == _canonical_fit(split_c, CRAFTED_SPEC)
        assert config == RoadGuardConfig(seed=42)

    def test_failed_call_does_not_mutate(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        before_segments = dataset_c.segments.copy(deep=True)
        before_observations = dataset_c.observations.copy(deep=True)
        forged = replace(split_c, validation=split_c.test)
        with pytest.raises(AdvancedClassificationError):
            evaluate_advanced_classifier(dataset_c, forged, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert_frame_equal(dataset_c.segments, before_segments)
        assert_frame_equal(dataset_c.observations, before_observations)


class TestInputValidation:
    def test_wrong_top_level_types_raise_typeerror(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        config = RoadGuardConfig()
        with pytest.raises(TypeError, match="dataset"):
            evaluate_advanced_classifier(object(), split_c, fit_c, CRAFTED_SPEC, config)
        with pytest.raises(TypeError, match="split"):
            evaluate_advanced_classifier(dataset_c, object(), fit_c, CRAFTED_SPEC, config)
        with pytest.raises(TypeError, match="fit"):
            evaluate_advanced_classifier(dataset_c, split_c, object(), CRAFTED_SPEC, config)
        with pytest.raises(TypeError, match="spec"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, object(), config)
        with pytest.raises(TypeError, match="config"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, object())

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
            evaluate_advanced_classifier(
                export_lookalike, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
            )
        split_lookalike = SimpleNamespace(
            train=split_c.train,
            validation=split_c.validation,
            test=split_c.test,
            train_dates=split_c.train_dates,
            validation_dates=split_c.validation_dates,
            test_dates=split_c.test_dates,
        )
        with pytest.raises(TypeError, match="split"):
            evaluate_advanced_classifier(
                dataset_c, split_lookalike, fit_c, CRAFTED_SPEC, RoadGuardConfig()
            )
        fit_lookalike = SimpleNamespace(
            scaled_columns=fit_c.scaled_columns,
            means=fit_c.means,
            stds=fit_c.stds,
            province_categories=fit_c.province_categories,
            road_type_categories=fit_c.road_type_categories,
        )
        with pytest.raises(TypeError, match="fit"):
            evaluate_advanced_classifier(
                dataset_c, split_c, fit_lookalike, CRAFTED_SPEC, RoadGuardConfig()
            )
        spec_lookalike = SimpleNamespace(
            dataset_segments=1, dataset_months_per_segment=48, dataset_observations=48
        )
        with pytest.raises(TypeError, match="spec"):
            evaluate_advanced_classifier(
                dataset_c, split_c, fit_c, spec_lookalike, RoadGuardConfig()
            )
        config_lookalike = SimpleNamespace(seed=42)
        with pytest.raises(TypeError, match="config"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, config_lookalike)

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
            evaluate_advanced_classifier(subclass, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())


class TestConfigValidation:
    def test_forged_invalid_seeds_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        class IntSubclass(int):
            pass

        forged_seeds: list[object] = [
            True,
            0,
            -3,
            "42",
            np.int64(42),
            IntSubclass(42),
            None,
        ]
        for seed in forged_seeds:
            forged = RoadGuardConfig.model_construct(seed=seed)
            with pytest.raises(AdvancedClassificationError, match="seed"):
                evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, forged)

    def test_missing_seed_field_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = RoadGuardConfig.model_construct()
        object.__setattr__(forged, "__dict__", {"env": "development"})
        with pytest.raises(AdvancedClassificationError, match="seed"):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, forged)

    def test_irrelevant_config_fields_cannot_change_result(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        baseline = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        altered = RoadGuardConfig(
            env="production",
            data_dir=Path("nope"),
            artifacts_dir=Path("also-nope"),
            database_url=SecretStr("ignored-database-url"),
        )
        assert (
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, altered)
            == baseline
        )

    def test_irrelevant_config_fields_are_never_read(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        seed_only = RoadGuardConfig.model_construct(seed=42)
        object.__setattr__(seed_only, "__dict__", {"seed": 42})
        result = evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, seed_only)
        assert result.contract_version == ADVANCED_CLASSIFIER_CONTRACT_VERSION

    def test_environment_variables_are_never_read(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        baseline = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig()
        )
        monkeypatch.setenv("ROADGUARD_SEED", "999")
        assert (
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
            == baseline
        )

    def test_load_config_is_never_called(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        def bomb(*args: object, **kwargs: object) -> Any:
            raise AssertionError("load_config must not be called")

        monkeypatch.setattr("roadguard.config.load_config", bomb)
        evaluate_advanced_classifier(dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_changed_seed_preserves_provenance_fields(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        first = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig(seed=42)
        )
        second = evaluate_advanced_classifier(
            dataset_c, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig(seed=7)
        )
        assert second.contract_version == first.contract_version
        assert second.feature_columns == first.feature_columns
        assert (second.train_rows, second.validation_rows, second.test_rows) == (
            first.train_rows,
            first.validation_rows,
            first.test_rows,
        )
        assert (
            tuple(record.classifier_name for record in second.candidates)
            == CANDIDATE_CLASSIFIER_NAMES
        )


class TestForgedInputs:
    @pytest.mark.parametrize(
        ("kind", "missing_field"),
        [
            ("export", "observations"),
            ("split", "train"),
            ("fit", "means"),
        ],
    )
    def test_missing_nested_field_is_sanitized(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
        kind: str,
        missing_field: str,
    ) -> None:
        dataset = replace(dataset_c)
        split = replace(split_c)
        fit = replace(fit_c)
        forged = {"export": dataset, "split": split, "fit": fit}[kind]
        object.__delattr__(forged, missing_field)

        with pytest.raises(AdvancedClassificationError) as excinfo:
            evaluate_advanced_classifier(dataset, split, fit, CRAFTED_SPEC, RoadGuardConfig())
        assert "AttributeError" not in str(excinfo.value)

    def test_missing_spec_field_is_sanitized(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged_spec = DatasetSpec.model_construct(
            dataset_segments=1,
            dataset_months_per_segment=48,
        )
        with pytest.raises(AdvancedClassificationError):
            evaluate_advanced_classifier(dataset_c, split_c, fit_c, forged_spec, RoadGuardConfig())

    def test_forged_split_partition_membership_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(split_c, validation=split_c.test)
        with pytest.raises(AdvancedClassificationError, match="split"):
            evaluate_advanced_classifier(dataset_c, forged, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_forged_split_dates_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(split_c, train_dates=split_c.train_dates[1:])
        with pytest.raises(AdvancedClassificationError, match="split"):
            evaluate_advanced_classifier(dataset_c, forged, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_forged_fit_state_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(fit_c, means=tuple(value + 1.0 for value in fit_c.means))
        with pytest.raises(AdvancedClassificationError, match="fit"):
            evaluate_advanced_classifier(
                dataset_c, split_c, forged, CRAFTED_SPEC, RoadGuardConfig()
            )

    def test_mismatched_spec_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        with pytest.raises(AdvancedClassificationError):
            evaluate_advanced_classifier(
                dataset_c, split_c, fit_c, GENERATED_SPEC, RoadGuardConfig()
            )


class TestTargetAlignment:
    def test_missing_target_row_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        forged = replace(dataset_c, targets=dataset_c.targets.iloc[:-1])
        with pytest.raises(AdvancedClassificationError):
            evaluate_advanced_classifier(forged, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_duplicate_target_key_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        targets = pd.concat([dataset_c.targets, dataset_c.targets.iloc[[0]]], ignore_index=True)
        forged = replace(dataset_c, targets=targets)
        with pytest.raises(AdvancedClassificationError):
            evaluate_advanced_classifier(forged, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())

    def test_forged_target_value_rejected(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        targets = dataset_c.targets.copy()
        targets.loc[0, "days_until_maintenance"] = 999
        forged = replace(dataset_c, targets=targets)
        with pytest.raises(AdvancedClassificationError):
            evaluate_advanced_classifier(forged, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())


class TestForbiddenFields:
    def test_feature_columns_match_transformed_features(
        self,
        fit_c: PreprocessorFit,
        result_c: AdvancedClassificationEvaluation,
    ) -> None:
        assert result_c.feature_columns == fit_c.transformed_feature_columns

    def test_forbidden_names_never_model_features(
        self, result_c: AdvancedClassificationEvaluation
    ) -> None:
        assert not FORBIDDEN_FEATURE_NAMES.intersection(result_c.feature_columns)
        assert "segment_id" not in result_c.feature_columns
        assert "date" not in result_c.feature_columns
        for column in TARGET_COLUMNS[2:]:
            assert column not in result_c.feature_columns


class TestSanitizedErrors:
    def test_expected_failures_become_advanced_classification_error(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        targets = dataset_c.targets.copy()
        targets.loc[0, "days_until_maintenance"] = 999
        forged = replace(dataset_c, targets=targets)
        with pytest.raises(AdvancedClassificationError) as excinfo:
            evaluate_advanced_classifier(forged, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        assert type(excinfo.value) is AdvancedClassificationError

    def test_error_message_is_fixed_and_contains_no_raw_values(
        self,
        dataset_c: RepositoryExport,
        split_c: ChronologicalSplit,
        fit_c: PreprocessorFit,
    ) -> None:
        targets = dataset_c.targets.copy()
        targets.loc[0, "days_until_maintenance"] = 999
        forged = replace(dataset_c, targets=targets)
        with pytest.raises(AdvancedClassificationError) as excinfo:
            evaluate_advanced_classifier(forged, split_c, fit_c, CRAFTED_SPEC, RoadGuardConfig())
        message = str(excinfo.value)
        assert message.startswith("evaluate_advanced_classifier failed: ")
        assert "999" not in message
        assert "target_event_inconsistency" not in message
        assert "days_until_maintenance" not in message
        assert "QL01" not in message
        assert "C:" not in message and "\\" not in message


class TestDegeneratePartitions:
    def test_single_class_train_rejected(self) -> None:
        dataset = _export_a()
        split = _canonical_split(dataset, CRAFTED_SPEC)
        fit = _canonical_fit(split, CRAFTED_SPEC)
        with pytest.raises(AdvancedClassificationError, match="training"):
            evaluate_advanced_classifier(dataset, split, fit, CRAFTED_SPEC, RoadGuardConfig())

    def test_single_class_validation_rejected(self) -> None:
        dataset = _export_b()
        split = _canonical_split(dataset, CRAFTED_SPEC)
        fit = _canonical_fit(split, CRAFTED_SPEC)
        with pytest.raises(AdvancedClassificationError, match="validation"):
            evaluate_advanced_classifier(dataset, split, fit, CRAFTED_SPEC, RoadGuardConfig())

    def test_single_class_test_rejected(self) -> None:
        dataset = _export_e()
        split = _canonical_split(dataset, CRAFTED_SPEC)
        fit = _canonical_fit(split, CRAFTED_SPEC)
        with pytest.raises(AdvancedClassificationError, match="test"):
            evaluate_advanced_classifier(dataset, split, fit, CRAFTED_SPEC, RoadGuardConfig())


class TestV1Profile:
    def test_v1_profile_complete_evaluation(self) -> None:
        dataset = _generated_export(V1_SPEC)
        split = _canonical_split(dataset, V1_SPEC)
        fit = _canonical_fit(split, V1_SPEC)
        result = evaluate_advanced_classifier(dataset, split, fit, V1_SPEC, RoadGuardConfig())
        assert result.contract_version == ADVANCED_CLASSIFIER_CONTRACT_VERSION
        assert result.feature_columns == fit.transformed_feature_columns
        assert (result.train_rows, result.validation_rows, result.test_rows) == (
            10_200,
            2_100,
            2_100,
        )
        assert result.selected_classifier_name in CANDIDATE_CLASSIFIER_NAMES
        assert (
            tuple(record.classifier_name for record in result.candidates)
            == CANDIDATE_CLASSIFIER_NAMES
        )
        for record in result.candidates:
            assert 0.0 <= record.validation_pr_auc <= 1.0
            assert 0.0 <= record.decision_threshold <= 1.0
            assert 0.0 <= record.validation_f1 <= 1.0
            assert 0.0 <= record.validation_recall <= 1.0
        test = result.test
        for value in (test.accuracy, test.precision, test.recall, test.f1, test.roc_auc):
            assert 0.0 <= value <= 1.0
            assert math.isfinite(value)
        (tn, fp), (fn, tp) = test.confusion_matrix
        assert min(tn, fp, fn, tp) >= 0
        assert tn + fp + fn + tp == 2_100
        repeated = evaluate_advanced_classifier(dataset, split, fit, V1_SPEC, RoadGuardConfig())
        assert repeated == result
