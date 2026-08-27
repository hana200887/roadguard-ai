"""Phase 11 advanced classification.

The module fits exactly two locked feature-dependent classifier candidates
(``LogisticRegression`` with L2 penalty and ``HistGradientBoostingClassifier``)
on the canonical 34-date training partition only, records their validation
evidence, selects exactly one candidate and its decision threshold by the
fixed validation ranking, and returns immutable selected-model test metrics.

``evaluate_advanced_classifier`` accepts a complete Phase 6
``RepositoryExport``, the canonical Phase 8 ``ChronologicalSplit``, its
matching train-only ``PreprocessorFit``, their ``DatasetSpec``, and an exact
``RoadGuardConfig``. It deep-copies the export, fresh-validates it through the
Phase 7 feature builder, rebuilds the canonical Phase 8 split, refits the
train-only preprocessor and rejects mismatched supplied split or fit. Each
candidate derives its own ``random_state`` from the supplied configuration
seed through the locked namespace; no global RNG, public seed argument, or
unseeded stochastic operation exists. Validation is used only for candidate
records and selection; the test partition is transformed exactly once and
passed to exactly one selected candidate inside one private test stage after
selection is frozen. No model, fitted state, prediction, coefficient, seed,
or config field is returned or persisted; the result is an immutable metrics
record frozen in docs/contracts.md section 18.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import numpy as np
import pandas as pd
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

from roadguard._db_types import RepositoryExport
from roadguard.config import RoadGuardConfig
from roadguard.contracts import DatasetSpec
from roadguard.features import FEATURE_KEY_COLUMNS, FeatureInputError, build_feature_frame
from roadguard.preprocessing import (
    ChronologicalSplit,
    PreprocessingError,
    PreprocessorFit,
    fit_preprocessor,
    split_chronologically,
    transform,
)
from roadguard.targets import TARGET_COLUMNS

ADVANCED_CLASSIFIER_CONTRACT_VERSION: Final[str] = "roadguard.phase11.v1"
ADVANCED_CLASSIFIER_RNG_NAMESPACE: Final[int] = 0x5247311
CANDIDATE_CLASSIFIER_NAMES: Final[tuple[str, str]] = (
    "logistic_l2",
    "hist_gradient_boosting",
)

_CLASSIFICATION_TARGET: Final[str] = TARGET_COLUMNS[3]
_TARGET_PROJECTION: Final[list[str]] = [*FEATURE_KEY_COLUMNS, _CLASSIFICATION_TARGET]

_ERROR_PREFIX: Final[str] = "evaluate_advanced_classifier failed: "


class AdvancedClassificationError(ValueError):
    """Raised when Phase 11 input, estimator, prediction, selection, or metric state is invalid."""


@dataclass(frozen=True)
class CandidateValidationMetrics:
    """Validation evidence for one locked classifier candidate."""

    classifier_name: str
    validation_pr_auc: float
    decision_threshold: float
    validation_f1: float
    validation_recall: float


@dataclass(frozen=True)
class TestClassificationMetrics:
    """Selected-candidate metrics on the frozen test partition."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class AdvancedClassificationEvaluation:
    """Immutable Phase 11 advanced-classification evaluation result."""

    contract_version: str
    selected_classifier_name: str
    feature_columns: tuple[str, ...]
    train_rows: int
    validation_rows: int
    test_rows: int
    candidates: tuple[CandidateValidationMetrics, CandidateValidationMetrics]
    test: TestClassificationMetrics


@dataclass(frozen=True)
class _TestStageResult:
    """Private selected-candidate test metrics produced exactly once per call."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]


def evaluate_advanced_classifier(
    dataset: RepositoryExport,
    split: ChronologicalSplit,
    fit: PreprocessorFit,
    spec: DatasetSpec,
    config: RoadGuardConfig,
) -> AdvancedClassificationEvaluation:
    """Fit the two locked candidates, select one on validation, and evaluate it.

    All five arguments must be exact instances of their declared types; wrong
    types and lookalikes raise ``TypeError`` before any field is read. Only
    ``config.seed`` is read and it must be an exact positive built-in ``int``.
    Expected lower-phase and scikit-learn failures are translated to
    :class:`AdvancedClassificationError` with fixed sanitized messages and
    suppressed chaining. The test partition is transformed and evaluated
    exactly once in a private stage after both candidate records, the selected
    name, and the selected threshold are frozen.
    """
    _require_exact_types(dataset, split, fit, spec, config)
    config_seed = _require_config_seed(config)
    _require_spec_state(spec)

    copied_dataset = _copy_repository_export(dataset)
    targets = copied_dataset.targets

    try:
        frame = build_feature_frame(copied_dataset, spec)
        canonical_split = split_chronologically(frame, spec)
    except (FeatureInputError, PreprocessingError):
        raise _error(
            "the repository export could not be freshly validated into a canonical "
            "feature frame and chronological split"
        ) from None

    _require_matching_split(split, canonical_split)

    try:
        canonical_fit = fit_preprocessor(canonical_split, spec)
    except PreprocessingError:
        raise _error("the train-only preprocessor could not be refitted") from None

    _require_matching_fit(fit, canonical_fit)

    try:
        train_transformed = transform(canonical_split.train, canonical_fit)
        validation_transformed = transform(canonical_split.validation, canonical_fit)
    except PreprocessingError:
        raise _error("a partition could not be transformed with the fitted preprocessor") from None

    train_targets = _join_classification_target(targets, train_transformed.keys, "training")
    validation_targets = _join_classification_target(
        targets, validation_transformed.keys, "validation"
    )

    y_train = train_targets[_CLASSIFICATION_TARGET].to_numpy(dtype="int64")
    y_validation = validation_targets[_CLASSIFICATION_TARGET].to_numpy(dtype="int64")
    _require_binary_classes(y_train, "training")
    _require_binary_classes(y_validation, "validation")

    x_train = train_transformed.features.to_numpy(dtype="float64")
    x_validation = validation_transformed.features.to_numpy(dtype="float64")

    candidate_records: list[CandidateValidationMetrics] = []
    fitted_candidates: list[Any] = []
    for index, name in enumerate(CANDIDATE_CLASSIFIER_NAMES):
        derived_seed = _derive_candidate_seed(config_seed, index)
        classifier = _build_candidate(name, derived_seed)
        _fit_candidate(classifier, x_train, y_train)
        probabilities = _positive_probas(classifier, x_validation)
        threshold = _select_threshold(y_validation, probabilities)
        hard = (np.asarray(probabilities) >= threshold).astype(np.int64)
        candidate_records.append(
            CandidateValidationMetrics(
                classifier_name=name,
                validation_pr_auc=_metric_probability(
                    average_precision_score,
                    y_validation,
                    probabilities,
                    "validation PR-AUC",
                ),
                decision_threshold=threshold,
                validation_f1=_metric_probability(
                    f1_score, y_validation, hard, "validation F1", zero_division=0
                ),
                validation_recall=_metric_probability(
                    recall_score, y_validation, hard, "validation recall", zero_division=0
                ),
            )
        )
        fitted_candidates.append(classifier)

    candidates = (candidate_records[0], candidate_records[1])
    selected_index = _select_candidate_index(candidates)
    selected = candidates[selected_index]

    test = _evaluate_test_stage(
        test_frame=canonical_split.test,
        fit=canonical_fit,
        classifier=fitted_candidates[selected_index],
        threshold=selected.decision_threshold,
        targets=targets,
    )

    return AdvancedClassificationEvaluation(
        contract_version=ADVANCED_CLASSIFIER_CONTRACT_VERSION,
        selected_classifier_name=selected.classifier_name,
        feature_columns=canonical_fit.transformed_feature_columns,
        train_rows=len(canonical_split.train),
        validation_rows=len(canonical_split.validation),
        test_rows=len(canonical_split.test),
        candidates=candidates,
        test=TestClassificationMetrics(
            accuracy=test.accuracy,
            precision=test.precision,
            recall=test.recall,
            f1=test.f1,
            roc_auc=test.roc_auc,
            confusion_matrix=test.confusion_matrix,
        ),
    )


def _error(message: str) -> AdvancedClassificationError:
    return AdvancedClassificationError(_ERROR_PREFIX + message)


def _require_exact_types(
    dataset: object,
    split: object,
    fit: object,
    spec: object,
    config: object,
) -> None:
    if type(dataset) is not RepositoryExport:
        raise TypeError("dataset must be a RepositoryExport")
    if type(split) is not ChronologicalSplit:
        raise TypeError("split must be a ChronologicalSplit")
    if type(fit) is not PreprocessorFit:
        raise TypeError("fit must be a PreprocessorFit")
    if type(spec) is not DatasetSpec:
        raise TypeError("spec must be a DatasetSpec")
    if type(config) is not RoadGuardConfig:
        raise TypeError("config must be a RoadGuardConfig")


def _require_config_seed(config: RoadGuardConfig) -> int:
    """Validate the only configuration field Phase 11 may read.

    ``env``, ``data_dir``, ``artifacts_dir``, and ``database_url`` are never
    read, validated, or represented in an error.
    """
    seed = getattr(config, "seed", None)
    if type(seed) is not int or seed < 1:
        raise _error("the configuration seed must be a positive integer")
    return seed


def _require_spec_state(spec: DatasetSpec) -> None:
    try:
        values = (
            spec.dataset_segments,
            spec.dataset_months_per_segment,
            spec.dataset_observations,
        )
    except AttributeError:
        raise _error("the dataset specification is invalid") from None
    if (
        any(type(value) is not int or value < 1 for value in values)
        or values[2] != values[0] * values[1]
    ):
        raise _error("the dataset specification is invalid")


def _copy_repository_export(dataset: RepositoryExport) -> RepositoryExport:
    try:
        frames = (
            dataset.segments,
            dataset.observations,
            dataset.targets,
            dataset.maintenance_events,
        )
    except AttributeError:
        raise _error("the repository export frames are invalid") from None
    if any(type(frame) is not pd.DataFrame for frame in frames):
        raise _error("the repository export frames are invalid")
    return RepositoryExport(*(frame.copy(deep=True) for frame in frames))


def _require_matching_split(supplied: ChronologicalSplit, canonical: ChronologicalSplit) -> None:
    try:
        supplied_partitions = (supplied.train, supplied.validation, supplied.test)
        supplied_dates = (supplied.train_dates, supplied.validation_dates, supplied.test_dates)
    except AttributeError:
        raise _error("the supplied split state is invalid") from None
    canonical_partitions = (canonical.train, canonical.validation, canonical.test)
    for label, supplied_frame, canonical_frame in zip(
        ("train", "validation", "test"), supplied_partitions, canonical_partitions, strict=True
    ):
        if type(supplied_frame) is not pd.DataFrame:
            raise _error(f"the supplied {label} partition is not a valid split frame")
        if tuple(supplied_frame.columns) != tuple(canonical_frame.columns):
            raise _error(
                f"the supplied {label} partition schema does not match the canonical split"
            )
        if list(supplied_frame.dtypes) != list(canonical_frame.dtypes):
            raise _error(f"the supplied {label} partition dtypes do not match the canonical split")
        if not supplied_frame.reset_index(drop=True).equals(canonical_frame.reset_index(drop=True)):
            raise _error(f"the supplied {label} partition does not match the canonical split")
    if any(
        type(values) is not tuple or any(type(value) is not date for value in values)
        for values in supplied_dates
    ):
        raise _error("the supplied split date provenance is invalid")
    canonical_dates = (canonical.train_dates, canonical.validation_dates, canonical.test_dates)
    if supplied_dates != canonical_dates:
        raise _error("the supplied split date provenance does not match the canonical split")


def _require_matching_fit(supplied: PreprocessorFit, canonical: PreprocessorFit) -> None:
    string_fields = ("scaled_columns", "province_categories", "road_type_categories")
    numeric_fields = ("means", "stds")
    for field in (*string_fields, *numeric_fields):
        try:
            supplied_value = getattr(supplied, field)
        except AttributeError:
            raise _error(
                "the supplied preprocessor fit does not match the canonical train-only fit"
            ) from None
        expected_scalar = str if field in string_fields else float
        if type(supplied_value) is not tuple or any(
            type(value) is not expected_scalar for value in supplied_value
        ):
            raise _error(
                "the supplied preprocessor fit does not match the canonical train-only fit"
            )
        if supplied_value != getattr(canonical, field):
            raise _error(
                "the supplied preprocessor fit does not match the canonical train-only fit"
            )
    try:
        supplied_columns = supplied.transformed_feature_columns
    except AttributeError:
        raise _error(
            "the supplied preprocessor fit does not match the canonical train-only fit"
        ) from None
    if supplied_columns != canonical.transformed_feature_columns:
        raise _error("the supplied preprocessor fit does not match the canonical train-only fit")


def _join_classification_target(
    targets: pd.DataFrame, keys: pd.DataFrame, label: str
) -> pd.DataFrame:
    projected = targets.loc[:, _TARGET_PROJECTION]
    try:
        joined = keys.merge(
            projected,
            how="left",
            on=list(FEATURE_KEY_COLUMNS),
            sort=False,
            validate="one_to_one",
        )
    except pd.errors.MergeError:
        raise _error(f"targets do not align one-to-one with the {label} partition keys") from None
    if (
        not joined.loc[:, list(FEATURE_KEY_COLUMNS)]
        .reset_index(drop=True)
        .equals(keys.reset_index(drop=True))
    ):
        raise _error(f"targets do not align with the {label} partition key order")
    if joined[_CLASSIFICATION_TARGET].isna().any():
        raise _error(f"targets are missing for the {label} partition keys")
    if str(joined[_CLASSIFICATION_TARGET].dtype) != "int64":
        raise _error(f"the {label} classification targets have an invalid dtype")
    return joined


def _require_binary_classes(y_true: np.ndarray, label: str) -> None:
    unique = np.unique(y_true)
    if unique.size != 2 or int(unique[0]) != 0 or int(unique[1]) != 1:
        raise _error(f"the {label} classification targets must contain exactly classes 0 and 1")


def _derive_candidate_seed(config_seed: int, candidate_index: int) -> int:
    try:
        state = np.random.SeedSequence(
            [config_seed, ADVANCED_CLASSIFIER_RNG_NAMESPACE, candidate_index]
        ).generate_state(1, dtype=np.uint32)
        return int(state[0])
    except (ValueError, ArithmeticError, OverflowError, TypeError, IndexError, AttributeError):
        raise _error("the candidate seed derivation failed") from None


def _build_candidate(name: str, derived_seed: int) -> Any:
    if name == CANDIDATE_CLASSIFIER_NAMES[0]:
        try:
            return LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                tol=1e-8,
                fit_intercept=True,
                class_weight=None,
                random_state=derived_seed,
            )
        except (ValueError, ArithmeticError):
            raise _error("the logistic_l2 candidate could not be constructed") from None
    try:
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=derived_seed,
        )
    except (ValueError, ArithmeticError):
        raise _error("the hist_gradient_boosting candidate could not be constructed") from None


def _fit_candidate(classifier: Any, x: np.ndarray, y: np.ndarray) -> None:
    try:
        classifier.fit(x, y)
    except (ValueError, ArithmeticError):
        raise _error("a candidate classifier could not be fitted") from None


def _positive_probas(classifier: Any, x: np.ndarray) -> tuple[float, ...]:
    """Positive-class probabilities, converted once to finite in-range floats.

    ``predict_proba`` is the only probability source; ``predict`` is never
    used for a reported hard-label metric.
    """
    try:
        probabilities = classifier.predict_proba(x)
    except (ValueError, ArithmeticError):
        raise _error("a candidate classifier produced invalid probabilities") from None
    if (
        not isinstance(probabilities, np.ndarray)
        or probabilities.ndim != 2
        or probabilities.shape != (len(x), 2)
    ):
        raise _error("a candidate classifier returned malformed probability output")
    _require_classifier_classes(classifier)
    try:
        positive = tuple(float(value) for value in probabilities[:, 1])
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        raise _error("a candidate classifier produced invalid probabilities") from None
    if not all(math.isfinite(value) for value in positive):
        raise _error("a candidate classifier returned non-finite probabilities")
    if any(value < 0.0 or value > 1.0 for value in positive):
        raise _error("a candidate classifier returned out-of-range probabilities")
    return positive


def _require_classifier_classes(classifier: Any) -> None:
    try:
        classes = classifier.classes_
    except (AttributeError, ValueError, ArithmeticError):
        raise _error("a candidate classifier returned an unexpected class ordering") from None
    if (
        not isinstance(classes, np.ndarray)
        or classes.shape != (2,)
        or any(
            not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            for value in classes
        )
        or tuple(int(value) for value in classes) != (0, 1)
    ):
        raise _error("a candidate classifier returned an unexpected class ordering")


def _select_threshold(y_true: np.ndarray, probabilities: Sequence[float]) -> float:
    """Freeze a candidate threshold from validation evidence only.

    Candidates are ``0.0``, ``1.0`` and every distinct finite validation
    probability; a hard prediction is one when ``probability >= threshold``.
    Candidates are ranked by higher validation F1, then higher validation
    recall, then higher threshold; the first candidate under that total
    ordering is returned.
    """
    if not all(math.isfinite(value) for value in probabilities):
        raise _error("threshold selection requires finite validation probabilities")
    candidates = sorted({0.0, 1.0, *probabilities})
    best_threshold = candidates[0]
    best_key: tuple[float, float, float] | None = None
    for threshold in candidates:
        hard = (np.asarray(probabilities, dtype="float64") >= threshold).astype(np.int64)
        f1 = _metric_probability(f1_score, y_true, hard, "threshold F1", zero_division=0)
        recall = _metric_probability(
            recall_score, y_true, hard, "threshold recall", zero_division=0
        )
        key = (f1, recall, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def _select_candidate_index(candidates: Sequence[CandidateValidationMetrics]) -> int:
    """Select the winning candidate from validation records only.

    The ranking is higher validation PR-AUC, then higher validation F1, then
    higher validation recall, then earlier position in the locked candidate
    order. No test metric, threshold value, baseline comparison, random draw,
    or model internals participate in the ranking.
    """
    best_index = 0
    best_key: tuple[float, float, float] | None = None
    for index, record in enumerate(candidates):
        key = (record.validation_pr_auc, record.validation_f1, record.validation_recall)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
    return best_index


def _metric_float(function: Any, y_true: Any, y_pred: Any, context: str, **kwargs: Any) -> float:
    try:
        value = function(y_true, y_pred, **kwargs)
    except (ValueError, ArithmeticError):
        raise _error(f"the {context} metric could not be computed") from None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        raise _error(f"the {context} metric is not a finite number") from None
    if not math.isfinite(converted):
        raise _error(f"the {context} metric is not finite")
    return converted


def _metric_probability(
    function: Any, y_true: Any, y_pred: Any, context: str, **kwargs: Any
) -> float:
    converted = _metric_float(function, y_true, y_pred, context, **kwargs)
    if not 0.0 <= converted <= 1.0:
        raise _error(f"the {context} metric is outside the valid [0, 1] range")
    return converted


def _confusion_matrix_tuple(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int]]:
    try:
        matrix = confusion_matrix(y_true, y_pred, labels=(0, 1))
    except (ValueError, ArithmeticError):
        raise _error("the test confusion matrix could not be computed") from None
    if not isinstance(matrix, np.ndarray) or matrix.shape != (2, 2):
        raise _error("the test confusion matrix is malformed")
    raw_counts = (matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1])
    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
        for value in raw_counts
    ):
        raise _error("the test confusion matrix is malformed")
    tn, fp, fn, tp = (int(value) for value in raw_counts)
    if min(tn, fp, fn, tp) < 0 or tn + fp + fn + tp != len(y_true):
        raise _error("the test confusion matrix is malformed")
    return ((tn, fp), (fn, tp))


def _evaluate_test_stage(
    test_frame: pd.DataFrame,
    fit: PreprocessorFit,
    classifier: Any,
    threshold: float,
    targets: pd.DataFrame,
) -> _TestStageResult:
    """Evaluate the frozen test partition with the selected candidate only.

    This is the single private test-evaluation boundary: it is reached only
    after both candidate records, the selected name, and the selected
    threshold are frozen. It transforms the test partition exactly once and
    calls the selected candidate's ``predict_proba`` exactly once; the losing
    candidate never receives the test matrix and ``predict`` is never used.
    """
    try:
        transformed = transform(test_frame, fit)
    except PreprocessingError:
        raise _error(
            "the test partition could not be transformed with the fitted preprocessor"
        ) from None
    test_targets = _join_classification_target(targets, transformed.keys, "test")
    y_test = test_targets[_CLASSIFICATION_TARGET].to_numpy(dtype="int64")
    _require_binary_classes(y_test, "test")

    x_test = transformed.features.to_numpy(dtype="float64")
    test_probabilities = _positive_probas(classifier, x_test)
    test_hard = (np.asarray(test_probabilities) >= threshold).astype(np.int64)

    return _TestStageResult(
        accuracy=_metric_probability(accuracy_score, y_test, test_hard, "test accuracy"),
        precision=_metric_probability(
            precision_score, y_test, test_hard, "test precision", zero_division=0
        ),
        recall=_metric_probability(recall_score, y_test, test_hard, "test recall", zero_division=0),
        f1=_metric_probability(f1_score, y_test, test_hard, "test F1", zero_division=0),
        roc_auc=_metric_probability(roc_auc_score, y_test, test_probabilities, "test ROC-AUC"),
        confusion_matrix=_confusion_matrix_tuple(y_test, test_hard),
    )


__all__ = [
    "evaluate_advanced_classifier",
    "AdvancedClassificationError",
    "CandidateValidationMetrics",
    "TestClassificationMetrics",
    "AdvancedClassificationEvaluation",
    "ADVANCED_CLASSIFIER_CONTRACT_VERSION",
    "ADVANCED_CLASSIFIER_RNG_NAMESPACE",
    "CANDIDATE_CLASSIFIER_NAMES",
]
