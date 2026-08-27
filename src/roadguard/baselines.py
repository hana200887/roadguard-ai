"""Phase 10 baseline supervised evaluation.

The module establishes one deterministic learned baseline per supervised
target: ``DummyClassifier(strategy="prior")`` and
``DummyRegressor(strategy="median")``. ``evaluate_baselines`` accepts a
complete Phase 6 ``RepositoryExport``, the canonical Phase 8
``ChronologicalSplit``, its matching train-only ``PreprocessorFit`` and their
``DatasetSpec``; it deep-copies the export, fresh-validates it through the
Phase 7 feature builder, rebuilds the canonical Phase 8 split, refits the
train-only preprocessor and rejects a mismatched supplied split or fit.

Both estimators fit exactly once on the transformed canonical 34-date
training partition. Validation is used only to record the locked
classifier PR-AUC/F1/recall evidence, select the decision threshold, and
record the locked regression MAE/RMSE evidence. The test partition is
transformed and evaluated exactly once, only after both estimators and the
threshold are frozen, inside one private test stage. No estimator, fitted
state, prediction, coefficient, or split detail is returned or persisted;
the result is an immutable metrics record whose schema is frozen in
docs/contracts.md section 17. The dummy estimators use no RNG, so Phase 10
introduces no seed argument or random namespace.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import numpy as np
import pandas as pd
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

from roadguard._db_types import RepositoryExport
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

BASELINE_CONTRACT_VERSION: Final[str] = "roadguard.phase10.v1"
BASELINE_CLASSIFIER_NAME: Final[str] = "dummy_prior"
BASELINE_REGRESSOR_NAME: Final[str] = "dummy_median"

_TARGET_VALUE_COLUMNS: Final[tuple[str, ...]] = TARGET_COLUMNS[2:]
_CLASSIFICATION_TARGET: Final[str] = _TARGET_VALUE_COLUMNS[1]
_REGRESSION_TARGET: Final[str] = _TARGET_VALUE_COLUMNS[0]

_ERROR_PREFIX: Final[str] = "evaluate_baselines failed: "


class BaselineEvaluationError(ValueError):
    """Raised when Phase 10 input, estimator, prediction, or metric state is invalid."""


@dataclass(frozen=True)
class ClassificationBaselineMetrics:
    """Locked classification metrics for one baseline evaluation."""

    validation_pr_auc: float
    decision_threshold: float
    validation_f1: float
    validation_recall: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_roc_auc: float
    test_confusion_matrix: tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class RegressionBaselineMetrics:
    """Locked regression metrics for one baseline evaluation."""

    validation_mae: float
    validation_rmse: float
    test_mae: float
    test_rmse: float
    test_r2: float


@dataclass(frozen=True)
class BaselineEvaluation:
    """Immutable Phase 10 baseline evaluation result."""

    contract_version: str
    classifier_name: str
    regressor_name: str
    feature_columns: tuple[str, ...]
    train_rows: int
    validation_rows: int
    test_rows: int
    classification: ClassificationBaselineMetrics
    regression: RegressionBaselineMetrics


@dataclass(frozen=True)
class _TestStageResult:
    """Private test-stage metric bundle produced exactly once per call."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    mae: float
    rmse: float
    r2: float


def evaluate_baselines(
    dataset: RepositoryExport,
    split: ChronologicalSplit,
    fit: PreprocessorFit,
    spec: DatasetSpec,
) -> BaselineEvaluation:
    """Evaluate the locked baseline estimators and return immutable metrics.

    All four arguments must be exact instances of the declared types; wrong
    types and lookalikes raise ``TypeError`` before any field is read. After
    exact-type validation, expected lower-phase and scikit-learn failures are
    translated to :class:`BaselineEvaluationError` with fixed sanitized
    messages and suppressed chaining. The test partition is transformed and
    evaluated exactly once in a private stage after both estimators and the
    decision threshold are frozen.
    """
    _require_exact_types(dataset, split, fit, spec)

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

    train_targets = _join_targets(targets, train_transformed.keys, "training")
    validation_targets = _join_targets(targets, validation_transformed.keys, "validation")

    y_train = train_targets[_CLASSIFICATION_TARGET].to_numpy(dtype="int64")
    y_validation = validation_targets[_CLASSIFICATION_TARGET].to_numpy(dtype="int64")
    _require_binary_classes(y_train, "training")
    _require_binary_classes(y_validation, "validation")

    y_train_regression = train_targets[_REGRESSION_TARGET].to_numpy(dtype="int64")
    y_validation_regression = validation_targets[_REGRESSION_TARGET].to_numpy(dtype="int64")

    x_train = train_transformed.features.to_numpy(dtype="float64")
    x_validation = validation_transformed.features.to_numpy(dtype="float64")

    classifier = _fit_classifier(x_train, y_train)
    regressor = _fit_regressor(x_train, y_train_regression)

    validation_probabilities = _positive_probas(classifier, x_validation)
    validation_predictions = _regression_predictions(regressor, x_validation)

    threshold = _select_threshold(y_validation, validation_probabilities)
    validation_hard = (np.asarray(validation_probabilities) >= threshold).astype(np.int64)

    validation_pr_auc = _metric_float(
        average_precision_score, y_validation, validation_probabilities, "validation PR-AUC"
    )
    validation_f1 = _metric_float(
        f1_score, y_validation, validation_hard, "validation F1", zero_division=0
    )
    validation_recall = _metric_float(
        recall_score, y_validation, validation_hard, "validation recall", zero_division=0
    )
    validation_mae = _metric_float(
        mean_absolute_error, y_validation_regression, validation_predictions, "validation MAE"
    )
    validation_rmse = _metric_float(
        root_mean_squared_error,
        y_validation_regression,
        validation_predictions,
        "validation RMSE",
    )

    test = _evaluate_test_stage(
        test_frame=canonical_split.test,
        fit=canonical_fit,
        classifier=classifier,
        regressor=regressor,
        threshold=threshold,
        targets=targets,
    )

    classification = ClassificationBaselineMetrics(
        validation_pr_auc=validation_pr_auc,
        decision_threshold=threshold,
        validation_f1=validation_f1,
        validation_recall=validation_recall,
        test_accuracy=test.accuracy,
        test_precision=test.precision,
        test_recall=test.recall,
        test_f1=test.f1,
        test_roc_auc=test.roc_auc,
        test_confusion_matrix=test.confusion_matrix,
    )
    regression = RegressionBaselineMetrics(
        validation_mae=validation_mae,
        validation_rmse=validation_rmse,
        test_mae=test.mae,
        test_rmse=test.rmse,
        test_r2=test.r2,
    )
    return BaselineEvaluation(
        contract_version=BASELINE_CONTRACT_VERSION,
        classifier_name=BASELINE_CLASSIFIER_NAME,
        regressor_name=BASELINE_REGRESSOR_NAME,
        feature_columns=canonical_fit.transformed_feature_columns,
        train_rows=len(canonical_split.train),
        validation_rows=len(canonical_split.validation),
        test_rows=len(canonical_split.test),
        classification=classification,
        regression=regression,
    )


def _error(message: str) -> BaselineEvaluationError:
    return BaselineEvaluationError(_ERROR_PREFIX + message)


def _require_exact_types(dataset: object, split: object, fit: object, spec: object) -> None:
    if type(dataset) is not RepositoryExport:
        raise TypeError("dataset must be a RepositoryExport")
    if type(split) is not ChronologicalSplit:
        raise TypeError("split must be a ChronologicalSplit")
    if type(fit) is not PreprocessorFit:
        raise TypeError("fit must be a PreprocessorFit")
    if type(spec) is not DatasetSpec:
        raise TypeError("spec must be a DatasetSpec")


def _copy_repository_export(dataset: RepositoryExport) -> RepositoryExport:
    frames = (
        dataset.segments,
        dataset.observations,
        dataset.targets,
        dataset.maintenance_events,
    )
    if any(type(frame) is not pd.DataFrame for frame in frames):
        raise _error("the repository export frames are invalid")
    return RepositoryExport(*(frame.copy(deep=True) for frame in frames))


def _require_matching_split(supplied: ChronologicalSplit, canonical: ChronologicalSplit) -> None:
    for label, supplied_frame, canonical_frame in (
        ("train", supplied.train, canonical.train),
        ("validation", supplied.validation, canonical.validation),
        ("test", supplied.test, canonical.test),
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
    supplied_dates = (supplied.train_dates, supplied.validation_dates, supplied.test_dates)
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
        supplied_value = getattr(supplied, field)
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
    if supplied.transformed_feature_columns != canonical.transformed_feature_columns:
        raise _error("the supplied preprocessor fit does not match the canonical train-only fit")


def _join_targets(targets: pd.DataFrame, keys: pd.DataFrame, label: str) -> pd.DataFrame:
    try:
        joined = keys.merge(
            targets,
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
    if joined.loc[:, list(_TARGET_VALUE_COLUMNS)].isna().any().any():
        raise _error(f"targets are missing for the {label} partition keys")
    return joined


def _require_binary_classes(y_true: np.ndarray, label: str) -> None:
    if np.unique(y_true).tolist() != [0, 1]:
        raise _error(f"the {label} classification targets must contain both classes 0 and 1")


def _require_non_constant(values: np.ndarray) -> None:
    if np.unique(values).size < 2:
        raise _error("the test regression target must contain at least two distinct values")


def _fit_classifier(x: np.ndarray, y: np.ndarray) -> DummyClassifier:
    try:
        classifier = DummyClassifier(strategy="prior")
        classifier.fit(x, y)
    except (ValueError, ArithmeticError):
        raise _error("the baseline classifier could not be fitted") from None
    _require_classifier_state(classifier, y)
    return classifier


def _fit_regressor(x: np.ndarray, y: np.ndarray) -> DummyRegressor:
    try:
        regressor = DummyRegressor(strategy="median")
        regressor.fit(x, y)
    except (ValueError, ArithmeticError):
        raise _error("the baseline regressor could not be fitted") from None
    _require_regressor_state(regressor, y)
    return regressor


def _require_classifier_state(classifier: Any, y: np.ndarray) -> None:
    _require_classifier_classes(classifier)
    try:
        raw_prior = classifier.class_prior_
    except AttributeError:
        raise _error("the baseline classifier fitted state is invalid") from None
    if not isinstance(raw_prior, np.ndarray) or raw_prior.shape != (2,):
        raise _error("the baseline classifier fitted state is invalid")
    try:
        prior = np.asarray(tuple(float(value) for value in raw_prior), dtype="float64")
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        raise _error("the baseline classifier fitted state is invalid") from None
    expected = np.asarray([(y == 0).mean(), (y == 1).mean()], dtype="float64")
    if not np.isfinite(prior).all() or not np.array_equal(prior, expected):
        raise _error("the baseline classifier fitted state is invalid")


def _require_regressor_state(regressor: Any, y: np.ndarray) -> None:
    try:
        raw_constant = regressor.constant_
    except AttributeError:
        raise _error("the baseline regressor fitted state is invalid") from None
    if not isinstance(raw_constant, np.ndarray) or raw_constant.shape != (1, 1):
        raise _error("the baseline regressor fitted state is invalid")
    try:
        constant = float(raw_constant[0, 0])
        expected = float(np.median(y))
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        raise _error("the baseline regressor fitted state is invalid") from None
    if not math.isfinite(constant) or constant != expected:
        raise _error("the baseline regressor fitted state is invalid")


def _require_classifier_classes(classifier: Any) -> None:
    try:
        classes = classifier.classes_
    except AttributeError:
        raise _error("the baseline classifier returned an unexpected class ordering") from None
    if (
        not isinstance(classes, np.ndarray)
        or classes.shape != (2,)
        or any(
            not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            for value in classes
        )
        or tuple(int(value) for value in classes) != (0, 1)
    ):
        raise _error("the baseline classifier returned an unexpected class ordering")


def _positive_probas(classifier: Any, x: np.ndarray) -> tuple[float, ...]:
    """Positive-class probabilities, converted once to finite in-range floats.

    ``predict_proba`` is the only probability source; ``predict`` is never
    used for reported hard-label metrics.
    """
    try:
        probabilities = classifier.predict_proba(x)
    except (ValueError, ArithmeticError):
        raise _error("the baseline classifier produced invalid probabilities") from None
    if (
        not isinstance(probabilities, np.ndarray)
        or probabilities.ndim != 2
        or probabilities.shape != (len(x), 2)
    ):
        raise _error("the baseline classifier returned malformed probability output")
    _require_classifier_classes(classifier)
    try:
        positive = tuple(float(value) for value in probabilities[:, 1])
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        raise _error("the baseline classifier produced invalid probabilities") from None
    if not all(math.isfinite(value) for value in positive):
        raise _error("the baseline classifier returned non-finite probabilities")
    if any(value < 0.0 or value > 1.0 for value in positive):
        raise _error("the baseline classifier returned out-of-range probabilities")
    return positive


def _regression_predictions(regressor: Any, x: np.ndarray) -> tuple[float, ...]:
    """Regression predictions, converted once to finite built-in floats."""
    try:
        predictions = regressor.predict(x)
    except (ValueError, ArithmeticError):
        raise _error("the baseline regressor produced invalid predictions") from None
    if (
        not isinstance(predictions, np.ndarray)
        or predictions.ndim != 1
        or len(predictions) != len(x)
    ):
        raise _error("the baseline regressor returned malformed predictions")
    try:
        converted = tuple(float(value) for value in predictions)
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        raise _error("the baseline regressor produced invalid predictions") from None
    if not all(math.isfinite(value) for value in converted):
        raise _error("the baseline regressor returned non-finite predictions")
    return converted


def _select_threshold(y_true: np.ndarray, probabilities: Sequence[float]) -> float:
    """Freeze the classifier threshold from validation evidence only.

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
        f1 = _metric_float(f1_score, y_true, hard, "threshold F1", zero_division=0)
        recall = _metric_float(recall_score, y_true, hard, "threshold recall", zero_division=0)
        key = (f1, recall, threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
    return best_threshold


def _metric_float(function: Any, y_true: Any, y_pred: Any, context: str, **kwargs: Any) -> float:
    try:
        value = function(y_true, y_pred, **kwargs)
    except (ValueError, ArithmeticError):
        raise _error(f"the {context} metric could not be computed") from None
    return _finite_float(value, context)


def _finite_float(value: Any, context: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        raise _error(f"the {context} metric is not a finite number") from None
    if not math.isfinite(converted):
        raise _error(f"the {context} metric is not finite")
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
    regressor: Any,
    threshold: float,
    targets: pd.DataFrame,
) -> _TestStageResult:
    """Evaluate the frozen test partition exactly once.

    This is the single private test-evaluation boundary: it is reached only
    after both estimators and the threshold are frozen, and it is invoked
    exactly once per workflow call. Test-target degeneracy checks (constant
    regression target, then single-class labels) run here.
    """
    try:
        transformed = transform(test_frame, fit)
    except PreprocessingError:
        raise _error(
            "the test partition could not be transformed with the fitted preprocessor"
        ) from None
    test_targets = _join_targets(targets, transformed.keys, "test")

    y_test = test_targets[_CLASSIFICATION_TARGET].to_numpy(dtype="int64")
    y_test_regression = test_targets[_REGRESSION_TARGET].to_numpy(dtype="int64")
    _require_non_constant(y_test_regression)
    _require_binary_classes(y_test, "test")

    x_test = transformed.features.to_numpy(dtype="float64")
    test_probabilities = _positive_probas(classifier, x_test)
    test_predictions = _regression_predictions(regressor, x_test)

    test_hard = (np.asarray(test_probabilities) >= threshold).astype(np.int64)

    accuracy = _metric_float(accuracy_score, y_test, test_hard, "test accuracy")
    precision = _metric_float(precision_score, y_test, test_hard, "test precision", zero_division=0)
    recall = _metric_float(recall_score, y_test, test_hard, "test recall", zero_division=0)
    f1 = _metric_float(f1_score, y_test, test_hard, "test F1", zero_division=0)
    roc_auc = _metric_float(roc_auc_score, y_test, test_probabilities, "test ROC-AUC")
    confusion = _confusion_matrix_tuple(y_test, test_hard)
    mae = _metric_float(mean_absolute_error, y_test_regression, test_predictions, "test MAE")
    rmse = _metric_float(root_mean_squared_error, y_test_regression, test_predictions, "test RMSE")
    r2 = _metric_float(
        r2_score,
        y_test_regression,
        test_predictions,
        "test R-squared",
        force_finite=False,
    )
    return _TestStageResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        confusion_matrix=confusion,
        mae=mae,
        rmse=rmse,
        r2=r2,
    )


__all__ = [
    "evaluate_baselines",
    "BaselineEvaluationError",
    "BaselineEvaluation",
    "ClassificationBaselineMetrics",
    "RegressionBaselineMetrics",
    "BASELINE_CONTRACT_VERSION",
    "BASELINE_CLASSIFIER_NAME",
    "BASELINE_REGRESSOR_NAME",
]
