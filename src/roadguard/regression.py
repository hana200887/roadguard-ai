"""Phase 12 validation-selected advanced regression."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from numbers import Real
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
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

ADVANCED_REGRESSOR_CONTRACT_VERSION: Final[str] = "roadguard.phase12.v1"
ADVANCED_REGRESSOR_RNG_NAMESPACE: Final[int] = 0x5247312
CANDIDATE_REGRESSOR_NAMES: Final[tuple[str, str]] = (
    "ridge_l2",
    "hist_gradient_boosting",
)
_REGRESSION_TARGET: Final[str] = TARGET_COLUMNS[2]
_TARGET_PROJECTION: Final[list[str]] = [*FEATURE_KEY_COLUMNS, _REGRESSION_TARGET]
_ERROR_PREFIX: Final[str] = "evaluate_advanced_regressor failed: "


class AdvancedRegressionError(ValueError):
    """Raised when Phase 12 input, estimator, prediction, or metric state is invalid."""


@dataclass(frozen=True)
class CandidateRegressionValidationMetrics:
    """Validation evidence for one locked regressor candidate."""

    regressor_name: str
    validation_mae: float
    validation_rmse: float


@dataclass(frozen=True)
class TestRegressionMetrics:
    """Selected-candidate metrics on the frozen test partition."""

    mae: float
    rmse: float
    r2: float


@dataclass(frozen=True)
class AdvancedRegressionEvaluation:
    """Immutable Phase 12 advanced-regression result."""

    contract_version: str
    selected_regressor_name: str
    feature_columns: tuple[str, ...]
    train_rows: int
    validation_rows: int
    test_rows: int
    candidates: tuple[
        CandidateRegressionValidationMetrics,
        CandidateRegressionValidationMetrics,
    ]
    test: TestRegressionMetrics


@dataclass(frozen=True)
class _TestStageResult:
    mae: float
    rmse: float
    r2: float


def evaluate_advanced_regressor(
    dataset: RepositoryExport,
    split: ChronologicalSplit,
    fit: PreprocessorFit,
    spec: DatasetSpec,
    config: RoadGuardConfig,
) -> AdvancedRegressionEvaluation:
    """Fit both candidates, validation-select one, and test it once."""
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

    train_targets = _join_regression_target(targets, train_transformed.keys, "training")
    validation_targets = _join_regression_target(targets, validation_transformed.keys, "validation")
    y_train = train_targets[_REGRESSION_TARGET].to_numpy(dtype="int64")
    y_validation = validation_targets[_REGRESSION_TARGET].to_numpy(dtype="int64")
    x_train = train_transformed.features.to_numpy(dtype="float64")
    x_validation = validation_transformed.features.to_numpy(dtype="float64")

    fitted_candidates = _build_candidates(config_seed)
    records: list[CandidateRegressionValidationMetrics] = []
    for name, candidate in zip(CANDIDATE_REGRESSOR_NAMES, fitted_candidates, strict=True):
        _fit_candidate(candidate, x_train, y_train)
        predictions = _regression_predictions(candidate, x_validation)
        records.append(
            CandidateRegressionValidationMetrics(
                regressor_name=name,
                validation_mae=_metric_nonnegative(
                    mean_absolute_error, y_validation, predictions, "validation MAE"
                ),
                validation_rmse=_metric_nonnegative(
                    root_mean_squared_error, y_validation, predictions, "validation RMSE"
                ),
            )
        )

    candidates = (records[0], records[1])
    selected_index = _select_candidate_index(candidates)
    selected = candidates[selected_index]
    test = _evaluate_test_stage(
        canonical_split.test,
        canonical_fit,
        fitted_candidates[selected_index],
        targets,
    )
    return AdvancedRegressionEvaluation(
        contract_version=ADVANCED_REGRESSOR_CONTRACT_VERSION,
        selected_regressor_name=selected.regressor_name,
        feature_columns=canonical_fit.transformed_feature_columns,
        train_rows=len(canonical_split.train),
        validation_rows=len(canonical_split.validation),
        test_rows=len(canonical_split.test),
        candidates=candidates,
        test=TestRegressionMetrics(test.mae, test.rmse, test.r2),
    )


def _error(message: str) -> AdvancedRegressionError:
    return AdvancedRegressionError(_ERROR_PREFIX + message)


def _require_exact_types(
    dataset: object, split: object, fit: object, spec: object, config: object
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
    return RepositoryExport(*(pd.DataFrame.copy(frame, deep=True) for frame in frames))


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
        normalized_supplied = pd.DataFrame.reset_index(supplied_frame, drop=True)
        normalized_canonical = pd.DataFrame.reset_index(canonical_frame, drop=True)
        if not pd.DataFrame.equals(normalized_supplied, normalized_canonical):
            raise _error(f"the supplied {label} partition does not match the canonical split")
    if any(
        type(values) is not tuple or any(type(value) is not date for value in values)
        for values in supplied_dates
    ):
        raise _error("the supplied split date provenance is invalid")
    if supplied_dates != (canonical.train_dates, canonical.validation_dates, canonical.test_dates):
        raise _error("the supplied split date provenance does not match the canonical split")


def _require_matching_fit(supplied: PreprocessorFit, canonical: PreprocessorFit) -> None:
    string_fields = ("scaled_columns", "province_categories", "road_type_categories")
    numeric_fields = ("means", "stds")
    for field in (*string_fields, *numeric_fields):
        try:
            value = getattr(supplied, field)
        except AttributeError:
            raise _error(
                "the supplied preprocessor fit does not match the canonical train-only fit"
            ) from None
        scalar_type = str if field in string_fields else float
        if type(value) is not tuple or any(type(item) is not scalar_type for item in value):
            raise _error(
                "the supplied preprocessor fit does not match the canonical train-only fit"
            )
        if value != getattr(canonical, field):
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


def _join_regression_target(targets: pd.DataFrame, keys: pd.DataFrame, label: str) -> pd.DataFrame:
    try:
        projected = targets.loc[:, _TARGET_PROJECTION]
        joined = keys.merge(
            projected,
            how="left",
            on=list(FEATURE_KEY_COLUMNS),
            sort=False,
            validate="one_to_one",
        )
    except (KeyError, pd.errors.MergeError):
        raise _error(f"targets do not align one-to-one with the {label} partition keys") from None
    if (
        not joined.loc[:, list(FEATURE_KEY_COLUMNS)]
        .reset_index(drop=True)
        .equals(keys.reset_index(drop=True))
    ):
        raise _error(f"targets do not align with the {label} partition key order")
    if joined[_REGRESSION_TARGET].isna().any():
        raise _error(f"targets are missing for the {label} partition keys")
    if str(joined[_REGRESSION_TARGET].dtype) != "int64":
        raise _error(f"the {label} regression targets have an invalid dtype")
    return joined


def _derive_hgb_seed(config_seed: int) -> int:
    try:
        state = np.random.SeedSequence(
            [config_seed, ADVANCED_REGRESSOR_RNG_NAMESPACE, 1]
        ).generate_state(1, dtype=np.uint32)
        return int(state[0])
    except (ValueError, ArithmeticError, OverflowError, TypeError, IndexError, AttributeError):
        raise _error("the regressor seed derivation failed") from None


def _build_candidates(config_seed: int) -> tuple[Any, Any]:
    try:
        ridge = Ridge(
            alpha=1.0,
            fit_intercept=True,
            solver="svd",
            tol=1e-8,
            positive=False,
        )
    except (ValueError, ArithmeticError):
        raise _error("the ridge_l2 candidate could not be constructed") from None
    hgb_seed = _derive_hgb_seed(config_seed)
    try:
        hgb = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=hgb_seed,
        )
    except (ValueError, ArithmeticError):
        raise _error("the hist_gradient_boosting candidate could not be constructed") from None
    return ridge, hgb


def _fit_candidate(candidate: Any, x: np.ndarray, y: np.ndarray) -> None:
    try:
        candidate.fit(x, y)
    except (ValueError, ArithmeticError):
        raise _error("a candidate regressor could not be fitted") from None


def _regression_predictions(candidate: Any, x: np.ndarray) -> tuple[float, ...]:
    try:
        predictions = candidate.predict(x)
    except (ValueError, ArithmeticError):
        raise _error("a candidate regressor produced invalid predictions") from None
    if type(predictions) is not np.ndarray or predictions.shape != (len(x),):
        raise _error("a candidate regressor returned malformed prediction output")
    converted: list[float] = []
    try:
        for value in predictions:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError
            converted.append(float(value))
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        raise _error("a candidate regressor produced invalid predictions") from None
    if not all(math.isfinite(value) for value in converted):
        raise _error("a candidate regressor returned non-finite predictions")
    return tuple(converted)


def _metric_float(
    function: Any,
    y_true: Any,
    prediction: Any,
    context: str,
    **kwargs: Any,
) -> float:
    try:
        value = function(y_true, prediction, **kwargs)
    except (ValueError, ArithmeticError):
        raise _error(f"the {context} metric could not be computed") from None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise _error(f"the {context} metric is not a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError, ArithmeticError):
        raise _error(f"the {context} metric is not a finite number") from None
    if not math.isfinite(converted):
        raise _error(f"the {context} metric is not finite")
    return converted


def _metric_nonnegative(
    function: Any, y_true: Any, prediction: Any, context: str, **kwargs: Any
) -> float:
    converted = _metric_float(function, y_true, prediction, context, **kwargs)
    if converted < 0.0:
        raise _error(f"the {context} metric is outside the valid non-negative range")
    return converted


def _select_candidate_index(
    candidates: Sequence[CandidateRegressionValidationMetrics],
) -> int:
    best_index = 0
    best_key: tuple[float, float] | None = None
    for index, record in enumerate(candidates):
        key = (record.validation_mae, record.validation_rmse)
        if best_key is None or key < best_key:
            best_key = key
            best_index = index
    return best_index


def _evaluate_test_stage(
    test_frame: pd.DataFrame,
    fit: PreprocessorFit,
    regressor: Any,
    targets: pd.DataFrame,
) -> _TestStageResult:
    """Transform and evaluate the frozen winning regressor exactly once."""
    try:
        transformed = transform(test_frame, fit)
    except PreprocessingError:
        raise _error(
            "the test partition could not be transformed with the fitted preprocessor"
        ) from None
    test_targets = _join_regression_target(targets, transformed.keys, "test")
    y_test = test_targets[_REGRESSION_TARGET].to_numpy(dtype="int64")
    if np.unique(y_test).size < 2:
        raise _error("the test regression targets must contain at least two distinct values")
    predictions = _regression_predictions(regressor, transformed.features.to_numpy(dtype="float64"))
    return _TestStageResult(
        mae=_metric_nonnegative(mean_absolute_error, y_test, predictions, "test MAE"),
        rmse=_metric_nonnegative(root_mean_squared_error, y_test, predictions, "test RMSE"),
        r2=_metric_float(
            r2_score,
            y_test,
            predictions,
            "test R-squared",
            force_finite=False,
        ),
    )


__all__ = [
    "evaluate_advanced_regressor",
    "AdvancedRegressionError",
    "CandidateRegressionValidationMetrics",
    "TestRegressionMetrics",
    "AdvancedRegressionEvaluation",
    "ADVANCED_REGRESSOR_CONTRACT_VERSION",
    "ADVANCED_REGRESSOR_RNG_NAMESPACE",
    "CANDIDATE_REGRESSOR_NAMES",
]
