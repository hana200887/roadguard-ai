"""Phase 9 train-only exploratory analysis and deterministic data card.

Accepts a complete Phase 6 ``RepositoryExport``, the canonical Phase 8
``ChronologicalSplit`` and their ``DatasetSpec``; fresh-validates the export,
rebuilds the Phase 7 feature frame and Phase 8 split, and computes all
descriptive statistics on the canonical 34-date training partition only.
Validation and test contribute only ``SplitInventory`` metadata. Rendering is
an in-memory deterministic Markdown data card with no I/O.
"""

from __future__ import annotations

import decimal
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Final, Literal

import numpy as np
import pandas as pd

from roadguard._db_types import RepositoryExport
from roadguard.contracts import DatasetSpec
from roadguard.data_quality import validate_cleaned_dataset
from roadguard.features import (
    FEATURE_COLUMNS,
    FEATURE_FRAME_COLUMNS,
    FEATURE_KEY_COLUMNS,
    FEATURE_REGISTRY,
    build_feature_frame,
)
from roadguard.preprocessing import ChronologicalSplit, split_chronologically
from roadguard.targets import TARGET_COLUMNS

CONTRACT_VERSION: Final[str] = "roadguard.phase9.v1"
_DECIMAL_PRECISION: Final[int] = 80
_RENDER_PRECISION: Final[int] = 1100
_DECIMAL_EMIN: Final[int] = -999999999
_DECIMAL_EMAX: Final[int] = 999999999
_NUMERIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    definition.name for definition in FEATURE_REGISTRY if definition.kind == "numeric"
)
_CATEGORICAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    definition.name for definition in FEATURE_REGISTRY if definition.kind == "categorical"
)
_DATETIME_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    definition.name for definition in FEATURE_REGISTRY if definition.kind == "datetime"
)
_TARGET_VALUE_COLUMNS: Final[tuple[str, ...]] = TARGET_COLUMNS[2:]
_CORRELATION_PAIRS: Final[tuple[tuple[str, str], ...]] = tuple(
    (feature, target) for feature in _NUMERIC_FEATURE_COLUMNS for target in _TARGET_VALUE_COLUMNS
)


class EDAError(ValueError):
    """Raised when Phase 9 data is invalid or a result cannot be produced safely."""


@dataclass(frozen=True)
class SplitInventory:
    """Partition metadata: name, row/date counts and date boundaries only."""

    name: Literal["train", "validation", "test"]
    row_count: int
    date_count: int
    first_date: date
    last_date: date


@dataclass(frozen=True)
class DataQualitySummary:
    """Training feature/target join integrity counts."""

    row_count: int
    segment_count: int
    date_count: int
    duplicate_key_count: int
    missing_cell_count: int
    non_finite_numeric_count: int


@dataclass(frozen=True)
class NumericSummary:
    """Exact Decimal-based descriptive statistics for one numeric column."""

    column: str
    count: int
    missing_count: int
    mean: float
    population_std: float
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    iqr_outlier_count: int
    iqr_outlier_rate: float
    zero_variance: bool


@dataclass(frozen=True)
class CategoricalLevel:
    """One categorical level with its count and proportion."""

    value: str
    count: int
    proportion: float


@dataclass(frozen=True)
class CategoricalSummary:
    """Level inventory for one categorical column."""

    column: str
    count: int
    missing_count: int
    cardinality: int
    levels: tuple[CategoricalLevel, ...]


@dataclass(frozen=True)
class DateSummary:
    """Date-boundary summary for one datetime column."""

    column: str
    count: int
    missing_count: int
    unique_count: int
    minimum: date
    maximum: date


@dataclass(frozen=True)
class ClassificationBalance:
    """Positive/negative balance for the binary classification target."""

    column: str
    negative_count: int
    positive_count: int
    positive_rate: float


@dataclass(frozen=True)
class TargetCorrelation:
    """Pearson correlation between one numeric feature and one target."""

    feature: str
    target: str
    pearson_r: float | None


@dataclass(frozen=True)
class EDAReport:
    """Immutable train-only descriptive evidence report."""

    contract_version: str
    training_fingerprint: str
    feature_columns: tuple[str, ...]
    split_inventory: tuple[SplitInventory, ...]
    data_quality: DataQualitySummary
    numeric_features: tuple[NumericSummary, ...]
    categorical_features: tuple[CategoricalSummary, ...]
    datetime_features: tuple[DateSummary, ...]
    regression_target: NumericSummary
    classification_target: ClassificationBalance
    target_correlations: tuple[TargetCorrelation, ...]


@dataclass(frozen=True)
class _NumericStatistics:
    mean: float
    population_std: float
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    iqr_outlier_count: int
    iqr_outlier_rate: float
    zero_variance: bool


def _fresh_decimal_context(precision: int) -> decimal.Context:
    ctx = decimal.getcontext().copy()
    ctx.prec = precision
    ctx.rounding = decimal.ROUND_HALF_EVEN
    ctx.Emin = _DECIMAL_EMIN
    ctx.Emax = _DECIMAL_EMAX
    ctx.traps[decimal.InvalidOperation] = True
    ctx.traps[decimal.DivisionByZero] = True
    ctx.traps[decimal.Overflow] = True
    ctx.clear_flags()
    return ctx


def _to_decimal(value: int | float) -> decimal.Decimal:
    if isinstance(value, int):
        return decimal.Decimal(value)
    return decimal.Decimal.from_float(float(value))


def _to_float(value: decimal.Decimal) -> float:
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as exc:
        raise EDAError(
            f"derived Decimal value {value} is not representable as a Python float"
        ) from exc
    if not math.isfinite(result):
        raise EDAError("derived statistic is not a finite Python float")
    return result


def _population_variance(
    values: Sequence[decimal.Decimal], mean: decimal.Decimal
) -> decimal.Decimal:
    total = decimal.Decimal(0)
    for value in values:
        deviation = value - mean
        total = total + deviation * deviation
    return total / decimal.Decimal(len(values))


def _quantile(
    ordered: Sequence[decimal.Decimal],
    count: int,
    proportion: decimal.Decimal,
) -> decimal.Decimal:
    position = decimal.Decimal(count - 1) * proportion
    floor_index = int(position.to_integral_value(rounding=decimal.ROUND_FLOOR))
    lower = ordered[floor_index]
    if position == decimal.Decimal(floor_index):
        return lower
    upper = ordered[floor_index + 1]
    fraction = position - decimal.Decimal(floor_index)
    return lower + (upper - lower) * fraction


def _numeric_statistics(values: Sequence[int | float]) -> _NumericStatistics:
    count = len(values)
    if count < 1:
        raise EDAError("numeric statistics require at least one value")
    with decimal.localcontext(_fresh_decimal_context(_DECIMAL_PRECISION)):
        decimals = [_to_decimal(value) for value in values]
        first = decimals[0]
        constant = all(value == first for value in decimals)
        if constant:
            mean = first
            variance = decimal.Decimal(0)
            minimum = first
            q1 = first
            median = first
            q3 = first
            maximum = first
            outlier_count = 0
        else:
            total = decimal.Decimal(0)
            for value in decimals:
                total = total + value
            mean = total / decimal.Decimal(count)
            variance = _population_variance(decimals, mean)
            if variance == 0:
                raise EDAError(
                    "non-constant numeric column produced a zero variance; "
                    "this is an arithmetic failure"
                )
            ordered = sorted(decimals)
            minimum = ordered[0]
            maximum = ordered[-1]
            q1 = _quantile(ordered, count, decimal.Decimal("0.25"))
            median = _quantile(ordered, count, decimal.Decimal("0.50"))
            q3 = _quantile(ordered, count, decimal.Decimal("0.75"))
            iqr = q3 - q1
            lower = q1 - decimal.Decimal("1.5") * iqr
            upper = q3 + decimal.Decimal("1.5") * iqr
            outlier_count = sum(1 for value in decimals if value < lower or value > upper)
        std = variance.sqrt()
        outlier_rate = decimal.Decimal(outlier_count) / decimal.Decimal(count)
    return _NumericStatistics(
        mean=_to_float(mean),
        population_std=_to_float(std),
        minimum=_to_float(minimum),
        q1=_to_float(q1),
        median=_to_float(median),
        q3=_to_float(q3),
        maximum=_to_float(maximum),
        iqr_outlier_count=outlier_count,
        iqr_outlier_rate=_to_float(outlier_rate),
        zero_variance=constant,
    )


def _pearson_correlation(
    x_values: Sequence[int | float], y_values: Sequence[int | float]
) -> float | None:
    count = len(x_values)
    if count != len(y_values) or count < 1:
        raise EDAError("correlation requires paired non-empty sequences")
    with decimal.localcontext(_fresh_decimal_context(_DECIMAL_PRECISION)):
        x = [_to_decimal(value) for value in x_values]
        y = [_to_decimal(value) for value in y_values]
        if all(value == x[0] for value in x) or all(value == y[0] for value in y):
            return None
        x_mean = sum(x) / decimal.Decimal(count)
        y_mean = sum(y) / decimal.Decimal(count)
        sxy = decimal.Decimal(0)
        sxx = decimal.Decimal(0)
        syy = decimal.Decimal(0)
        for x_value, y_value in zip(x, y, strict=True):
            x_deviation = x_value - x_mean
            y_deviation = y_value - y_mean
            sxy = sxy + x_deviation * y_deviation
            sxx = sxx + x_deviation * x_deviation
            syy = syy + y_deviation * y_deviation
        if sxx == 0 or syy == 0:
            raise EDAError("non-constant input produced a zero variance sum during correlation")
        correlation = sxy / (sxx * syy).sqrt()
    return _to_float(correlation)


def _categorical_levels(counts: Mapping[str, int], total: int) -> tuple[CategoricalLevel, ...]:
    if total < 1:
        raise EDAError("categorical summary requires at least one value")
    with decimal.localcontext(_fresh_decimal_context(_DECIMAL_PRECISION)):
        levels = tuple(
            CategoricalLevel(
                value=value,
                count=count,
                proportion=_to_float(decimal.Decimal(count) / decimal.Decimal(total)),
            )
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        )
    return levels


def _canonical_scalar(value: object) -> object:
    """Canonical fingerprint scalar: dates as ISO, floats as lowercase hex."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number == 0.0:
            return "0x0.0p+0"
        return number.hex()
    if isinstance(value, str):
        return value
    if isinstance(value, (pd.Timestamp, datetime, date)):
        if isinstance(value, pd.Timestamp):
            return value.date().isoformat()
        if isinstance(value, datetime):
            return value.date().isoformat()
        return value.isoformat()
    raise EDAError(f"value of type {type(value).__name__} cannot be fingerprinted")


def _training_fingerprint(
    join: pd.DataFrame,
    split: ChronologicalSplit,
    spec: DatasetSpec,
) -> str:
    columns = list(FEATURE_FRAME_COLUMNS) + list(_TARGET_VALUE_COLUMNS)
    rows: list[list[object]] = []
    for _, row in join.iterrows():
        rows.append([_canonical_scalar(row[column]) for column in columns])
    payload = {
        "columns": columns,
        "contract": CONTRACT_VERSION,
        "spec": {
            "dataset_months_per_segment": spec.dataset_months_per_segment,
            "dataset_observations": spec.dataset_observations,
            "dataset_segments": spec.dataset_segments,
        },
        "split": {
            "test": [value.isoformat() for value in split.test_dates],
            "train": [value.isoformat() for value in split.train_dates],
            "validation": [value.isoformat() for value in split.validation_dates],
        },
        "train_rows": rows,
    }
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (ValueError, TypeError) as exc:
        raise EDAError("fingerprint payload contains non-serializable values") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_matching_split(supplied: ChronologicalSplit, canonical: ChronologicalSplit) -> None:
    for label, supplied_frame, canonical_frame in (
        ("train", supplied.train, canonical.train),
        ("validation", supplied.validation, canonical.validation),
        ("test", supplied.test, canonical.test),
    ):
        if tuple(supplied_frame.columns) != tuple(canonical_frame.columns):
            raise EDAError(
                f"supplied split {label} frame schema does not match the canonical split"
            )
        if list(supplied_frame.dtypes) != list(canonical_frame.dtypes):
            raise EDAError(f"supplied split {label} frame dtypes do not match the canonical split")
        if not supplied_frame.reset_index(drop=True).equals(canonical_frame):
            raise EDAError(
                f"supplied split {label} frame does not match the canonical rebuilt split"
            )
    supplied_dates = (
        supplied.train_dates,
        supplied.validation_dates,
        supplied.test_dates,
    )
    canonical_dates = (
        canonical.train_dates,
        canonical.validation_dates,
        canonical.test_dates,
    )
    if supplied_dates != canonical_dates:
        raise EDAError("supplied split date provenance does not match the canonical split")


def _split_inventory(split: ChronologicalSplit) -> tuple[SplitInventory, ...]:
    entries: tuple[
        tuple[Literal["train", "validation", "test"], pd.DataFrame, tuple[date, ...]],
        ...,
    ] = (
        ("train", split.train, split.train_dates),
        ("validation", split.validation, split.validation_dates),
        ("test", split.test, split.test_dates),
    )
    return tuple(
        SplitInventory(
            name=name,
            row_count=len(frame),
            date_count=len(dates),
            first_date=dates[0],
            last_date=dates[-1],
        )
        for name, frame, dates in entries
    )


def _data_quality_summary(join: pd.DataFrame) -> DataQualitySummary:
    row_count = len(join)
    segment_count = int(join["segment_id"].nunique())
    date_count = int(join["date"].nunique())
    duplicate_key_count = int(join.duplicated(subset=list(FEATURE_KEY_COLUMNS)).sum())
    missing_cell_count = int(join.isna().sum().sum())
    numeric_cells = join.loc[:, list(_NUMERIC_FEATURE_COLUMNS) + list(_TARGET_VALUE_COLUMNS)]
    non_finite_numeric_count = 0
    for column in numeric_cells.columns:
        non_finite_numeric_count += int(
            (~np.isfinite(numeric_cells[column].to_numpy(dtype="float64"))).sum()
        )
    return DataQualitySummary(
        row_count=row_count,
        segment_count=segment_count,
        date_count=date_count,
        duplicate_key_count=duplicate_key_count,
        missing_cell_count=missing_cell_count,
        non_finite_numeric_count=non_finite_numeric_count,
    )


def _numeric_summary(column: str, series: pd.Series) -> NumericSummary:
    values = series.tolist()
    count = len(values)
    stats = _numeric_statistics(values)
    return NumericSummary(
        column=column,
        count=count,
        missing_count=0,
        mean=stats.mean,
        population_std=stats.population_std,
        minimum=stats.minimum,
        q1=stats.q1,
        median=stats.median,
        q3=stats.q3,
        maximum=stats.maximum,
        iqr_outlier_count=stats.iqr_outlier_count,
        iqr_outlier_rate=stats.iqr_outlier_rate,
        zero_variance=stats.zero_variance,
    )


def _categorical_summary(column: str, series: pd.Series) -> CategoricalSummary:
    values = series.tolist()
    count = len(values)
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return CategoricalSummary(
        column=column,
        count=count,
        missing_count=0,
        cardinality=len(counts),
        levels=_categorical_levels(counts, total=count),
    )


def _date_summary(column: str, series: pd.Series) -> DateSummary:
    values = sorted(series.dt.date)
    return DateSummary(
        column=column,
        count=len(values),
        missing_count=0,
        unique_count=len(set(values)),
        minimum=values[0],
        maximum=values[-1],
    )


def _classification_balance(column: str, series: pd.Series) -> ClassificationBalance:
    values = series.tolist()
    count = len(values)
    negative_count = sum(1 for value in values if value == 0)
    positive_count = sum(1 for value in values if value == 1)
    if negative_count + positive_count != count:
        raise EDAError("classification target must contain only 0 and 1 values")
    with decimal.localcontext(_fresh_decimal_context(_DECIMAL_PRECISION)):
        positive_rate = _to_float(decimal.Decimal(positive_count) / decimal.Decimal(count))
    return ClassificationBalance(
        column=column,
        negative_count=negative_count,
        positive_count=positive_count,
        positive_rate=positive_rate,
    )


def build_eda_report(
    dataset: RepositoryExport,
    split: ChronologicalSplit,
    spec: DatasetSpec,
) -> EDAReport:
    """Build the immutable train-only descriptive evidence report."""
    if type(dataset) is not RepositoryExport:
        raise TypeError("dataset must be a RepositoryExport")
    if type(split) is not ChronologicalSplit:
        raise TypeError("split must be a ChronologicalSplit")
    if type(spec) is not DatasetSpec:
        raise TypeError("spec must be a DatasetSpec")

    segments = dataset.segments.copy(deep=True)
    observations = dataset.observations.copy(deep=True)
    targets = dataset.targets.copy(deep=True)
    maintenance_events = dataset.maintenance_events.copy(deep=True)
    validation = validate_cleaned_dataset(segments, observations, targets, maintenance_events, spec)
    if validation.error_count > 0:
        codes = ", ".join(sorted(validation.counts_by_code))
        raise EDAError(f"cleaned dataset validation failed with errors: {codes}")

    exported = RepositoryExport(
        segments=segments,
        observations=observations,
        targets=targets,
        maintenance_events=maintenance_events,
    )
    frame = build_feature_frame(exported, spec)
    canonical_split = split_chronologically(frame, spec)
    _require_matching_split(supplied=split, canonical=canonical_split)

    train = canonical_split.train
    join = train.merge(
        targets,
        how="inner",
        on=list(FEATURE_KEY_COLUMNS),
        validate="one_to_one",
    )
    if len(join) != len(train):
        raise EDAError("target frame does not contain exactly one row per training key")
    join = join.sort_values(list(FEATURE_KEY_COLUMNS), kind="stable").reset_index(drop=True)

    numeric_features = tuple(
        _numeric_summary(column, join[column]) for column in _NUMERIC_FEATURE_COLUMNS
    )
    categorical_features = tuple(
        _categorical_summary(column, join[column]) for column in _CATEGORICAL_FEATURE_COLUMNS
    )
    datetime_features = tuple(
        _date_summary(column, join[column]) for column in _DATETIME_FEATURE_COLUMNS
    )
    regression_target = _numeric_summary(_TARGET_VALUE_COLUMNS[0], join[_TARGET_VALUE_COLUMNS[0]])
    classification_target = _classification_balance(
        _TARGET_VALUE_COLUMNS[1], join[_TARGET_VALUE_COLUMNS[1]]
    )
    target_correlations = tuple(
        TargetCorrelation(
            feature=feature,
            target=target,
            pearson_r=_pearson_correlation(join[feature].tolist(), join[target].tolist()),
        )
        for feature, target in _CORRELATION_PAIRS
    )
    return EDAReport(
        contract_version=CONTRACT_VERSION,
        training_fingerprint=_training_fingerprint(join, canonical_split, spec),
        feature_columns=FEATURE_COLUMNS,
        split_inventory=_split_inventory(canonical_split),
        data_quality=_data_quality_summary(join),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        datetime_features=datetime_features,
        regression_target=regression_target,
        classification_target=classification_target,
        target_correlations=target_correlations,
    )


from roadguard._eda_render import render_data_card  # noqa: E402

__all__ = [
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
]
