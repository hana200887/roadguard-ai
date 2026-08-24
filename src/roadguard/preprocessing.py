"""Phase 8 chronological splitting and train-only preprocessing.

Accepts only the exact target-free Phase 7 feature frame, splits it by
sorted unique observation dates into fixed 34/7/7 partitions, and fits
deterministic encoders/scalers on the training partition only. Validation
and test frames are only ever transformed with the fitted state; the public
workflow makes fitting on validation/test data structurally impossible
because ``fit_preprocessor`` accepts only a complete, provenance-checked
``ChronologicalSplit`` and always fits its canonical first 34 dates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import pandas as pd

from roadguard.contracts import DatasetSpec
from roadguard.features import (
    FEATURE_COLUMNS,
    FEATURE_FRAME_COLUMNS,
    FEATURE_KEY_COLUMNS,
    FEATURE_REGISTRY,
)
from roadguard.segments import PROVINCES, ROAD_TYPES, SEGMENT_ID_PATTERN

TRAIN_DATE_COUNT: Final[int] = 34
VALIDATION_DATE_COUNT: Final[int] = 7
TEST_DATE_COUNT: Final[int] = 7
V1_TRAIN_ROWS: Final[int] = 10_200
V1_VALIDATION_ROWS: Final[int] = 2_100
V1_TEST_ROWS: Final[int] = 2_100
CONSTRUCTION_DATE_DAY_COLUMN: Final[str] = "construction_date_days"
DAY_EPOCH: Final[date] = date(1970, 1, 1)

_CATEGORICAL_FEATURE_COLUMNS: Final[tuple[str, str]] = ("province", "road_type")
_DATETIME_FEATURE_COLUMNS: Final[tuple[str, ...]] = ("construction_date",)
_INT_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    definition.name
    for definition in FEATURE_REGISTRY
    if definition.kind == "numeric"
    and definition.name
    not in ("road_length_km", "heavy_vehicle_ratio", "rainfall_mm", "temperature", "humidity")
)
_FLOAT_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "road_length_km",
    "heavy_vehicle_ratio",
    "rainfall_mm",
    "temperature",
    "humidity",
)
_NUMERIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = tuple(
    definition.name for definition in FEATURE_REGISTRY if definition.kind == "numeric"
)


class PreprocessingError(ValueError):
    """Raised when a Phase 7 feature frame cannot be split or preprocessed safely."""


@dataclass(frozen=True)
class ChronologicalSplit:
    """Disjoint, contiguous, complete chronological partitions plus their dates."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_dates: tuple[date, ...]
    validation_dates: tuple[date, ...]
    test_dates: tuple[date, ...]


@dataclass(frozen=True)
class PreprocessorFit:
    """Immutable train-only fitted state for one-hot encoding and scaling."""

    scaled_columns: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    province_categories: tuple[str, ...]
    road_type_categories: tuple[str, ...]

    @property
    def one_hot_columns(self) -> tuple[str, ...]:
        return tuple(f"province_{category}" for category in self.province_categories) + tuple(
            f"road_type_{category}" for category in self.road_type_categories
        )

    @property
    def transformed_feature_columns(self) -> tuple[str, ...]:
        return self.scaled_columns + self.one_hot_columns


@dataclass(frozen=True)
class TransformedData:
    """Partition keys kept separately from the transformed model features."""

    keys: pd.DataFrame
    features: pd.DataFrame


def split_chronologically(frame: pd.DataFrame, spec: DatasetSpec) -> ChronologicalSplit:
    """Split an exact Phase 7 feature frame into 34/7/7 unique-date partitions.

    Partitions are disjoint, contiguous, together reproduce the complete
    input, and are canonically sorted by (``segment_id``, ``date``). For the
    V1 profile the exact 10,200/2,100/2,100 row counts are enforced.
    """
    if type(spec) is not DatasetSpec:
        raise TypeError("spec must be a DatasetSpec")
    validated = _validated_frame(
        frame,
        spec,
        required_date_count=48,
        allow_unknown_categories=False,
    )
    unique_dates = sorted(validated["date"].dt.date.unique())
    train_dates = tuple(unique_dates[:TRAIN_DATE_COUNT])
    validation_dates = tuple(
        unique_dates[TRAIN_DATE_COUNT : TRAIN_DATE_COUNT + VALIDATION_DATE_COUNT]
    )
    test_dates = tuple(unique_dates[-TEST_DATE_COUNT:])
    train_mask = validated["date"].dt.date.isin(train_dates)
    validation_mask = validated["date"].dt.date.isin(validation_dates)
    test_mask = validated["date"].dt.date.isin(test_dates)
    train = (
        validated.loc[train_mask]
        .sort_values(list(FEATURE_KEY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )
    validation = (
        validated.loc[validation_mask]
        .sort_values(list(FEATURE_KEY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )
    test = (
        validated.loc[test_mask]
        .sort_values(list(FEATURE_KEY_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )
    if _is_v1(spec) and (
        len(train) != V1_TRAIN_ROWS
        or len(validation) != V1_VALIDATION_ROWS
        or len(test) != V1_TEST_ROWS
    ):
        raise PreprocessingError(
            f"V1 split row counts must be exactly {V1_TRAIN_ROWS}/"
            f"{V1_VALIDATION_ROWS}/{V1_TEST_ROWS}"
        )
    return ChronologicalSplit(train, validation, test, train_dates, validation_dates, test_dates)


def fit_preprocessor(split: ChronologicalSplit, spec: DatasetSpec) -> PreprocessorFit:
    """Fit one-hot categories and scaling statistics from the train partition only.

    The supplied split is reconstructed and checked against the locked
    chronological boundary before any statistic is learned. This prevents a
    caller from presenting an arbitrary future-contaminated 34-date frame as
    training data.
    """
    if type(spec) is not DatasetSpec:
        raise TypeError("spec must be a DatasetSpec")
    if type(split) is not ChronologicalSplit:
        raise TypeError("split must be a ChronologicalSplit")
    validated = _validated_training_partition(split, spec)
    scaled_columns = _scaled_feature_columns()
    raw = _raw_scaled_matrix(validated)
    means, stds = _fit_scaling_statistics(raw)
    fitted = PreprocessorFit(
        scaled_columns=scaled_columns,
        means=means,
        stds=stds,
        province_categories=tuple(sorted(validated["province"].unique())),
        road_type_categories=tuple(sorted(validated["road_type"].unique())),
    )
    return _validated_fit(fitted)


def transform(frame: pd.DataFrame, fit: PreprocessorFit) -> TransformedData:
    """Transform any frame using already-fitted train-only state.

    Never refits and never consults validation/test statistics. Unknown
    categories encode as all-zero rows without changing the fitted schema;
    zero-variance training columns transform to a constant zero.
    """
    validated_fit = _validated_fit(fit)
    validated = _validated_frame(
        frame,
        None,
        required_date_count=None,
        allow_unknown_categories=True,
    )
    keys = validated.loc[:, list(FEATURE_KEY_COLUMNS)].copy()
    raw = _raw_scaled_matrix(validated)
    features = pd.DataFrame(index=validated.index)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for position, column in enumerate(validated_fit.scaled_columns):
            scale = validated_fit.stds[position] if validated_fit.stds[position] > 0 else 1.0
            features[column] = (raw[:, position] - validated_fit.means[position]) / scale
    for column, categories in (
        ("province", validated_fit.province_categories),
        ("road_type", validated_fit.road_type_categories),
    ):
        for category in categories:
            features[f"{column}_{category}"] = (validated[column] == category).astype("float64")
    features = features.loc[:, list(validated_fit.transformed_feature_columns)]
    if not np.isfinite(features.to_numpy(dtype="float64")).all():
        raise PreprocessingError("transformation produced non-finite feature values")
    return TransformedData(keys=keys, features=features)


def _validated_training_partition(
    split: ChronologicalSplit,
    spec: DatasetSpec,
) -> pd.DataFrame:
    combined = pd.concat(
        [split.train, split.validation, split.test],
        ignore_index=True,
        copy=True,
    )
    canonical = split_chronologically(combined, spec)
    supplied_partitions = (split.train, split.validation, split.test)
    canonical_partitions = (canonical.train, canonical.validation, canonical.test)
    for supplied, expected in zip(supplied_partitions, canonical_partitions, strict=True):
        supplied_keys = (
            supplied.loc[:, list(FEATURE_KEY_COLUMNS)]
            .sort_values(list(FEATURE_KEY_COLUMNS), kind="stable")
            .reset_index(drop=True)
        )
        if not supplied_keys.equals(expected.loc[:, list(FEATURE_KEY_COLUMNS)]):
            raise PreprocessingError(
                "ChronologicalSplit partition membership does not match the locked 34/7/7 boundary"
            )
    if (
        split.train_dates != canonical.train_dates
        or split.validation_dates != canonical.validation_dates
        or split.test_dates != canonical.test_dates
    ):
        raise PreprocessingError("ChronologicalSplit date provenance does not match its frames")
    return canonical.train


def _validated_fit(fit: PreprocessorFit) -> PreprocessorFit:
    if type(fit) is not PreprocessorFit:
        raise TypeError("fit must be a PreprocessorFit")
    expected_scaled = _scaled_feature_columns()
    tuple_fields = (
        fit.scaled_columns,
        fit.means,
        fit.stds,
        fit.province_categories,
        fit.road_type_categories,
    )
    if any(type(value) is not tuple for value in tuple_fields):
        raise PreprocessingError("fitted state fields must be immutable tuples")
    if fit.scaled_columns != expected_scaled:
        raise PreprocessingError("fitted scaled-column schema does not match Phase 8")
    if len(fit.means) != len(expected_scaled) or len(fit.stds) != len(expected_scaled):
        raise PreprocessingError("fitted scaling statistics do not match the feature schema")
    if any(type(value) is not float for value in fit.means + fit.stds):
        raise PreprocessingError("fitted scaling statistics must contain exact float values")
    statistics = np.asarray(fit.means + fit.stds, dtype="float64")
    if not np.isfinite(statistics).all() or any(value < 0 for value in fit.stds):
        raise PreprocessingError("fitted scaling statistics must be finite and non-negative")
    for name, categories in (
        ("province", fit.province_categories),
        ("road_type", fit.road_type_categories),
    ):
        if (
            not categories
            or categories != tuple(sorted(set(categories)))
            or any(type(value) is not str or not value for value in categories)
        ):
            raise PreprocessingError(f"fitted {name} categories must be sorted unique strings")
    if not set(fit.province_categories).issubset(PROVINCES) or not set(
        fit.road_type_categories
    ).issubset(ROAD_TYPES):
        raise PreprocessingError("fitted categories must come from the locked Phase 7 registries")
    if len(set(fit.transformed_feature_columns)) != len(fit.transformed_feature_columns):
        raise PreprocessingError("fitted transformed feature names must be unique")
    return fit


def _scaled_feature_columns() -> tuple[str, ...]:
    return tuple(
        column
        for column in FEATURE_COLUMNS
        if column not in _CATEGORICAL_FEATURE_COLUMNS + _DATETIME_FEATURE_COLUMNS
    ) + (CONSTRUCTION_DATE_DAY_COLUMN,)


def _construction_date_days(frame: pd.DataFrame) -> np.ndarray:
    return np.array(
        [(value.date() - DAY_EPOCH).days for value in frame["construction_date"]],
        dtype="float64",
    )


def _raw_scaled_matrix(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.loc[:, list(_NUMERIC_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    return np.column_stack([numeric, _construction_date_days(frame)])


def _fit_scaling_statistics(raw: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    means: list[float] = []
    stds: list[float] = []
    with np.errstate(over="ignore", invalid="ignore"):
        for column in raw.T:
            if np.equal(column, column[0]).all():
                means.append(float(column[0]))
                stds.append(0.0)
            else:
                means.append(float(column.mean()))
                stds.append(float(column.std()))
    return tuple(means), tuple(stds)


def _is_v1(spec: DatasetSpec) -> bool:
    return (
        spec.dataset_segments == 300
        and spec.dataset_months_per_segment == 48
        and spec.dataset_observations == 14_400
    )


def _validated_frame(
    frame: pd.DataFrame,
    spec: DatasetSpec | None,
    required_date_count: int | None,
    *,
    allow_unknown_categories: bool,
) -> pd.DataFrame:
    if type(frame) is not pd.DataFrame:
        raise PreprocessingError("frame must be a pandas DataFrame")
    if tuple(frame.columns) != FEATURE_FRAME_COLUMNS:
        raise PreprocessingError(
            f"frame columns must be exactly {FEATURE_FRAME_COLUMNS}, found {tuple(frame.columns)}"
        )
    validated = frame.copy(deep=True)

    _check_dtype(validated, "segment_id", "object")
    _check_dtype(validated, "province", "object")
    _check_dtype(validated, "road_type", "object")
    _check_dtype(validated, "date", "datetime64")
    _check_dtype(validated, "construction_date", "datetime64")
    for column in _INT_FEATURE_COLUMNS:
        _check_dtype(validated, column, "int64")
    for column in _FLOAT_FEATURE_COLUMNS:
        _check_dtype(validated, column, "float64")

    if validated.duplicated(subset=list(FEATURE_KEY_COLUMNS)).any():
        raise PreprocessingError("frame contains duplicate natural keys (segment_id, date)")
    if validated.isna().any().any():
        raise PreprocessingError("frame contains null or missing values")
    for column in ("segment_id", "province", "road_type"):
        for value in validated[column]:
            if type(value) is not str:
                raise PreprocessingError(f"column {column} must contain only exact string values")
    for column in _FLOAT_FEATURE_COLUMNS:
        if not np.isfinite(validated[column].to_numpy(dtype="float64")).all():
            raise PreprocessingError(f"column {column} contains non-finite values")
    for column in ("date", "construction_date"):
        series = validated[column]
        if series.dt.tz is not None:
            raise PreprocessingError(f"column {column} must be timezone-naive")
        components = (
            series.dt.hour
            + series.dt.minute
            + series.dt.second
            + series.dt.microsecond
            + series.dt.nanosecond
        )
        if (components != 0).any():
            raise PreprocessingError(f"column {column} must contain midnight dates")

    _validate_feature_domains(validated, allow_unknown_categories=allow_unknown_categories)

    if required_date_count is not None:
        if spec is None:
            raise AssertionError("grid validation requires a spec")
        if spec.dataset_months_per_segment != required_date_count:
            raise PreprocessingError(
                f"Phase 8 requires a {required_date_count}-month source specification, "
                f"found {spec.dataset_months_per_segment}"
            )
        n_dates = validated["date"].nunique()
        n_segments = validated["segment_id"].nunique()
        if n_dates != required_date_count:
            raise PreprocessingError(
                f"frame must contain exactly {required_date_count} unique "
                f"observation dates, found {n_dates}"
            )
        if n_segments != spec.dataset_segments:
            raise PreprocessingError(
                f"frame must contain exactly {spec.dataset_segments} segments, found {n_segments}"
            )
        expected_rows = required_date_count * spec.dataset_segments
        if len(validated) != expected_rows:
            raise PreprocessingError(
                f"frame must contain exactly {expected_rows} rows, found {len(validated)}"
            )
        per_date = validated.groupby("date")["segment_id"].nunique()
        if (per_date != n_segments).any():
            raise PreprocessingError(
                "frame has an incomplete observation grid: some dates are missing segments"
            )
        unique_dates = pd.DatetimeIndex(sorted(validated["date"].unique()))
        if any(value.day != 1 for value in unique_dates):
            raise PreprocessingError("observation dates must be calendar month starts")
        expected_dates = pd.date_range(unique_dates[0], periods=required_date_count, freq="MS")
        if not unique_dates.equals(expected_dates):
            raise PreprocessingError("observation dates must form a complete monthly calendar")
    return validated


def _validate_feature_domains(
    frame: pd.DataFrame,
    *,
    allow_unknown_categories: bool,
) -> None:
    for column in ("segment_id", "province", "road_type"):
        if frame[column].str.len().eq(0).any():
            raise PreprocessingError(f"column {column} must not contain empty strings")
    if frame["segment_id"].map(lambda value: re.fullmatch(SEGMENT_ID_PATTERN, value) is None).any():
        raise PreprocessingError("segment_id contains a value outside the Phase 7 pattern")
    if not allow_unknown_categories:
        if not set(frame["province"]).issubset(PROVINCES):
            raise PreprocessingError("province contains a value outside the Phase 7 registry")
        if not set(frame["road_type"]).issubset(ROAD_TYPES):
            raise PreprocessingError("road_type contains a value outside the Phase 7 registry")
    static_columns = ("province", "road_type", "construction_date", "road_length_km")
    static_counts = frame.groupby("segment_id", sort=False)[list(static_columns)].nunique()
    if static_counts.ne(1).any().any():
        raise PreprocessingError("static segment attributes must be invariant by segment_id")
    if frame["date"].dt.day.ne(1).any():
        raise PreprocessingError("observation dates must be calendar month starts")
    if (frame["construction_date"] > frame["date"]).any():
        raise PreprocessingError("construction_date must not be after the observation date")
    expected_age = (frame["date"] - frame["construction_date"]).dt.days
    if not frame["road_age_days"].equals(expected_age.astype("int64")):
        raise PreprocessingError("road_age_days must equal date minus construction_date")
    if (frame["road_length_km"] <= 0).any():
        raise PreprocessingError("road_length_km must be positive")
    if (frame["traffic_volume"] < 0).any():
        raise PreprocessingError("traffic_volume must be non-negative")
    if not frame["heavy_vehicle_ratio"].between(0.0, 1.0).all():
        raise PreprocessingError("heavy_vehicle_ratio must be between 0 and 1")
    if (frame["rainfall_mm"] < 0).any():
        raise PreprocessingError("rainfall_mm must be non-negative")
    if not frame["temperature"].between(-50.0, 60.0).all():
        raise PreprocessingError("temperature must be between -50 and 60")
    if not frame["humidity"].between(0.0, 100.0).all():
        raise PreprocessingError("humidity must be between 0 and 100")
    if (frame["days_since_last_maintenance"] < 0).any() or (
        frame["days_since_last_maintenance"] > frame["road_age_days"]
    ).any():
        raise PreprocessingError("days_since_last_maintenance is outside the valid road age")
    if (frame["previous_repairs"] < 0).any():
        raise PreprocessingError("previous_repairs must be non-negative")
    for column in (
        "road_condition_score",
        "marking_condition_score",
        "guardrail_condition_score",
        "sign_condition_score",
    ):
        if not frame[column].between(1, 100).all():
            raise PreprocessingError(f"{column} must be between 1 and 100")
    if (frame["accident_count_30d"] < 0).any() or (
        frame["accident_count_365d"] < frame["accident_count_30d"]
    ).any():
        raise PreprocessingError("accident counts must satisfy 0 <= 30d <= 365d")


def _check_dtype(frame: pd.DataFrame, column: str, expected: str) -> None:
    dtype = str(frame[column].dtype)
    if expected == "datetime64" and isinstance(frame[column].dtype, pd.DatetimeTZDtype):
        raise PreprocessingError(f"column {column} must be timezone-naive")
    valid = dtype == "datetime64[ns]" if expected == "datetime64" else dtype == expected
    if not valid:
        raise PreprocessingError(
            f"feature column {column} has invalid dtype {dtype}, expected {expected}"
        )


__all__ = [
    "CONSTRUCTION_DATE_DAY_COLUMN",
    "TEST_DATE_COUNT",
    "TRAIN_DATE_COUNT",
    "VALIDATION_DATE_COUNT",
    "V1_TEST_ROWS",
    "V1_TRAIN_ROWS",
    "V1_VALIDATION_ROWS",
    "ChronologicalSplit",
    "PreprocessingError",
    "PreprocessorFit",
    "TransformedData",
    "fit_preprocessor",
    "split_chronologically",
    "transform",
]
