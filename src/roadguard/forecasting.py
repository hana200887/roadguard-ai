"""Deterministic rolling-origin material forecasts for Phase 14.

The module intentionally accepts only already-authenticated in-memory data.
It does not load configuration, artifacts, or database state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Final

import pandas as pd

from roadguard._db_models import MAINTENANCE_HISTORY_COLUMNS
from roadguard._db_types import RepositoryExport
from roadguard.contracts import V1_OBSERVATION_START, DatasetSpec
from roadguard.data_quality import validate_cleaned_dataset
from roadguard.events import EVENT_COLUMNS, observation_dates
from roadguard.segments import SEGMENT_ID_PATTERN

MATERIAL_FORECAST_CONTRACT_VERSION: Final[str] = "roadguard.phase14.v1"
FORECAST_MATERIAL_NAMES: Final[tuple[str, ...]] = (
    "thermoplastic_paint_kg",
    "reflective_sheet_m2",
    "guardrail_meter",
    "traffic_sign_quantity",
)
FORECAST_CANDIDATE_NAMES: Final[tuple[str, ...]] = (
    "seasonal_naive_12",
    "trailing_mean_3",
)
INITIAL_TRAIN_MONTHS: Final[int] = 24
FROZEN_TEST_ORIGIN_COUNT: Final[int] = 7
FORECAST_HORIZON_MONTHS: Final[int] = 1

_INPUT_ERROR: Final[str] = "Phase 14 input validation failed."
_EVALUATION_ERROR: Final[str] = "Phase 14 rolling-origin evaluation failed."
_OUTPUT_ERROR: Final[str] = "Phase 14 forecast output failed."
_SEGMENT_PATTERN: Final[re.Pattern[str]] = re.compile(SEGMENT_ID_PATTERN)


class MaterialForecastError(ValueError):
    """Raised when a locked Phase 14 material-forecast contract is violated."""


@dataclass(frozen=True)
class ForecastCandidateMetrics:
    """Validation metrics for one locked forecasting candidate."""

    candidate_name: str
    validation_mae: float
    validation_rmse: float


@dataclass(frozen=True)
class MaterialForecastMetrics:
    """Frozen validation choice and frozen-test metrics for one material."""

    material: str
    candidates: tuple[ForecastCandidateMetrics, ...]
    selected_candidate_name: str
    test_mae: float
    test_rmse: float


@dataclass(frozen=True)
class MaterialForecast:
    """One next-month network material forecast."""

    period: date
    material: str
    forecast_quantity: float


@dataclass(frozen=True)
class MaterialForecastEvaluation:
    """Immutable evidence from the complete deterministic forecast workflow."""

    contract_version: str
    forecast_input_fingerprint: str
    history_start: date
    history_end: date
    forecast_period: date
    validation_origins: tuple[date, ...]
    test_origins: tuple[date, ...]
    material_metrics: tuple[MaterialForecastMetrics, ...]
    forecasts: tuple[MaterialForecast, ...]


def forecast_materials(
    dataset: RepositoryExport,
    maintenance_history: pd.DataFrame,
    spec: DatasetSpec,
) -> MaterialForecastEvaluation:
    """Forecast each locked material using expanding one-step rolling origins."""
    if (
        type(dataset) is not RepositoryExport
        or type(maintenance_history) is not pd.DataFrame
        or type(spec) is not DatasetSpec
    ):
        raise TypeError("Phase 14 requires exact public input types")

    try:
        prepared = _prepare_inputs(dataset, maintenance_history, spec)
        forecast_input_fingerprint = _forecast_fingerprint(prepared)
    except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError):
        raise MaterialForecastError(_INPUT_ERROR) from None

    try:
        material_metrics = _evaluate_materials(prepared)
    except (ArithmeticError, TypeError, ValueError):
        raise MaterialForecastError(_EVALUATION_ERROR) from None

    try:
        forecasts = _forecast_next_month(prepared, material_metrics)
        return MaterialForecastEvaluation(
            contract_version=MATERIAL_FORECAST_CONTRACT_VERSION,
            forecast_input_fingerprint=forecast_input_fingerprint,
            history_start=prepared.months[0],
            history_end=prepared.months[-1],
            forecast_period=_next_month(prepared.months[-1]),
            validation_origins=prepared.validation_origins,
            test_origins=prepared.test_origins,
            material_metrics=material_metrics,
            forecasts=forecasts,
        )
    except (ArithmeticError, TypeError, ValueError):
        raise MaterialForecastError(_OUTPUT_ERROR) from None


@dataclass(frozen=True)
class _PreparedInputs:
    spec: DatasetSpec
    months: tuple[date, ...]
    validation_origins: tuple[date, ...]
    test_origins: tuple[date, ...]
    series_by_material: tuple[tuple[float, ...], ...]
    canonical_rows: tuple[tuple[date, str, float], ...]


def _prepare_inputs(
    dataset: RepositoryExport,
    maintenance_history: pd.DataFrame,
    spec: DatasetSpec,
) -> _PreparedInputs:
    frames = (
        dataset.segments,
        dataset.observations,
        dataset.targets,
        dataset.maintenance_events,
    )
    if any(type(frame) is not pd.DataFrame for frame in frames):
        raise TypeError("repository export frames must be exact DataFrames")
    _require_columns(dataset.maintenance_events, EVENT_COLUMNS)
    _require_columns(maintenance_history, MAINTENANCE_HISTORY_COLUMNS)

    validated_spec = DatasetSpec(**spec.model_dump())
    if validated_spec.dataset_months_per_segment < (
        INITIAL_TRAIN_MONTHS + FROZEN_TEST_ORIGIN_COUNT + 1
    ):
        raise ValueError("insufficient forecast history")

    copied_frames = tuple(frame.copy(deep=True) for frame in frames)
    copied_history = maintenance_history.copy(deep=True)
    report = validate_cleaned_dataset(
        copied_frames[0],
        copied_frames[1],
        copied_frames[2],
        copied_frames[3],
        validated_spec,
    )
    if not report.is_valid:
        raise ValueError("cleaned dataset validation failed")

    months = tuple(
        timestamp.date()
        for timestamp in observation_dates(
            validated_spec.dataset_months_per_segment,
            V1_OBSERVATION_START,
        )
    )
    observed_months = tuple(
        sorted({_as_date(value) for value in copied_frames[1]["date"].tolist()})
    )
    if observed_months != months:
        raise ValueError("observation calendar is not the locked V1 calendar")

    event_keys = _event_keys(copied_frames[3])
    material_rows = _validated_history_rows(copied_history, event_keys)
    canonical_rows = _canonical_history(material_rows, event_keys, months)
    first_test_index = len(months) - FROZEN_TEST_ORIGIN_COUNT
    return _PreparedInputs(
        spec=validated_spec,
        months=months,
        validation_origins=months[INITIAL_TRAIN_MONTHS:first_test_index],
        test_origins=months[first_test_index:],
        series_by_material=tuple(
            tuple(
                quantity
                for _period, row_material, quantity in canonical_rows
                if row_material == material
            )
            for material in FORECAST_MATERIAL_NAMES
        ),
        canonical_rows=canonical_rows,
    )


def _require_columns(frame: pd.DataFrame, expected: tuple[str, ...]) -> None:
    columns = tuple(frame.columns)
    if any(type(column) is not str for column in columns) or columns != expected:
        raise ValueError("frame schema is invalid")


def _event_keys(events: pd.DataFrame) -> frozenset[tuple[str, date]]:
    keys: set[tuple[str, date]] = set()
    for segment_id, maintenance_date in events.itertuples(index=False, name=None):
        key = (_validated_segment_id(segment_id), _as_date(maintenance_date))
        if key in keys:
            raise ValueError("duplicate maintenance event")
        keys.add(key)
    return frozenset(keys)


def _validated_history_rows(
    history: pd.DataFrame,
    event_keys: frozenset[tuple[str, date]],
) -> tuple[tuple[str, date, tuple[float, ...]], ...]:
    records: list[tuple[str, date, tuple[float, ...]]] = []
    history_keys: set[tuple[str, date]] = set()
    for row in history.itertuples(index=False, name=None):
        segment_id = _validated_segment_id(row[0])
        maintenance_date = _as_date(row[1])
        key = (segment_id, maintenance_date)
        if key in history_keys or key not in event_keys:
            raise ValueError("maintenance history key is invalid")
        history_keys.add(key)
        _positive_builtin_int(row[2])
        quantities = (
            _nonnegative_builtin_float(row[3]),
            _nonnegative_builtin_float(row[4]),
            _nonnegative_builtin_float(row[5]),
            float(_nonnegative_builtin_int(row[6])),
        )
        records.append((segment_id, maintenance_date, quantities))

    if not history_keys.issubset(event_keys):
        raise ValueError("maintenance history does not match events")
    return tuple(records)


def _canonical_history(
    records: tuple[tuple[str, date, tuple[float, ...]], ...],
    event_keys: frozenset[tuple[str, date]],
    months: tuple[date, ...],
) -> tuple[tuple[date, str, float], ...]:
    history_start, history_end = months[0], months[-1]
    in_window_event_keys = {
        key for key in event_keys if history_start <= _month_start(key[1]) <= history_end
    }
    in_window_history_keys = {
        (segment_id, maintenance_date)
        for segment_id, maintenance_date, _quantities in records
        if history_start <= _month_start(maintenance_date) <= history_end
    }
    if in_window_history_keys != in_window_event_keys:
        raise ValueError("in-window maintenance history is incomplete")

    grouped: dict[tuple[date, str], list[float]] = {}
    for _segment_id, maintenance_date, quantities in sorted(
        records, key=lambda item: (item[0], item[1])
    ):
        if history_start <= _month_start(maintenance_date) <= history_end:
            period = _month_start(maintenance_date)
            for material, quantity in zip(FORECAST_MATERIAL_NAMES, quantities, strict=True):
                grouped.setdefault((period, material), []).append(quantity)

    rows: list[tuple[date, str, float]] = []
    for period in months:
        for material in FORECAST_MATERIAL_NAMES:
            quantity = float(math.fsum(grouped.get((period, material), ())))
            rows.append((period, material, _normalized_nonnegative_float(quantity)))
    return tuple(rows)


def _evaluate_materials(prepared: _PreparedInputs) -> tuple[MaterialForecastMetrics, ...]:
    test_start = len(prepared.months) - FROZEN_TEST_ORIGIN_COUNT
    validation_records: list[
        tuple[str, tuple[float, ...], tuple[ForecastCandidateMetrics, ...], str]
    ] = []
    for material, values in zip(FORECAST_MATERIAL_NAMES, prepared.series_by_material, strict=True):
        candidate_metrics: list[ForecastCandidateMetrics] = []
        for candidate_name in FORECAST_CANDIDATE_NAMES:
            predictions = tuple(
                _candidate_output(_candidate_forecast(candidate_name, values[:index]))
                for index in range(INITIAL_TRAIN_MONTHS, test_start)
            )
            mae, rmse = _metrics(values[INITIAL_TRAIN_MONTHS:test_start], predictions)
            candidate_metrics.append(ForecastCandidateMetrics(candidate_name, mae, rmse))
        selected_index = min(
            range(len(candidate_metrics)),
            key=lambda index: (
                candidate_metrics[index].validation_mae,
                candidate_metrics[index].validation_rmse,
                index,
            ),
        )
        candidate_metrics_tuple = tuple(candidate_metrics)
        validation_records.append(
            (
                material,
                values,
                candidate_metrics_tuple,
                candidate_metrics_tuple[selected_index].candidate_name,
            )
        )

    metrics: list[MaterialForecastMetrics] = []
    for material, values, candidate_records, selected_name in validation_records:
        test_predictions = tuple(
            _candidate_output(_candidate_forecast(selected_name, values[:index]))
            for index in range(test_start, len(values))
        )
        test_mae, test_rmse = _metrics(values[test_start:], test_predictions)
        metrics.append(
            MaterialForecastMetrics(
                material=material,
                candidates=candidate_records,
                selected_candidate_name=selected_name,
                test_mae=test_mae,
                test_rmse=test_rmse,
            )
        )
    return tuple(metrics)


def _forecast_next_month(
    prepared: _PreparedInputs,
    metrics: tuple[MaterialForecastMetrics, ...],
) -> tuple[MaterialForecast, ...]:
    forecast_period = _next_month(prepared.months[-1])
    return tuple(
        MaterialForecast(
            period=forecast_period,
            material=material,
            forecast_quantity=_candidate_output(
                _candidate_forecast(metric.selected_candidate_name, values)
            ),
        )
        for material, values, metric in zip(
            FORECAST_MATERIAL_NAMES,
            prepared.series_by_material,
            metrics,
            strict=True,
        )
    )


def _candidate_forecast(candidate_name: str, prefix: tuple[float, ...]) -> float:
    """Return one locked candidate prediction through the monkeypatchable seam."""
    if candidate_name == "seasonal_naive_12":
        return _normalized_nonnegative_float(float(prefix[-12]))
    if candidate_name == "trailing_mean_3":
        return _normalized_nonnegative_float(float(math.fsum(prefix[-3:]) / 3.0))
    raise ValueError("unknown forecast candidate")


def _metrics(actual: tuple[float, ...], predicted: tuple[float, ...]) -> tuple[float, float]:
    if not actual or len(actual) != len(predicted):
        raise ValueError("metric vectors are invalid")
    mae = math.fsum(
        abs(value - estimate) for value, estimate in zip(actual, predicted, strict=True)
    )
    rmse = math.sqrt(
        math.fsum(
            (value - estimate) ** 2 for value, estimate in zip(actual, predicted, strict=True)
        )
        / len(actual)
    )
    return _normalized_nonnegative_float(float(mae / len(actual))), _normalized_nonnegative_float(
        float(rmse)
    )


def _forecast_fingerprint(prepared: _PreparedInputs) -> str:
    payload = {
        "candidates": [
            {"lag_months": 12, "name": "seasonal_naive_12"},
            {"name": "trailing_mean_3", "window_months": 3},
        ],
        "columns": ["period", "material", "quantity"],
        "contract": MATERIAL_FORECAST_CONTRACT_VERSION,
        "forecast_horizon_months": FORECAST_HORIZON_MONTHS,
        "history_end": prepared.months[-1].isoformat(),
        "history_rows": [
            [period.isoformat(), material, quantity.hex()]
            for period, material, quantity in prepared.canonical_rows
        ],
        "history_start": prepared.months[0].isoformat(),
        "initial_train_months": INITIAL_TRAIN_MONTHS,
        "materials": list(FORECAST_MATERIAL_NAMES),
        "spec": {
            "dataset_months_per_segment": prepared.spec.dataset_months_per_segment,
            "dataset_observations": prepared.spec.dataset_observations,
            "dataset_segments": prepared.spec.dataset_segments,
        },
        "test_origins": [period.isoformat() for period in prepared.test_origins],
        "validation_origins": [period.isoformat() for period in prepared.validation_origins],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _validated_segment_id(value: object) -> str:
    if type(value) is not str or not value.isascii() or _SEGMENT_PATTERN.fullmatch(value) is None:
        raise ValueError("segment id is invalid")
    return value


def _as_date(value: object) -> date:
    if type(value) is date:
        return value
    if type(value) is pd.Timestamp and value.tz is None and value == value.normalize():
        return value.date()
    raise ValueError("date scalar is invalid")


def _positive_builtin_int(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError("positive integer scalar is invalid")
    return value


def _nonnegative_builtin_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("non-negative integer scalar is invalid")
    return value


def _nonnegative_builtin_float(value: object) -> float:
    if not isinstance(value, (int, float)) or type(value) not in (int, float):
        raise ValueError("non-negative finite scalar is invalid")
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0:
        raise ValueError("non-negative finite scalar is invalid")
    return _normalized_nonnegative_float(scalar)


def _candidate_output(value: object) -> float:
    if type(value) is not float:
        raise ValueError("candidate output is invalid")
    return _normalized_nonnegative_float(value)


def _normalized_nonnegative_float(value: float) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise ValueError("non-negative finite float is invalid")
    return 0.0 if value == 0.0 else value


def _next_month(value: date) -> date:
    return date(value.year + (value.month == 12), value.month % 12 + 1, 1)


def _month_start(value: date) -> date:
    return value.replace(day=1)


__all__ = [
    "forecast_materials",
    "MaterialForecastError",
    "ForecastCandidateMetrics",
    "MaterialForecastMetrics",
    "MaterialForecast",
    "MaterialForecastEvaluation",
    "MATERIAL_FORECAST_CONTRACT_VERSION",
    "FORECAST_MATERIAL_NAMES",
    "FORECAST_CANDIDATE_NAMES",
    "INITIAL_TRAIN_MONTHS",
    "FROZEN_TEST_ORIGIN_COUNT",
    "FORECAST_HORIZON_MONTHS",
]
