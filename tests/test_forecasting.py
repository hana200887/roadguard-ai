"""Phase 14 RED-first material-forecasting contract tests.

The production module is deliberately absent while this file is introduced.
These tests freeze the public surface and the temporal, numerical, provenance,
and isolation guarantees in ``docs/contracts.md`` section 21.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import inspect
import json
import math
import os
import random
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import roadguard.forecasting as forecasting
from pandas.testing import assert_frame_equal
from roadguard.forecasting import (
    FORECAST_CANDIDATE_NAMES,
    FORECAST_HORIZON_MONTHS,
    FORECAST_MATERIAL_NAMES,
    FROZEN_TEST_ORIGIN_COUNT,
    INITIAL_TRAIN_MONTHS,
    MATERIAL_FORECAST_CONTRACT_VERSION,
    ForecastCandidateMetrics,
    MaterialForecast,
    MaterialForecastError,
    MaterialForecastEvaluation,
    MaterialForecastMetrics,
    forecast_materials,
)

import roadguard
import roadguard._artifact_io as artifact_io
import roadguard.artifacts as artifacts
import roadguard.config as config
import roadguard.database as database
import roadguard.preprocessing as preprocessing
from roadguard import (
    DatasetSpec,
    RepositoryExport,
    clean_raw_dataset,
    derive_observation_targets,
    generate_accident_timeline,
    generate_observations,
    generate_segments,
    observation_dates,
)
from roadguard._db_models import MAINTENANCE_HISTORY_COLUMNS
from roadguard.contracts import V1_OBSERVATION_START
from roadguard.events import EVENT_COLUMNS

SPEC = DatasetSpec(dataset_segments=3, dataset_months_per_segment=48, dataset_observations=144)
PUBLIC_NAMES = (
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
)
ALLOWED_INPUT_ERROR = "Phase 14 input validation failed."
ALLOWED_EVALUATION_ERROR = "Phase 14 rolling-origin evaluation failed."
ALLOWED_OUTPUT_ERROR = "Phase 14 forecast output failed."


def _next_month(value: date) -> date:
    return date(value.year + (value.month == 12), value.month % 12 + 1, 1)


def _history_frame(
    dataset: RepositoryExport,
    spec: DatasetSpec = SPEC,
    *,
    zero_day_20: bool = False,
) -> pd.DataFrame:
    """Build complete realized facts, including one deliberately ignored future row."""
    rows: list[dict[str, object]] = []
    ordered_events = dataset.maintenance_events.sort_values(
        ["segment_id", "maintenance_date"], kind="stable"
    ).reset_index(drop=True)
    segment_rank = {
        segment_id: rank
        for rank, segment_id in enumerate(sorted(dataset.segments["segment_id"].tolist()))
    }
    for _, event in ordered_events.iterrows():
        segment_id = cast(str, event["segment_id"])
        event_date = cast(pd.Timestamp, event["maintenance_date"]).date()
        rank = segment_rank[segment_id]
        period_index = (event_date.year - V1_OBSERVATION_START.year) * 12 + (
            event_date.month - V1_OBSERVATION_START.month
        )
        quantities: dict[str, int | float] = {
            # The first material proves canonical math.fsum rather than sum.
            "thermoplastic_paint_kg": (1e16, 1.0, 1.0)[rank],
            "reflective_sheet_m2": float(max(period_index, 0) + rank),
            "guardrail_meter": float((max(period_index, 0) + rank) % 5),
            "traffic_sign_quantity": rank + 1,
        }
        if zero_day_20 and event_date.day == 20:
            quantities = {material: 0 for material in FORECAST_MATERIAL_NAMES}
        rows.append(
            {
                "segment_id": segment_id,
                "maintenance_date": pd.Timestamp(event_date),
                "maintenance_cost": 1_000_000 + rank + max(period_index, 0),
                **quantities,
            }
        )
    history = pd.DataFrame(rows, columns=MAINTENANCE_HISTORY_COLUMNS)
    history["segment_id"] = history["segment_id"].astype(object)
    history["maintenance_date"] = pd.to_datetime(history["maintenance_date"])
    history["maintenance_cost"] = history["maintenance_cost"].astype("int64")
    for material in FORECAST_MATERIAL_NAMES[:-1]:
        history[material] = history[material].astype("float64")
    history["traffic_sign_quantity"] = history["traffic_sign_quantity"].astype("int64")
    return history


def _build_dataset(
    spec: DatasetSpec,
    *,
    event_day: int = 15,
    add_zero_quantity_event: bool = False,
) -> RepositoryExport:
    segments = generate_segments(spec, 42, observation_start=V1_OBSERVATION_START)
    observation_months = observation_dates(spec.dataset_months_per_segment, V1_OBSERVATION_START)
    event_rows = [
        {"segment_id": segment_id, "maintenance_date": month.replace(day=event_day)}
        for segment_id in segments["segment_id"].tolist()
        for month in observation_months
    ]
    # A valid row after the forecast window must be authenticated but ignored.
    event_rows.append(
        {
            "segment_id": cast(str, segments.iloc[0]["segment_id"]),
            "maintenance_date": _next_month(observation_months[-1]).replace(day=event_day),
        }
    )
    if add_zero_quantity_event:
        event_rows.append(
            {
                "segment_id": cast(str, segments.iloc[0]["segment_id"]),
                "maintenance_date": observation_months[12].replace(day=20),
            }
        )
    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    events["maintenance_date"] = pd.to_datetime(events["maintenance_date"])
    accidents = generate_accident_timeline(segments, spec, 42, start_date=V1_OBSERVATION_START)
    observations = generate_observations(
        segments,
        events,
        accidents,
        spec,
        42,
        start_date=V1_OBSERVATION_START,
    )
    targets = derive_observation_targets(observations, events)
    cleaned = clean_raw_dataset(segments, observations, targets, events, spec)
    return RepositoryExport(
        segments=cleaned.segments,
        observations=cleaned.observations,
        targets=cleaned.targets,
        maintenance_events=cleaned.maintenance_events,
    )


@pytest.fixture(scope="module")
def dataset() -> RepositoryExport:
    return _build_dataset(SPEC)


@pytest.fixture(scope="module")
def history(dataset: RepositoryExport) -> pd.DataFrame:
    return _history_frame(dataset)


def _canonical_history(
    dataset: RepositoryExport, history: pd.DataFrame, spec: DatasetSpec = SPEC
) -> tuple[tuple[date, str, float], ...]:
    months = observation_dates(spec.dataset_months_per_segment, V1_OBSERVATION_START)
    rows: list[tuple[date, str, float]] = []
    ordered = history.sort_values(["segment_id", "maintenance_date"], kind="stable")
    for month in months:
        active = ordered.loc[
            ordered["maintenance_date"].dt.date.map(
                lambda value, active_month=month: (
                    value.year == active_month.year and value.month == active_month.month
                )
            )
        ]
        for material in FORECAST_MATERIAL_NAMES:
            value = float(math.fsum(float(item) for item in active[material].tolist()))
            rows.append((month, material, 0.0 if value == 0.0 else value))
    return tuple(rows)


def _series(rows: tuple[tuple[date, str, float], ...], material: str) -> tuple[float, ...]:
    return tuple(quantity for _period, item, quantity in rows if item == material)


def _metrics(actual: tuple[float, ...], predicted: tuple[float, ...]) -> tuple[float, float]:
    mae = math.fsum(
        abs(value - estimate) for value, estimate in zip(actual, predicted, strict=True)
    )
    rmse = math.sqrt(
        math.fsum(
            (value - estimate) ** 2 for value, estimate in zip(actual, predicted, strict=True)
        )
        / len(actual)
    )
    return float(mae / len(actual)), float(rmse)


def _expected_fingerprint(
    rows: tuple[tuple[date, str, float], ...], spec: DatasetSpec = SPEC
) -> str:
    months = observation_dates(spec.dataset_months_per_segment, V1_OBSERVATION_START)
    validation = months[INITIAL_TRAIN_MONTHS:-FROZEN_TEST_ORIGIN_COUNT]
    test = months[-FROZEN_TEST_ORIGIN_COUNT:]
    payload = {
        "candidates": [
            {"lag_months": 12, "name": "seasonal_naive_12"},
            {"name": "trailing_mean_3", "window_months": 3},
        ],
        "columns": ["period", "material", "quantity"],
        "contract": MATERIAL_FORECAST_CONTRACT_VERSION,
        "forecast_horizon_months": FORECAST_HORIZON_MONTHS,
        "history_end": months[-1].isoformat(),
        "history_rows": [
            [period.isoformat(), material, quantity.hex()] for period, material, quantity in rows
        ],
        "history_start": months[0].isoformat(),
        "initial_train_months": INITIAL_TRAIN_MONTHS,
        "materials": list(FORECAST_MATERIAL_NAMES),
        "spec": {
            "dataset_months_per_segment": spec.dataset_months_per_segment,
            "dataset_observations": spec.dataset_observations,
            "dataset_segments": spec.dataset_segments,
        },
        "test_origins": [item.isoformat() for item in test],
        "validation_origins": [item.isoformat() for item in validation],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _renamed_export(dataset: RepositoryExport, spec: DatasetSpec = SPEC) -> RepositoryExport:
    mapping = {
        segment_id: f"QL99-{segment_id.split('-', maxsplit=1)[1]}"
        for segment_id in dataset.segments["segment_id"].tolist()
    }
    assert len(set(mapping.values())) == len(mapping)

    def renamed(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.assign(segment_id=lambda source: source["segment_id"].map(mapping))

    cleaned = clean_raw_dataset(
        renamed(dataset.segments),
        renamed(dataset.observations),
        renamed(dataset.targets),
        renamed(dataset.maintenance_events),
        spec,
    )
    return RepositoryExport(
        segments=cleaned.segments,
        observations=cleaned.observations,
        targets=cleaned.targets,
        maintenance_events=cleaned.maintenance_events,
    )


def test_exact_public_surface_signature_constants_and_frozen_schemas() -> None:
    assert tuple(forecasting.__all__) == PUBLIC_NAMES
    assert all(
        name in roadguard.__all__ and vars(roadguard)[name] is vars(forecasting)[name]
        for name in PUBLIC_NAMES
    )
    expected_signature = (
        "(dataset: 'RepositoryExport', maintenance_history: 'pd.DataFrame', "
        "spec: 'DatasetSpec') -> 'MaterialForecastEvaluation'"
    )
    assert str(inspect.signature(forecast_materials)) == expected_signature
    assert MATERIAL_FORECAST_CONTRACT_VERSION == "roadguard.phase14.v1"
    assert FORECAST_MATERIAL_NAMES == (
        "thermoplastic_paint_kg",
        "reflective_sheet_m2",
        "guardrail_meter",
        "traffic_sign_quantity",
    )
    assert FORECAST_CANDIDATE_NAMES == ("seasonal_naive_12", "trailing_mean_3")
    assert (INITIAL_TRAIN_MONTHS, FROZEN_TEST_ORIGIN_COUNT, FORECAST_HORIZON_MONTHS) == (24, 7, 1)
    assert issubclass(MaterialForecastError, ValueError)
    assert tuple(field.name for field in dataclasses.fields(ForecastCandidateMetrics)) == (
        "candidate_name",
        "validation_mae",
        "validation_rmse",
    )
    assert tuple(field.name for field in dataclasses.fields(MaterialForecastMetrics)) == (
        "material",
        "candidates",
        "selected_candidate_name",
        "test_mae",
        "test_rmse",
    )
    assert tuple(field.name for field in dataclasses.fields(MaterialForecast)) == (
        "period",
        "material",
        "forecast_quantity",
    )
    assert tuple(field.name for field in dataclasses.fields(MaterialForecastEvaluation)) == (
        "contract_version",
        "forecast_input_fingerprint",
        "history_start",
        "history_end",
        "forecast_period",
        "validation_origins",
        "test_origins",
        "material_metrics",
        "forecasts",
    )
    assert all(
        schema.__dataclass_params__.frozen
        for schema in (
            ForecastCandidateMetrics,
            MaterialForecastMetrics,
            MaterialForecast,
            MaterialForecastEvaluation,
        )
    )


def test_private_candidate_seam_has_the_locked_formulas_and_rejects_unknown_names() -> None:
    prefix = tuple(float(value) for value in range(15))

    assert forecasting._candidate_forecast("seasonal_naive_12", prefix) == 3.0
    assert forecasting._candidate_forecast("trailing_mean_3", prefix) == 13.0
    assert (
        math.copysign(
            1.0,
            forecasting._candidate_forecast("seasonal_naive_12", (-0.0,) * 12),
        )
        == 1.0
    )
    with pytest.raises(ValueError):
        forecasting._candidate_forecast("unlocked-candidate", prefix)


def test_minimum_month_boundary_has_one_validation_origin_and_seven_frozen_test_origins() -> None:
    too_short = DatasetSpec(
        dataset_segments=3,
        dataset_months_per_segment=31,
        dataset_observations=93,
    )
    short_export = _build_dataset(too_short)
    with pytest.raises(MaterialForecastError, match=f"^{ALLOWED_INPUT_ERROR}$"):
        forecast_materials(short_export, _history_frame(short_export, too_short), too_short)

    minimum = DatasetSpec(
        dataset_segments=3,
        dataset_months_per_segment=32,
        dataset_observations=96,
    )
    minimum_export = _build_dataset(minimum)
    result = forecast_materials(minimum_export, _history_frame(minimum_export, minimum), minimum)

    assert len(result.validation_origins) == 1
    assert len(result.test_origins) == FROZEN_TEST_ORIGIN_COUNT
    assert result.validation_origins == (date(2024, 1, 1),)
    assert result.test_origins == tuple(date(2024, month, 1) for month in range(2, 9))
    assert result.forecast_period == date(2024, 9, 1)


def test_v1_calendar_completeness_densification_and_canonical_fsum(
    dataset: RepositoryExport, history: pd.DataFrame
) -> None:
    calls: list[tuple[str, tuple[float, ...]]] = []

    def candidate(name: str, prefix: tuple[float, ...]) -> float:
        calls.append((name, prefix))
        if name == "seasonal_naive_12":
            return float(prefix[-12])
        return float(math.fsum(prefix[-3:]) / 3.0)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(forecasting, "_candidate_forecast", candidate)
    try:
        result = forecast_materials(dataset, history, SPEC)
    finally:
        monkeypatch.undo()

    canonical = _canonical_history(dataset, history)
    assert canonical[0] == (date(2022, 1, 1), FORECAST_MATERIAL_NAMES[0], 1e16 + 2.0)
    assert len(canonical) == 192
    assert result.history_start == date(2022, 1, 1)
    assert result.history_end == date(2025, 12, 1)
    assert result.forecast_period == date(2026, 1, 1)
    assert result.validation_origins == tuple(observation_dates(48, V1_OBSERVATION_START)[24:41])
    assert result.test_origins == tuple(observation_dates(48, V1_OBSERVATION_START)[41:])
    assert tuple(metric.material for metric in result.material_metrics) == FORECAST_MATERIAL_NAMES
    assert tuple(forecast.material for forecast in result.forecasts) == FORECAST_MATERIAL_NAMES
    assert tuple(forecast.period for forecast in result.forecasts) == (date(2026, 1, 1),) * 4
    assert all(
        type(forecast.forecast_quantity) is float and forecast.forecast_quantity >= 0.0
        for forecast in result.forecasts
    )
    assert any(prefix[0] == 1e16 + 2.0 for _name, prefix in calls)
    assert result.forecast_input_fingerprint == _expected_fingerprint(canonical)


def test_rolling_origin_prefixes_selection_freeze_and_single_selected_test_pass(
    dataset: RepositoryExport, history: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, tuple[float, ...]]] = []

    def candidate(name: str, prefix: tuple[float, ...]) -> float:
        calls.append((name, prefix))
        return float(prefix[-12] if name == "seasonal_naive_12" else math.fsum(prefix[-3:]) / 3.0)

    monkeypatch.setattr(forecasting, "_candidate_forecast", candidate)
    result = forecast_materials(dataset, history, SPEC)

    validation_call_count = 4 * len(result.validation_origins) * len(FORECAST_CANDIDATE_NAMES)
    test_call_count = 4 * len(result.test_origins)
    assert len(calls) == validation_call_count + test_call_count + 4
    assert all(24 <= len(prefix) <= 40 for _name, prefix in calls[:validation_call_count])
    assert all(41 <= len(prefix) <= 47 for _name, prefix in calls[validation_call_count:-4])
    assert all(len(prefix) == 48 for _name, prefix in calls[-4:])
    selected = {
        metric.material: metric.selected_candidate_name for metric in result.material_metrics
    }
    test_and_final = calls[validation_call_count:]
    for material, expected_name in selected.items():
        material_prefix = _series(_canonical_history(dataset, history), material)[0]
        material_calls = [name for name, prefix in test_and_final if prefix[0] == material_prefix]
        assert material_calls == [expected_name] * 8

    changed_test = history.copy(deep=True)
    changed_test.loc[
        changed_test["maintenance_date"].dt.to_period("M") >= pd.Period("2025-06", freq="M"),
        "guardrail_meter",
    ] = 0.0
    changed = forecast_materials(dataset, changed_test, SPEC)
    assert tuple(item.selected_candidate_name for item in changed.material_metrics) == tuple(
        selected.values()
    )
    assert tuple(item.candidates for item in changed.material_metrics) == tuple(
        item.candidates for item in result.material_metrics
    )


def test_metrics_formulas_per_material_and_candidate_order_tie_break(
    dataset: RepositoryExport, history: pd.DataFrame
) -> None:
    result = forecast_materials(dataset, history, SPEC)
    canonical = _canonical_history(dataset, history)
    for metric in result.material_metrics:
        values = _series(canonical, metric.material)
        expected: dict[str, tuple[float, float]] = {}
        for candidate in FORECAST_CANDIDATE_NAMES:
            predictions = tuple(
                values[index - 12]
                if candidate == "seasonal_naive_12"
                else math.fsum(values[index - 3 : index]) / 3.0
                for index in range(24, 41)
            )
            expected[candidate] = _metrics(values[24:41], predictions)
        assert tuple(item.candidate_name for item in metric.candidates) == FORECAST_CANDIDATE_NAMES
        for candidate_metric in metric.candidates:
            assert (candidate_metric.validation_mae, candidate_metric.validation_rmse) == expected[
                candidate_metric.candidate_name
            ]
        expected_name = min(
            FORECAST_CANDIDATE_NAMES,
            key=lambda name: (
                expected[name][0],
                expected[name][1],
                FORECAST_CANDIDATE_NAMES.index(name),
            ),
        )
        assert metric.selected_candidate_name == expected_name

    zero_history = history.copy(deep=True)
    for material in FORECAST_MATERIAL_NAMES:
        zero_history[material] = 0 if material == "traffic_sign_quantity" else 0.0
    zero = forecast_materials(dataset, zero_history, SPEC)
    assert (
        tuple(item.selected_candidate_name for item in zero.material_metrics)
        == ("seasonal_naive_12",) * 4
    )
    assert tuple(item.forecast_quantity for item in zero.forecasts) == (0.0,) * 4


def test_exact_input_types_columns_calendar_and_complete_history_are_required(
    dataset: RepositoryExport, history: pd.DataFrame
) -> None:
    class ExportSubclass(RepositoryExport):
        pass

    class FrameSubclass(pd.DataFrame):
        pass

    class SpecSubclass(DatasetSpec):
        pass

    for bad_dataset, bad_history, bad_spec in (
        (cast(RepositoryExport, object()), history, SPEC),
        (ExportSubclass(**dataclasses.asdict(dataset)), history, SPEC),
        (dataset, cast(pd.DataFrame, object()), SPEC),
        (dataset, FrameSubclass(history), SPEC),
        (dataset, history, SpecSubclass(**SPEC.model_dump())),
    ):
        with pytest.raises(TypeError):
            forecast_materials(bad_dataset, bad_history, bad_spec)

    incomplete = history.iloc[1:].copy(deep=True)
    unknown = history.copy(deep=True)
    unknown.loc[0, "segment_id"] = "QL99-KM999-9"
    duplicate = pd.concat([history, history.iloc[[0]]], ignore_index=True)
    bad_columns = history.loc[:, tuple(reversed(MAINTENANCE_HISTORY_COLUMNS))]
    duplicate_labels = history.copy(deep=True)
    duplicate_labels.columns = (
        "segment_id",
        "segment_id",
        *MAINTENANCE_HISTORY_COLUMNS[2:],
    )
    non_string_label = history.copy(deep=True)
    non_string_label.columns = (
        1,
        *MAINTENANCE_HISTORY_COLUMNS[1:],
    )
    hostile_label = history.copy(deep=True)

    class HostileLabel(str):
        def __eq__(self, _other: object) -> bool:
            raise AssertionError("hostile label evaluated")

        __hash__ = str.__hash__

    hostile_label.columns = (
        HostileLabel("segment_id"),
        *MAINTENANCE_HISTORY_COLUMNS[1:],
    )
    bad_key = history.copy(deep=True)
    bad_key["segment_id"] = bad_key["segment_id"].astype(object)
    bad_key.iloc[0, bad_key.columns.get_loc("segment_id")] = True
    timezone_date = history.copy(deep=True)
    timezone_date["maintenance_date"] = timezone_date["maintenance_date"].astype(object)
    timezone_date.iloc[0, timezone_date.columns.get_loc("maintenance_date")] = pd.Timestamp(
        "2022-01-15",
        tz="UTC",
    )
    shifted = RepositoryExport(
        segments=dataset.segments.copy(deep=True),
        observations=dataset.observations.assign(
            date=lambda frame: frame["date"] + pd.DateOffset(months=1)
        ),
        targets=dataset.targets.assign(date=lambda frame: frame["date"] + pd.DateOffset(months=1)),
        maintenance_events=dataset.maintenance_events.copy(deep=True),
    )
    for bad_history, bad_export in (
        (incomplete, dataset),
        (unknown, dataset),
        (duplicate, dataset),
        (bad_columns, dataset),
        (duplicate_labels, dataset),
        (non_string_label, dataset),
        (hostile_label, dataset),
        (bad_key, dataset),
        (timezone_date, dataset),
        (history, shifted),
    ):
        with pytest.raises(MaterialForecastError, match=f"^{ALLOWED_INPUT_ERROR}$") as exc_info:
            forecast_materials(bad_export, bad_history, SPEC)
        assert exc_info.value.__cause__ is None


def test_scalar_rules_cost_isolation_outside_window_and_caller_immutability(
    dataset: RepositoryExport, history: pd.DataFrame
) -> None:
    before_dataset = tuple(frame.copy(deep=True) for frame in dataclasses.astuple(dataset))
    before_history = history.copy(deep=True)
    baseline = forecast_materials(dataset, history, SPEC)
    cost_changed = history.copy(deep=True)
    cost_changed["maintenance_cost"] += 7
    assert forecast_materials(dataset, cost_changed, SPEC) == baseline
    ignored_changed = history.copy(deep=True)
    ignored_changed.loc[
        ignored_changed["maintenance_date"].dt.date > baseline.history_end,
        "thermoplastic_paint_kg",
    ] = 999.0
    assert forecast_materials(dataset, ignored_changed, SPEC) == baseline
    for before, after in zip(before_dataset, dataclasses.astuple(dataset), strict=True):
        assert_frame_equal(before, after)
    assert_frame_equal(before_history, history)

    for column, value in (
        ("maintenance_cost", True),
        ("maintenance_cost", 0),
        ("traffic_sign_quantity", 1.5),
        ("traffic_sign_quantity", True),
        ("thermoplastic_paint_kg", float("nan")),
        ("reflective_sheet_m2", -1.0),
    ):
        invalid = history.copy(deep=True)
        invalid[column] = invalid[column].astype(object)
        invalid.iloc[0, invalid.columns.get_loc(column)] = value
        with pytest.raises(MaterialForecastError, match="^Phase 14 input validation failed\\.$"):
            forecast_materials(dataset, invalid, SPEC)


@pytest.mark.parametrize("returned", [-1.0, float("nan"), float("inf"), 1, True])
def test_invalid_candidate_outputs_are_never_clipped_or_coerced(
    dataset: RepositoryExport,
    history: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    returned: object,
) -> None:
    monkeypatch.setattr(forecasting, "_candidate_forecast", lambda _name, _prefix: returned)
    with pytest.raises(
        MaterialForecastError, match="^Phase 14 rolling-origin evaluation failed\\.$"
    ) as exc_info:
        forecast_materials(dataset, history, SPEC)
    assert exc_info.value.__cause__ is None


def test_final_candidate_failure_is_mapped_to_forecast_output_error(
    dataset: RepositoryExport, history: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invalid_final(_name: str, prefix: tuple[float, ...]) -> float:
        return float("nan") if len(prefix) == 48 else float(prefix[-12])

    monkeypatch.setattr(forecasting, "_candidate_forecast", invalid_final)
    with pytest.raises(
        MaterialForecastError, match="^Phase 14 forecast output failed\\.$"
    ) as exc_info:
        forecast_materials(dataset, history, SPEC)
    assert exc_info.value.__cause__ is None


def test_shuffle_index_and_irrelevant_lower_phase_fields_do_not_change_evidence(
    dataset: RepositoryExport, history: pd.DataFrame
) -> None:
    baseline = forecast_materials(dataset, history, SPEC)
    shuffled = RepositoryExport(
        segments=dataset.segments.sample(frac=1, random_state=1).set_axis(range(10, 13)),
        observations=dataset.observations.sample(frac=1, random_state=2).set_axis(range(100, 244)),
        targets=dataset.targets.sample(frac=1, random_state=3).set_axis(range(200, 344)),
        maintenance_events=dataset.maintenance_events.sample(frac=1, random_state=4).set_axis(
            range(len(dataset.maintenance_events))
        ),
    )
    shuffled_history = history.sample(frac=1, random_state=5).set_axis(
        range(1000, 1000 + len(history))
    )
    irrelevant = replace(
        shuffled,
        segments=shuffled.segments.assign(
            road_length_km=lambda frame: frame["road_length_km"] + 0.25
        ),
    )
    assert forecast_materials(shuffled, shuffled_history, SPEC) == baseline
    assert forecast_materials(irrelevant, shuffled_history, SPEC) == baseline


def test_fingerprint_is_sensitive_to_accepted_history_and_spec_but_isolates_event_identity(
    dataset: RepositoryExport, history: pd.DataFrame
) -> None:
    baseline = forecast_materials(dataset, history, SPEC)
    changed_history = history.copy(deep=True)
    changed_history.loc[
        changed_history["maintenance_date"].dt.date == date(2024, 1, 15),
        "reflective_sheet_m2",
    ] += 0.5
    changed = forecast_materials(dataset, changed_history, SPEC)
    assert changed.forecast_input_fingerprint != baseline.forecast_input_fingerprint
    assert changed != baseline

    different_spec = DatasetSpec(
        dataset_segments=3,
        dataset_months_per_segment=49,
        dataset_observations=147,
    )
    different_export = _build_dataset(different_spec)
    different = forecast_materials(
        different_export,
        _history_frame(different_export, different_spec),
        different_spec,
    )
    assert different.forecast_input_fingerprint != baseline.forecast_input_fingerprint

    changed_days_export = _build_dataset(SPEC, event_day=20)
    changed_days = forecast_materials(
        changed_days_export, _history_frame(changed_days_export), SPEC
    )
    assert changed_days == baseline

    extra_event_export = _build_dataset(SPEC, add_zero_quantity_event=True)
    extra_event_history = _history_frame(extra_event_export, zero_day_20=True)
    extra_event = forecast_materials(extra_event_export, extra_event_history, SPEC)
    assert extra_event == baseline

    renamed = _renamed_export(dataset)
    renamed_history = history.assign(
        segment_id=lambda frame: frame["segment_id"].map(
            {
                segment_id: f"QL99-{segment_id.split('-', maxsplit=1)[1]}"
                for segment_id in dataset.segments["segment_id"].tolist()
            }
        )
    )
    assert forecast_materials(renamed, renamed_history, SPEC) == baseline


def test_pure_workflow_has_no_configuration_database_artifact_supervised_split_filesystem_or_rng_io(
    dataset: RepositoryExport, history: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("forbidden Phase 14 I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(config, "load_config", forbidden)
    monkeypatch.setattr(database, "create_database_engine", forbidden)
    monkeypatch.setattr(database, "PostgresRepository", forbidden)
    monkeypatch.setattr(artifacts, "persist_selected_artifacts", forbidden)
    monkeypatch.setattr(artifact_io.joblib, "load", forbidden)
    monkeypatch.setattr(artifact_io.joblib, "dump", forbidden)
    monkeypatch.setattr(preprocessing, "split_chronologically", forbidden)
    monkeypatch.setattr(preprocessing, "fit_preprocessor", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(np.random, "seed", forbidden)
    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(random, "Random", forbidden)
    result = forecast_materials(dataset, history, SPEC)
    assert result.contract_version == MATERIAL_FORECAST_CONTRACT_VERSION
