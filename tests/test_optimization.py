"""Phase 15 RED contract tests for offline maintenance prioritization.

The local source fixture structurally reconstructs a canonical in-memory Phase
13 manifest and its 2,100-row risk JSONL. This exercises the fixed 300
candidate boundary without models, database, or filesystem work.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import itertools
import json
import os
import random
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import roadguard
from roadguard.artifacts import (
    FROZEN_SELECTION_CONTRACT_VERSION,
    ArtifactFile,
    FrozenSelectionResult,
    RiskOutput,
    SelectedArtifactManifest,
)
from roadguard.classification import CandidateValidationMetrics
from roadguard.regression import CandidateRegressionValidationMetrics
from roadguard.risk import risk_score_from_probability

PUBLIC_NAMES = (
    "optimize_maintenance",
    "MaintenanceOptimizationError",
    "MaintenanceCostInput",
    "MaintenanceRecommendation",
    "MaintenanceOptimizationResult",
    "MAINTENANCE_OPTIMIZATION_CONTRACT_VERSION",
    "MAINTENANCE_OPTIMIZATION_USE_CASE",
    "MAX_EXACT_VND",
    "V1_OPTIMIZATION_CANDIDATE_COUNT",
)
SEGMENTS = tuple(f"QL01-KM{index:03d}-{index + 1:03d}" for index in range(300))
TEST_DATES = tuple(date(2025, month, 1) for month in range(6, 13))
INPUT_ERROR = "Phase 15 input validation failed."


def _optimization() -> Any:
    """Import during test execution, preserving absent-module RED evidence."""
    return importlib.import_module("roadguard.optimization")


def _band(score: int) -> str:
    return (
        "LOW" if score <= 30 else "MEDIUM" if score <= 60 else "HIGH" if score <= 80 else "CRITICAL"
    )


def _month_dates(year: int, month: int, count: int) -> tuple[date, ...]:
    return tuple(
        date(year + ((month - 1 + i) // 12), ((month - 1 + i) % 12) + 1, 1) for i in range(count)
    )


def _risk_bytes(rows: tuple[RiskOutput, ...]) -> bytes:
    return b"".join(
        json.dumps(
            {
                "date": row.date.isoformat(),
                "maintenance_probability": row.maintenance_probability,
                "risk_band": row.risk_band,
                "risk_score": row.risk_score,
                "segment_id": row.segment_id,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _selection(final_probabilities: dict[str, float] | None = None) -> FrozenSelectionResult:
    probabilities = final_probabilities or {}
    risk = tuple(
        RiskOutput(
            segment_id,
            risk_date,
            probability,
            risk_score_from_probability(probability),
            _band(risk_score_from_probability(probability)),
        )
        for segment_id in SEGMENTS
        for risk_date in TEST_DATES
        for probability in (
            probabilities.get(segment_id, 0.0) if risk_date == TEST_DATES[-1] else 0.0,
        )
    )
    payload = _risk_bytes(risk)
    artifacts = (
        ArtifactFile("preprocessor", "preprocessor.json", "a" * 64, 1),
        ArtifactFile("classifier", "classifier.joblib", "b" * 64, 1),
        ArtifactFile("regressor", "regressor.joblib", "c" * 64, 1),
        ArtifactFile(
            "test_risk", "test-risk.jsonl", hashlib.sha256(payload).hexdigest(), len(payload)
        ),
    )
    manifest = SelectedArtifactManifest(
        FROZEN_SELECTION_CONTRACT_VERSION,
        "d" * 64,
        "e" * 64,
        "f" * 64,
        "roadguard.phase11.v1",
        "roadguard.phase12.v1",
        "logistic_l2",
        "ridge_l2",
        0.5,
        42,
        42,
        43,
        ("road_type",),
        10200,
        2100,
        2100,
        _month_dates(2022, 1, 34),
        _month_dates(2024, 11, 7),
        TEST_DATES,
        (
            CandidateValidationMetrics("logistic_l2", 0.5, 0.5, 0.5, 0.5),
            CandidateValidationMetrics("hist_gradient_boosting", 0.4, 0.5, 0.4, 0.4),
        ),
        (
            CandidateRegressionValidationMetrics("ridge_l2", 1.0, 1.0),
            CandidateRegressionValidationMetrics("hist_gradient_boosting", 2.0, 2.0),
        ),
        (("LOW", 0, 30), ("MEDIUM", 31, 60), ("HIGH", 61, 80), ("CRITICAL", 81, 100)),
        2100,
        (
            ("python", "3.12"),
            ("numpy", "2.0"),
            ("pandas", "2.2"),
            ("scikit-learn", "1.5"),
            ("joblib", "1.5"),
        ),
        "win-amd64",
        artifacts,
    )
    from roadguard import artifacts as phase13

    digest = hashlib.sha256(
        phase13._canonical_json_bytes(phase13._manifest_projection(manifest))
    ).hexdigest()
    return FrozenSelectionResult(digest, f"roadguard.phase13.v1/{digest}", manifest, risk)


def _replace_risk(
    selection: FrozenSelectionResult,
    risk: tuple[RiskOutput, ...],
    *,
    test_dates: tuple[date, ...] | None = None,
) -> FrozenSelectionResult:
    """Re-authenticate a deliberately forged in-memory Phase 13 risk payload."""
    payload = _risk_bytes(risk)
    test_risk = ArtifactFile(
        "test_risk", "test-risk.jsonl", hashlib.sha256(payload).hexdigest(), len(payload)
    )
    manifest = replace(
        selection.manifest,
        test_dates=selection.manifest.test_dates if test_dates is None else test_dates,
        test_rows=len(risk),
        risk_output_rows=len(risk),
        artifacts=(*selection.manifest.artifacts[:3], test_risk),
    )
    from roadguard import artifacts as phase13

    digest = hashlib.sha256(
        phase13._canonical_json_bytes(phase13._manifest_projection(manifest))
    ).hexdigest()
    return FrozenSelectionResult(digest, f"roadguard.phase13.v1/{digest}", manifest, risk)


def _costs(module: Any, overrides: dict[str, tuple[int, date]] | None = None) -> tuple[Any, ...]:
    values = overrides or {}
    return tuple(
        module.MaintenanceCostInput(segment, *values.get(segment, (1_000_000, TEST_DATES[-1])))
        for segment in SEGMENTS
    )


def _run(probabilities: dict[str, float], costs: dict[str, tuple[int, date]], budget: int) -> Any:
    module = _optimization()
    selection = _selection(probabilities)
    return module.optimize_maintenance(
        selection, selection.manifest_sha256, _costs(module, costs), budget
    )


def _ids(result: Any) -> tuple[str, ...]:
    return tuple(item.segment_id for item in result.recommendations)


def _expected_fingerprint(
    selection: FrozenSelectionResult, costs: tuple[Any, ...], budget: int
) -> str:
    """Independent Phase 15 canonical JSON calculation, not a production helper."""
    final = {row.segment_id: row for row in selection.risk_output if row.date == TEST_DATES[-1]}
    cost_by_segment = {item.segment_id: item for item in costs}
    rows = []
    for segment_id in sorted(final):
        risk = final[segment_id]
        cost = cost_by_segment[segment_id]
        probability = 0.0 if risk.maintenance_probability == 0.0 else risk.maintenance_probability
        rows.append(
            (
                segment_id,
                risk.date.isoformat(),
                probability.hex(),
                risk.risk_score,
                risk.risk_band,
                cost.cost_vnd,
                cost.cost_as_of_date.isoformat(),
            )
        )
    payload = {
        "budget_vnd": budget,
        "candidates": rows,
        "columns": [
            "segment_id",
            "evidence_date",
            "maintenance_probability",
            "risk_score",
            "risk_band",
            "cost_vnd",
            "cost_as_of_date",
        ],
        "contract": "roadguard.phase15.v1",
        "objective": [
            "maximize_total_risk_score",
            "minimize_total_cost_vnd",
            "prefer_selected_lower_segment_id_at_first_difference",
        ],
        "source_manifest_sha256": selection.manifest_sha256,
        "source_risk_input_fingerprint": selection.manifest.risk_input_fingerprint,
        "use_case": "OFFLINE_EVALUATION_ONLY",
    }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def test_public_surface_signature_constants_and_frozen_field_order_are_exact() -> None:
    module = _optimization()
    assert tuple(module.__all__) == PUBLIC_NAMES
    assert all(name in roadguard.__all__ and hasattr(roadguard, name) for name in PUBLIC_NAMES)
    assert module.MAINTENANCE_OPTIMIZATION_CONTRACT_VERSION == "roadguard.phase15.v1"
    assert module.MAINTENANCE_OPTIMIZATION_USE_CASE == "OFFLINE_EVALUATION_ONLY"
    assert module.MAX_EXACT_VND == 2**63 - 1
    assert module.V1_OPTIMIZATION_CANDIDATE_COUNT == 300
    assert tuple(
        importlib.import_module("inspect").signature(module.optimize_maintenance).parameters
    ) == ("selection", "expected_manifest_sha256", "candidate_costs", "budget_vnd")
    assert tuple(field.name for field in dataclasses.fields(module.MaintenanceCostInput)) == (
        "segment_id",
        "cost_vnd",
        "cost_as_of_date",
    )
    assert tuple(field.name for field in dataclasses.fields(module.MaintenanceRecommendation)) == (
        "priority_rank",
        "segment_id",
        "evidence_date",
        "maintenance_probability",
        "risk_score",
        "risk_band",
        "cost_vnd",
        "cost_as_of_date",
    )
    assert tuple(
        field.name for field in dataclasses.fields(module.MaintenanceOptimizationResult)
    ) == (
        "contract_version",
        "use_case",
        "source_manifest_sha256",
        "source_risk_input_fingerprint",
        "optimization_input_fingerprint",
        "evidence_date",
        "risk_window_end",
        "budget_vnd",
        "selected_cost_vnd",
        "remaining_budget_vnd",
        "candidate_count",
        "selected_count",
        "total_risk_score",
        "recommendations",
    )
    assert all(
        record.__dataclass_params__.frozen
        for record in (
            module.MaintenanceCostInput,
            module.MaintenanceRecommendation,
            module.MaintenanceOptimizationResult,
        )
    )


def test_trusted_manifest_digest_phase13_manifest_and_risk_jsonl_are_authenticated() -> None:
    module = _optimization()
    selection = _selection({SEGMENTS[0]: 0.805, SEGMENTS[1]: 0.605, SEGMENTS[2]: 0.305})
    costs = _costs(module)
    result = module.optimize_maintenance(selection, selection.manifest_sha256, costs, 3_000_000)
    assert result.source_manifest_sha256 == selection.manifest_sha256
    assert result.source_risk_input_fingerprint == selection.manifest.risk_input_fingerprint
    bad_values = (
        replace(selection, manifest_sha256="0" * 64),
        replace(selection, relative_artifact_directory="roadguard.phase13.v1/" + "0" * 64),
        replace(selection, risk_output=selection.risk_output[:-1]),
        replace(
            selection,
            manifest=replace(selection.manifest, artifacts=selection.manifest.artifacts[:-1]),
        ),
    )
    for bad_selection in bad_values:
        with pytest.raises(module.MaintenanceOptimizationError, match=f"^{INPUT_ERROR}$") as info:
            module.optimize_maintenance(bad_selection, selection.manifest_sha256, costs, 3_000_000)
        assert info.value.__cause__ is None
    with pytest.raises(module.MaintenanceOptimizationError, match=f"^{INPUT_ERROR}$"):
        module.optimize_maintenance(selection, "0" * 64, costs, 3_000_000)


def test_v1_source_requires_exact_seven_dates_2100_rows_and_300_final_candidates() -> None:
    module = _optimization()
    selection = _selection({SEGMENTS[0]: 0.5})
    six_dates = TEST_DATES[:-1]
    six_date_risk = tuple(row for row in selection.risk_output if row.date in six_dates)
    too_many_rows = selection.risk_output + (selection.risk_output[-1],)
    for forged in (
        _replace_risk(selection, six_date_risk, test_dates=six_dates),
        _replace_risk(selection, too_many_rows),
    ):
        with pytest.raises(module.MaintenanceOptimizationError, match=f"^{INPUT_ERROR}$"):
            module.optimize_maintenance(forged, forged.manifest_sha256, _costs(module), 1)


def test_authenticated_risk_rows_reject_forged_probability_score_band_date_and_order() -> None:
    module = _optimization()
    selection = _selection({SEGMENTS[0]: 0.5})
    first = selection.risk_output[0]
    for changed in (
        (replace(first, maintenance_probability=1.1), *selection.risk_output[1:]),
        (replace(first, risk_score=99), *selection.risk_output[1:]),
        (replace(first, risk_band="CRITICAL"), *selection.risk_output[1:]),
        (replace(first, date=date(2024, 1, 1)), *selection.risk_output[1:]),
        (selection.risk_output[1], selection.risk_output[0], *selection.risk_output[2:]),
    ):
        forged = _replace_risk(selection, changed)
        with pytest.raises(module.MaintenanceOptimizationError, match=f"^{INPUT_ERROR}$"):
            module.optimize_maintenance(forged, forged.manifest_sha256, _costs(module), 1)


def test_exact_risk_boundary_scores_bands_dates_and_cost_coverage_as_of_rules() -> None:
    module = _optimization()
    probabilities = {SEGMENTS[0]: 0.305, SEGMENTS[1]: 0.605, SEGMENTS[2]: 0.805}
    selection = _selection(probabilities)
    costs = _costs(module, {segment: (1, TEST_DATES[-1]) for segment in probabilities})
    result = module.optimize_maintenance(selection, selection.manifest_sha256, costs, 3)
    assert result.evidence_date == date(2025, 12, 1)
    assert result.risk_window_end == date(2025, 12, 31)
    assert {
        item.segment_id: (item.risk_score, item.risk_band) for item in result.recommendations
    } == {SEGMENTS[0]: (31, "MEDIUM"), SEGMENTS[1]: (61, "HIGH"), SEGMENTS[2]: (81, "CRITICAL")}
    invalid_costs = (
        costs[:-1],
        costs + (costs[0],),
        costs[:-1] + (costs[0],),
        _costs(module, {SEGMENTS[0]: (1, date(2025, 12, 2))}),
    )
    for invalid in invalid_costs:
        with pytest.raises(module.MaintenanceOptimizationError, match=f"^{INPUT_ERROR}$"):
            module.optimize_maintenance(selection, selection.manifest_sha256, invalid, 3)


def test_exact_vnd_maximum_multiple_max_costs_and_all_empty_optima() -> None:
    module = _optimization()
    maximum = module.MAX_EXACT_VND
    selection = _selection({SEGMENTS[0]: 1.0, SEGMENTS[1]: 0.9})
    costs = _costs(
        module, {SEGMENTS[0]: (maximum, TEST_DATES[-1]), SEGMENTS[1]: (maximum, TEST_DATES[-1])}
    )
    exact = module.optimize_maintenance(selection, selection.manifest_sha256, costs, maximum)
    assert (
        _ids(exact) == (SEGMENTS[0],)
        and exact.selected_cost_vnd == maximum
        and exact.remaining_budget_vnd == 0
    )
    for budget, probabilities, overrides in (
        (0, {SEGMENTS[0]: 1.0}, {SEGMENTS[0]: (1, TEST_DATES[-1])}),
        (1, {SEGMENTS[0]: 1.0}, {SEGMENTS[0]: (2, TEST_DATES[-1])}),
        (3, {}, {SEGMENTS[0]: (1, TEST_DATES[-1])}),
    ):
        result = _run(probabilities, overrides, budget)
        assert (
            result.recommendations == ()
            and result.selected_cost_vnd == result.total_risk_score == 0
            and result.remaining_budget_vnd == budget
        )
    all_fit = _run(
        {SEGMENTS[0]: 0.4, SEGMENTS[1]: 0.8},
        {SEGMENTS[0]: (2, TEST_DATES[-1]), SEGMENTS[1]: (3, TEST_DATES[-1])},
        5,
    )
    assert set(_ids(all_fit)) == {SEGMENTS[0], SEGMENTS[1]}
    for bad_budget in (-1, maximum + 1, True):
        with pytest.raises((TypeError, module.MaintenanceOptimizationError)):
            module.optimize_maintenance(selection, selection.manifest_sha256, costs, bad_budget)


def test_expected_digest_and_cost_scalar_types_are_strictly_separated_from_domain_errors() -> None:
    module = _optimization()
    selection = _selection({SEGMENTS[0]: 0.5})
    costs = _costs(module)
    for digest in (True, 7, b"a" * 64):
        with pytest.raises(TypeError):
            module.optimize_maintenance(selection, cast(Any, digest), costs, 1)
    for digest in ("A" * 64, "a" * 63, "a" * 65):
        with pytest.raises(module.MaintenanceOptimizationError, match=f"^{INPUT_ERROR}$"):
            module.optimize_maintenance(selection, digest, costs, 1)

    class IntSubclass(int):
        pass

    for value in (True, 1.0, np.int64(1), IntSubclass(1)):
        changed = _costs(module)
        object.__setattr__(changed[0], "cost_vnd", value)
        with pytest.raises(TypeError):
            module.optimize_maintenance(selection, selection.manifest_sha256, changed, 1)
    for value in (0, module.MAX_EXACT_VND + 1):
        changed = _costs(module)
        object.__setattr__(changed[0], "cost_vnd", value)
        with pytest.raises(module.MaintenanceOptimizationError, match=f"^{INPUT_ERROR}$"):
            module.optimize_maintenance(selection, selection.manifest_sha256, changed, 1)


def test_exact_dynamic_programming_beats_greedy_and_matches_small_exhaustive_oracle() -> None:
    probabilities = {SEGMENTS[0]: 0.6, SEGMENTS[1]: 1.0, SEGMENTS[2]: 1.0}
    costs = {
        SEGMENTS[0]: (10, TEST_DATES[-1]),
        SEGMENTS[1]: (20, TEST_DATES[-1]),
        SEGMENTS[2]: (30, TEST_DATES[-1]),
    }
    result = _run(probabilities, costs, 50)
    scores = {
        segment: risk_score_from_probability(probability)
        for segment, probability in probabilities.items()
    }
    ordered = tuple(sorted(probabilities))

    def objective(ids: tuple[str, ...]) -> tuple[int, int, tuple[int, ...]]:
        return (
            -sum(scores[item] for item in ids),
            sum(costs[item][0] for item in ids),
            tuple(-int(item in ids) for item in ordered),
        )

    feasible = tuple(
        subset
        for size in range(4)
        for subset in itertools.combinations(ordered, size)
        if sum(costs[item][0] for item in subset) <= 50
    )
    expected = min(feasible, key=objective)
    assert set(_ids(result)) == set(expected) == {SEGMENTS[1], SEGMENTS[2]}


def test_all_objective_tie_breaks_and_presentation_ranking_are_independent() -> None:
    lower_cost = _run(
        {SEGMENTS[0]: 0.5, SEGMENTS[1]: 0.5},
        {SEGMENTS[0]: (20, TEST_DATES[-1]), SEGMENTS[1]: (10, TEST_DATES[-1])},
        20,
    )
    assert _ids(lower_cost) == (SEGMENTS[1],)
    earlier_id = _run(
        {SEGMENTS[0]: 0.5, SEGMENTS[1]: 0.5},
        {SEGMENTS[0]: (10, TEST_DATES[-1]), SEGMENTS[1]: (10, TEST_DATES[-1])},
        10,
    )
    assert _ids(earlier_id) == (SEGMENTS[0],)
    display = _run(
        {SEGMENTS[0]: 0.7, SEGMENTS[1]: 0.8, SEGMENTS[2]: 0.8},
        {
            SEGMENTS[0]: (1, TEST_DATES[-1]),
            SEGMENTS[1]: (3, TEST_DATES[-1]),
            SEGMENTS[2]: (2, TEST_DATES[-1]),
        },
        6,
    )
    assert _ids(display) == (SEGMENTS[2], SEGMENTS[1], SEGMENTS[0])
    assert tuple(item.priority_rank for item in display.recommendations) == (1, 2, 3)


def test_canonical_fingerprint_sensitivity_and_shuffled_cost_invariance() -> None:
    module = _optimization()
    selection = _selection({SEGMENTS[0]: 0.5, SEGMENTS[1]: 0.8})
    costs = _costs(module, {SEGMENTS[0]: (2, TEST_DATES[-1]), SEGMENTS[1]: (3, TEST_DATES[-1])})
    baseline = module.optimize_maintenance(selection, selection.manifest_sha256, costs, 5)
    shuffled = module.optimize_maintenance(
        selection, selection.manifest_sha256, tuple(reversed(costs)), 5
    )
    changed_cost = module.optimize_maintenance(
        selection,
        selection.manifest_sha256,
        _costs(module, {SEGMENTS[0]: (4, TEST_DATES[-1]), SEGMENTS[1]: (3, TEST_DATES[-1])}),
        5,
    )
    changed_as_of = module.optimize_maintenance(
        selection,
        selection.manifest_sha256,
        _costs(module, {SEGMENTS[0]: (2, date(2025, 11, 30)), SEGMENTS[1]: (3, TEST_DATES[-1])}),
        5,
    )
    assert shuffled == baseline
    assert baseline.optimization_input_fingerprint == _expected_fingerprint(selection, costs, 5)
    assert changed_cost.optimization_input_fingerprint != baseline.optimization_input_fingerprint
    assert changed_as_of.optimization_input_fingerprint != baseline.optimization_input_fingerprint
    assert (
        len(baseline.optimization_input_fingerprint) == 64
        and baseline.optimization_input_fingerprint
        == baseline.optimization_input_fingerprint.lower()
    )


def test_recursive_exact_types_immutability_and_sanitized_errors() -> None:
    module = _optimization()
    selection, costs = _selection({SEGMENTS[0]: 0.5}), _costs(module)

    class CostSubclass(module.MaintenanceCostInput):
        pass

    forged_manifest = replace(selection, manifest=cast(SelectedArtifactManifest, object()))
    forged_risk_tuple = replace(
        selection, risk_output=cast(tuple[RiskOutput, ...], list(selection.risk_output))
    )
    forged_artifact_tuple = replace(
        selection,
        manifest=replace(
            selection.manifest, artifacts=cast(Any, list(selection.manifest.artifacts))
        ),
    )
    forged_metric_tuple = replace(
        selection,
        manifest=replace(
            selection.manifest,
            classification_candidates=cast(Any, list(selection.manifest.classification_candidates)),
        ),
    )
    forged_runtime_pair = replace(
        selection,
        manifest=replace(
            selection.manifest, runtime_versions=cast(Any, (("python", "3.12"), ["numpy", "2.0"]))
        ),
    )
    for arguments in (
        (cast(Any, object()), selection.manifest_sha256, costs, 1),
        (selection, cast(Any, 7), costs, 1),
        (selection, selection.manifest_sha256, cast(Any, list(costs)), 1),
        (
            selection,
            selection.manifest_sha256,
            costs[:-1] + (CostSubclass(SEGMENTS[-1], 1, TEST_DATES[-1]),),
            1,
        ),
        (forged_manifest, selection.manifest_sha256, costs, 1),
        (forged_risk_tuple, selection.manifest_sha256, costs, 1),
        (forged_artifact_tuple, selection.manifest_sha256, costs, 1),
        (forged_metric_tuple, selection.manifest_sha256, costs, 1),
        (forged_runtime_pair, selection.manifest_sha256, costs, 1),
    ):
        with pytest.raises(TypeError):
            module.optimize_maintenance(*arguments)
    with pytest.raises(dataclasses.FrozenInstanceError):
        costs[0].cost_vnd = 7
    bad_costs = _costs(module, {SEGMENTS[0]: (-9, TEST_DATES[-1])})
    with pytest.raises(module.MaintenanceOptimizationError, match=f"^{INPUT_ERROR}$") as info:
        module.optimize_maintenance(selection, selection.manifest_sha256, bad_costs, 1)
    assert info.value.__cause__ is None and "-9" not in str(info.value)


def test_call_preserves_authenticated_selection_and_cost_scenario_objects() -> None:
    module = _optimization()
    selection = _selection({SEGMENTS[0]: 0.5, SEGMENTS[1]: 0.8})
    costs = _costs(module, {SEGMENTS[0]: (2, TEST_DATES[-1]), SEGMENTS[1]: (3, TEST_DATES[-1])})
    before_selection = selection
    before_costs = costs
    result = module.optimize_maintenance(selection, selection.manifest_sha256, costs, 5)
    assert selection == before_selection
    assert costs == before_costs
    assert tuple((item.segment_id, item.cost_vnd, item.cost_as_of_date) for item in costs) == tuple(
        (item.segment_id, item.cost_vnd, item.cost_as_of_date) for item in before_costs
    )
    assert all(
        item.segment_id in {cost.segment_id for cost in costs} for item in result.recommendations
    )


def test_workflow_is_pure_against_poisoned_environment_io_models_forecasts_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _optimization()

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("forbidden Phase 15 side effect")

    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    for module_name, attribute in (
        ("roadguard.config", "load_config"),
        ("roadguard.database", "create_database_engine"),
        ("roadguard.artifacts", "persist_selected_artifacts"),
        ("roadguard.forecasting", "forecast_materials"),
    ):
        monkeypatch.setattr(importlib.import_module(module_name), attribute, forbidden)
    assert _run({SEGMENTS[0]: 0.9}, {SEGMENTS[0]: (1, TEST_DATES[-1])}, 1).selected_count == 1


def test_300_candidate_scale_reconstruction_and_total_invariants() -> None:
    module = _optimization()
    selection = _selection({segment: 0.5 for segment in SEGMENTS})
    costs = _costs(module, {segment: (1, TEST_DATES[-1]) for segment in SEGMENTS})
    result = module.optimize_maintenance(selection, selection.manifest_sha256, costs, 300)
    assert (
        result.candidate_count,
        result.selected_count,
        result.total_risk_score,
        result.selected_cost_vnd,
        result.remaining_budget_vnd,
    ) == (300, 300, 15000, 300, 0)
    assert len({item.segment_id for item in result.recommendations}) == 300
    assert sum(item.cost_vnd for item in result.recommendations) == result.selected_cost_vnd
    assert sum(item.risk_score for item in result.recommendations) == result.total_risk_score
