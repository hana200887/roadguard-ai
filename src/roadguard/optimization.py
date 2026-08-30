"""Exact offline Phase 15 maintenance-prioritization workflow."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Final, Literal, cast

from roadguard import artifacts as _artifacts
from roadguard._optimization_manifest import (
    preflight_manifest as _preflight_manifest,
)
from roadguard._optimization_manifest import (
    snapshot_selection as _snapshot_selection,
)
from roadguard._optimization_manifest import (
    validate_manifest as _validate_canonical_manifest,
)
from roadguard.artifacts import (
    FrozenSelectionResult,
    RiskOutput,
    SelectedArtifactManifest,
)
from roadguard.risk import risk_score_from_probability
from roadguard.segments import SEGMENT_ID_PATTERN

MAINTENANCE_OPTIMIZATION_CONTRACT_VERSION: Final[str] = "roadguard.phase15.v1"
MAINTENANCE_OPTIMIZATION_USE_CASE: Final[Literal["OFFLINE_EVALUATION_ONLY"]] = (
    "OFFLINE_EVALUATION_ONLY"
)
MAX_EXACT_VND: Final[int] = 2**63 - 1
V1_OPTIMIZATION_CANDIDATE_COUNT: Final[int] = 300

_INPUT_ERROR: Final[str] = "Phase 15 input validation failed."
_OPTIMIZATION_ERROR: Final[str] = "Phase 15 optimization failed."
_OUTPUT_ERROR: Final[str] = "Phase 15 output validation failed."
_RISK_BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("LOW", 0, 30),
    ("MEDIUM", 31, 60),
    ("HIGH", 61, 80),
    ("CRITICAL", 81, 100),
)
_SEGMENT_ID_RE: Final[re.Pattern[str]] = re.compile(SEGMENT_ID_PATTERN, re.ASCII)


class MaintenanceOptimizationError(ValueError):
    """Raised when authenticated offline optimization input or output is invalid."""


@dataclass(frozen=True)
class MaintenanceCostInput:
    segment_id: str
    cost_vnd: int
    cost_as_of_date: date


@dataclass(frozen=True)
class MaintenanceRecommendation:
    priority_rank: int
    segment_id: str
    evidence_date: date
    maintenance_probability: float
    risk_score: int
    risk_band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    cost_vnd: int
    cost_as_of_date: date


@dataclass(frozen=True)
class MaintenanceOptimizationResult:
    contract_version: str
    use_case: Literal["OFFLINE_EVALUATION_ONLY"]
    source_manifest_sha256: str
    source_risk_input_fingerprint: str
    optimization_input_fingerprint: str
    evidence_date: date
    risk_window_end: date
    budget_vnd: int
    selected_cost_vnd: int
    remaining_budget_vnd: int
    candidate_count: int
    selected_count: int
    total_risk_score: int
    recommendations: tuple[MaintenanceRecommendation, ...]


@dataclass(frozen=True)
class _Candidate:
    segment_id: str
    probability: float
    score: int
    band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    cost: int
    cost_as_of: date


@dataclass(frozen=True)
class _State:
    cost: int
    selected_mask: int


def optimize_maintenance(
    selection: FrozenSelectionResult,
    expected_manifest_sha256: str,
    candidate_costs: tuple[MaintenanceCostInput, ...],
    budget_vnd: int,
) -> MaintenanceOptimizationResult:
    """Authenticate the Phase 13 snapshot and exactly optimize its final-date candidates."""
    try:
        _preflight_types(selection, expected_manifest_sha256, candidate_costs, budget_vnd)
    except AttributeError:
        raise MaintenanceOptimizationError(_INPUT_ERROR) from None
    try:
        selection_snapshot = _snapshot_selection(selection)
        cost_snapshot = tuple(
            MaintenanceCostInput(cost.segment_id, cost.cost_vnd, cost.cost_as_of_date)
            for cost in candidate_costs
        )
        candidates, evidence_date = _validate_input(
            selection_snapshot, expected_manifest_sha256, cost_snapshot, budget_vnd
        )
        source_manifest_sha256 = expected_manifest_sha256
        source_risk_input_fingerprint = selection_snapshot.manifest.risk_input_fingerprint
        fingerprint = _optimization_fingerprint(
            source_manifest_sha256,
            source_risk_input_fingerprint,
            candidates,
            budget_vnd,
            evidence_date,
        )
    except MaintenanceOptimizationError:
        raise
    except (ArithmeticError, AttributeError, KeyError, TypeError, ValueError):
        raise MaintenanceOptimizationError(_INPUT_ERROR) from None

    try:
        chosen = _exact_optimum(candidates, budget_vnd)
    except MaintenanceOptimizationError:
        raise
    except (ArithmeticError, OverflowError, ValueError):
        raise MaintenanceOptimizationError(_OPTIMIZATION_ERROR) from None

    try:
        result = _build_result(
            source_manifest_sha256,
            source_risk_input_fingerprint,
            candidates,
            chosen,
            evidence_date,
            budget_vnd,
            fingerprint,
        )
        _validate_output(
            result,
            candidates,
            budget_vnd,
            evidence_date,
            source_manifest_sha256,
            source_risk_input_fingerprint,
            fingerprint,
        )
        return result
    except MaintenanceOptimizationError:
        raise
    except (ArithmeticError, AttributeError, TypeError, ValueError):
        raise MaintenanceOptimizationError(_OUTPUT_ERROR) from None


def _preflight_types(
    selection: object,
    expected_manifest_sha256: object,
    candidate_costs: object,
    budget_vnd: object,
) -> None:
    if type(selection) is not FrozenSelectionResult:
        raise TypeError("selection must be a FrozenSelectionResult")
    if type(expected_manifest_sha256) is not str:
        raise TypeError("expected_manifest_sha256 must be a str")
    if type(candidate_costs) is not tuple:
        raise TypeError("candidate_costs must be a tuple")
    if type(budget_vnd) is not int:
        raise TypeError("budget_vnd must be an int")
    if len(candidate_costs) != V1_OPTIMIZATION_CANDIDATE_COUNT:
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    for cost in candidate_costs:
        if type(cost) is not MaintenanceCostInput:
            raise TypeError("candidate costs must be MaintenanceCostInput records")
        _exact(cost.segment_id, str, "cost segment")
        _exact(cost.cost_vnd, int, "cost VND")
        _exact(cost.cost_as_of_date, date, "cost as-of date")
    _preflight_selection(selection)


def _preflight_selection(selection: FrozenSelectionResult) -> None:
    _exact(selection.manifest_sha256, str, "manifest digest")
    _exact(selection.relative_artifact_directory, str, "relative artifact directory")
    if type(selection.manifest) is not SelectedArtifactManifest:
        raise TypeError("manifest must be a SelectedArtifactManifest")
    if type(selection.risk_output) is not tuple:
        raise TypeError("risk output must be a tuple")
    if len(selection.risk_output) != 2100:
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    _preflight_manifest(selection.manifest)
    for row in selection.risk_output:
        if type(row) is not RiskOutput:
            raise TypeError("risk rows must be RiskOutput records")
        _exact(row.segment_id, str, "risk segment")
        _exact(row.date, date, "risk date")
        _exact(row.maintenance_probability, float, "risk probability")
        _exact(row.risk_score, int, "risk score")
        _exact(row.risk_band, str, "risk band")


def _exact(value: object, expected: type[Any], name: str) -> None:
    if type(value) is not expected:
        raise TypeError(f"{name} has the wrong type")


def _validate_input(
    selection: FrozenSelectionResult,
    expected_digest: str,
    costs: tuple[MaintenanceCostInput, ...],
    budget: int,
) -> tuple[tuple[_Candidate, ...], date]:
    _require_digest(expected_digest)
    _require_digest(selection.manifest_sha256)
    if selection.manifest_sha256 != expected_digest:
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    _validate_manifest(selection)
    evidence_date, final_rows = _validate_risk(selection)
    candidates = _validate_costs(final_rows, costs, evidence_date, budget)
    return candidates, evidence_date


def _validate_manifest(selection: FrozenSelectionResult) -> None:
    try:
        _validate_canonical_manifest(selection)
    except (ArithmeticError, AttributeError, IndexError, TypeError, ValueError):
        raise MaintenanceOptimizationError(_INPUT_ERROR) from None


def _validate_risk(selection: FrozenSelectionResult) -> tuple[date, tuple[RiskOutput, ...]]:
    manifest = selection.manifest
    rows = selection.risk_output
    if len(rows) != 2100:
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    if tuple((row.segment_id, row.date) for row in rows) != tuple(
        sorted((row.segment_id, row.date) for row in rows)
    ):
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    expected_dates = manifest.test_dates
    expected_set = set(expected_dates)
    segment_sets: dict[date, set[str]] = {value: set() for value in expected_dates}
    for row in rows:
        if not _ascii_identifier(row.segment_id) or row.date not in expected_set:
            raise MaintenanceOptimizationError(_INPUT_ERROR)
        if (
            not math.isfinite(row.maintenance_probability)
            or not 0.0 <= row.maintenance_probability <= 1.0
        ):
            raise MaintenanceOptimizationError(_INPUT_ERROR)
        if row.risk_score != risk_score_from_probability(row.maintenance_probability):
            raise MaintenanceOptimizationError(_INPUT_ERROR)
        if row.risk_band != _band_for_score(row.risk_score):
            raise MaintenanceOptimizationError(_INPUT_ERROR)
        if row.segment_id in segment_sets[row.date]:
            raise MaintenanceOptimizationError(_INPUT_ERROR)
        segment_sets[row.date].add(row.segment_id)
    if any(len(values) != V1_OPTIMIZATION_CANDIDATE_COUNT for values in segment_sets.values()):
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    canonical_segments = next(iter(segment_sets.values()))
    if any(values != canonical_segments for values in segment_sets.values()):
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    payload = _risk_jsonl(rows)
    record = selection.manifest.artifacts[3]
    if len(payload) != record.size_bytes or hashlib.sha256(payload).hexdigest() != record.sha256:
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    evidence_date = expected_dates[-1]
    final = tuple(row for row in rows if row.date == evidence_date)
    if len(final) != V1_OPTIMIZATION_CANDIDATE_COUNT:
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    return evidence_date, final


def _validate_costs(
    final_rows: tuple[RiskOutput, ...],
    costs: tuple[MaintenanceCostInput, ...],
    evidence_date: date,
    budget: int,
) -> tuple[_Candidate, ...]:
    if budget < 0 or budget > MAX_EXACT_VND:
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    by_segment: dict[str, MaintenanceCostInput] = {}
    for cost in costs:
        if (
            not _ascii_identifier(cost.segment_id)
            or cost.cost_vnd <= 0
            or cost.cost_vnd > MAX_EXACT_VND
        ):
            raise MaintenanceOptimizationError(_INPUT_ERROR)
        if cost.cost_as_of_date > evidence_date or cost.segment_id in by_segment:
            raise MaintenanceOptimizationError(_INPUT_ERROR)
        by_segment[cost.segment_id] = cost
    risk_by_segment = {row.segment_id: row for row in final_rows}
    if set(by_segment) != set(risk_by_segment) or len(costs) != V1_OPTIMIZATION_CANDIDATE_COUNT:
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    return tuple(
        _Candidate(
            segment_id,
            risk_by_segment[segment_id].maintenance_probability,
            risk_by_segment[segment_id].risk_score,
            cast_band(risk_by_segment[segment_id].risk_band),
            by_segment[segment_id].cost_vnd,
            by_segment[segment_id].cost_as_of_date,
        )
        for segment_id in sorted(risk_by_segment)
    )


def cast_band(value: str) -> Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
    if value not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
        raise MaintenanceOptimizationError(_INPUT_ERROR)
    return cast(Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"], value)


def _exact_optimum(candidates: tuple[_Candidate, ...], budget: int) -> tuple[_Candidate, ...]:
    states: list[_State | None] = [_State(0, 0)]
    total_score = 0
    for index, candidate in enumerate(candidates):
        if candidate.score == 0:
            continue
        previous_max = total_score
        total_score += candidate.score
        states.extend([None] * candidate.score)
        bit = 1 << (len(candidates) - index - 1)
        for score in range(previous_max, -1, -1):
            state = states[score]
            if state is None:
                continue
            cost = state.cost + candidate.cost
            if cost > budget:
                continue
            proposed = _State(cost, state.selected_mask | bit)
            target = score + candidate.score
            existing = states[target]
            if existing is None or _prefer_state(proposed, existing):
                states[target] = proposed
    for score in range(len(states) - 1, -1, -1):
        state = states[score]
        if state is not None:
            return tuple(
                candidate
                for index, candidate in enumerate(candidates)
                if state.selected_mask & (1 << (len(candidates) - index - 1))
            )
    raise MaintenanceOptimizationError(_OPTIMIZATION_ERROR)


def _prefer_state(left: _State, right: _State) -> bool:
    return left.cost < right.cost or (
        left.cost == right.cost and left.selected_mask > right.selected_mask
    )


def _independent_optimum(candidates: tuple[_Candidate, ...], budget: int) -> tuple[str, ...]:
    """Recheck the optimum with a separate sparse tuple-state oracle."""
    states: dict[int, tuple[int, tuple[str, ...]]] = {0: (0, ())}
    for candidate in candidates:
        if candidate.score == 0:
            continue
        additions: dict[int, tuple[int, tuple[str, ...]]] = {}
        for score, (cost, selected_ids) in tuple(states.items()):
            proposed_cost = cost + candidate.cost
            if proposed_cost > budget:
                continue
            target = score + candidate.score
            proposed = (proposed_cost, (*selected_ids, candidate.segment_id))
            existing = additions.get(target, states.get(target))
            if existing is None or proposed < existing:
                additions[target] = proposed
        for score, proposed in additions.items():
            existing = states.get(score)
            if existing is None or proposed < existing:
                states[score] = proposed
    best_score = max(states)
    return states[best_score][1]


def _build_result(
    source_manifest_sha256: str,
    source_risk_input_fingerprint: str,
    candidates: tuple[_Candidate, ...],
    selected: tuple[_Candidate, ...],
    evidence_date: date,
    budget: int,
    fingerprint: str,
) -> MaintenanceOptimizationResult:
    ordered = tuple(
        sorted(
            selected,
            key=lambda value: (-value.score, -value.probability, value.cost, value.segment_id),
        )
    )
    recommendations = tuple(
        MaintenanceRecommendation(
            index,
            value.segment_id,
            evidence_date,
            value.probability,
            value.score,
            value.band,
            value.cost,
            value.cost_as_of,
        )
        for index, value in enumerate(ordered, start=1)
    )
    selected_cost = sum(value.cost for value in selected)
    total_score = sum(value.score for value in selected)
    return MaintenanceOptimizationResult(
        MAINTENANCE_OPTIMIZATION_CONTRACT_VERSION,
        MAINTENANCE_OPTIMIZATION_USE_CASE,
        source_manifest_sha256,
        source_risk_input_fingerprint,
        fingerprint,
        evidence_date,
        evidence_date + timedelta(days=30),
        budget,
        selected_cost,
        budget - selected_cost,
        len(candidates),
        len(recommendations),
        total_score,
        recommendations,
    )


def _validate_output(
    result: MaintenanceOptimizationResult,
    candidates: tuple[_Candidate, ...],
    budget: int,
    evidence_date: date,
    source_manifest_sha256: str,
    source_risk_input_fingerprint: str,
    optimization_input_fingerprint: str,
) -> None:
    if (
        type(result) is not MaintenanceOptimizationResult
        or type(result.recommendations) is not tuple
    ):
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    if any(type(value) is not MaintenanceRecommendation for value in result.recommendations):
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    string_fields = (
        result.contract_version,
        result.use_case,
        result.source_manifest_sha256,
        result.source_risk_input_fingerprint,
        result.optimization_input_fingerprint,
    )
    date_fields = (result.evidence_date, result.risk_window_end)
    int_fields = (
        result.budget_vnd,
        result.selected_cost_vnd,
        result.remaining_budget_vnd,
        result.candidate_count,
        result.selected_count,
        result.total_risk_score,
    )
    if (
        any(type(value) is not str for value in string_fields)
        or any(type(value) is not date for value in date_fields)
        or any(type(value) is not int for value in int_fields)
    ):
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    for recommendation in result.recommendations:
        if (
            type(recommendation.priority_rank) is not int
            or type(recommendation.segment_id) is not str
            or type(recommendation.evidence_date) is not date
            or type(recommendation.maintenance_probability) is not float
            or type(recommendation.risk_score) is not int
            or type(recommendation.risk_band) is not str
            or type(recommendation.cost_vnd) is not int
            or type(recommendation.cost_as_of_date) is not date
        ):
            raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    if (
        result.contract_version != MAINTENANCE_OPTIMIZATION_CONTRACT_VERSION
        or result.use_case != MAINTENANCE_OPTIMIZATION_USE_CASE
        or result.source_manifest_sha256 != source_manifest_sha256
        or result.source_risk_input_fingerprint != source_risk_input_fingerprint
        or result.optimization_input_fingerprint != optimization_input_fingerprint
        or result.evidence_date != evidence_date
        or result.risk_window_end != evidence_date + timedelta(days=30)
        or result.budget_vnd != budget
        or result.candidate_count != len(candidates)
        or result.selected_count != len(result.recommendations)
    ):
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    candidates_by_id = {value.segment_id: value for value in candidates}
    if len({value.segment_id for value in result.recommendations}) != len(result.recommendations):
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    selected = tuple(candidates_by_id.get(value.segment_id) for value in result.recommendations)
    if any(value is None for value in selected):
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    selected_values = tuple(value for value in selected if value is not None)
    for recommendation, candidate in zip(result.recommendations, selected_values, strict=True):
        if (
            recommendation.evidence_date != evidence_date
            or recommendation.maintenance_probability != candidate.probability
            or recommendation.risk_score != candidate.score
            or recommendation.risk_band != candidate.band
            or recommendation.cost_vnd != candidate.cost
            or recommendation.cost_as_of_date != candidate.cost_as_of
        ):
            raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    if sum(value.cost for value in selected_values) != result.selected_cost_vnd:
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    if sum(value.score for value in selected_values) != result.total_risk_score:
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    if (
        result.selected_cost_vnd > budget
        or result.remaining_budget_vnd != budget - result.selected_cost_vnd
    ):
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    expected_order = tuple(
        sorted(
            selected_values,
            key=lambda value: (-value.score, -value.probability, value.cost, value.segment_id),
        )
    )
    if tuple(value.segment_id for value in result.recommendations) != tuple(
        value.segment_id for value in expected_order
    ):
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    if tuple(value.priority_rank for value in result.recommendations) != tuple(
        range(1, len(result.recommendations) + 1)
    ):
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)
    exact_ids = _independent_optimum(candidates, budget)
    if tuple(sorted(value.segment_id for value in selected_values)) != exact_ids:
        raise MaintenanceOptimizationError(_OUTPUT_ERROR)


def _optimization_fingerprint(
    source_manifest_sha256: str,
    source_risk_input_fingerprint: str,
    candidates: tuple[_Candidate, ...],
    budget: int,
    evidence_date: date,
) -> str:
    payload = {
        "budget_vnd": budget,
        "candidates": [
            [
                value.segment_id,
                evidence_date.isoformat(),
                _float_hex(value.probability),
                value.score,
                value.band,
                value.cost,
                value.cost_as_of.isoformat(),
            ]
            for value in candidates
        ],
        "columns": [
            "segment_id",
            "evidence_date",
            "maintenance_probability",
            "risk_score",
            "risk_band",
            "cost_vnd",
            "cost_as_of_date",
        ],
        "contract": MAINTENANCE_OPTIMIZATION_CONTRACT_VERSION,
        "objective": [
            "maximize_total_risk_score",
            "minimize_total_cost_vnd",
            "prefer_selected_lower_segment_id_at_first_difference",
        ],
        "source_manifest_sha256": source_manifest_sha256,
        "source_risk_input_fingerprint": source_risk_input_fingerprint,
        "use_case": MAINTENANCE_OPTIMIZATION_USE_CASE,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float_hex(value: float) -> str:
    return (0.0 if value == 0.0 else value).hex()


def _risk_jsonl(rows: tuple[RiskOutput, ...]) -> bytes:
    return b"".join(
        _artifacts._canonical_json(
            {
                "date": row.date.isoformat(),
                "maintenance_probability": row.maintenance_probability,
                "risk_band": row.risk_band,
                "risk_score": row.risk_score,
                "segment_id": row.segment_id,
            }
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _require_digest(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise MaintenanceOptimizationError(_INPUT_ERROR)


def _ascii_identifier(value: str) -> bool:
    return _SEGMENT_ID_RE.fullmatch(value) is not None


def _band_for_score(score: int) -> str:
    for name, lower, upper in _RISK_BANDS:
        if lower <= score <= upper:
            return name
    raise MaintenanceOptimizationError(_INPUT_ERROR)


__all__ = [
    "optimize_maintenance",
    "MaintenanceOptimizationError",
    "MaintenanceCostInput",
    "MaintenanceRecommendation",
    "MaintenanceOptimizationResult",
    "MAINTENANCE_OPTIMIZATION_CONTRACT_VERSION",
    "MAINTENANCE_OPTIMIZATION_USE_CASE",
    "MAX_EXACT_VND",
    "V1_OPTIMIZATION_CANDIDATE_COUNT",
]
