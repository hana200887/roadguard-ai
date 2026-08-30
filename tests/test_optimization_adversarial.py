"""Adversarial review regressions for Phase 15 maintenance optimization."""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import test_optimization as base


def test_source_rejects_malformed_ids_and_noncanonical_manifest() -> None:
    module = base._optimization()
    selection = base._selection({base.SEGMENTS[0]: 0.5})
    malformed_rows = tuple(
        replace(row, segment_id="BAD") if row.segment_id == base.SEGMENTS[0] else row
        for row in selection.risk_output
    )
    malformed = base._replace_risk(selection, malformed_rows)
    malformed_costs = tuple(
        replace(cost, segment_id="BAD") if cost.segment_id == base.SEGMENTS[0] else cost
        for cost in base._costs(module)
    )
    forged_manifests = (
        replace(selection.manifest, classifier_contract_version="forged.phase11"),
        replace(selection.manifest, selected_classifier_name="forged-model"),
        replace(selection.manifest, runtime_versions=(("python", "3.12"),) * 5),
        replace(
            selection.manifest,
            train_dates=base._month_dates(2023, 1, 34),
            validation_dates=base._month_dates(2025, 11, 7),
            test_dates=base._month_dates(2026, 6, 7),
        ),
        replace(
            selection.manifest,
            feature_columns=("not_a_phase8_feature", *base.EXPECTED_FEATURE_COLUMNS[1:]),
        ),
        replace(selection.manifest, classifier_seed=selection.manifest.classifier_seed + 1),
    )
    with pytest.raises(module.MaintenanceOptimizationError, match=f"^{base.INPUT_ERROR}$"):
        module.optimize_maintenance(
            malformed, malformed.manifest_sha256, malformed_costs, 1_000_000
        )
    for forged_manifest in forged_manifests:
        forged = base._replace_manifest(selection, forged_manifest)
        with pytest.raises(module.MaintenanceOptimizationError, match=f"^{base.INPUT_ERROR}$"):
            module.optimize_maintenance(forged, forged.manifest_sha256, base._costs(module), 1)


def test_output_rechecks_provenance_fingerprint_and_optimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = base._optimization()
    selection = base._selection({base.SEGMENTS[0]: 1.0})
    costs = base._costs(module, {base.SEGMENTS[0]: (1, base.TEST_DATES[-1])})
    result = module.optimize_maintenance(selection, selection.manifest_sha256, costs, 1)
    candidates, evidence_date = module._validate_input(
        selection, selection.manifest_sha256, costs, 1
    )
    arguments = (
        candidates,
        1,
        evidence_date,
        selection.manifest_sha256,
        selection.manifest.risk_input_fingerprint,
        result.optimization_input_fingerprint,
    )
    forged_results = (
        replace(result, source_manifest_sha256="0" * 64),
        replace(result, source_risk_input_fingerprint="0" * 64),
        replace(result, optimization_input_fingerprint="0" * 64),
        replace(result, selected_count=True),
        replace(
            result,
            recommendations=(replace(result.recommendations[0], priority_rank=True),),
        ),
        replace(
            result,
            recommendations=(
                replace(
                    result.recommendations[0],
                    maintenance_probability=np.float64(
                        result.recommendations[0].maintenance_probability
                    ),
                ),
            ),
        ),
        replace(
            result,
            recommendations=(),
            selected_count=0,
            selected_cost_vnd=0,
            remaining_budget_vnd=1,
            total_risk_score=0,
        ),
    )
    for forged in forged_results:
        with pytest.raises(module.MaintenanceOptimizationError, match=f"^{base.OUTPUT_ERROR}$"):
            module._validate_output(forged, *arguments)
    monkeypatch.setattr(module, "_exact_optimum", lambda _candidates, _budget: ())
    with pytest.raises(module.MaintenanceOptimizationError, match=f"^{base.OUTPUT_ERROR}$"):
        module.optimize_maintenance(selection, selection.manifest_sha256, costs, 1)


def test_authenticated_provenance_is_snapshotted_before_optimization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = base._optimization()
    selection = base._selection({base.SEGMENTS[0]: 1.0})
    trusted_digest = selection.manifest_sha256
    trusted_risk_fingerprint = selection.manifest.risk_input_fingerprint
    original = module._exact_optimum

    def mutate_after_validation(candidates: Any, budget: int) -> Any:
        object.__setattr__(selection, "manifest_sha256", "0" * 64)
        object.__setattr__(selection.manifest, "risk_input_fingerprint", "1" * 64)
        return original(candidates, budget)

    monkeypatch.setattr(module, "_exact_optimum", mutate_after_validation)
    result = module.optimize_maintenance(
        selection,
        trusted_digest,
        base._costs(module, {base.SEGMENTS[0]: (1, base.TEST_DATES[-1])}),
        1,
    )
    assert result.source_manifest_sha256 == trusted_digest
    assert result.source_risk_input_fingerprint == trusted_risk_fingerprint


def test_snapshot_ignores_hostile_undeclared_attributes() -> None:
    module = base._optimization()
    selection = base._selection({base.SEGMENTS[0]: 1.0})
    costs = base._costs(module, {base.SEGMENTS[0]: (1, base.TEST_DATES[-1])})
    calls: list[str] = []

    class Poison:
        def __deepcopy__(self, _memo: object) -> object:
            calls.append("called")
            raise RuntimeError("SECRET_MARKER")

    poisoned = Poison()
    for record in (
        selection,
        selection.manifest,
        selection.manifest.artifacts[0],
        selection.manifest.classification_candidates[0],
        selection.risk_output[0],
        costs[0],
    ):
        object.__setattr__(record, "undeclared_poison", poisoned)
    result = module.optimize_maintenance(selection, selection.manifest_sha256, costs, 1)
    assert result.selected_count == 1
    assert calls == []


def test_seed_authentication_uses_no_rng_api(monkeypatch: pytest.MonkeyPatch) -> None:
    module = base._optimization()
    selection = base._selection({base.SEGMENTS[0]: 1.0})
    validation = importlib.import_module("roadguard._optimization_manifest")
    known_vectors = (
        (1, 1332403962, 3965004449, 3190592876),
        (42, 2546419119, 1504567447, 426177242),
        (2**32, 2546735900, 901563092, 2398659316),
        (2**64 + 123, 181850524, 2755056499, 1002102983),
    )
    for master_seed, classifier_zero, classifier_one, regressor_one in known_vectors:
        assert validation._seed_sequence_word((master_seed, 0x5247311, 0)) == classifier_zero
        assert validation._seed_sequence_word((master_seed, 0x5247311, 1)) == classifier_one
        assert validation._seed_sequence_word((master_seed, 0x5247312, 1)) == regressor_one

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Phase 15 must not invoke an RNG API")

    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    result = module.optimize_maintenance(
        selection,
        selection.manifest_sha256,
        base._costs(module, {base.SEGMENTS[0]: (1, base.TEST_DATES[-1])}),
        1,
    )
    assert result.selected_count == 1
