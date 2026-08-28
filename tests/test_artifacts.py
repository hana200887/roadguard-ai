"""Phase 13 frozen-selection artifact contract tests.

The contract under test is ``docs/contracts.md`` section 20.  The focused
tests deliberately use a small, valid deterministic export; the full suite
adds the locked V1 integration coverage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import roadguard
from roadguard import (
    DatasetSpec,
    RepositoryExport,
    RoadGuardConfig,
    clean_raw_dataset,
    derive_observation_targets,
    generate_accident_timeline,
    generate_maintenance_events,
    generate_observations,
    generate_segments,
)
from roadguard.features import build_feature_frame
from roadguard.preprocessing import fit_preprocessor, split_chronologically


SPEC = DatasetSpec(dataset_segments=5, dataset_months_per_segment=48, dataset_observations=240)
PUBLIC_NAMES = (
    "persist_selected_artifacts",
    "FrozenSelectionError",
    "ArtifactPersistenceError",
    "ArtifactFile",
    "RiskOutput",
    "SelectedArtifactManifest",
    "FrozenSelectionResult",
    "FROZEN_SELECTION_CONTRACT_VERSION",
    "ARTIFACT_FILENAMES",
    "RISK_BAND_NAMES",
)


@pytest.fixture(scope="module")
def dataset() -> RepositoryExport:
    segments = generate_segments(SPEC, 42)
    events = generate_maintenance_events(segments, SPEC, 42)
    accidents = generate_accident_timeline(segments, SPEC, 42)
    observations = generate_observations(segments, events, accidents, SPEC, 42)
    targets = derive_observation_targets(observations, events)
    cleaned = clean_raw_dataset(segments, observations, targets, events, SPEC)
    return RepositoryExport(
        segments=cleaned.segments,
        observations=cleaned.observations,
        targets=cleaned.targets,
        maintenance_events=cleaned.maintenance_events,
    )


@pytest.fixture(scope="module")
def canonical_state(dataset: RepositoryExport):  # type: ignore[no-untyped-def]
    split = split_chronologically(build_feature_frame(dataset, SPEC), SPEC)
    return split, fit_preprocessor(split, SPEC)


def _config(root: Path) -> RoadGuardConfig:
    return RoadGuardConfig(seed=42, artifacts_dir=root)


def test_public_surface_and_frozen_schema_are_exact() -> None:
    from roadguard.artifacts import (
        ARTIFACT_FILENAMES,
        FROZEN_SELECTION_CONTRACT_VERSION,
        RISK_BAND_NAMES,
        ArtifactFile,
        FrozenSelectionResult,
        RiskOutput,
        SelectedArtifactManifest,
    )

    import roadguard.artifacts as artifacts

    assert tuple(artifacts.__all__) == PUBLIC_NAMES
    assert all(hasattr(roadguard, name) for name in PUBLIC_NAMES)
    assert FROZEN_SELECTION_CONTRACT_VERSION == "roadguard.phase13.v1"
    assert ARTIFACT_FILENAMES == (
        "preprocessor.json",
        "classifier.joblib",
        "regressor.joblib",
        "test-risk.jsonl",
        "manifest.json",
    )
    assert RISK_BAND_NAMES == ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert tuple(field.name for field in dataclasses.fields(ArtifactFile)) == (
        "role",
        "filename",
        "sha256",
        "size_bytes",
    )
    assert tuple(field.name for field in dataclasses.fields(RiskOutput)) == (
        "segment_id",
        "date",
        "maintenance_probability",
        "risk_score",
        "risk_band",
    )
    assert tuple(field.name for field in dataclasses.fields(SelectedArtifactManifest))[:4] == (
        "contract_version",
        "training_fingerprint",
        "selection_fingerprint",
        "risk_input_fingerprint",
    )
    assert tuple(field.name for field in dataclasses.fields(FrozenSelectionResult)) == (
        "manifest_sha256",
        "relative_artifact_directory",
        "manifest",
        "risk_output",
    )
    assert all(
        cls.__dataclass_params__.frozen
        for cls in (ArtifactFile, RiskOutput, SelectedArtifactManifest, FrozenSelectionResult)
    )


def test_persists_canonical_target_free_bundle_and_is_idempotent(
    dataset: RepositoryExport, canonical_state: tuple[object, object], tmp_path: Path
) -> None:
    from roadguard.artifacts import ARTIFACT_FILENAMES, persist_selected_artifacts

    split, fit = canonical_state
    result = persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path))
    repeated = persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path))

    assert repeated == result
    assert result.relative_artifact_directory == (
        f"roadguard.phase13.v1/{result.manifest_sha256}"
    )
    bundle = tmp_path.joinpath(*result.relative_artifact_directory.split("/"))
    assert tuple(path.name for path in sorted(bundle.iterdir())) == tuple(sorted(ARTIFACT_FILENAMES))
    assert hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest() == result.manifest_sha256
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["risk_input_fingerprint"] == result.manifest.risk_input_fingerprint
    assert len(result.risk_output) == len(split.test)
    assert all(
        0.0 <= row.maintenance_probability <= 1.0
        and 0 <= row.risk_score <= 100
        and row.risk_band in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        for row in result.risk_output
    )
    risk_text = (bundle / "test-risk.jsonl").read_text(encoding="utf-8")
    assert "maintenance_within_30_days" not in risk_text
    assert "days_until_maintenance" not in risk_text


def test_rebuild_in_an_isolated_root_has_identical_bytes(
    dataset: RepositoryExport, canonical_state: tuple[object, object], tmp_path: Path
) -> None:
    from roadguard.artifacts import ARTIFACT_FILENAMES, persist_selected_artifacts

    split, fit = canonical_state
    first = persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path / "one"))
    second = persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path / "two"))
    assert first == second
    for filename in ARTIFACT_FILENAMES:
        first_path = (tmp_path / "one").joinpath(*first.relative_artifact_directory.split("/"), filename)
        second_path = (tmp_path / "two").joinpath(*second.relative_artifact_directory.split("/"), filename)
        assert first_path.read_bytes() == second_path.read_bytes()


def test_wrong_top_level_types_fail_before_fields_are_read() -> None:
    from roadguard.artifacts import persist_selected_artifacts

    with pytest.raises(TypeError, match="dataset must be a RepositoryExport"):
        persist_selected_artifacts(object(), object(), object(), object(), object())
