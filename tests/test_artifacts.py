"""Phase 13 frozen-selection artifact contract tests.

The contract under test is ``docs/contracts.md`` section 20.  The focused
tests deliberately use a small, valid deterministic export; the full suite
adds the locked V1 integration coverage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pickle
import shutil
import subprocess
from dataclasses import replace
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
    import roadguard.artifacts as artifacts
    from roadguard.artifacts import (
        ARTIFACT_FILENAMES,
        FROZEN_SELECTION_CONTRACT_VERSION,
        RISK_BAND_NAMES,
        ArtifactFile,
        FrozenSelectionResult,
        RiskOutput,
        SelectedArtifactManifest,
    )

    assert tuple(artifacts.__all__) == PUBLIC_NAMES
    assert all(hasattr(roadguard, name) and name in roadguard.__all__ for name in PUBLIC_NAMES)
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
    assert tuple(field.name for field in dataclasses.fields(SelectedArtifactManifest)) == (
        "contract_version",
        "training_fingerprint",
        "selection_fingerprint",
        "risk_input_fingerprint",
        "classifier_contract_version",
        "regressor_contract_version",
        "selected_classifier_name",
        "selected_regressor_name",
        "classifier_decision_threshold",
        "master_seed",
        "classifier_seed",
        "regressor_seed",
        "feature_columns",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_dates",
        "validation_dates",
        "test_dates",
        "classification_candidates",
        "regression_candidates",
        "risk_bands",
        "risk_output_rows",
        "runtime_versions",
        "platform_tag",
        "artifacts",
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
    assert result.relative_artifact_directory == (f"roadguard.phase13.v1/{result.manifest_sha256}")
    bundle = tmp_path.joinpath(*result.relative_artifact_directory.split("/"))
    assert tuple(path.name for path in sorted(bundle.iterdir())) == tuple(
        sorted(ARTIFACT_FILENAMES)
    )
    assert (
        hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
        == result.manifest_sha256
    )
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
        first_path = (tmp_path / "one").joinpath(
            *first.relative_artifact_directory.split("/"), filename
        )
        second_path = (tmp_path / "two").joinpath(
            *second.relative_artifact_directory.split("/"), filename
        )
        assert first_path.read_bytes() == second_path.read_bytes()


def test_valid_test_feature_change_updates_only_risk_input_provenance(
    dataset: RepositoryExport, canonical_state: tuple[object, object], tmp_path: Path
) -> None:
    from roadguard.artifacts import persist_selected_artifacts

    original_split, original_fit = canonical_state
    original = persist_selected_artifacts(
        dataset, original_split, original_fit, SPEC, _config(tmp_path / "original")
    )
    observations = dataset.observations.copy(deep=True)
    test_dates = pd.to_datetime(list(original_split.test_dates))
    observations.loc[observations["date"].isin(test_dates), "rainfall_mm"] += 1.0
    changed_dataset = replace(dataset, observations=observations)
    changed_split = split_chronologically(build_feature_frame(changed_dataset, SPEC), SPEC)
    changed_fit = fit_preprocessor(changed_split, SPEC)
    changed = persist_selected_artifacts(
        changed_dataset, changed_split, changed_fit, SPEC, _config(tmp_path / "changed")
    )

    assert changed.manifest.training_fingerprint == original.manifest.training_fingerprint
    assert changed.manifest.selection_fingerprint == original.manifest.selection_fingerprint
    assert changed.manifest.risk_input_fingerprint != original.manifest.risk_input_fingerprint
    assert changed.manifest_sha256 != original.manifest_sha256
    for filename in ("preprocessor.json", "classifier.joblib", "regressor.joblib"):
        original_path = (tmp_path / "original").joinpath(
            *original.relative_artifact_directory.split("/"), filename
        )
        changed_path = (tmp_path / "changed").joinpath(
            *changed.relative_artifact_directory.split("/"), filename
        )
        assert original_path.read_bytes() == changed_path.read_bytes()


def test_private_orchestration_serializes_winners_before_one_selected_test_call(
    dataset: RepositoryExport,
    canonical_state: tuple[object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import joblib
    import joblib.numpy_pickle as joblib_numpy_pickle

    import roadguard.artifacts as artifacts
    import roadguard.classification as classification
    import roadguard.regression as regression

    split, fit = canonical_state
    transform = artifacts.transform
    write_pretest_payloads = artifacts._write_pretest_payloads
    positive_probas = artifacts._positive_probas
    regression_predictions = artifacts._regression_predictions
    events: list[tuple[str, int | str]] = []
    serialized_classifier_ids: list[int] = []

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Phase 13 must not call a public evaluator or model loader")

    def partition_name(frame: object) -> str:
        assert isinstance(frame, pd.DataFrame)
        dates = tuple(sorted(frame["date"].dt.date.unique()))
        expected = {
            "train": split.train_dates,
            "validation": split.validation_dates,
            "test": split.test_dates,
        }
        return next(name for name, partition_dates in expected.items() if dates == partition_dates)

    def track_transform(frame: object, supplied_fit: object) -> object:
        events.append((f"transform:{partition_name(frame)}", ""))
        return transform(frame, supplied_fit)

    def track_serialization(
        stage: object, supplied_fit: object, classifier: object, regressor: object
    ) -> object:
        serialized_classifier_ids.append(id(classifier))
        events.append(("serialize", id(classifier)))
        return write_pretest_payloads(stage, supplied_fit, classifier, regressor)

    def track_probabilities(classifier: object, matrix: object) -> object:
        events.append(("probabilities", id(classifier)))
        return positive_probas(classifier, matrix)

    def track_regression_predictions(regressor: object, matrix: object) -> object:
        events.append(("regression_predictions", id(regressor)))
        return regression_predictions(regressor, matrix)

    monkeypatch.setattr(artifacts, "transform", track_transform)
    monkeypatch.setattr(artifacts, "_write_pretest_payloads", track_serialization)
    monkeypatch.setattr(artifacts, "_positive_probas", track_probabilities)
    monkeypatch.setattr(artifacts, "_regression_predictions", track_regression_predictions)
    monkeypatch.setattr(classification, "evaluate_advanced_classifier", forbidden)
    monkeypatch.setattr(regression, "evaluate_advanced_regressor", forbidden)
    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(joblib_numpy_pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "load", forbidden)
    monkeypatch.setattr(pickle, "loads", forbidden)

    artifacts.persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path))

    transform_events = [event[0] for event in events if event[0].startswith("transform:")]
    assert transform_events == ["transform:train", "transform:validation", "transform:test"]
    serialization_index = next(
        index for index, event in enumerate(events) if event[0] == "serialize"
    )
    test_transform_index = next(
        index for index, event in enumerate(events) if event[0] == "transform:test"
    )
    assert test_transform_index > serialization_index
    assert [
        event[0] for event in events[:serialization_index] if event[0].startswith("transform:")
    ] == ["transform:train", "transform:validation"]
    after_serialization = events[serialization_index + 1 :]
    test_probability_events = [
        event for event in after_serialization if event[0] == "probabilities"
    ]
    assert len(serialized_classifier_ids) == 1
    assert test_probability_events == [("probabilities", serialized_classifier_ids[0])]
    assert not any(event[0] == "regression_predictions" for event in after_serialization)


def test_tampered_existing_bundle_fails_closed(
    dataset: RepositoryExport, canonical_state: tuple[object, object], tmp_path: Path
) -> None:
    from roadguard.artifacts import ArtifactPersistenceError, persist_selected_artifacts

    split, fit = canonical_state
    result = persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path))
    bundle = tmp_path.joinpath(*result.relative_artifact_directory.split("/"))
    (bundle / "classifier.joblib").write_bytes(b"tampered")
    with pytest.raises(ArtifactPersistenceError, match="Phase 13 artifact verification failed\\."):
        persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path))


def test_oversized_existing_manifest_fails_without_unbounded_read(
    dataset: RepositoryExport, canonical_state: tuple[object, object], tmp_path: Path
) -> None:
    import roadguard.artifacts as artifacts

    split, fit = canonical_state
    result = artifacts.persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path))
    bundle = tmp_path.joinpath(*result.relative_artifact_directory.split("/"))
    (bundle / "manifest.json").write_bytes(b"0" * (artifacts._MAX_MANIFEST_BYTES + 1))

    with pytest.raises(
        artifacts.ArtifactPersistenceError, match="Phase 13 artifact verification failed\\."
    ):
        artifacts.persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path))


def test_root_path_is_rejected(
    dataset: RepositoryExport, canonical_state: tuple[object, object]
) -> None:
    from roadguard.artifacts import ArtifactPersistenceError, persist_selected_artifacts

    split, fit = canonical_state
    root = Path(Path.cwd().anchor)
    with pytest.raises(
        ArtifactPersistenceError, match="Phase 13 artifact path validation failed\\."
    ):
        persist_selected_artifacts(dataset, split, fit, SPEC, _config(root))


def test_non_anchor_mount_component_is_rejected(
    dataset: RepositoryExport,
    canonical_state: tuple[object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import roadguard.artifacts as artifacts

    split, fit = canonical_state
    original_ismount = os.path.ismount
    monkeypatch.setattr(
        os.path,
        "ismount",
        lambda path: Path(path) == tmp_path or original_ismount(path),
    )

    with pytest.raises(
        artifacts.ArtifactPersistenceError, match="Phase 13 artifact path validation failed\\."
    ):
        artifacts.persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path))


def test_publication_uses_no_replace_rename(
    dataset: RepositoryExport,
    canonical_state: tuple[object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import roadguard.artifacts as artifacts

    split, fit = canonical_state

    def replacement_rename(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Phase 13 must not use a replacing rename")

    monkeypatch.setattr(os, "replace", replacement_rename)
    result = artifacts.persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path))

    assert result.manifest_sha256


def test_changed_stage_identity_is_never_removed(
    dataset: RepositoryExport,
    canonical_state: tuple[object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import roadguard.artifacts as artifacts

    split, fit = canonical_state
    replaced_stages: list[Path] = []

    def replace_stage(*args: object, **_kwargs: object) -> object:
        stage = args[0]
        assert isinstance(stage, Path)
        shutil.rmtree(stage)
        stage.mkdir()
        replaced_stages.append(stage)
        raise artifacts.ArtifactPersistenceError("Phase 13 artifact serialization failed.")

    monkeypatch.setattr(artifacts, "_write_pretest_payloads", replace_stage)

    with pytest.raises(artifacts.ArtifactPersistenceError):
        artifacts.persist_selected_artifacts(dataset, split, fit, SPEC, _config(tmp_path))

    assert len(replaced_stages) == 1
    assert replaced_stages[0].is_dir()


def test_private_io_boundaries_fail_closed_with_sanitized_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import roadguard._artifact_io as artifact_io

    def dump_failure(*_args: object, **_kwargs: object) -> object:
        raise ValueError("sensitive serialized value")

    monkeypatch.setattr(artifact_io.joblib, "dump", dump_failure)
    with pytest.raises(
        artifact_io.ArtifactPersistenceError, match="Phase 13 artifact serialization failed\\."
    ):
        artifact_io.dump_joblib(object(), tmp_path / "classifier.joblib")
    with pytest.raises(
        artifact_io.ArtifactPersistenceError, match="Phase 13 artifact write failed\\."
    ):
        artifact_io.write_bytes(tmp_path / "missing" / "payload.json", b"payload")
    with pytest.raises(
        artifact_io.ArtifactPersistenceError,
        match="Phase 13 artifact path validation failed\\.",
    ):
        artifact_io.prepare_root(tmp_path / "missing" / "nested")
    blocked = tmp_path / "not-a-directory"
    blocked.write_bytes(b"not a directory")
    with pytest.raises(
        artifact_io.ArtifactPersistenceError,
        match="Phase 13 artifact path validation failed\\.",
    ):
        artifact_io.mkdir_checked(blocked)
    with pytest.raises(
        artifact_io.ArtifactPersistenceError,
        match="Phase 13 artifact path validation failed\\.",
    ):
        artifact_io.require_safe_path(tmp_path / "missing", tmp_path)
    with pytest.raises(
        artifact_io.ArtifactPersistenceError,
        match="Phase 13 artifact verification failed\\.",
    ):
        artifact_io.verify_bundle(
            tmp_path,
            "0" * 64,
            b"{}\n",
            (),
            ("manifest.json",),
        )


def test_prepare_root_rejects_symlink_ancestor_before_creating_leaf(tmp_path: Path) -> None:
    import roadguard._artifact_io as artifact_io

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_ancestor = tmp_path / "linked-ancestor"
    try:
        linked_ancestor.symlink_to(outside, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        command = f'mklink /J "{linked_ancestor}" "{outside}"'
        completed = subprocess.run(
            ["cmd", "/d", "/c", command],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip("the current Windows policy does not permit symlinks or junctions")

    with pytest.raises(
        artifact_io.ArtifactPersistenceError,
        match="Phase 13 artifact path validation failed\\.",
    ):
        artifact_io.prepare_root(linked_ancestor / "artifacts")

    assert not (outside / "artifacts").exists()


def test_publish_syncs_parent_and_translates_sync_failure(
    dataset: RepositoryExport,
    canonical_state: tuple[object, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import roadguard._artifact_io as artifact_io
    import roadguard.artifacts as artifacts

    split, fit = canonical_state
    synchronized: list[Path] = []
    original_sync = artifact_io.sync_directory

    def track_sync(directory: Path) -> None:
        synchronized.append(directory)
        original_sync(directory)

    monkeypatch.setattr(artifact_io, "sync_directory", track_sync)
    successful = artifacts.persist_selected_artifacts(
        dataset, split, fit, SPEC, _config(tmp_path / "successful")
    )
    bundle = (tmp_path / "successful").joinpath(*successful.relative_artifact_directory.split("/"))
    assert synchronized == [bundle.parent]

    def sync_failure(_directory: Path) -> None:
        raise OSError("sync failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(artifact_io, "sync_directory", sync_failure)
        with pytest.raises(
            artifacts.ArtifactPersistenceError,
            match="Phase 13 artifact publication failed\\.",
        ):
            artifacts.persist_selected_artifacts(
                dataset, split, fit, SPEC, _config(tmp_path / "sync-failure")
            )


def test_wrong_top_level_types_fail_before_fields_are_read() -> None:
    from roadguard.artifacts import persist_selected_artifacts

    with pytest.raises(TypeError, match="dataset must be a RepositoryExport"):
        persist_selected_artifacts(object(), object(), object(), object(), object())
