"""Phase 13 frozen selection, local artifact publication, and risk mapping."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import secrets
import sysconfig
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (  # type: ignore[import-untyped]
    average_precision_score,
    f1_score,
    mean_absolute_error,
    recall_score,
    root_mean_squared_error,
)

import roadguard._artifact_io as _artifact_io
from roadguard._db_types import RepositoryExport
from roadguard.classification import (
    ADVANCED_CLASSIFIER_CONTRACT_VERSION,
    CANDIDATE_CLASSIFIER_NAMES,
    CandidateValidationMetrics,
    _build_candidate,
    _derive_candidate_seed,
    _metric_probability,
    _positive_probas,
    _select_threshold,
)
from roadguard.classification import (
    _fit_candidate as _fit_classifier,
)
from roadguard.classification import (
    _select_candidate_index as _select_classifier,
)
from roadguard.config import RoadGuardConfig
from roadguard.contracts import DatasetSpec
from roadguard.eda import _training_fingerprint
from roadguard.features import FEATURE_FRAME_COLUMNS, FEATURE_KEY_COLUMNS, build_feature_frame
from roadguard.preprocessing import (
    ChronologicalSplit,
    PreprocessorFit,
    fit_preprocessor,
    split_chronologically,
    transform,
)
from roadguard.regression import (
    ADVANCED_REGRESSOR_CONTRACT_VERSION,
    CANDIDATE_REGRESSOR_NAMES,
    CandidateRegressionValidationMetrics,
    _build_candidates,
    _metric_nonnegative,
    _regression_predictions,
)
from roadguard.regression import (
    _fit_candidate as _fit_regressor,
)
from roadguard.regression import (
    _select_candidate_index as _select_regressor,
)
from roadguard.risk import risk_score_from_probability
from roadguard.targets import TARGET_COLUMNS

FROZEN_SELECTION_CONTRACT_VERSION: Final[str] = "roadguard.phase13.v1"
ARTIFACT_FILENAMES: Final[tuple[str, str, str, str, str]] = (
    "preprocessor.json",
    "classifier.joblib",
    "regressor.joblib",
    "test-risk.jsonl",
    "manifest.json",
)
RISK_BAND_NAMES: Final[tuple[str, str, str, str]] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

_CLASSIFICATION_TARGET: Final[str] = TARGET_COLUMNS[3]
_REGRESSION_TARGET: Final[str] = TARGET_COLUMNS[2]
_VERSION_DIRECTORY: Final[str] = FROZEN_SELECTION_CONTRACT_VERSION
_MAX_MANIFEST_BYTES: Final[int] = _artifact_io.MAX_MANIFEST_BYTES
ArtifactPersistenceError = _artifact_io.ArtifactPersistenceError


class FrozenSelectionError(ValueError):
    """Raised when frozen Phase 13 selection or risk data is invalid."""


@dataclass(frozen=True)
class ArtifactFile:
    role: Literal["preprocessor", "classifier", "regressor", "test_risk"]
    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class RiskOutput:
    segment_id: str
    date: date
    maintenance_probability: float
    risk_score: int
    risk_band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass(frozen=True)
class SelectedArtifactManifest:
    contract_version: str
    training_fingerprint: str
    selection_fingerprint: str
    risk_input_fingerprint: str
    classifier_contract_version: str
    regressor_contract_version: str
    selected_classifier_name: str
    selected_regressor_name: str
    classifier_decision_threshold: float
    master_seed: int
    classifier_seed: int
    regressor_seed: int | None
    feature_columns: tuple[str, ...]
    train_rows: int
    validation_rows: int
    test_rows: int
    train_dates: tuple[date, ...]
    validation_dates: tuple[date, ...]
    test_dates: tuple[date, ...]
    classification_candidates: tuple[CandidateValidationMetrics, CandidateValidationMetrics]
    regression_candidates: tuple[
        CandidateRegressionValidationMetrics, CandidateRegressionValidationMetrics
    ]
    risk_bands: tuple[tuple[Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"], int, int], ...]
    risk_output_rows: int
    runtime_versions: tuple[tuple[str, str], ...]
    platform_tag: str
    artifacts: tuple[ArtifactFile, ArtifactFile, ArtifactFile, ArtifactFile]


@dataclass(frozen=True)
class FrozenSelectionResult:
    manifest_sha256: str
    relative_artifact_directory: str
    manifest: SelectedArtifactManifest
    risk_output: tuple[RiskOutput, ...]


def persist_selected_artifacts(
    dataset: RepositoryExport,
    split: ChronologicalSplit,
    fit: PreprocessorFit,
    spec: DatasetSpec,
    config: RoadGuardConfig,
) -> FrozenSelectionResult:
    """Freeze validation-selected models, publish a verified bundle, and map test risk."""
    _require_types(dataset, split, fit, spec, config)
    seed = _require_seed(config)
    root = _prepare_root(config.artifacts_dir)
    copied = _copy_export(dataset)
    try:
        frame = build_feature_frame(copied, spec)
        canonical_split = split_chronologically(frame, spec)
        _require_matching_split(split, canonical_split)
        canonical_fit = fit_preprocessor(canonical_split, spec)
        _require_matching_fit(fit, canonical_fit)
        train = transform(canonical_split.train, canonical_fit)
        validation = transform(canonical_split.validation, canonical_fit)
    except FrozenSelectionError:
        raise
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, RuntimeError)):
            raise
        raise FrozenSelectionError("Phase 13 input validation failed.") from None

    train_targets = _join_targets(copied.targets, train.keys, "training")
    validation_targets = _join_targets(copied.targets, validation.keys, "validation")
    x_train = train.features.to_numpy(dtype="float64")
    x_validation = validation.features.to_numpy(dtype="float64")
    classifier_state = _select_classifier_state(
        x_train,
        x_validation,
        train_targets[_CLASSIFICATION_TARGET].to_numpy(dtype="int64"),
        validation_targets[_CLASSIFICATION_TARGET].to_numpy(dtype="int64"),
        seed,
    )
    regressor_state = _select_regressor_state(
        x_train,
        x_validation,
        train_targets[_REGRESSION_TARGET].to_numpy(dtype="int64"),
        validation_targets[_REGRESSION_TARGET].to_numpy(dtype="int64"),
        seed,
    )

    training_join = train.keys.merge(
        copied.targets.loc[:, list(TARGET_COLUMNS)],
        on=list(FEATURE_KEY_COLUMNS),
        how="left",
        sort=False,
        validate="one_to_one",
    )
    training_join = canonical_split.train.merge(
        training_join,
        on=list(FEATURE_KEY_COLUMNS),
        how="left",
        sort=False,
        validate="one_to_one",
    )
    training_fingerprint = _training_fingerprint(training_join, canonical_split, spec)
    selection_fingerprint = _partition_fingerprint(
        "validation_dates",
        "validation_rows",
        canonical_split.validation,
        validation_targets,
        list(FEATURE_FRAME_COLUMNS) + list(TARGET_COLUMNS[2:]),
        canonical_split.validation_dates,
        spec,
    )
    risk_input_fingerprint = _partition_fingerprint(
        "test_dates",
        "test_rows",
        canonical_split.test,
        None,
        list(FEATURE_FRAME_COLUMNS),
        canonical_split.test_dates,
        spec,
    )

    version_dir = root / _VERSION_DIRECTORY
    _artifact_io.mkdir_checked(version_dir)
    stage = version_dir / f".stage-{secrets.token_hex(16)}"
    stage_identity: tuple[int, int] | None = None
    try:
        stage.mkdir()
        _artifact_io.require_safe_path(stage, version_dir)
        stage_identity = _artifact_io.file_identity(stage)
        artifacts = _write_pretest_payloads(
            stage, canonical_fit, classifier_state[0], regressor_state[0]
        )
        test = transform(canonical_split.test, canonical_fit)
        probabilities = _positive_probas(
            classifier_state[0], test.features.to_numpy(dtype="float64")
        )
        risk_output = _risk_output(test.keys, probabilities)
        risk_record = _write_risk_payload(stage, risk_output)
        artifact_records = (*artifacts, risk_record)
        manifest = _manifest(
            training_fingerprint,
            selection_fingerprint,
            risk_input_fingerprint,
            classifier_state,
            regressor_state,
            canonical_split,
            canonical_fit,
            seed,
            artifact_records,
            len(risk_output),
        )
        manifest_bytes = _canonical_json_bytes(_manifest_projection(manifest))
        _artifact_io.write_bytes(stage / "manifest.json", manifest_bytes)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        destination = version_dir / manifest_digest
        result = _publish_or_verify(stage, destination, manifest_digest, manifest, risk_output)
        return result
    except FrozenSelectionError:
        raise
    except ArtifactPersistenceError:
        raise
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, RuntimeError)):
            raise
        raise ArtifactPersistenceError("Phase 13 artifact publication failed.") from None
    finally:
        if stage_identity is not None:
            _artifact_io.cleanup_stage(stage, version_dir, stage_identity)


def _require_types(
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


def _require_seed(config: RoadGuardConfig) -> int:
    seed = config.seed
    if type(seed) is not int or seed < 1:
        raise FrozenSelectionError("Phase 13 input validation failed.")
    return seed


def _copy_export(dataset: RepositoryExport) -> RepositoryExport:
    frames = (dataset.segments, dataset.observations, dataset.targets, dataset.maintenance_events)
    if any(type(frame) is not pd.DataFrame for frame in frames):
        raise FrozenSelectionError("Phase 13 input validation failed.")
    return RepositoryExport(*(frame.copy(deep=True) for frame in frames))


def _require_matching_split(supplied: ChronologicalSplit, canonical: ChronologicalSplit) -> None:
    supplied_values = (supplied.train, supplied.validation, supplied.test)
    canonical_values = (canonical.train, canonical.validation, canonical.test)
    if any(type(value) is not pd.DataFrame for value in supplied_values):
        raise FrozenSelectionError("Phase 13 input validation failed.")
    for supplied_frame, canonical_frame in zip(supplied_values, canonical_values, strict=True):
        if tuple(supplied_frame.columns) != tuple(canonical_frame.columns):
            raise FrozenSelectionError("Phase 13 input validation failed.")
        if list(supplied_frame.dtypes) != list(canonical_frame.dtypes):
            raise FrozenSelectionError("Phase 13 input validation failed.")
        if not supplied_frame.reset_index(drop=True).equals(canonical_frame.reset_index(drop=True)):
            raise FrozenSelectionError("Phase 13 input validation failed.")
    if (
        supplied.train_dates,
        supplied.validation_dates,
        supplied.test_dates,
    ) != (canonical.train_dates, canonical.validation_dates, canonical.test_dates):
        raise FrozenSelectionError("Phase 13 input validation failed.")


def _require_matching_fit(supplied: PreprocessorFit, canonical: PreprocessorFit) -> None:
    for name in ("scaled_columns", "means", "stds", "province_categories", "road_type_categories"):
        if getattr(supplied, name) != getattr(canonical, name):
            raise FrozenSelectionError("Phase 13 input validation failed.")
    if supplied.transformed_feature_columns != canonical.transformed_feature_columns:
        raise FrozenSelectionError("Phase 13 input validation failed.")


def _join_targets(targets: pd.DataFrame, keys: pd.DataFrame, label: str) -> pd.DataFrame:
    try:
        projected = targets.loc[
            :, [*FEATURE_KEY_COLUMNS, _REGRESSION_TARGET, _CLASSIFICATION_TARGET]
        ]
        joined = keys.merge(
            projected, on=list(FEATURE_KEY_COLUMNS), how="left", sort=False, validate="one_to_one"
        )
    except (KeyError, pd.errors.MergeError):
        raise FrozenSelectionError("Phase 13 input validation failed.") from None
    if joined[[_REGRESSION_TARGET, _CLASSIFICATION_TARGET]].isna().any().any():
        raise FrozenSelectionError("Phase 13 input validation failed.")
    if (
        str(joined[_REGRESSION_TARGET].dtype) != "int64"
        or str(joined[_CLASSIFICATION_TARGET].dtype) != "int64"
    ):
        raise FrozenSelectionError("Phase 13 input validation failed.")
    return joined


def _select_classifier_state(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> tuple[Any, tuple[CandidateValidationMetrics, CandidateValidationMetrics], int, float, int]:
    if tuple(np.unique(y_train)) != (0, 1) or tuple(np.unique(y_validation)) != (0, 1):
        raise FrozenSelectionError("Phase 13 selection failed.")
    candidates: list[Any] = []
    records: list[CandidateValidationMetrics] = []
    try:
        for index, name in enumerate(CANDIDATE_CLASSIFIER_NAMES):
            candidate_seed = _derive_candidate_seed(seed, index)
            classifier = _build_candidate(name, candidate_seed)
            _fit_classifier(classifier, x_train, y_train)
            probabilities = _positive_probas(classifier, x_validation)
            threshold = _select_threshold(y_validation, probabilities)
            hard = (np.asarray(probabilities) >= threshold).astype(np.int64)
            records.append(
                CandidateValidationMetrics(
                    classifier_name=name,
                    validation_pr_auc=_metric_probability(
                        average_precision_score, y_validation, probabilities, "validation PR-AUC"
                    ),
                    decision_threshold=threshold,
                    validation_f1=_metric_probability(
                        f1_score, y_validation, hard, "validation F1", zero_division=0
                    ),
                    validation_recall=_metric_probability(
                        recall_score, y_validation, hard, "validation recall", zero_division=0
                    ),
                )
            )
            candidates.append(classifier)
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, RuntimeError)):
            raise
        raise FrozenSelectionError("Phase 13 selection failed.") from None
    locked = (records[0], records[1])
    selected_index = _select_classifier(locked)
    selected = locked[selected_index]
    return (
        candidates[selected_index],
        locked,
        selected_index,
        selected.decision_threshold,
        _derive_candidate_seed(seed, selected_index),
    )


def _select_regressor_state(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> tuple[
    Any,
    tuple[CandidateRegressionValidationMetrics, CandidateRegressionValidationMetrics],
    int,
    int | None,
]:
    try:
        candidates = _build_candidates(seed)
        records: list[CandidateRegressionValidationMetrics] = []
        for name, candidate in zip(CANDIDATE_REGRESSOR_NAMES, candidates, strict=True):
            _fit_regressor(candidate, x_train, y_train)
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
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit, RuntimeError)):
            raise
        raise FrozenSelectionError("Phase 13 selection failed.") from None
    locked = (records[0], records[1])
    selected_index = _select_regressor(locked)
    regressor_seed = None if selected_index == 0 else int(candidates[1].random_state)
    return candidates[selected_index], locked, selected_index, regressor_seed


def _canonical_scalar(value: object) -> object:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise FrozenSelectionError("Phase 13 input validation failed.")
        return "0x0.0p+0" if number == 0.0 else number.hex()
    if isinstance(value, str):
        return value
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return (
            value.date().isoformat()
            if isinstance(value, (pd.Timestamp, datetime))
            else value.isoformat()
        )
    raise FrozenSelectionError("Phase 13 input validation failed.")


def _partition_fingerprint(
    dates_key: str,
    rows_key: str,
    frame: pd.DataFrame,
    targets: pd.DataFrame | None,
    columns: list[str],
    dates: tuple[date, ...],
    spec: DatasetSpec,
) -> str:
    source = (
        frame
        if targets is None
        else frame.merge(
            targets, on=list(FEATURE_KEY_COLUMNS), how="left", sort=False, validate="one_to_one"
        )
    )
    ordered = source.sort_values(list(FEATURE_KEY_COLUMNS), kind="stable")
    payload = {
        "columns": columns,
        "contract": FROZEN_SELECTION_CONTRACT_VERSION,
        "spec": {
            "dataset_months_per_segment": spec.dataset_months_per_segment,
            "dataset_observations": spec.dataset_observations,
            "dataset_segments": spec.dataset_segments,
        },
        dates_key: [value.isoformat() for value in dates],
        rows_key: [
            [_canonical_scalar(row[column]) for column in columns] for _, row in ordered.iterrows()
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _risk_output(keys: pd.DataFrame, probabilities: tuple[float, ...]) -> tuple[RiskOutput, ...]:
    if len(keys) != len(probabilities):
        raise FrozenSelectionError("Phase 13 risk mapping failed.")
    rows: list[RiskOutput] = []
    for (_, key), probability in zip(keys.iterrows(), probabilities, strict=True):
        if (
            type(probability) is not float
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise FrozenSelectionError("Phase 13 risk mapping failed.")
        score = risk_score_from_probability(probability)
        band: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        if score <= 30:
            band = "LOW"
        elif score <= 60:
            band = "MEDIUM"
        elif score <= 80:
            band = "HIGH"
        else:
            band = "CRITICAL"
        value = key["date"]
        rows.append(RiskOutput(str(key["segment_id"]), value.date(), probability, score, band))
    return tuple(rows)


def _manifest(
    training: str,
    selection: str,
    risk_input: str,
    classifier_state: tuple[
        Any, tuple[CandidateValidationMetrics, CandidateValidationMetrics], int, float, int
    ],
    regressor_state: tuple[
        Any,
        tuple[CandidateRegressionValidationMetrics, CandidateRegressionValidationMetrics],
        int,
        int | None,
    ],
    split: ChronologicalSplit,
    fit: PreprocessorFit,
    seed: int,
    artifacts: tuple[ArtifactFile, ArtifactFile, ArtifactFile, ArtifactFile],
    risk_rows: int,
) -> SelectedArtifactManifest:
    classifier, classification, classifier_index, threshold, classifier_seed = classifier_state
    _, regression, regressor_index, regressor_seed = regressor_state
    del classifier
    return SelectedArtifactManifest(
        FROZEN_SELECTION_CONTRACT_VERSION,
        training,
        selection,
        risk_input,
        ADVANCED_CLASSIFIER_CONTRACT_VERSION,
        ADVANCED_REGRESSOR_CONTRACT_VERSION,
        classification[classifier_index].classifier_name,
        regression[regressor_index].regressor_name,
        threshold,
        seed,
        classifier_seed,
        regressor_seed,
        fit.transformed_feature_columns,
        len(split.train),
        len(split.validation),
        len(split.test),
        split.train_dates,
        split.validation_dates,
        split.test_dates,
        classification,
        regression,
        (("LOW", 0, 30), ("MEDIUM", 31, 60), ("HIGH", 61, 80), ("CRITICAL", 81, 100)),
        risk_rows,
        _runtime_versions(),
        sysconfig.get_platform(),
        artifacts,
    )


def _runtime_versions() -> tuple[tuple[str, str], ...]:
    return (
        ("python", platform.python_version()),
        ("numpy", importlib.metadata.version("numpy")),
        ("pandas", importlib.metadata.version("pandas")),
        ("scikit-learn", importlib.metadata.version("scikit-learn")),
        ("joblib", importlib.metadata.version("joblib")),
    )


def _preprocessor_projection(fit: PreprocessorFit) -> dict[str, object]:
    return {
        "means": [_canonical_scalar(value) for value in fit.means],
        "province_categories": list(fit.province_categories),
        "road_type_categories": list(fit.road_type_categories),
        "scaled_columns": list(fit.scaled_columns),
        "stds": [_canonical_scalar(value) for value in fit.stds],
    }


def _manifest_projection(value: SelectedArtifactManifest) -> object:
    return _json_projection(value)


def _json_projection(value: object) -> object:
    if value is None:
        return None
    if is_dataclass(value):
        return {field.name: _json_projection(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_json_projection(item) for item in value]
    if isinstance(value, list):
        return [_json_projection(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_projection(item) for key, item in value.items()}
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return _canonical_scalar(value)
    if isinstance(value, (float, np.floating)):
        return _canonical_scalar(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        return value
    raise ArtifactPersistenceError("Phase 13 artifact serialization failed.")


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError):
        raise ArtifactPersistenceError("Phase 13 artifact serialization failed.") from None


def _canonical_json_bytes(value: object) -> bytes:
    return (_canonical_json(value) + "\n").encode("utf-8")


def _write_pretest_payloads(
    stage: Path, fit: PreprocessorFit, classifier: Any, regressor: Any
) -> tuple[ArtifactFile, ArtifactFile, ArtifactFile]:
    preprocessor = stage / "preprocessor.json"
    _artifact_io.write_bytes(preprocessor, _canonical_json_bytes(_preprocessor_projection(fit)))
    classifier_path = stage / "classifier.joblib"
    regressor_path = stage / "regressor.joblib"
    _artifact_io.dump_joblib(classifier, classifier_path)
    _artifact_io.dump_joblib(regressor, regressor_path)
    return (
        _file_record("preprocessor", preprocessor),
        _file_record("classifier", classifier_path),
        _file_record("regressor", regressor_path),
    )


def _write_risk_payload(stage: Path, risk: tuple[RiskOutput, ...]) -> ArtifactFile:
    path = stage / "test-risk.jsonl"
    try:
        with path.open("xb") as handle:
            for row in risk:
                value = {
                    "date": row.date.isoformat(),
                    "maintenance_probability": row.maintenance_probability,
                    "risk_band": row.risk_band,
                    "risk_score": row.risk_score,
                    "segment_id": row.segment_id,
                }
                handle.write(_canonical_json(value).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError):
        raise ArtifactPersistenceError("Phase 13 artifact write failed.") from None
    return _file_record("test_risk", path)


def _file_record(
    role: Literal["preprocessor", "classifier", "regressor", "test_risk"], path: Path
) -> ArtifactFile:
    try:
        return ArtifactFile(role, path.name, _artifact_io.sha256(path), path.stat().st_size)
    except OSError:
        raise ArtifactPersistenceError("Phase 13 artifact verification failed.") from None


def _prepare_root(value: Path) -> Path:
    if type(value) is not type(Path()):
        raise FrozenSelectionError("Phase 13 input validation failed.")
    return _artifact_io.prepare_root(value)


def _publish_or_verify(
    stage: Path,
    destination: Path,
    digest: str,
    manifest: SelectedArtifactManifest,
    risk: tuple[RiskOutput, ...],
) -> FrozenSelectionResult:
    expected_manifest_bytes = _canonical_json_bytes(_manifest_projection(manifest))
    expected_artifacts = tuple(
        (record.filename, record.size_bytes, record.sha256) for record in manifest.artifacts
    )
    if destination.exists():
        _artifact_io.verify_bundle(
            destination, digest, expected_manifest_bytes, expected_artifacts, ARTIFACT_FILENAMES
        )
        return FrozenSelectionResult(digest, f"{_VERSION_DIRECTORY}/{digest}", manifest, risk)
    try:
        _artifact_io.rename_without_replacement(stage, destination)
    except OSError:
        if destination.exists():
            _artifact_io.verify_bundle(
                destination, digest, expected_manifest_bytes, expected_artifacts, ARTIFACT_FILENAMES
            )
            return FrozenSelectionResult(digest, f"{_VERSION_DIRECTORY}/{digest}", manifest, risk)
        raise ArtifactPersistenceError("Phase 13 artifact publication failed.") from None
    try:
        _artifact_io.sync_directory(destination.parent)
    except OSError:
        raise ArtifactPersistenceError("Phase 13 artifact publication failed.") from None
    _artifact_io.verify_bundle(
        destination, digest, expected_manifest_bytes, expected_artifacts, ARTIFACT_FILENAMES
    )
    return FrozenSelectionResult(digest, f"{_VERSION_DIRECTORY}/{digest}", manifest, risk)


__all__ = [
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
]
