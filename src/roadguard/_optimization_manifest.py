"""Private canonical Phase 13 manifest validation for Phase 15."""

from __future__ import annotations

import hashlib
import math
from datetime import date
from typing import Any, Final

from roadguard import artifacts as _artifacts
from roadguard.artifacts import (
    ARTIFACT_FILENAMES,
    FROZEN_SELECTION_CONTRACT_VERSION,
    ArtifactFile,
    FrozenSelectionResult,
    RiskOutput,
    SelectedArtifactManifest,
)
from roadguard.classification import (
    ADVANCED_CLASSIFIER_CONTRACT_VERSION,
    ADVANCED_CLASSIFIER_RNG_NAMESPACE,
    CANDIDATE_CLASSIFIER_NAMES,
    CandidateValidationMetrics,
)
from roadguard.contracts import V1_OBSERVATION_START
from roadguard.features import FEATURE_COLUMNS
from roadguard.preprocessing import CONSTRUCTION_DATE_DAY_COLUMN
from roadguard.regression import (
    ADVANCED_REGRESSOR_CONTRACT_VERSION,
    ADVANCED_REGRESSOR_RNG_NAMESPACE,
    CANDIDATE_REGRESSOR_NAMES,
    CandidateRegressionValidationMetrics,
)
from roadguard.segments import PROVINCES, ROAD_TYPES

_RISK_BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("LOW", 0, 30),
    ("MEDIUM", 31, 60),
    ("HIGH", 61, 80),
    ("CRITICAL", 81, 100),
)
_RUNTIME_NAMES: Final[tuple[str, ...]] = (
    "python",
    "numpy",
    "pandas",
    "scikit-learn",
    "joblib",
)
_EXPECTED_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    tuple(
        column
        for column in FEATURE_COLUMNS
        if column not in ("province", "road_type", "construction_date")
    )
    + (CONSTRUCTION_DATE_DAY_COLUMN,)
    + tuple(f"province_{category}" for category in sorted(PROVINCES))
    + tuple(f"road_type_{category}" for category in sorted(ROAD_TYPES))
)
_UINT32_MASK: Final[int] = 2**32 - 1
_SEED_INIT_A: Final[int] = 0x43B0D7E5
_SEED_MULT_A: Final[int] = 0x931E8875
_SEED_INIT_B: Final[int] = 0x8B51F9DD
_SEED_MULT_B: Final[int] = 0x58F38DED
_SEED_MIX_MULT_L: Final[int] = 0xCA01F9DD
_SEED_MIX_MULT_R: Final[int] = 0x4973F715


def preflight_manifest(manifest: SelectedArtifactManifest) -> None:
    """Bound nested input and require exact dataclass leaf types before copying."""
    for name in (
        "contract_version",
        "training_fingerprint",
        "selection_fingerprint",
        "risk_input_fingerprint",
        "classifier_contract_version",
        "regressor_contract_version",
        "selected_classifier_name",
        "selected_regressor_name",
        "platform_tag",
    ):
        _exact(getattr(manifest, name), str)
    _exact(manifest.classifier_decision_threshold, float)
    for name in (
        "master_seed",
        "classifier_seed",
        "train_rows",
        "validation_rows",
        "test_rows",
        "risk_output_rows",
    ):
        _exact(getattr(manifest, name), int)
    if manifest.regressor_seed is not None:
        _exact(manifest.regressor_seed, int)
    tuple_fields = (
        "feature_columns",
        "train_dates",
        "validation_dates",
        "test_dates",
        "classification_candidates",
        "regression_candidates",
        "risk_bands",
        "runtime_versions",
        "artifacts",
    )
    if any(type(getattr(manifest, name)) is not tuple for name in tuple_fields):
        raise TypeError("manifest collections must be tuples")
    lengths = (
        (manifest.feature_columns, len(_EXPECTED_FEATURE_COLUMNS)),
        (manifest.train_dates, 34),
        (manifest.validation_dates, 7),
        (manifest.test_dates, 7),
        (manifest.classification_candidates, 2),
        (manifest.regression_candidates, 2),
        (manifest.risk_bands, 4),
        (manifest.runtime_versions, 5),
        (manifest.artifacts, 4),
    )
    if any(len(values) > maximum for values, maximum in lengths):
        raise ValueError("manifest collection is oversized")
    for feature_name in manifest.feature_columns:
        _exact(feature_name, str)
    for manifest_date in (*manifest.train_dates, *manifest.validation_dates, *manifest.test_dates):
        _exact(manifest_date, date)
    for classification_record in manifest.classification_candidates:
        if type(classification_record) is not CandidateValidationMetrics:
            raise TypeError("invalid classification record")
        _exact(classification_record.classifier_name, str)
        for metric in (
            classification_record.validation_pr_auc,
            classification_record.decision_threshold,
            classification_record.validation_f1,
            classification_record.validation_recall,
        ):
            _exact(metric, float)
    for regression_record in manifest.regression_candidates:
        if type(regression_record) is not CandidateRegressionValidationMetrics:
            raise TypeError("invalid regression record")
        _exact(regression_record.regressor_name, str)
        _exact(regression_record.validation_mae, float)
        _exact(regression_record.validation_rmse, float)
    for risk_band in manifest.risk_bands:
        if type(risk_band) is not tuple or len(risk_band) != 3:
            raise TypeError("invalid risk band")
        _exact(risk_band[0], str)
        _exact(risk_band[1], int)
        _exact(risk_band[2], int)
    for runtime_version in manifest.runtime_versions:
        if type(runtime_version) is not tuple or len(runtime_version) != 2:
            raise TypeError("invalid runtime version")
        _exact(runtime_version[0], str)
        _exact(runtime_version[1], str)
    for artifact in manifest.artifacts:
        if type(artifact) is not ArtifactFile:
            raise TypeError("invalid artifact record")
        _exact(artifact.role, str)
        _exact(artifact.filename, str)
        _exact(artifact.sha256, str)
        _exact(artifact.size_bytes, int)


def snapshot_selection(selection: FrozenSelectionResult) -> FrozenSelectionResult:
    """Copy only declared exact Phase 13 fields, ignoring hostile attached state."""
    source = selection.manifest
    manifest = SelectedArtifactManifest(
        source.contract_version,
        source.training_fingerprint,
        source.selection_fingerprint,
        source.risk_input_fingerprint,
        source.classifier_contract_version,
        source.regressor_contract_version,
        source.selected_classifier_name,
        source.selected_regressor_name,
        source.classifier_decision_threshold,
        source.master_seed,
        source.classifier_seed,
        source.regressor_seed,
        tuple(source.feature_columns),
        source.train_rows,
        source.validation_rows,
        source.test_rows,
        tuple(source.train_dates),
        tuple(source.validation_dates),
        tuple(source.test_dates),
        tuple(
            CandidateValidationMetrics(
                record.classifier_name,
                record.validation_pr_auc,
                record.decision_threshold,
                record.validation_f1,
                record.validation_recall,
            )
            for record in source.classification_candidates
        ),  # type: ignore[arg-type]
        tuple(
            CandidateRegressionValidationMetrics(
                record.regressor_name,
                record.validation_mae,
                record.validation_rmse,
            )
            for record in source.regression_candidates
        ),  # type: ignore[arg-type]
        tuple((record[0], record[1], record[2]) for record in source.risk_bands),
        source.risk_output_rows,
        tuple((record[0], record[1]) for record in source.runtime_versions),
        source.platform_tag,
        tuple(
            ArtifactFile(record.role, record.filename, record.sha256, record.size_bytes)
            for record in source.artifacts
        ),  # type: ignore[arg-type]
    )
    risk_output = tuple(
        RiskOutput(
            row.segment_id,
            row.date,
            row.maintenance_probability,
            row.risk_score,
            row.risk_band,
        )
        for row in selection.risk_output
    )
    return FrozenSelectionResult(
        selection.manifest_sha256,
        selection.relative_artifact_directory,
        manifest,
        risk_output,
    )


def validate_manifest(selection: FrozenSelectionResult) -> None:
    """Require an authenticated, fully canonical fixed-shape Phase 13 manifest."""
    manifest = selection.manifest
    if (
        manifest.contract_version != FROZEN_SELECTION_CONTRACT_VERSION
        or manifest.classifier_contract_version != ADVANCED_CLASSIFIER_CONTRACT_VERSION
        or manifest.regressor_contract_version != ADVANCED_REGRESSOR_CONTRACT_VERSION
        or manifest.feature_columns != _EXPECTED_FEATURE_COLUMNS
        or (manifest.train_rows, manifest.validation_rows, manifest.test_rows)
        != (
            10_200,
            2_100,
            2_100,
        )
        or manifest.risk_output_rows != 2_100
        or manifest.master_seed <= 0
        or not manifest.platform_tag
    ):
        raise ValueError("noncanonical Phase 13 manifest")
    expected_dates = _month_dates(V1_OBSERVATION_START, 48)
    if (
        manifest.train_dates != expected_dates[:34]
        or manifest.validation_dates != expected_dates[34:41]
        or manifest.test_dates != expected_dates[41:]
    ):
        raise ValueError("noncanonical Phase 13 dates")
    for digest in (
        manifest.training_fingerprint,
        manifest.selection_fingerprint,
        manifest.risk_input_fingerprint,
    ):
        _require_digest(digest)
    classifier_index = _validate_classifier(manifest)
    regressor_index = _validate_regressor(manifest)
    expected_classifier_seed = _seed_sequence_word(
        (manifest.master_seed, ADVANCED_CLASSIFIER_RNG_NAMESPACE, classifier_index)
    )
    expected_regressor_seed = (
        None
        if regressor_index == 0
        else _seed_sequence_word((manifest.master_seed, ADVANCED_REGRESSOR_RNG_NAMESPACE, 1))
    )
    if (
        manifest.classifier_seed != expected_classifier_seed
        or manifest.regressor_seed != expected_regressor_seed
        or tuple(name for name, _ in manifest.runtime_versions) != _RUNTIME_NAMES
        or any(not version for _, version in manifest.runtime_versions)
        or manifest.risk_bands != _RISK_BANDS
    ):
        raise ValueError("noncanonical Phase 13 metadata")
    _validate_artifacts(selection)


def _validate_classifier(manifest: SelectedArtifactManifest) -> int:
    if tuple(record.classifier_name for record in manifest.classification_candidates) != (
        CANDIDATE_CLASSIFIER_NAMES
    ) or any(
        not _unit_metric(metric)
        for record in manifest.classification_candidates
        for metric in (
            record.validation_pr_auc,
            record.decision_threshold,
            record.validation_f1,
            record.validation_recall,
        )
    ):
        raise ValueError("invalid classifier evidence")
    index = max(
        range(2),
        key=lambda item: (
            manifest.classification_candidates[item].validation_pr_auc,
            manifest.classification_candidates[item].validation_f1,
            manifest.classification_candidates[item].validation_recall,
            -item,
        ),
    )
    selected = manifest.classification_candidates[index]
    if (
        manifest.selected_classifier_name != selected.classifier_name
        or manifest.classifier_decision_threshold != selected.decision_threshold
    ):
        raise ValueError("invalid classifier selection")
    return index


def _validate_regressor(manifest: SelectedArtifactManifest) -> int:
    if tuple(record.regressor_name for record in manifest.regression_candidates) != (
        CANDIDATE_REGRESSOR_NAMES
    ) or any(
        not math.isfinite(metric) or metric < 0.0
        for record in manifest.regression_candidates
        for metric in (record.validation_mae, record.validation_rmse)
    ):
        raise ValueError("invalid regressor evidence")
    index = min(
        range(2),
        key=lambda item: (
            manifest.regression_candidates[item].validation_mae,
            manifest.regression_candidates[item].validation_rmse,
            item,
        ),
    )
    if manifest.selected_regressor_name != manifest.regression_candidates[index].regressor_name:
        raise ValueError("invalid regressor selection")
    return index


def _validate_artifacts(selection: FrozenSelectionResult) -> None:
    manifest = selection.manifest
    expected = zip(
        manifest.artifacts,
        ("preprocessor", "classifier", "regressor", "test_risk"),
        ARTIFACT_FILENAMES[:-1],
        strict=True,
    )
    for record, role, filename in expected:
        if record.role != role or record.filename != filename or record.size_bytes <= 0:
            raise ValueError("invalid artifact inventory")
        _require_digest(record.sha256)
    expected_directory = f"{FROZEN_SELECTION_CONTRACT_VERSION}/{selection.manifest_sha256}"
    digest = hashlib.sha256(
        _artifacts._canonical_json_bytes(_artifacts._manifest_projection(manifest))
    ).hexdigest()
    if (
        selection.relative_artifact_directory != expected_directory
        or digest != selection.manifest_sha256
    ):
        raise ValueError("invalid manifest authentication")


def _month_dates(start: date, count: int) -> tuple[date, ...]:
    return tuple(
        date(start.year + ((start.month - 1 + index) // 12), (start.month - 1 + index) % 12 + 1, 1)
        for index in range(count)
    )


def _seed_sequence_word(entropy: tuple[int, int, int]) -> int:
    """Reproduce one locked NumPy SeedSequence uint32 without invoking RNG APIs."""
    entropy_words = tuple(word for value in entropy for word in _uint32_words(value))
    pool = [0, 0, 0, 0]
    hash_constant = _SEED_INIT_A
    for index in range(4):
        value = entropy_words[index] if index < len(entropy_words) else 0
        pool[index], hash_constant = _seed_hash(value, hash_constant)
    for source in range(4):
        for destination in range(4):
            if source != destination:
                hashed, hash_constant = _seed_hash(pool[source], hash_constant)
                pool[destination] = _seed_mix(pool[destination], hashed)
    for value in entropy_words[4:]:
        for destination in range(4):
            hashed, hash_constant = _seed_hash(value, hash_constant)
            pool[destination] = _seed_mix(pool[destination], hashed)
    value = (pool[0] ^ _SEED_INIT_B) & _UINT32_MASK
    output_constant = (_SEED_INIT_B * _SEED_MULT_B) & _UINT32_MASK
    value = (value * output_constant) & _UINT32_MASK
    return (value ^ (value >> 16)) & _UINT32_MASK


def _seed_hash(value: int, constant: int) -> tuple[int, int]:
    value = (value ^ constant) & _UINT32_MASK
    next_constant = (constant * _SEED_MULT_A) & _UINT32_MASK
    value = (value * next_constant) & _UINT32_MASK
    return (value ^ (value >> 16)) & _UINT32_MASK, next_constant


def _seed_mix(left: int, right: int) -> int:
    value = (_SEED_MIX_MULT_L * left - _SEED_MIX_MULT_R * right) & _UINT32_MASK
    return (value ^ (value >> 16)) & _UINT32_MASK


def _uint32_words(value: int) -> tuple[int, ...]:
    words: list[int] = []
    while value:
        words.append(value & _UINT32_MASK)
        value >>= 32
    return tuple(words or (0,))


def _unit_metric(value: float) -> bool:
    return math.isfinite(value) and 0.0 <= value <= 1.0


def _require_digest(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("invalid digest")


def _exact(value: object, expected: type[Any]) -> None:
    if type(value) is not expected:
        raise TypeError("invalid exact manifest type")
