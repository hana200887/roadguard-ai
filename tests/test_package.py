"""Tests for package metadata and the public API surface."""

from __future__ import annotations

import re

import roadguard


def test_package_imports_and_exposes_semver_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", roadguard.__version__) is not None


def test_public_api_surface() -> None:
    expected = (
        "CleaningResult",
        "ChronologicalSplit",
        "ClassificationBalance",
        "CategoricalLevel",
        "CategoricalSummary",
        "ConfigError",
        "CorruptionEntry",
        "CorruptionManifest",
        "DataQualitySummary",
        "DateSummary",
        "EDAError",
        "EDAReport",
        "ENV_PREFIX",
        "DatasetSpec",
        "DatabaseConfigurationError",
        "DatabaseUnavailableError",
        "GenerationError",
        "LoadReport",
        "PersistenceConflict",
        "PersistenceError",
        "PostgresRepository",
        "PreprocessingError",
        "PreprocessorFit",
        "RiskBand",
        "RiskBands",
        "RoadGuardConfig",
        "RepositoryExport",
        "RepositoryInputError",
        "SegmentHistory",
        "SegmentMaster",
        "SplitInventory",
        "TARGET_COLUMNS",
        "TargetCorrelation",
        "TransformedData",
        "V1Contract",
        "ValidationIssue",
        "ValidationReport",
        "clean_raw_dataset",
        "build_eda_report",
        "create_database_engine",
        "days_until_maintenance",
        "decay_condition",
        "derive_observation_targets",
        "generate_accident_timeline",
        "generate_maintenance_events",
        "generate_observations",
        "generate_segments",
        "fit_preprocessor",
        "inject_observation_corruption",
        "initialize_database",
        "load_config",
        "load_cleaning_result",
        "maintenance_within_30_days",
        "month_transition",
        "monthly_hazard",
        "observation_dates",
        "risk_score_from_probability",
        "render_data_card",
        "split_chronologically",
        "transform",
        "validate_cleaned_dataset",
        "validate_raw_dataset",
    )
    for name in expected:
        assert hasattr(roadguard, name), f"missing public attribute: {name}"
