"""Tests for package metadata and the public API surface."""

from __future__ import annotations

import re

import roadguard


def test_package_imports_and_exposes_semver_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", roadguard.__version__) is not None


def test_public_api_surface() -> None:
    expected = (
        "CleaningResult",
        "ConfigError",
        "CorruptionEntry",
        "CorruptionManifest",
        "ENV_PREFIX",
        "DatasetSpec",
        "GenerationError",
        "RiskBand",
        "RiskBands",
        "RoadGuardConfig",
        "SegmentMaster",
        "TARGET_COLUMNS",
        "V1Contract",
        "ValidationIssue",
        "ValidationReport",
        "clean_raw_dataset",
        "days_until_maintenance",
        "decay_condition",
        "derive_observation_targets",
        "generate_accident_timeline",
        "generate_maintenance_events",
        "generate_observations",
        "generate_segments",
        "inject_observation_corruption",
        "load_config",
        "maintenance_within_30_days",
        "month_transition",
        "monthly_hazard",
        "observation_dates",
        "risk_score_from_probability",
        "validate_cleaned_dataset",
        "validate_raw_dataset",
    )
    for name in expected:
        assert hasattr(roadguard, name), f"missing public attribute: {name}"
