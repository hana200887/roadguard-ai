"""Tests for package metadata and the public API surface."""

from __future__ import annotations

import re

import roadguard


def test_package_imports_and_exposes_semver_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", roadguard.__version__) is not None


def test_public_api_surface() -> None:
    expected = (
        "ConfigError",
        "ENV_PREFIX",
        "DatasetSpec",
        "GenerationError",
        "RiskBand",
        "RiskBands",
        "RoadGuardConfig",
        "SegmentMaster",
        "V1Contract",
        "days_until_maintenance",
        "decay_condition",
        "generate_accident_timeline",
        "generate_maintenance_events",
        "generate_segments",
        "load_config",
        "maintenance_within_30_days",
        "month_transition",
        "monthly_hazard",
        "observation_dates",
        "risk_score_from_probability",
    )
    for name in expected:
        assert hasattr(roadguard, name), f"missing public attribute: {name}"
