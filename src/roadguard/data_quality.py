"""Phase 5 public facade: raw-data corruption, staged validation and safe cleaning.

Public entry points re-exported from the focused implementation modules:

- ``_dq_corruption``: deterministic corruption and manifest types;
- ``_dq_validation``: staged raw/cleaned validation and report types;
- ``_dq_cleaning``: structural cleaning and the cleaned result.

The facade is the stable public API surface; import from
``roadguard.data_quality`` only.
"""

from __future__ import annotations

from roadguard._dq_cleaning import CleaningResult, clean_raw_dataset
from roadguard._dq_corruption import (
    CORRUPTION_RNG_NAMESPACE,
    DUPLICATE_RATE_DEFAULT,
    MISSING_RATE_DEFAULT,
    OUTLIER_COLUMNS,
    OUTLIER_MULTIPLIER_DEFAULT,
    OUTLIER_RATE_DEFAULT,
    WEATHER_COLUMNS,
    CorruptionEntry,
    CorruptionManifest,
    inject_observation_corruption,
)
from roadguard._dq_rowchecks import (
    validate_cleaned_dataset,
    validate_raw_dataset,
)
from roadguard._dq_validation import (
    PUBLIC_SEGMENT_COLUMNS,
    RAINFALL_OUTLIER_MAX,
    TRAFFIC_VOLUME_OUTLIER_MAX,
    ValidationIssue,
    ValidationReport,
)

MAX_CORRUPTION_RATE = 0.25

__all__ = [
    "CORRUPTION_RNG_NAMESPACE",
    "CleaningResult",
    "CorruptionEntry",
    "CorruptionManifest",
    "DUPLICATE_RATE_DEFAULT",
    "MAX_CORRUPTION_RATE",
    "MISSING_RATE_DEFAULT",
    "OUTLIER_COLUMNS",
    "OUTLIER_MULTIPLIER_DEFAULT",
    "OUTLIER_RATE_DEFAULT",
    "PUBLIC_SEGMENT_COLUMNS",
    "RAINFALL_OUTLIER_MAX",
    "TRAFFIC_VOLUME_OUTLIER_MAX",
    "ValidationIssue",
    "ValidationReport",
    "WEATHER_COLUMNS",
    "clean_raw_dataset",
    "inject_observation_corruption",
    "validate_cleaned_dataset",
    "validate_raw_dataset",
]
