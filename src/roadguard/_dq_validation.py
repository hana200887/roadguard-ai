"""Validation types, severity semantics and shared constants.

The staged validation driver and row/cross-field checks live in
``_dq_rowchecks``; this module holds the immutable report types, severity
semantics and the canonical table/column constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Literal

from roadguard.observations import OBSERVATION_COLUMNS

PUBLIC_SEGMENT_COLUMNS: Final[tuple[str, ...]] = (
    "segment_id",
    "province",
    "road_type",
    "construction_date",
    "road_length_km",
)
WEATHER_COLUMNS: Final[tuple[str, ...]] = ("rainfall_mm", "temperature", "humidity")
TRAFFIC_VOLUME_OUTLIER_MAX: Final[int] = 100_000
RAINFALL_OUTLIER_MAX: Final[float] = 1_000.0

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ValidationIssue:
    """A single deterministic validation finding (never a full row dump)."""

    severity: Severity
    code: str
    table: str
    column: str | None = None
    row_key: str | None = None
    message: str = ""


@dataclass(frozen=True)
class ValidationReport:
    """Deterministic, grouped summary of validation issues."""

    issues: tuple[ValidationIssue, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return self.error_count == 0

    @property
    def counts_by_code(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.code] = counts.get(issue.code, 0) + 1
        return counts

    @property
    def counts_by_table(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.table] = counts.get(issue.table, 0) + 1
        return counts

    @property
    def counts_by_column(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            column = issue.column or ""
            counts[column] = counts.get(column, 0) + 1
        return counts


def extra_column_label(position: int, column: object) -> str:
    """Deterministic, non-colliding locator for an unexpected column label.

    Uses the column position plus an exact type name so two distinct
    unsupported labels of the same type never collide and no repr or
    equality method of arbitrary labels is ever invoked.
    """
    if type(column) is str:
        return f"extra[{position}]:str:{column}"
    return f"extra[{position}]:{type(column).__name__}"


def rows_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return True when two observation row dicts are value-identical."""
    for column in OBSERVATION_COLUMNS:
        a = left[column]
        b = right[column]
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            continue
        if a != b:
            return False
    return True


__all__ = [
    "PUBLIC_SEGMENT_COLUMNS",
    "RAINFALL_OUTLIER_MAX",
    "TRAFFIC_VOLUME_OUTLIER_MAX",
    "WEATHER_COLUMNS",
    "ValidationIssue",
    "ValidationReport",
    "extra_column_label",
    "rows_equal",
]
