"""Public value types and sanitized errors for PostgreSQL persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd


class PersistenceError(RuntimeError):
    """Raised when a transactional persistence operation fails."""


class PersistenceConflict(PersistenceError):
    """Raised when an existing natural key has different persisted values."""


class RepositoryInputError(ValueError):
    """Raised when repository input violates the public persistence contract."""


@dataclass(frozen=True)
class LoadReport:
    """Immutable row reconciliation counts for one committed ETL transaction."""

    inserted: Mapping[str, int]
    existing: Mapping[str, int]
    persisted: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "inserted", MappingProxyType(dict(self.inserted)))
        object.__setattr__(self, "existing", MappingProxyType(dict(self.existing)))
        object.__setattr__(self, "persisted", MappingProxyType(dict(self.persisted)))


@dataclass(frozen=True)
class RepositoryExport:
    """Deterministic physical exports with targets kept separate from observations."""

    segments: pd.DataFrame
    observations: pd.DataFrame
    targets: pd.DataFrame
    maintenance_events: pd.DataFrame


@dataclass(frozen=True)
class SegmentHistory:
    """Point-in-time history for one segment."""

    segment: pd.DataFrame
    observations: pd.DataFrame
    maintenance_events: pd.DataFrame


__all__ = [
    "LoadReport",
    "PersistenceConflict",
    "PersistenceError",
    "RepositoryExport",
    "RepositoryInputError",
    "SegmentHistory",
]
