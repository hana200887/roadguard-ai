"""Public Phase 6 PostgreSQL persistence API."""

from roadguard._db_engine import (
    DatabaseConfigurationError,
    DatabaseUnavailableError,
    create_database_engine,
    initialize_database,
)
from roadguard._db_etl import load_cleaning_result
from roadguard._db_repository import PostgresRepository
from roadguard._db_types import (
    LoadReport,
    PersistenceConflict,
    PersistenceError,
    RepositoryExport,
    RepositoryInputError,
    SegmentHistory,
)

__all__ = [
    "DatabaseConfigurationError",
    "DatabaseUnavailableError",
    "LoadReport",
    "PersistenceConflict",
    "PersistenceError",
    "PostgresRepository",
    "RepositoryExport",
    "RepositoryInputError",
    "SegmentHistory",
    "create_database_engine",
    "initialize_database",
    "load_cleaning_result",
]
