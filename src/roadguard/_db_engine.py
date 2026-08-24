"""Credential-safe PostgreSQL engine construction and schema initialization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Final

import sqlalchemy as sa
from pydantic import SecretStr
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError, SQLAlchemyError

from roadguard._db_models import DB_SCHEMA, metadata
from roadguard._db_schema_contract import (
    CHECK_DEFINITIONS,
    INDEX_DEFINITIONS,
    normalize_definition,
    normalize_optional_definition,
)
from roadguard.config import RoadGuardConfig

_DRIVER: Final[str] = "postgresql+psycopg"
_SCHEMA_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


class DatabaseConfigurationError(ValueError):
    """Raised when database configuration is missing or unsupported."""


class DatabaseUnavailableError(RuntimeError):
    """Raised when PostgreSQL cannot be reached without leaking credentials."""


def create_database_engine(
    database_url: RoadGuardConfig | SecretStr | str | None,
) -> Engine:
    """Build a synchronous PostgreSQL/psycopg engine without opening a connection."""
    raw = _extract_database_url(database_url)
    try:
        url = make_url(raw)
    except (ArgumentError, TypeError, ValueError) as exc:
        raise DatabaseConfigurationError("database URL is invalid") from exc
    if url.drivername != _DRIVER:
        raise DatabaseConfigurationError("database URL must use PostgreSQL with psycopg")
    if "connect_timeout" not in url.query:
        url = url.update_query_dict({"connect_timeout": "5"})
    return sa.create_engine(
        url,
        future=True,
        hide_parameters=True,
        pool_pre_ping=True,
    )


def initialize_database(engine: Engine) -> None:
    """Create or verify the fixed RoadGuard schema idempotently."""
    _require_postgresql(engine)
    schema = _effective_schema(engine)
    try:
        with engine.begin() as connection:
            connection.execute(sa.schema.CreateSchema(schema, if_not_exists=True))
            existing = set(sa.inspect(connection).get_table_names(schema=schema))
            if existing:
                _verify_schema(connection, schema)
            else:
                metadata.create_all(connection)
                _verify_schema(connection, schema)
    except _SchemaDriftError:
        raise DatabaseUnavailableError("PostgreSQL schema drift detected") from None
    except SQLAlchemyError:
        raise DatabaseUnavailableError("PostgreSQL schema initialization failed") from None


def _extract_database_url(value: RoadGuardConfig | SecretStr | str | None) -> str:
    candidate = value.database_url if isinstance(value, RoadGuardConfig) else value
    if isinstance(candidate, SecretStr):
        raw = candidate.get_secret_value()
    elif type(candidate) is str:
        raw = candidate
    else:
        raise DatabaseConfigurationError("database URL is required")
    if not raw.strip():
        raise DatabaseConfigurationError("database URL is required")
    return raw


def _require_postgresql(engine: Engine) -> None:
    if engine.dialect.name != "postgresql" or engine.dialect.driver != "psycopg":
        raise DatabaseConfigurationError("a PostgreSQL psycopg engine is required")


class _SchemaDriftError(RuntimeError):
    """Internal signal for an incompatible pre-existing physical schema."""


def _effective_schema(engine: Engine) -> str:
    options = engine.get_execution_options()
    raw_map = options.get("schema_translate_map", {})
    schema_map = raw_map if isinstance(raw_map, Mapping) else {}
    candidate = schema_map.get(DB_SCHEMA, DB_SCHEMA)
    if type(candidate) is not str or _SCHEMA_PATTERN.fullmatch(candidate) is None:
        raise DatabaseConfigurationError("database schema mapping is invalid")
    return candidate


def _verify_schema(connection: sa.Connection, schema: str) -> None:
    inspector = sa.inspect(connection)
    expected_tables = {table.name: table for table in metadata.sorted_tables}
    actual_names = set(inspector.get_table_names(schema=schema))
    if actual_names != set(expected_tables):
        raise _SchemaDriftError
    for table in expected_tables.values():
        _verify_table(inspector, schema, table)


def _verify_table(inspector: sa.Inspector, schema: str, table: sa.Table) -> None:
    actual_columns = inspector.get_columns(table.name, schema=schema)
    if [column["name"] for column in actual_columns] != list(table.c.keys()):
        raise _SchemaDriftError
    for actual, expected in zip(actual_columns, table.columns, strict=True):
        if bool(actual["nullable"]) != bool(expected.nullable):
            raise _SchemaDriftError
        if not _compatible_type(actual["type"], expected.type):
            raise _SchemaDriftError

    actual_pk = tuple(
        inspector.get_pk_constraint(table.name, schema=schema).get("constrained_columns", ())
    )
    expected_pk = tuple(column.name for column in table.primary_key.columns)
    if actual_pk != expected_pk:
        raise _SchemaDriftError

    actual_fks = {
        (
            tuple(item.get("constrained_columns") or ()),
            item.get("referred_schema"),
            item.get("referred_table"),
            tuple(item.get("referred_columns") or ()),
            (item.get("options") or {}).get("ondelete"),
        )
        for item in inspector.get_foreign_keys(table.name, schema=schema)
    }
    expected_fks = {
        (
            tuple(element.parent.name for element in constraint.elements),
            schema,
            constraint.elements[0].column.table.name,
            tuple(element.column.name for element in constraint.elements),
            constraint.ondelete,
        )
        for constraint in table.foreign_key_constraints
    }
    if actual_fks != expected_fks:
        raise _SchemaDriftError

    actual_checks = {
        str(item.get("name")): normalize_definition(str(item.get("sqltext")))
        for item in inspector.get_check_constraints(table.name, schema=schema)
    }
    expected_checks = CHECK_DEFINITIONS[table.name]
    if actual_checks != expected_checks:
        raise _SchemaDriftError

    actual_unique_constraints = {
        (
            str(item.get("name")),
            tuple(item.get("column_names") or ()),
            bool((item.get("dialect_options") or {}).get("postgresql_nulls_not_distinct")),
        )
        for item in inspector.get_unique_constraints(table.name, schema=schema)
    }
    if actual_unique_constraints:
        raise _SchemaDriftError

    actual_indexes = {
        (
            str(item.get("name")),
            bool(item.get("unique")),
            tuple(
                normalize_definition(str(expression))
                for expression in (item.get("expressions") or item.get("column_names") or ())
            ),
            tuple(item.get("include_columns") or ()),
            normalize_optional_definition(
                (item.get("dialect_options") or {}).get("postgresql_where")
            ),
        )
        for item in inspector.get_indexes(table.name, schema=schema)
        if not item.get("duplicates_constraint")
    }
    expected_indexes = INDEX_DEFINITIONS[table.name]
    if actual_indexes != expected_indexes:
        raise _SchemaDriftError


def _compatible_type(actual: Any, expected: sa.types.TypeEngine[Any]) -> bool:
    if isinstance(expected, sa.Text):
        return isinstance(actual, sa.Text)
    if isinstance(expected, sa.Date):
        return isinstance(actual, sa.Date) and not isinstance(actual, sa.DateTime)
    if isinstance(expected, sa.SmallInteger):
        return isinstance(actual, sa.SmallInteger)
    if isinstance(expected, sa.BigInteger):
        return isinstance(actual, sa.BigInteger)
    if isinstance(expected, sa.Float):
        return type(actual) is postgresql.DOUBLE_PRECISION
    return type(actual) is type(expected)


__all__ = [
    "DatabaseConfigurationError",
    "DatabaseUnavailableError",
    "create_database_engine",
    "initialize_database",
]
