"""Transactional, insert-or-verify ETL for validated Phase 5 output."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from math import isfinite
from typing import Any, Final

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from roadguard._db_engine import _require_postgresql
from roadguard._db_models import (
    MAINTENANCE_HISTORY_COLUMNS,
    maintenance_events,
    maintenance_history,
    observation_targets,
    road_observations,
    road_segments,
)
from roadguard._db_types import (
    LoadReport,
    PersistenceConflict,
    PersistenceError,
    RepositoryInputError,
)
from roadguard._dq_cleaning import CleaningResult
from roadguard._dq_validation import ValidationReport
from roadguard.contracts import DatasetSpec
from roadguard.data_quality import PUBLIC_SEGMENT_COLUMNS, validate_cleaned_dataset
from roadguard.events import EVENT_COLUMNS
from roadguard.observations import OBSERVATION_COLUMNS
from roadguard.targets import TARGET_COLUMNS

_LOAD_TABLES: Final[tuple[sa.Table, ...]] = (
    road_segments,
    road_observations,
    maintenance_events,
    observation_targets,
    maintenance_history,
)
_QUERY_CHUNK_SIZE: Final[int] = 500


def load_cleaning_result(
    engine: Engine,
    result: CleaningResult,
    spec: DatasetSpec,
    *,
    maintenance_history: pd.DataFrame | None = None,
) -> LoadReport:
    """Commit validated cleaned data atomically using natural-key reconciliation.

    Existing identical rows are counted as already present. An existing natural
    key with different values raises :class:`PersistenceConflict`; the enclosing
    transaction then rolls back every table in the load.
    """
    _require_postgresql(engine)
    records_by_table = _prepare_records(result, spec, maintenance_history)
    inserted: dict[str, int] = {}
    existing: dict[str, int] = {}
    try:
        with engine.begin() as connection:
            for table in _LOAD_TABLES:
                added, present = _insert_or_verify(
                    connection,
                    table,
                    records_by_table[table.name],
                )
                inserted[table.name] = added
                existing[table.name] = present
            persisted = {
                table.name: int(
                    connection.execute(sa.select(sa.func.count()).select_from(table)).scalar_one()
                )
                for table in _LOAD_TABLES
            }
    except (PersistenceConflict, RepositoryInputError):
        raise
    except SQLAlchemyError as exc:
        raise PersistenceError("transactional PostgreSQL load failed") from exc
    return LoadReport(inserted=inserted, existing=existing, persisted=persisted)


def _prepare_records(
    result: CleaningResult,
    spec: DatasetSpec,
    realized_history: pd.DataFrame | None,
) -> dict[str, list[dict[str, Any]]]:
    if (
        type(result) is not CleaningResult
        or type(spec) is not DatasetSpec
        or type(result.report) is not ValidationReport
        or any(
            type(frame) is not pd.DataFrame
            for frame in (
                result.segments,
                result.observations,
                result.targets,
                result.maintenance_events,
            )
        )
        or (realized_history is not None and type(realized_history) is not pd.DataFrame)
        or result.report.error_count
    ):
        raise RepositoryInputError("a validated CleaningResult is required")
    try:
        validated_spec = DatasetSpec.model_validate(spec.model_dump(mode="python"))
    except Exception:
        raise RepositoryInputError("a validated DatasetSpec is required") from None
    working = CleaningResult(
        segments=result.segments.copy(deep=True),
        observations=result.observations.copy(deep=True),
        targets=result.targets.copy(deep=True),
        maintenance_events=result.maintenance_events.copy(deep=True),
        report=result.report,
    )
    fresh_report = validate_cleaned_dataset(
        working.segments,
        working.observations,
        working.targets,
        working.maintenance_events,
        validated_spec,
    )
    if fresh_report.error_count:
        raise RepositoryInputError("a validated CleaningResult is required")
    history = (
        pd.DataFrame(columns=MAINTENANCE_HISTORY_COLUMNS)
        if realized_history is None
        else realized_history.copy(deep=True)
    )
    frames = (
        (road_segments, working.segments, PUBLIC_SEGMENT_COLUMNS),
        (road_observations, working.observations, OBSERVATION_COLUMNS),
        (maintenance_events, working.maintenance_events, EVENT_COLUMNS),
        (observation_targets, working.targets, TARGET_COLUMNS),
        (maintenance_history, history, MAINTENANCE_HISTORY_COLUMNS),
    )
    prepared: dict[str, list[dict[str, Any]]] = {}
    for table, frame, expected in frames:
        label = table.name
        if not isinstance(frame, pd.DataFrame):
            raise RepositoryInputError(f"{label} must be a pandas DataFrame")
        labels = list(frame.columns)
        if len(labels) != len(expected) or any(
            type(label) is not str or label != required
            for label, required in zip(labels, expected, strict=True)
        ):
            raise RepositoryInputError(f"{label} schema does not match the contract")
        prepared[label] = _normalize_records(table, frame)
    event_keys = {
        _primary_key(maintenance_events, record) for record in prepared[maintenance_events.name]
    }
    history_keys = {
        _primary_key(maintenance_history, record) for record in prepared[maintenance_history.name]
    }
    if not history_keys.issubset(event_keys):
        raise RepositoryInputError(
            "maintenance_history keys must belong to current maintenance events"
        )
    return prepared


def _normalize_records(table: sa.Table, frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for values in frame.itertuples(index=False, name=None):
        raw = dict(zip(table.c.keys(), values, strict=True))
        record = {
            column.name: _normalize_value(column, raw[column.name]) for column in table.columns
        }
        records.append(record)
    keys = [_primary_key(table, record) for record in records]
    if len(keys) != len(set(keys)):
        raise RepositoryInputError(f"{table.name} contains duplicate natural keys")
    records.sort(key=lambda record: _primary_key(table, record))
    return records


def _normalize_value(column: sa.Column[Any], value: Any) -> Any:
    if isinstance(column.type, sa.Date):
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is not None or value.time() != datetime.min.time():
                raise RepositoryInputError(f"{column.name} must be a timezone-free date")
            return value.date()
        if type(value) is date:
            return value
        raise RepositoryInputError(f"{column.name} must be a date")
    if isinstance(column.type, (sa.BigInteger, sa.SmallInteger, sa.Integer)):
        if type(value) is int:
            return value
        raise RepositoryInputError(f"{column.name} must be an integer")
    if isinstance(column.type, sa.Float):
        if type(value) not in {int, float}:
            raise RepositoryInputError(f"{column.name} must be numeric")
        number = float(value)
        if not isfinite(number):
            raise RepositoryInputError(f"{column.name} must be finite")
        return number
    if isinstance(column.type, sa.Text):
        if type(value) is not str:
            raise RepositoryInputError(f"{column.name} must be a string")
        return value
    raise RepositoryInputError(f"unsupported database type for {column.name}")


def _insert_or_verify(
    connection: Connection,
    table: sa.Table,
    records: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    if not records:
        return 0, 0
    inserted = 0
    for chunk in _chunks(records, _QUERY_CHUNK_SIZE):
        keys = [_primary_key(table, record) for record in chunk]
        statement = (
            postgresql_insert(table)
            .values(list(chunk))
            .on_conflict_do_nothing()
            .returning(*table.primary_key.columns)
        )
        inserted += len(connection.execute(statement).all())
        existing_by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        verify = sa.select(table).where(_keys_predicate(table, keys))
        for row in connection.execute(verify).mappings():
            current_row = dict(row)
            existing_by_key[_primary_key(table, current_row)] = current_row
        for record in chunk:
            key = _primary_key(table, record)
            current = existing_by_key.get(key)
            if current is None or any(
                current[column.name] != record[column.name] for column in table.columns
            ):
                raise PersistenceConflict(
                    f"{table.name} natural key already exists with different values"
                )
    return inserted, len(records) - inserted


def _keys_predicate(table: sa.Table, keys: Sequence[tuple[Any, ...]]) -> sa.ColumnElement[bool]:
    columns = list(table.primary_key.columns)
    if len(columns) == 1:
        return columns[0].in_([key[0] for key in keys])
    return sa.tuple_(*columns).in_(keys)


def _primary_key(table: sa.Table, record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[column.name] for column in table.primary_key.columns)


def _chunks(
    values: Sequence[Mapping[str, Any]],
    size: int,
) -> Iterable[Sequence[Mapping[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


__all__ = ["load_cleaning_result"]
