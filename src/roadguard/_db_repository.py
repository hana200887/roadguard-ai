"""Read-only PostgreSQL repository queries for approved Phase 6 use cases."""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from typing import Final

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from roadguard._db_engine import _require_postgresql
from roadguard._db_models import (
    MAINTENANCE_HISTORY_COLUMNS,
    MATERIALS,
    maintenance_events,
    maintenance_history,
    observation_targets,
    road_observations,
    road_segments,
)
from roadguard._db_types import (
    PersistenceError,
    RepositoryExport,
    RepositoryInputError,
    SegmentHistory,
)
from roadguard.data_quality import PUBLIC_SEGMENT_COLUMNS
from roadguard.events import EVENT_COLUMNS
from roadguard.observations import OBSERVATION_COLUMNS
from roadguard.segments import SEGMENT_ID_PATTERN
from roadguard.targets import TARGET_COLUMNS

_SEGMENT_PATTERN: Final[re.Pattern[str]] = re.compile(SEGMENT_ID_PATTERN)
_OBSERVATION_INT_COLUMNS: Final[tuple[str, ...]] = (
    "traffic_volume",
    "road_age_days",
    "days_since_last_maintenance",
    "previous_repairs",
    "road_condition_score",
    "marking_condition_score",
    "guardrail_condition_score",
    "sign_condition_score",
    "accident_count_30d",
    "accident_count_365d",
)
_OBSERVATION_FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    "heavy_vehicle_ratio",
    "rainfall_mm",
    "temperature",
    "humidity",
)


class PostgresRepository:
    """Deterministic, parameter-bound queries over the fixed RoadGuard schema."""

    def __init__(self, engine: Engine) -> None:
        _require_postgresql(engine)
        self._engine = engine

    def export_dataset(self) -> RepositoryExport:
        """Export physical inputs and targets independently in stable order."""
        try:
            with _read_snapshot(self._engine) as connection:
                segments = _read_frame(
                    connection,
                    sa.select(road_segments).order_by(road_segments.c.segment_id),
                    PUBLIC_SEGMENT_COLUMNS,
                )
                observations = _read_frame(
                    connection,
                    sa.select(road_observations).order_by(
                        road_observations.c.segment_id,
                        road_observations.c.date,
                    ),
                    OBSERVATION_COLUMNS,
                )
                targets = _read_frame(
                    connection,
                    sa.select(observation_targets).order_by(
                        observation_targets.c.segment_id,
                        observation_targets.c.date,
                    ),
                    TARGET_COLUMNS,
                )
                events = _read_frame(
                    connection,
                    sa.select(maintenance_events).order_by(
                        maintenance_events.c.segment_id,
                        maintenance_events.c.maintenance_date,
                    ),
                    EVENT_COLUMNS,
                )
        except SQLAlchemyError as exc:
            raise PersistenceError("PostgreSQL dataset export failed") from exc
        return RepositoryExport(
            segments=_normalize_segments(segments),
            observations=_normalize_observations(observations),
            targets=_normalize_targets(targets),
            maintenance_events=_normalize_events(events),
        )

    def export_material_forecast_inputs(self) -> tuple[RepositoryExport, pd.DataFrame]:
        """Return one repeatable-read snapshot for the Phase 14 pure workflow."""
        try:
            with _read_snapshot(self._engine) as connection:
                segments = _read_frame(
                    connection,
                    sa.select(road_segments).order_by(road_segments.c.segment_id),
                    PUBLIC_SEGMENT_COLUMNS,
                )
                observations = _read_frame(
                    connection,
                    sa.select(road_observations).order_by(
                        road_observations.c.segment_id,
                        road_observations.c.date,
                    ),
                    OBSERVATION_COLUMNS,
                )
                targets = _read_frame(
                    connection,
                    sa.select(observation_targets).order_by(
                        observation_targets.c.segment_id,
                        observation_targets.c.date,
                    ),
                    TARGET_COLUMNS,
                )
                events = _read_frame(
                    connection,
                    sa.select(maintenance_events).order_by(
                        maintenance_events.c.segment_id,
                        maintenance_events.c.maintenance_date,
                    ),
                    EVENT_COLUMNS,
                )
                history = _read_frame(
                    connection,
                    sa.select(maintenance_history).order_by(
                        maintenance_history.c.segment_id,
                        maintenance_history.c.maintenance_date,
                    ),
                    MAINTENANCE_HISTORY_COLUMNS,
                )
        except SQLAlchemyError as exc:
            raise PersistenceError("PostgreSQL material forecast input export failed") from exc
        return (
            RepositoryExport(
                segments=_normalize_segments(segments),
                observations=_normalize_observations(observations),
                targets=_normalize_targets(targets),
                maintenance_events=_normalize_events(events),
            ),
            _normalize_history(history),
        )

    def get_segment_history(self, segment_id: str, as_of_date: date) -> SegmentHistory:
        """Return observations through ``as_of_date`` and strictly prior events."""
        _validate_history_input(segment_id, as_of_date)
        try:
            with _read_snapshot(self._engine) as connection:
                segment = _read_frame(
                    connection,
                    sa.select(road_segments)
                    .where(road_segments.c.segment_id == segment_id)
                    .order_by(road_segments.c.segment_id),
                    PUBLIC_SEGMENT_COLUMNS,
                )
                observations = _read_frame(
                    connection,
                    sa.select(road_observations)
                    .where(
                        road_observations.c.segment_id == segment_id,
                        road_observations.c.date <= as_of_date,
                    )
                    .order_by(road_observations.c.date),
                    OBSERVATION_COLUMNS,
                )
                events = _read_frame(
                    connection,
                    sa.select(maintenance_events)
                    .where(
                        maintenance_events.c.segment_id == segment_id,
                        maintenance_events.c.maintenance_date < as_of_date,
                    )
                    .order_by(maintenance_events.c.maintenance_date),
                    EVENT_COLUMNS,
                )
        except SQLAlchemyError as exc:
            raise PersistenceError("PostgreSQL history query failed") from exc
        if segment.empty:
            raise RepositoryInputError("segment_id does not exist")
        if observations.empty:
            raise RepositoryInputError("observation history is unavailable at this date")
        return SegmentHistory(
            segment=_normalize_segments(segment),
            observations=_normalize_observations(observations),
            maintenance_events=_normalize_events(events),
        )

    def aggregate_monthly_material_usage(self) -> pd.DataFrame:
        """Aggregate only fully realized maintenance-history material quantities."""
        statements = [
            sa.select(
                sa.func.date_trunc("month", maintenance_history.c.maintenance_date)
                .cast(sa.Date)
                .label("period"),
                sa.literal(material).label("material"),
                maintenance_history.c[material].cast(sa.Float).label("quantity"),
            )
            for material in MATERIALS
        ]
        rows = sa.union_all(*statements).subquery()
        query = (
            sa.select(
                rows.c.period,
                rows.c.material,
                sa.func.sum(rows.c.quantity).label("quantity"),
            )
            .group_by(rows.c.period, rows.c.material)
            .order_by(rows.c.period, rows.c.material)
        )
        try:
            with _read_snapshot(self._engine) as connection:
                frame = _read_frame(connection, query, ("period", "material", "quantity"))
        except SQLAlchemyError as exc:
            raise PersistenceError("PostgreSQL material aggregation failed") from exc
        frame["period"] = pd.to_datetime(frame["period"])
        frame["material"] = frame["material"].astype(object)
        frame["quantity"] = frame["quantity"].astype("float64")
        return frame


def _validate_history_input(segment_id: str, as_of_date: date) -> None:
    if type(segment_id) is not str or _SEGMENT_PATTERN.fullmatch(segment_id) is None:
        raise RepositoryInputError("segment_id format is invalid")
    if type(as_of_date) is not date:
        raise RepositoryInputError("as_of_date must be a date")


@contextmanager
def _read_snapshot(engine: Engine) -> Iterator[Connection]:
    with engine.connect() as raw_connection:
        connection = raw_connection.execution_options(isolation_level="REPEATABLE READ")
        with connection.begin():
            connection.execute(sa.text("SET TRANSACTION READ ONLY"))
            yield connection


def _read_frame(
    connection: Connection,
    statement: sa.Executable,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    rows = connection.execute(statement).mappings().all()
    return pd.DataFrame((dict(row) for row in rows), columns=columns)


def _normalize_segments(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["segment_id"] = normalized["segment_id"].astype(object)
    normalized["province"] = normalized["province"].astype(object)
    normalized["road_type"] = normalized["road_type"].astype(object)
    normalized["construction_date"] = pd.to_datetime(normalized["construction_date"])
    normalized["road_length_km"] = normalized["road_length_km"].astype("float64")
    return normalized


def _normalize_observations(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["segment_id"] = normalized["segment_id"].astype(object)
    normalized["date"] = pd.to_datetime(normalized["date"])
    for column in _OBSERVATION_INT_COLUMNS:
        normalized[column] = normalized[column].astype("int64")
    for column in _OBSERVATION_FLOAT_COLUMNS:
        normalized[column] = normalized[column].astype("float64")
    return normalized


def _normalize_targets(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["segment_id"] = normalized["segment_id"].astype(object)
    normalized["date"] = pd.to_datetime(normalized["date"])
    for column in TARGET_COLUMNS[2:]:
        normalized[column] = normalized[column].astype("int64")
    return normalized


def _normalize_events(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["segment_id"] = normalized["segment_id"].astype(object)
    normalized["maintenance_date"] = pd.to_datetime(normalized["maintenance_date"])
    return normalized


def _normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["segment_id"] = normalized["segment_id"].astype(object)
    normalized["maintenance_date"] = pd.to_datetime(normalized["maintenance_date"])
    normalized["maintenance_cost"] = normalized["maintenance_cost"].astype("int64")
    for column in MATERIALS[:-1]:
        normalized[column] = normalized[column].astype("float64")
    normalized["traffic_sign_quantity"] = normalized["traffic_sign_quantity"].astype("int64")
    return normalized


__all__ = ["PostgresRepository"]
