"""SQLAlchemy Core metadata for the Phase 6 PostgreSQL persistence boundary."""

from __future__ import annotations

from typing import Final

import sqlalchemy as sa

from roadguard.data_quality import PUBLIC_SEGMENT_COLUMNS
from roadguard.events import EVENT_COLUMNS
from roadguard.observations import OBSERVATION_COLUMNS
from roadguard.targets import TARGET_COLUMNS

DB_SCHEMA: Final[str] = "roadguard"
MAINTENANCE_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "segment_id",
    "maintenance_date",
    "maintenance_cost",
    "thermoplastic_paint_kg",
    "reflective_sheet_m2",
    "guardrail_meter",
    "traffic_sign_quantity",
)
MATERIALS: Final[tuple[str, ...]] = (
    "thermoplastic_paint_kg",
    "reflective_sheet_m2",
    "guardrail_meter",
    "traffic_sign_quantity",
)

_NAMING_CONVENTION: Final[dict[str, str]] = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = sa.MetaData(schema=DB_SCHEMA, naming_convention=_NAMING_CONVENTION)

road_segments = sa.Table(
    "road_segments",
    metadata,
    sa.Column("segment_id", sa.Text, primary_key=True),
    sa.Column("province", sa.Text, nullable=False),
    sa.Column("road_type", sa.Text, nullable=False),
    sa.Column("construction_date", sa.Date, nullable=False),
    sa.Column("road_length_km", sa.Float, nullable=False),
    sa.CheckConstraint(
        r"segment_id ~ '^QL[0-9]{2}-KM[0-9]+-[0-9]+$'",
        name="segment_id_format",
    ),
    sa.CheckConstraint("province IN ('NA', 'TH', 'QB', 'DN', 'GL', 'LA')", name="province"),
    sa.CheckConstraint(
        "road_type IN ('highway', 'national', 'provincial', 'urban', 'rural')",
        name="road_type",
    ),
    sa.CheckConstraint(
        "road_length_km > 0 AND road_length_km < 'Infinity'::float8",
        name="road_length",
    ),
)

road_observations = sa.Table(
    "road_observations",
    metadata,
    sa.Column(
        "segment_id",
        sa.Text,
        sa.ForeignKey(f"{DB_SCHEMA}.road_segments.segment_id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("traffic_volume", sa.BigInteger, nullable=False),
    sa.Column("heavy_vehicle_ratio", sa.Float, nullable=False),
    sa.Column("road_age_days", sa.BigInteger, nullable=False),
    sa.Column("rainfall_mm", sa.Float, nullable=False),
    sa.Column("temperature", sa.Float, nullable=False),
    sa.Column("humidity", sa.Float, nullable=False),
    sa.Column("days_since_last_maintenance", sa.BigInteger, nullable=False),
    sa.Column("previous_repairs", sa.BigInteger, nullable=False),
    sa.Column("road_condition_score", sa.SmallInteger, nullable=False),
    sa.Column("marking_condition_score", sa.SmallInteger, nullable=False),
    sa.Column("guardrail_condition_score", sa.SmallInteger, nullable=False),
    sa.Column("sign_condition_score", sa.SmallInteger, nullable=False),
    sa.Column("accident_count_30d", sa.BigInteger, nullable=False),
    sa.Column("accident_count_365d", sa.BigInteger, nullable=False),
    sa.CheckConstraint("traffic_volume >= 0", name="traffic_volume"),
    sa.CheckConstraint("heavy_vehicle_ratio BETWEEN 0 AND 1", name="heavy_vehicle_ratio"),
    sa.CheckConstraint("road_age_days >= 0", name="road_age_days"),
    sa.CheckConstraint(
        "rainfall_mm >= 0 AND rainfall_mm < 'Infinity'::float8",
        name="rainfall_mm",
    ),
    sa.CheckConstraint("temperature BETWEEN -50 AND 60", name="temperature"),
    sa.CheckConstraint("humidity BETWEEN 0 AND 100", name="humidity"),
    sa.CheckConstraint(
        "days_since_last_maintenance BETWEEN 0 AND road_age_days",
        name="maintenance_age",
    ),
    sa.CheckConstraint("previous_repairs >= 0", name="previous_repairs"),
    sa.CheckConstraint("road_condition_score BETWEEN 1 AND 100", name="road_score"),
    sa.CheckConstraint("marking_condition_score BETWEEN 1 AND 100", name="marking_score"),
    sa.CheckConstraint("guardrail_condition_score BETWEEN 1 AND 100", name="guardrail_score"),
    sa.CheckConstraint("sign_condition_score BETWEEN 1 AND 100", name="sign_score"),
    sa.CheckConstraint(
        "accident_count_30d >= 0 AND accident_count_365d >= accident_count_30d",
        name="accident_counts",
    ),
    sa.Index("ix_road_observations_date", "date"),
)

maintenance_events = sa.Table(
    "maintenance_events",
    metadata,
    sa.Column(
        "segment_id",
        sa.Text,
        sa.ForeignKey(f"{DB_SCHEMA}.road_segments.segment_id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    sa.Column("maintenance_date", sa.Date, primary_key=True),
    sa.Index("ix_maintenance_events_date", "maintenance_date"),
    sa.Index(
        "uq_maintenance_events_segment_month",
        "segment_id",
        sa.extract("year", sa.column("maintenance_date")),
        sa.extract("month", sa.column("maintenance_date")),
        unique=True,
    ),
)

observation_targets = sa.Table(
    "observation_targets",
    metadata,
    sa.Column("segment_id", sa.Text, primary_key=True),
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("days_until_maintenance", sa.BigInteger, nullable=False),
    sa.Column("maintenance_within_30_days", sa.SmallInteger, nullable=False),
    sa.ForeignKeyConstraint(
        ["segment_id", "date"],
        [
            f"{DB_SCHEMA}.road_observations.segment_id",
            f"{DB_SCHEMA}.road_observations.date",
        ],
        ondelete="CASCADE",
    ),
    sa.CheckConstraint("days_until_maintenance >= 0", name="days_until_maintenance"),
    sa.CheckConstraint(
        "maintenance_within_30_days IN (0, 1)",
        name="maintenance_within_30_days",
    ),
    sa.CheckConstraint(
        "maintenance_within_30_days = CASE WHEN days_until_maintenance <= 30 THEN 1 ELSE 0 END",
        name="target_consistency",
    ),
)

maintenance_history = sa.Table(
    "maintenance_history",
    metadata,
    sa.Column("segment_id", sa.Text, primary_key=True),
    sa.Column("maintenance_date", sa.Date, primary_key=True),
    sa.Column("maintenance_cost", sa.BigInteger, nullable=False),
    sa.Column("thermoplastic_paint_kg", sa.Float, nullable=False),
    sa.Column("reflective_sheet_m2", sa.Float, nullable=False),
    sa.Column("guardrail_meter", sa.Float, nullable=False),
    sa.Column("traffic_sign_quantity", sa.BigInteger, nullable=False),
    sa.ForeignKeyConstraint(
        ["segment_id", "maintenance_date"],
        [
            f"{DB_SCHEMA}.maintenance_events.segment_id",
            f"{DB_SCHEMA}.maintenance_events.maintenance_date",
        ],
        ondelete="CASCADE",
    ),
    sa.CheckConstraint("maintenance_cost > 0", name="maintenance_cost"),
    sa.CheckConstraint(
        "thermoplastic_paint_kg >= 0 AND thermoplastic_paint_kg < 'Infinity'::float8",
        name="paint_quantity",
    ),
    sa.CheckConstraint(
        "reflective_sheet_m2 >= 0 AND reflective_sheet_m2 < 'Infinity'::float8",
        name="sheet_quantity",
    ),
    sa.CheckConstraint(
        "guardrail_meter >= 0 AND guardrail_meter < 'Infinity'::float8",
        name="guardrail_quantity",
    ),
    sa.CheckConstraint("traffic_sign_quantity >= 0", name="sign_quantity"),
    sa.Index("ix_maintenance_history_date", "maintenance_date"),
)

predictions = sa.Table(
    "predictions",
    metadata,
    sa.Column("segment_id", sa.Text, primary_key=True),
    sa.Column("date", sa.Date, primary_key=True),
    sa.Column("maintenance_probability", sa.Float, nullable=False),
    sa.Column("risk_score", sa.SmallInteger, nullable=False),
    sa.Column("risk_band", sa.Text, nullable=False),
    sa.ForeignKeyConstraint(
        ["segment_id", "date"],
        [
            f"{DB_SCHEMA}.road_observations.segment_id",
            f"{DB_SCHEMA}.road_observations.date",
        ],
        ondelete="CASCADE",
    ),
    sa.CheckConstraint(
        "maintenance_probability BETWEEN 0 AND 1",
        name="maintenance_probability",
    ),
    sa.CheckConstraint("risk_score BETWEEN 0 AND 100", name="risk_score"),
    sa.CheckConstraint(
        "(risk_band = 'LOW' AND risk_score BETWEEN 0 AND 30) OR "
        "(risk_band = 'MEDIUM' AND risk_score BETWEEN 31 AND 60) OR "
        "(risk_band = 'HIGH' AND risk_score BETWEEN 61 AND 80) OR "
        "(risk_band = 'CRITICAL' AND risk_score BETWEEN 81 AND 100)",
        name="risk_band",
    ),
    sa.Index("ix_predictions_date", "date"),
)

material_forecasts = sa.Table(
    "material_forecasts",
    metadata,
    sa.Column("period", sa.Date, primary_key=True),
    sa.Column("material", sa.Text, primary_key=True),
    sa.Column("forecast_quantity", sa.Float, nullable=False),
    sa.CheckConstraint(
        "material IN ('thermoplastic_paint_kg', 'reflective_sheet_m2', "
        "'guardrail_meter', 'traffic_sign_quantity')",
        name="material",
    ),
    sa.CheckConstraint(
        "forecast_quantity >= 0 AND forecast_quantity < 'Infinity'::float8",
        name="forecast_quantity",
    ),
    sa.Index("ix_material_forecasts_period", "period"),
)

assert tuple(road_segments.c.keys()) == PUBLIC_SEGMENT_COLUMNS
assert tuple(road_observations.c.keys()) == OBSERVATION_COLUMNS
assert tuple(observation_targets.c.keys()) == TARGET_COLUMNS
assert tuple(maintenance_events.c.keys()) == EVENT_COLUMNS
assert tuple(maintenance_history.c.keys()) == MAINTENANCE_HISTORY_COLUMNS

__all__ = [
    "DB_SCHEMA",
    "MAINTENANCE_HISTORY_COLUMNS",
    "MATERIALS",
    "maintenance_events",
    "maintenance_history",
    "material_forecasts",
    "metadata",
    "observation_targets",
    "predictions",
    "road_observations",
    "road_segments",
]
