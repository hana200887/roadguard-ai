from __future__ import annotations

from typing import cast

import pytest
from pydantic import SecretStr
from sqlalchemy.engine import Engine

from roadguard._db_models import DB_SCHEMA, metadata, road_observations
from roadguard.config import RoadGuardConfig
from roadguard.database import (
    DatabaseConfigurationError,
    create_database_engine,
)
from roadguard.observations import OBSERVATION_COLUMNS
from roadguard.targets import TARGET_COLUMNS

EXPECTED_TABLES = {
    "maintenance_events",
    "maintenance_history",
    "material_forecasts",
    "observation_targets",
    "predictions",
    "road_observations",
    "road_segments",
}


def test_runtime_database_url_is_optional_and_secret() -> None:
    config = RoadGuardConfig(database_url="postgresql+psycopg://user:private@localhost/db")

    assert isinstance(config.database_url, SecretStr)
    assert "private" not in repr(config)
    assert RoadGuardConfig().database_url is None


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        42,
        "",
        "sqlite:///roadguard.db",
        "postgresql://user:password@localhost/db",
    ],
)
def test_engine_factory_rejects_missing_or_non_psycopg_urls_without_leaking(
    value: object,
) -> None:
    with pytest.raises(DatabaseConfigurationError) as exc_info:
        create_database_engine(cast(str | SecretStr | None, value))

    message = str(exc_info.value)
    assert "password" not in message
    assert "sqlite" not in message


def test_engine_factory_builds_postgresql_psycopg_engine_without_connecting() -> None:
    engine = create_database_engine(
        SecretStr("postgresql+psycopg://roadguard:private@127.0.0.1:1/roadguard")
    )

    assert isinstance(engine, Engine)
    assert engine.dialect.name == "postgresql"
    assert "private" not in repr(engine)
    engine.dispose()


def test_metadata_has_exact_phase6_tables_in_fixed_schema() -> None:
    assert set(metadata.tables) == {f"{DB_SCHEMA}.{name}" for name in EXPECTED_TABLES}
    assert {table.name for table in metadata.sorted_tables} == EXPECTED_TABLES


def test_observation_table_contains_only_approved_feature_columns() -> None:
    assert tuple(road_observations.c.keys()) == OBSERVATION_COLUMNS
    assert set(TARGET_COLUMNS[2:]).isdisjoint(road_observations.c.keys())


def test_metadata_uses_natural_keys_and_required_foreign_keys() -> None:
    expected_primary_keys = {
        "road_segments": ("segment_id",),
        "road_observations": ("segment_id", "date"),
        "observation_targets": ("segment_id", "date"),
        "maintenance_events": ("segment_id", "maintenance_date"),
        "maintenance_history": ("segment_id", "maintenance_date"),
        "predictions": ("segment_id", "date"),
        "material_forecasts": ("period", "material"),
    }

    for table in metadata.sorted_tables:
        actual = tuple(column.name for column in table.primary_key.columns)
        assert actual == expected_primary_keys[table.name]
        assert all(column.name != "id" for column in table.columns)

    foreign_key_targets = {
        element.target_fullname
        for table in metadata.sorted_tables
        for constraint in table.foreign_key_constraints
        for element in constraint.elements
    }
    assert f"{DB_SCHEMA}.road_segments.segment_id" in foreign_key_targets
    assert f"{DB_SCHEMA}.road_observations.segment_id" in foreign_key_targets
    assert f"{DB_SCHEMA}.maintenance_events.maintenance_date" in foreign_key_targets
