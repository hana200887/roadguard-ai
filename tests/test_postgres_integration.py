from __future__ import annotations

import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date
from threading import Barrier
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from pandas.testing import assert_frame_equal
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

import roadguard._db_etl as db_etl
import roadguard._db_repository as db_repository
from roadguard._db_models import DB_SCHEMA, MAINTENANCE_HISTORY_COLUMNS, metadata
from roadguard._dq_cleaning import CleaningResult
from roadguard._dq_validation import ValidationIssue, ValidationReport
from roadguard.contracts import DatasetSpec
from roadguard.database import (
    DatabaseUnavailableError,
    PersistenceConflict,
    PersistenceError,
    PostgresRepository,
    RepositoryInputError,
    create_database_engine,
    initialize_database,
    load_cleaning_result,
)
from roadguard.events import EVENT_COLUMNS
from roadguard.observations import OBSERVATION_COLUMNS
from roadguard.targets import TARGET_COLUMNS

TEST_DATABASE_URL_ENV = "TEST_DATABASE_URL"
MATERIAL_COLUMNS = (
    "maintenance_cost",
    "thermoplastic_paint_kg",
    "reflective_sheet_m2",
    "guardrail_meter",
    "traffic_sign_quantity",
)


@pytest.fixture(scope="session")
def pg_engine() -> Iterator[Engine]:
    raw = os.getenv(TEST_DATABASE_URL_ENV)
    if raw is None:
        pytest.fail("real PostgreSQL required; Phase 6 INCOMPLETE")
    base_engine = create_database_engine(raw)
    try:
        require_disposable_test_engine(base_engine)
    except ValueError as exc:
        pytest.fail(str(exc))
    test_schema = f"roadguard_test_{uuid4().hex}"
    engine = base_engine.execution_options(
        schema_translate_map={DB_SCHEMA: test_schema},
    )
    if engine.dialect.name != "postgresql":
        pytest.fail("real PostgreSQL required; Phase 6 INCOMPLETE")
    try:
        with engine.connect() as connection:
            assert connection.execute(sa.text("SELECT 1")).scalar_one() == 1
    except Exception as exc:
        raise AssertionError("real PostgreSQL unavailable; Phase 6 INCOMPLETE") from exc
    yield engine
    engine.dispose()
    with base_engine.begin() as connection:
        connection.execute(sa.schema.DropSchema(test_schema, cascade=True, if_exists=True))
    base_engine.dispose()


def active_schema(engine: Engine) -> str:
    schema_map = engine.get_execution_options().get("schema_translate_map")
    assert isinstance(schema_map, dict)
    return cast(str, schema_map[DB_SCHEMA])


def require_disposable_test_engine(engine: Engine) -> None:
    url = engine.url
    if (
        url.host not in {"127.0.0.1", "localhost"}
        or not (url.database or "").endswith("_test")
        or not (url.username or "").endswith("_test")
    ):
        raise ValueError("disposable local PostgreSQL test database required")


@pytest.fixture(autouse=True)
def isolated_schema(pg_engine: Engine) -> Iterator[None]:
    initialize_database(pg_engine)
    yield
    with pg_engine.begin() as connection:
        connection.execute(
            sa.schema.DropSchema(active_schema(pg_engine), cascade=True, if_exists=True)
        )


@pytest.fixture
def dataset_spec() -> DatasetSpec:
    return DatasetSpec(
        dataset_segments=1,
        dataset_months_per_segment=3,
        dataset_observations=3,
    )


@pytest.fixture
def cleaned_result() -> CleaningResult:
    segments = pd.DataFrame(
        {
            "segment_id": ["QL01-KM1-2"],
            "province": ["NA"],
            "road_type": ["national"],
            "construction_date": pd.to_datetime(["2010-01-01"]),
            "road_length_km": pd.Series([1.0], dtype="float64"),
        }
    )
    observations = pd.DataFrame(
        {
            "segment_id": ["QL01-KM1-2", "QL01-KM1-2", "QL01-KM1-2"],
            "date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
            "traffic_volume": pd.Series([1000, 1100, 1200], dtype="int64"),
            "heavy_vehicle_ratio": pd.Series([0.2, 0.21, 0.22], dtype="float64"),
            "road_age_days": pd.Series([4383, 4414, 4442], dtype="int64"),
            "rainfall_mm": pd.Series([10.0, 20.0, 30.0], dtype="float64"),
            "temperature": pd.Series([25.0, 26.0, 27.0], dtype="float64"),
            "humidity": pd.Series([60.0, 61.0, 62.0], dtype="float64"),
            "days_since_last_maintenance": pd.Series([100, 10, 38], dtype="int64"),
            "previous_repairs": pd.Series([0, 1, 1], dtype="int64"),
            "road_condition_score": pd.Series([80, 90, 88], dtype="int64"),
            "marking_condition_score": pd.Series([75, 85, 83], dtype="int64"),
            "guardrail_condition_score": pd.Series([78, 88, 86], dtype="int64"),
            "sign_condition_score": pd.Series([77, 87, 85], dtype="int64"),
            "accident_count_30d": pd.Series([0, 1, 0], dtype="int64"),
            "accident_count_365d": pd.Series([1, 2, 2], dtype="int64"),
        },
        columns=OBSERVATION_COLUMNS,
    )
    targets = pd.DataFrame(
        {
            "segment_id": ["QL01-KM1-2", "QL01-KM1-2", "QL01-KM1-2"],
            "date": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01"]),
            "days_until_maintenance": pd.Series([14, 28, 0], dtype="int64"),
            "maintenance_within_30_days": pd.Series([1, 1, 1], dtype="int64"),
        },
        columns=TARGET_COLUMNS,
    )
    events = pd.DataFrame(
        {
            "segment_id": ["QL01-KM1-2", "QL01-KM1-2", "QL01-KM1-2"],
            "maintenance_date": pd.to_datetime(["2022-01-15", "2022-03-01", "2022-04-01"]),
        },
        columns=EVENT_COLUMNS,
    )
    return CleaningResult(
        segments=segments,
        observations=observations,
        targets=targets,
        maintenance_events=events,
        report=ValidationReport(issues=()),
    )


@pytest.fixture
def maintenance_history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["QL01-KM1-2"],
            "maintenance_date": pd.to_datetime(["2022-01-15"]),
            "maintenance_cost": pd.Series([1_000_000], dtype="int64"),
            "thermoplastic_paint_kg": pd.Series([5.0], dtype="float64"),
            "reflective_sheet_m2": pd.Series([0.0], dtype="float64"),
            "guardrail_meter": pd.Series([2.0], dtype="float64"),
            "traffic_sign_quantity": pd.Series([1], dtype="int64"),
        }
    )


def table_counts(engine: Engine) -> dict[str, int]:
    with engine.connect() as connection:
        return {
            table.name: int(
                connection.execute(sa.select(sa.func.count()).select_from(table)).scalar_one()
            )
            for table in metadata.sorted_tables
        }


def test_schema_initialization_is_repeatable_and_postgresql_native(pg_engine: Engine) -> None:
    initialize_database(pg_engine)
    inspector = sa.inspect(pg_engine)
    schema = active_schema(pg_engine)

    assert pg_engine.dialect.name == "postgresql"
    assert set(inspector.get_table_names(schema=schema)) == {
        table.name for table in metadata.sorted_tables
    }
    assert inspector.has_table("road_segments", schema=schema)
    assert not inspector.has_table("road_segments", schema=DB_SCHEMA)


def test_test_harness_refuses_non_disposable_database_url() -> None:
    unsafe = create_database_engine(
        "postgresql+psycopg://roadguard_app:private@db.example/roadguard"
    )
    try:
        with pytest.raises(ValueError, match="disposable local PostgreSQL"):
            require_disposable_test_engine(unsafe)
    finally:
        unsafe.dispose()


def test_database_foreign_keys_and_natural_keys_are_enforced(pg_engine: Engine) -> None:
    observations = metadata.tables[f"{DB_SCHEMA}.road_observations"]
    segments = metadata.tables[f"{DB_SCHEMA}.road_segments"]
    orphan = {column.name: 0 for column in observations.columns if column.name != "segment_id"}
    orphan.update(segment_id="QL01-KM9-10", date=date(2022, 1, 1))

    with pytest.raises(IntegrityError), pg_engine.begin() as connection:
        connection.execute(observations.insert().values(orphan))

    valid_segment = {
        "segment_id": "QL01-KM1-2",
        "province": "NA",
        "road_type": "national",
        "construction_date": date(2010, 1, 1),
        "road_length_km": 1.0,
    }
    with pg_engine.begin() as connection:
        connection.execute(segments.insert().values(valid_segment))
    with pytest.raises(IntegrityError), pg_engine.begin() as connection:
        connection.execute(segments.insert().values(valid_segment))


def test_transactional_load_is_idempotent_and_reconciles_rows(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    original = tuple(
        frame.copy(deep=True)
        for frame in (
            cleaned_result.segments,
            cleaned_result.observations,
            cleaned_result.targets,
            cleaned_result.maintenance_events,
        )
    )

    first = load_cleaning_result(pg_engine, cleaned_result, dataset_spec)
    second = load_cleaning_result(pg_engine, cleaned_result, dataset_spec)

    assert first.inserted == {
        "maintenance_events": 3,
        "maintenance_history": 0,
        "observation_targets": 3,
        "road_observations": 3,
        "road_segments": 1,
    }
    assert all(count == 0 for count in second.inserted.values())
    assert second.existing == first.inserted
    assert table_counts(pg_engine) | {} == {
        "maintenance_events": 3,
        "maintenance_history": 0,
        "material_forecasts": 0,
        "observation_targets": 3,
        "predictions": 0,
        "road_observations": 3,
        "road_segments": 1,
    }
    for before, after in zip(
        original,
        (
            cleaned_result.segments,
            cleaned_result.observations,
            cleaned_result.targets,
            cleaned_result.maintenance_events,
        ),
        strict=True,
    ):
        assert_frame_equal(before, after)


def test_conflicting_replay_fails_without_overwriting(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    load_cleaning_result(pg_engine, cleaned_result, dataset_spec)
    before = table_counts(pg_engine)
    changed = cleaned_result.observations.copy(deep=True)
    changed.loc[0, "traffic_volume"] = 9999
    conflict = replace(cleaned_result, observations=changed)

    with pytest.raises(PersistenceConflict):
        load_cleaning_result(pg_engine, conflict, dataset_spec)

    assert table_counts(pg_engine) == before
    exported = PostgresRepository(pg_engine).export_dataset()
    assert int(exported.observations.loc[0, "traffic_volume"]) == 1000


def test_late_constraint_failure_rolls_back_earlier_inserts(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
    maintenance_history: pd.DataFrame,
) -> None:
    bad = maintenance_history.copy(deep=True)
    bad.loc[0, "maintenance_cost"] = -1

    with pytest.raises(PersistenceError):
        load_cleaning_result(
            pg_engine,
            cleaned_result,
            dataset_spec,
            maintenance_history=bad,
        )

    assert all(count == 0 for count in table_counts(pg_engine).values())


def test_maintenance_history_requires_complete_realized_schema_before_write(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
    maintenance_history: pd.DataFrame,
) -> None:
    partial = maintenance_history.drop(columns=[MATERIAL_COLUMNS[-1]])

    with pytest.raises(RepositoryInputError, match="maintenance_history schema"):
        load_cleaning_result(
            pg_engine,
            cleaned_result,
            dataset_spec,
            maintenance_history=partial,
        )

    assert all(count == 0 for count in table_counts(pg_engine).values())


def test_repository_export_is_deterministic_and_targets_remain_separate(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    shuffled = replace(
        cleaned_result,
        observations=cleaned_result.observations.sample(frac=1, random_state=7),
        targets=cleaned_result.targets.sample(frac=1, random_state=8),
        maintenance_events=cleaned_result.maintenance_events.sample(frac=1, random_state=9),
    )
    load_cleaning_result(pg_engine, shuffled, dataset_spec)
    first = PostgresRepository(pg_engine).export_dataset()
    second = PostgresRepository(pg_engine).export_dataset()

    assert tuple(first.observations.columns) == OBSERVATION_COLUMNS
    assert tuple(first.targets.columns) == TARGET_COLUMNS
    assert tuple(first.maintenance_events.columns) == EVENT_COLUMNS
    assert set(TARGET_COLUMNS[2:]).isdisjoint(first.observations.columns)
    assert_frame_equal(first.segments, second.segments)
    assert_frame_equal(first.observations, second.observations)
    assert_frame_equal(first.targets, second.targets)
    assert_frame_equal(first.maintenance_events, second.maintenance_events)
    assert first.observations["traffic_volume"].dtype == "int64"
    assert first.observations["heavy_vehicle_ratio"].dtype == "float64"
    assert str(first.observations["date"].dtype) == "datetime64[ns]"


def test_history_query_is_temporally_safe_and_rejects_injection(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    load_cleaning_result(pg_engine, cleaned_result, dataset_spec)
    repository = PostgresRepository(pg_engine)

    history = repository.get_segment_history("QL01-KM1-2", date(2022, 3, 1))

    assert list(history.observations["date"].dt.date) == [
        date(2022, 1, 1),
        date(2022, 2, 1),
        date(2022, 3, 1),
    ]
    assert list(history.maintenance_events["maintenance_date"].dt.date) == [date(2022, 1, 15)]
    with pytest.raises(RepositoryInputError):
        repository.get_segment_history(
            "QL01-KM1-2'; DROP SCHEMA roadguard CASCADE;--",
            date(2022, 3, 1),
        )
    assert sa.inspect(pg_engine).has_table(
        "road_segments",
        schema=active_schema(pg_engine),
    )


def test_material_aggregation_uses_only_realized_history(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
    maintenance_history: pd.DataFrame,
) -> None:
    load_cleaning_result(
        pg_engine,
        cleaned_result,
        dataset_spec,
        maintenance_history=maintenance_history,
    )

    aggregated = PostgresRepository(pg_engine).aggregate_monthly_material_usage()

    assert list(aggregated.columns) == ["period", "material", "quantity"]
    assert len(aggregated) == 4
    assert set(aggregated["material"]) == set(MATERIAL_COLUMNS[1:])
    assert (aggregated["quantity"] >= 0).all()


def test_phase14_combined_export_returns_exact_frames_and_never_reads_material_forecasts(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
    maintenance_history: pd.DataFrame,
) -> None:
    load_cleaning_result(
        pg_engine,
        cleaned_result,
        dataset_spec,
        maintenance_history=maintenance_history,
    )
    statements: list[str] = []
    select_isolation_levels: list[str] = []

    def collect_sql(
        connection: sa.Connection,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)
        if statement.lstrip().upper().startswith("SELECT"):
            select_isolation_levels.append(connection.get_isolation_level())

    event.listen(pg_engine, "after_cursor_execute", collect_sql)
    try:
        exported, history = PostgresRepository(pg_engine).export_material_forecast_inputs()
    finally:
        event.remove(pg_engine, "after_cursor_execute", collect_sql)

    assert tuple(exported.segments.columns) == (
        "segment_id",
        "province",
        "road_type",
        "construction_date",
        "road_length_km",
    )
    assert tuple(exported.observations.columns) == OBSERVATION_COLUMNS
    assert tuple(exported.targets.columns) == TARGET_COLUMNS
    assert tuple(exported.maintenance_events.columns) == EVENT_COLUMNS
    assert tuple(history.columns) == MAINTENANCE_HISTORY_COLUMNS
    assert history["segment_id"].dtype == object
    assert str(history["maintenance_date"].dtype) == "datetime64[ns]"
    assert history["maintenance_cost"].dtype == "int64"
    assert history["traffic_sign_quantity"].dtype == "int64"
    assert all(history[material].dtype == "float64" for material in MATERIAL_COLUMNS[1:-1])
    assert history.equals(
        history.sort_values(["segment_id", "maintenance_date"], kind="stable").reset_index(
            drop=True
        )
    )
    assert all("material_forecasts" not in statement.lower() for statement in statements)
    assert any(statement.strip().upper() == "SET TRANSACTION READ ONLY" for statement in statements)
    assert select_isolation_levels and set(select_isolation_levels) == {"REPEATABLE READ"}


def test_phase14_combined_export_preserves_empty_history_dtypes(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    load_cleaning_result(pg_engine, cleaned_result, dataset_spec)

    _exported, history = PostgresRepository(pg_engine).export_material_forecast_inputs()

    assert history.empty
    assert tuple(history.columns) == MAINTENANCE_HISTORY_COLUMNS
    assert history["segment_id"].dtype == object
    assert str(history["maintenance_date"].dtype) == "datetime64[ns]"
    assert history["maintenance_cost"].dtype == "int64"
    assert history["traffic_sign_quantity"].dtype == "int64"
    assert all(history[material].dtype == "float64" for material in MATERIAL_COLUMNS[1:-1])


def test_invalid_cleaning_report_is_rejected_before_database_write(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    invalid = replace(
        cleaned_result,
        report=ValidationReport(
            issues=(
                ValidationIssue(
                    severity="error",
                    code="invalid",
                    table="road_observations",
                ),
            )
        ),
    )

    with pytest.raises(RepositoryInputError, match="validated CleaningResult"):
        load_cleaning_result(pg_engine, invalid, dataset_spec)

    assert all(count == 0 for count in table_counts(pg_engine).values())


def test_forged_clean_report_cannot_bypass_fresh_validation(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    poisoned_targets = cleaned_result.targets.copy(deep=True)
    poisoned_targets.loc[1, "days_until_maintenance"] = 40
    poisoned_targets.loc[1, "maintenance_within_30_days"] = 0
    forged = replace(cleaned_result, targets=poisoned_targets)

    with pytest.raises(RepositoryInputError, match="validated CleaningResult"):
        load_cleaning_result(pg_engine, forged, dataset_spec)

    assert all(count == 0 for count in table_counts(pg_engine).values())


def test_history_before_first_observation_fails_explicitly(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    load_cleaning_result(pg_engine, cleaned_result, dataset_spec)

    with pytest.raises(RepositoryInputError, match="observation history"):
        PostgresRepository(pg_engine).get_segment_history(
            "QL01-KM1-2",
            date(2021, 12, 31),
        )


def test_database_enforces_one_event_per_segment_month(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    load_cleaning_result(pg_engine, cleaned_result, dataset_spec)
    events = metadata.tables[f"{DB_SCHEMA}.maintenance_events"]

    with pytest.raises(IntegrityError), pg_engine.begin() as connection:
        connection.execute(
            events.insert().values(
                segment_id="QL01-KM1-2",
                maintenance_date=date(2022, 1, 20),
            )
        )


def test_realized_history_must_belong_to_current_cleaning_result(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
    maintenance_history: pd.DataFrame,
) -> None:
    load_cleaning_result(pg_engine, cleaned_result, dataset_spec)
    events = metadata.tables[f"{DB_SCHEMA}.maintenance_events"]
    with pg_engine.begin() as connection:
        connection.execute(
            events.insert().values(
                segment_id="QL01-KM1-2",
                maintenance_date=date(2022, 5, 1),
            )
        )
    unrelated = maintenance_history.copy(deep=True)
    unrelated.loc[0, "maintenance_date"] = pd.Timestamp("2022-05-01")

    with pytest.raises(RepositoryInputError, match="current maintenance events"):
        load_cleaning_result(
            pg_engine,
            cleaned_result,
            dataset_spec,
            maintenance_history=unrelated,
        )


def test_schema_drift_fails_without_creating_missing_tables(pg_engine: Engine) -> None:
    schema = active_schema(pg_engine)
    with pg_engine.begin() as connection:
        connection.execute(sa.schema.DropSchema(schema, cascade=True, if_exists=True))
        connection.execute(sa.schema.CreateSchema(schema))
        translated = connection.execution_options(
            schema_translate_map={DB_SCHEMA: schema},
        )
        malformed = sa.Table(
            "road_segments",
            sa.MetaData(schema=schema),
            sa.Column("segment_id", sa.Text, primary_key=True),
        )
        malformed.create(translated)

    with pytest.raises(DatabaseUnavailableError, match="schema drift"):
        initialize_database(pg_engine)

    assert sa.inspect(pg_engine).get_table_names(schema=schema) == ["road_segments"]


def test_schema_drift_detects_same_named_weakened_check(pg_engine: Engine) -> None:
    schema = active_schema(pg_engine)
    with pg_engine.begin() as connection:
        quoted = connection.dialect.identifier_preparer.quote(schema)
        connection.exec_driver_sql(
            f"ALTER TABLE {quoted}.road_segments DROP CONSTRAINT ck_road_segments_road_length"
        )
        connection.exec_driver_sql(
            f"ALTER TABLE {quoted}.road_segments "
            "ADD CONSTRAINT ck_road_segments_road_length CHECK (road_length_km > -999)"
        )

    with pytest.raises(DatabaseUnavailableError, match="schema drift"):
        initialize_database(pg_engine)


def test_schema_drift_preserves_case_sensitive_check_literals(pg_engine: Engine) -> None:
    schema = active_schema(pg_engine)
    with pg_engine.begin() as connection:
        quoted = connection.dialect.identifier_preparer.quote(schema)
        connection.exec_driver_sql(
            f"ALTER TABLE {quoted}.road_segments DROP CONSTRAINT ck_road_segments_province"
        )
        connection.exec_driver_sql(
            f"ALTER TABLE {quoted}.road_segments "
            "ADD CONSTRAINT ck_road_segments_province "
            "CHECK (province IN ('na', 'th', 'qb', 'dn', 'gl', 'la'))"
        )

    with pytest.raises(DatabaseUnavailableError, match="schema drift"):
        initialize_database(pg_engine)


def test_schema_drift_detects_same_named_wrong_index(pg_engine: Engine) -> None:
    schema = active_schema(pg_engine)
    with pg_engine.begin() as connection:
        quoted = connection.dialect.identifier_preparer.quote(schema)
        connection.exec_driver_sql(f"DROP INDEX {quoted}.uq_maintenance_events_segment_month")
        connection.exec_driver_sql(
            f"CREATE UNIQUE INDEX uq_maintenance_events_segment_month "
            f"ON {quoted}.maintenance_events (segment_id, maintenance_date)"
        )

    with pytest.raises(DatabaseUnavailableError, match="schema drift"):
        initialize_database(pg_engine)


def test_schema_drift_detects_single_precision_float(pg_engine: Engine) -> None:
    schema = active_schema(pg_engine)
    with pg_engine.begin() as connection:
        quoted = connection.dialect.identifier_preparer.quote(schema)
        connection.exec_driver_sql(
            f"ALTER TABLE {quoted}.road_segments ALTER COLUMN road_length_km TYPE REAL"
        )

    with pytest.raises(DatabaseUnavailableError, match="schema drift"):
        initialize_database(pg_engine)


def test_schema_drift_detects_unexpected_unique_constraint(pg_engine: Engine) -> None:
    schema = active_schema(pg_engine)
    with pg_engine.begin() as connection:
        quoted = connection.dialect.identifier_preparer.quote(schema)
        connection.exec_driver_sql(
            f"ALTER TABLE {quoted}.road_segments "
            "ADD CONSTRAINT uq_road_segments_province UNIQUE (province)"
        )

    with pytest.raises(DatabaseUnavailableError, match="schema drift"):
        initialize_database(pg_engine)


def test_concurrent_identical_loads_are_idempotent(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = Barrier(2)
    original = db_etl._insert_or_verify

    def synchronized_insert(*args: object, **kwargs: object) -> tuple[int, int]:
        table = cast(sa.Table, args[1])
        if table.name == "road_segments":
            barrier.wait(timeout=5)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(db_etl, "_insert_or_verify", synchronized_insert)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                load_cleaning_result,
                pg_engine,
                cleaned_result,
                dataset_spec,
            )
            for _ in range(2)
        ]
        reports = [future.result(timeout=10) for future in futures]

    assert sum(report.inserted["road_segments"] for report in reports) == 1
    assert table_counts(pg_engine)["road_observations"] == 3


def test_concurrent_conflicting_load_is_classified_without_overwrite(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = Barrier(2)
    original = db_etl._insert_or_verify
    changed_observations = cleaned_result.observations.copy(deep=True)
    changed_observations.loc[0, "traffic_volume"] = 9999
    conflicting = replace(cleaned_result, observations=changed_observations)

    def synchronized_insert(*args: object, **kwargs: object) -> tuple[int, int]:
        table = cast(sa.Table, args[1])
        if table.name == "road_segments":
            barrier.wait(timeout=5)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(db_etl, "_insert_or_verify", synchronized_insert)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(load_cleaning_result, pg_engine, result, dataset_spec)
            for result in (cleaned_result, conflicting)
        ]
        outcomes: list[object] = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=10))
            except PersistenceConflict as exc:
                outcomes.append(exc)

    assert sum(isinstance(outcome, PersistenceConflict) for outcome in outcomes) == 1
    exported = PostgresRepository(pg_engine).export_dataset()
    assert int(exported.observations.loc[0, "traffic_volume"]) in {1000, 9999}


def test_concurrent_reversed_key_order_is_deadlock_safe(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    second_id = "QL01-KM2-3"

    def with_second_segment(frame: pd.DataFrame) -> pd.DataFrame:
        second = frame.copy(deep=True)
        second["segment_id"] = second_id
        return pd.concat([frame, second], ignore_index=True)

    doubled = CleaningResult(
        segments=with_second_segment(cleaned_result.segments),
        observations=with_second_segment(cleaned_result.observations),
        targets=with_second_segment(cleaned_result.targets),
        maintenance_events=with_second_segment(cleaned_result.maintenance_events),
        report=ValidationReport(issues=()),
    )
    reversed_result = replace(
        doubled,
        segments=doubled.segments.iloc[::-1].reset_index(drop=True),
        observations=doubled.observations.iloc[::-1].reset_index(drop=True),
        targets=doubled.targets.iloc[::-1].reset_index(drop=True),
        maintenance_events=doubled.maintenance_events.iloc[::-1].reset_index(drop=True),
    )
    spec = DatasetSpec(
        dataset_segments=2,
        dataset_months_per_segment=3,
        dataset_observations=6,
    )
    barrier = Barrier(2)
    original = db_etl._insert_or_verify

    def synchronized_insert(*args: object, **kwargs: object) -> tuple[int, int]:
        table = cast(sa.Table, args[1])
        if table.name == "road_segments":
            barrier.wait(timeout=5)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(db_etl, "_insert_or_verify", synchronized_insert)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(load_cleaning_result, pg_engine, result, spec)
            for result in (doubled, reversed_result)
        ]
        reports = [future.result(timeout=10) for future in futures]

    assert sum(report.inserted["road_segments"] for report in reports) == 2
    assert table_counts(pg_engine)["road_observations"] == 6


def test_export_uses_one_repeatable_read_snapshot(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    state = {"loaded": False}

    def load_after_first_select(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if not state["loaded"] and statement.lstrip().upper().startswith("SELECT"):
            state["loaded"] = True
            load_cleaning_result(pg_engine, cleaned_result, dataset_spec)

    event.listen(pg_engine, "after_cursor_execute", load_after_first_select)
    try:
        exported = PostgresRepository(pg_engine).export_dataset()
    finally:
        event.remove(pg_engine, "after_cursor_execute", load_after_first_select)

    assert state["loaded"]
    assert exported.segments.empty
    assert exported.observations.empty
    assert exported.targets.empty
    assert exported.maintenance_events.empty


def test_phase14_combined_export_uses_one_snapshot_across_export_and_history(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
    maintenance_history: pd.DataFrame,
) -> None:
    state = {"loaded": False}

    def load_after_first_select(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if not state["loaded"] and statement.lstrip().upper().startswith("SELECT"):
            state["loaded"] = True
            load_cleaning_result(
                pg_engine,
                cleaned_result,
                dataset_spec,
                maintenance_history=maintenance_history,
            )

    event.listen(pg_engine, "after_cursor_execute", load_after_first_select)
    try:
        exported, history = PostgresRepository(pg_engine).export_material_forecast_inputs()
    finally:
        event.remove(pg_engine, "after_cursor_execute", load_after_first_select)

    assert state["loaded"]
    assert exported.segments.empty
    assert exported.observations.empty
    assert exported.targets.empty
    assert exported.maintenance_events.empty
    assert history.empty


def test_phase14_combined_export_sanitizes_sqlalchemy_failure(
    pg_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = SQLAlchemyError("postgresql://user:credential@host/private")

    def fail_read(*_args: object, **_kwargs: object) -> pd.DataFrame:
        raise original

    monkeypatch.setattr(db_repository, "_read_frame", fail_read)
    with pytest.raises(
        PersistenceError,
        match="^PostgreSQL material forecast input export failed$",
    ) as exc_info:
        PostgresRepository(pg_engine).export_material_forecast_inputs()

    assert str(exc_info.value) == "PostgreSQL material forecast input export failed"
    assert exc_info.value.__cause__ is original


def test_hostile_realized_scalars_and_labels_fail_safely(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
    maintenance_history: pd.DataFrame,
) -> None:
    class HostileInt(int):
        def __int__(self) -> int:
            raise RuntimeError("hostile-int-called")

    class HostileLabel(str):
        def __eq__(self, other: object) -> bool:
            raise RuntimeError("hostile-label-called")

        __hash__ = str.__hash__

    class LiarInt(np.int64):
        __module__ = "numpy.evil"

        def __int__(self) -> int:
            return 999

    class LiarFloat(np.float64):
        __module__ = "numpy.evil"

        def __float__(self) -> float:
            return 999.0

    hostile_value = maintenance_history.copy(deep=True)
    hostile_value["maintenance_cost"] = pd.Series(
        [HostileInt(1_000_000)],
        dtype=object,
    )
    with pytest.raises(RepositoryInputError):
        load_cleaning_result(
            pg_engine,
            cleaned_result,
            dataset_spec,
            maintenance_history=hostile_value,
        )

    hostile_label = maintenance_history.copy(deep=True)
    hostile_label.columns = [
        HostileLabel(column) if column == "maintenance_cost" else column
        for column in hostile_label.columns
    ]
    with pytest.raises(RepositoryInputError):
        load_cleaning_result(
            pg_engine,
            cleaned_result,
            dataset_spec,
            maintenance_history=hostile_label,
        )

    for column, liar in (
        ("maintenance_cost", LiarInt(1)),
        ("thermoplastic_paint_kg", LiarFloat(1.0)),
    ):
        lying_value = maintenance_history.copy(deep=True)
        lying_value[column] = pd.Series([liar], dtype=object)
        with pytest.raises(RepositoryInputError):
            load_cleaning_result(
                pg_engine,
                cleaned_result,
                dataset_spec,
                maintenance_history=lying_value,
            )


def test_non_dataframe_and_hostile_copy_inputs_fail_safely(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
    dataset_spec: DatasetSpec,
) -> None:
    class DerivedFrame(pd.DataFrame):
        pass

    class HostileCopy:
        def copy(self, *, deep: bool) -> object:
            raise RuntimeError(f"hostile-copy-called-{deep}")

    forged_results = (
        replace(cleaned_result, segments=cast(Any, 42)),
        replace(
            cleaned_result,
            observations=cast(Any, DerivedFrame(cleaned_result.observations)),
        ),
        replace(cleaned_result, targets=cast(Any, HostileCopy())),
    )
    for forged in forged_results:
        with pytest.raises(RepositoryInputError):
            load_cleaning_result(pg_engine, forged, dataset_spec)

    with pytest.raises(RepositoryInputError):
        load_cleaning_result(
            pg_engine,
            cleaned_result,
            dataset_spec,
            maintenance_history=cast(Any, HostileCopy()),
        )


def test_model_construct_cannot_bypass_dataset_spec_validation(
    pg_engine: Engine,
    cleaned_result: CleaningResult,
) -> None:
    empty = CleaningResult(
        segments=cleaned_result.segments.iloc[0:0].copy(),
        observations=cleaned_result.observations.iloc[0:0].copy(),
        targets=cleaned_result.targets.iloc[0:0].copy(),
        maintenance_events=cleaned_result.maintenance_events.iloc[0:0].copy(),
        report=ValidationReport(issues=()),
    )
    bypassed = DatasetSpec.model_construct(
        dataset_segments=0,
        dataset_months_per_segment=0,
        dataset_observations=0,
    )

    with pytest.raises(RepositoryInputError, match="validated DatasetSpec"):
        load_cleaning_result(pg_engine, empty, bypassed)

    assert all(count == 0 for count in table_counts(pg_engine).values())
