"""Phase 7 point-in-time feature-frame contract tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import roadguard
from roadguard import (
    DatasetSpec,
    RepositoryExport,
    clean_raw_dataset,
    derive_observation_targets,
    generate_accident_timeline,
    generate_maintenance_events,
    generate_observations,
    generate_segments,
)
from roadguard.features import (
    FEATURE_COLUMNS,
    FEATURE_FRAME_COLUMNS,
    FEATURE_KEY_COLUMNS,
    FEATURE_REGISTRY,
    FeatureInputError,
    build_feature_frame,
)

SPEC = DatasetSpec(dataset_segments=2, dataset_months_per_segment=12, dataset_observations=24)
START = date(2022, 1, 1)


@pytest.fixture
def dataset() -> RepositoryExport:
    segments = generate_segments(SPEC, 42, observation_start=START)
    events = generate_maintenance_events(segments, SPEC, 42, start_date=START)
    timeline = generate_accident_timeline(segments, SPEC, 42, start_date=START)
    observations = generate_observations(segments, events, timeline, SPEC, 42, start_date=START)
    targets = derive_observation_targets(observations, events)
    cleaned = clean_raw_dataset(segments, observations, targets, events, SPEC)
    return RepositoryExport(
        segments=cleaned.segments,
        observations=cleaned.observations,
        targets=cleaned.targets,
        maintenance_events=cleaned.maintenance_events,
    )


def test_registry_has_exact_source_feature_order() -> None:
    assert FEATURE_KEY_COLUMNS == ("segment_id", "date")
    assert tuple(definition.name for definition in FEATURE_REGISTRY) == FEATURE_COLUMNS
    assert FEATURE_FRAME_COLUMNS == FEATURE_KEY_COLUMNS + FEATURE_COLUMNS
    assert FEATURE_COLUMNS == (
        "province",
        "road_type",
        "construction_date",
        "road_length_km",
        "traffic_volume",
        "heavy_vehicle_ratio",
        "road_age_days",
        "rainfall_mm",
        "temperature",
        "humidity",
        "days_since_last_maintenance",
        "previous_repairs",
        "road_condition_score",
        "marking_condition_score",
        "guardrail_condition_score",
        "sign_condition_score",
        "accident_count_30d",
        "accident_count_365d",
    )
    assert all(definition.source_columns == (definition.name,) for definition in FEATURE_REGISTRY)
    assert all(definition.classification_allowed for definition in FEATURE_REGISTRY)
    assert all(definition.regression_allowed for definition in FEATURE_REGISTRY)


def test_feature_builder_is_publicly_exported() -> None:
    assert roadguard.build_feature_frame is build_feature_frame
    assert roadguard.FEATURE_REGISTRY is FEATURE_REGISTRY


def test_build_feature_frame_has_exact_target_free_schema(dataset: RepositoryExport) -> None:
    frame = build_feature_frame(dataset, SPEC)

    assert tuple(frame.columns) == FEATURE_FRAME_COLUMNS
    assert len(frame) == SPEC.dataset_observations
    assert not {
        "days_until_maintenance",
        "maintenance_within_30_days",
        "maintenance_date",
        "maintenance_cost",
        "thermoplastic_paint_kg",
        "traffic_base",
    }.intersection(frame.columns)


def test_build_feature_frame_is_canonical_and_does_not_mutate_inputs(
    dataset: RepositoryExport,
) -> None:
    before_segments = dataset.segments.copy(deep=True)
    before_observations = dataset.observations.copy(deep=True)
    shuffled = replace(
        dataset,
        segments=dataset.segments.sample(frac=1.0, random_state=3).reset_index(drop=True),
        observations=dataset.observations.sample(frac=1.0, random_state=7).reset_index(drop=True),
        targets=dataset.targets.sample(frac=1.0, random_state=11).reset_index(drop=True),
        maintenance_events=dataset.maintenance_events.sample(frac=1.0, random_state=13).reset_index(
            drop=True
        ),
    )

    expected = build_feature_frame(dataset, SPEC)
    actual = build_feature_frame(shuffled, SPEC)

    assert_frame_equal(actual, expected)
    assert_frame_equal(dataset.segments, before_segments)
    assert_frame_equal(dataset.observations, before_observations)


def test_valid_future_event_and_rederived_targets_cannot_change_feature_values(
    dataset: RepositoryExport,
) -> None:
    last_observation_date = dataset.observations["date"].max()
    future = dataset.maintenance_events[
        dataset.maintenance_events["maintenance_date"] > last_observation_date
    ]
    assert not future.empty
    changed_events = dataset.maintenance_events.copy()
    event_index = future.index[0]
    changed_events.loc[event_index, "maintenance_date"] = pd.Timestamp("2099-01-15")
    changed_targets = derive_observation_targets(dataset.observations, changed_events)
    changed = replace(
        dataset,
        targets=changed_targets,
        maintenance_events=changed_events,
    )

    assert_frame_equal(build_feature_frame(changed, SPEC), build_feature_frame(dataset, SPEC))


def test_build_feature_frame_revalidates_forged_export(dataset: RepositoryExport) -> None:
    targets = dataset.targets.copy()
    targets.loc[0, "days_until_maintenance"] += 1
    forged = replace(dataset, targets=targets)

    with pytest.raises(ValueError, match="target_event_inconsistency"):
        build_feature_frame(forged, SPEC)


def test_build_feature_frame_rejects_target_encoded_maintenance_feature(
    dataset: RepositoryExport,
) -> None:
    observations = dataset.observations.copy()
    observations["previous_repairs"] = dataset.targets["maintenance_within_30_days"].astype("int64")
    forged = replace(dataset, observations=observations)

    with pytest.raises(FeatureInputError, match="previous_repairs"):
        build_feature_frame(forged, SPEC)


def test_build_feature_frame_normalizes_phase6_datetime_output(dataset: RepositoryExport) -> None:
    segments = dataset.segments.copy()
    segments["construction_date"] = pd.Series(
        segments["construction_date"].dt.date,
        dtype=object,
    )
    normalized = build_feature_frame(replace(dataset, segments=segments), SPEC)

    assert str(normalized["construction_date"].dtype) == "datetime64[ns]"
    assert_frame_equal(normalized, build_feature_frame(dataset, SPEC))


def test_build_feature_frame_rejects_non_export(dataset: RepositoryExport) -> None:
    with pytest.raises(TypeError, match="RepositoryExport"):
        build_feature_frame(dataset.observations, SPEC)


def test_v1_feature_frame_preserves_all_canonical_rows() -> None:
    v1_spec = DatasetSpec(
        dataset_segments=300,
        dataset_months_per_segment=48,
        dataset_observations=14_400,
    )
    segments = generate_segments(v1_spec, 42, observation_start=START)
    events = generate_maintenance_events(segments, v1_spec, 42, start_date=START)
    timeline = generate_accident_timeline(segments, v1_spec, 42, start_date=START)
    observations = generate_observations(segments, events, timeline, v1_spec, 42, start_date=START)
    targets = derive_observation_targets(observations, events)
    cleaned = clean_raw_dataset(segments, observations, targets, events, v1_spec)
    frame = build_feature_frame(
        RepositoryExport(
            segments=cleaned.segments,
            observations=cleaned.observations,
            targets=cleaned.targets,
            maintenance_events=cleaned.maintenance_events,
        ),
        v1_spec,
    )

    assert len(frame) == 14_400
    assert frame["segment_id"].nunique() == 300
    assert tuple(frame.columns) == FEATURE_FRAME_COLUMNS
